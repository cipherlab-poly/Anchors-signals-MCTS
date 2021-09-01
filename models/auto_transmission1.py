import numpy as np
from joblib import load
import os.path
from stl import Simulator

class AutoTransmission1(Simulator):
    def __init__(self, throttles, thetas, tdelta, gears=None, params={}):
        """
        Parameters
        ----------
        throttles : iterable (e.g. list) of float
            throttle opening value at each timestamp (within [0, 1])
        thetas : iterable (e.g. list) of float
            road slope at each timestamp (in rad, within [0, pi/2])
        tdelta: float
            duration between two successive timestamps
        """
        if len(throttles) != len(thetas):
            raise ValueError('throttles and thetas should have same duration')
        if any(throttle < 0 or throttle > 1 for throttle in throttles):
            raise ValueError(f'throttles should be within [0, 1], got {throttles}')
        if any(theta < -np.pi/2 or theta > np.pi/2 for theta in thetas):
            raise ValueError(f'thetas should be within [-pi/2, pi/2], got {thetas}')
        if gears is not None and len(throttles) != len(gears):
            raise ValueError('gears wrong duration')
        
        self.slen = len(throttles)
        self.throttles = throttles  # Throttle: [0, 1]
        self.thetas = thetas        # Road slope (rad): [0, pi/2]
        self.tdelta = tdelta        # Time step

        self.auto = (gears is None)
        if self.auto:
            self.gears = np.zeros((1, self.slen))
        else:
            self.gears = np.array(gears).reshape((1, -1))
        
        self.vspd = 0               # Vehicle speed (km/h)
        self.espd = 1000            # Engine speed (rpm)
        self.gear = 0               # Gear: 0, 1, 2, 3, 4
        self.params = params
        self.clock = 0
        self.shifts = { 
            '2-1': (0.5, 0.9, 5, 30), '1-2': (0.25, 0.9, 10, 40),
            '3-2': (0.05, 0.9, 20, 50), '2-3': (0.35, 0.9, 30, 70),
            '4-3': (0.05, 0.9, 35, 80), '3-4': (0.35, 0.9, 50, 100)
        }

        self.max_espd = 0.0

    def shift_gear(self):
        """Inspired from:
            https://www.mathworks.com/help/simulink/slref/modeling-an-automatic-transmission-controller.html
        """
        def speed(shift):
            throttle = self.throttles[self.clock]
            x1, x2, y1, y2 = self.shifts[shift]
            if throttle <= x1:
                return y1 * 1.61
            if throttle >= x2:
                return y2 * 1.61
            return (y1 + (y2 - y1)/(x2 - x1) * (throttle - x1)) * 1.61

        def nearest_gear(x):
            return abs(x - self.gear)
        
        if self.auto:
            shift = min(self.shifts, key=lambda shift: abs(speed(shift) - self.vspd))
            if speed(shift) >= self.vspd:
                if shift == '2-1':
                    self.gear = 1
                elif shift == '1-2':
                    self.gear = min([1, 2], key=nearest_gear)
                elif shift == '3-2':
                    self.gear = 2
                elif shift == '2-3':
                    self.gear = min([2, 3], key=nearest_gear)
                elif shift == '4-3':
                    self.gear = 3
                elif shift == '3-4':
                    self.gear = min([3, 4], key=nearest_gear)
            elif speed(shift) <= self.vspd:
                if shift == '2-1':
                    self.gear = min([1, 2], key=nearest_gear)
                elif shift == '1-2':
                    self.gear = 2
                elif shift == '3-2':
                    self.gear = min([2, 3], key=nearest_gear)
                elif shift == '2-3':
                    self.gear = 3
                elif shift == '4-3':
                    self.gear = min([3, 4], key=nearest_gear)
                elif shift == '3-4':
                    self.gear = 4
            self.gears[0, self.clock] = self.gear
        else:
            self.gear = self.gears[0, self.clock]

    def update(self):
        """Updates the state machine. Modified from:
            https://python-control.readthedocs.io/en/0.8.3/cruise-control.html

        Parameters
        ----------
        noise: bool
            add Gaussian random noise to the sensors (vehicle speed, engine speed)
        """
        m = self.params.get('m', 1600.)
        g = self.params.get('g', 9.8)
        Cr = self.params.get('Cr', 0.01)
        Cd = self.params.get('Cd', 0.32)
        rho = self.params.get('rho', 1.3)
        A = self.params.get('A', 2.4)
        alpha = self.params.get(
            'alpha', [40, 25, 16, 12])              # gear ratio / wheel radius
        Tm = self.params.get('Tm', 1400.)           # engine torque constant
        omega_m = self.params.get('omega_m', 420.)  # peak engine angular speed
        beta = self.params.get('beta', 0.4)         # peak engine rolloff

        throttle = self.throttles[self.clock]
        theta = self.thetas[self.clock]
        ratio = alpha[max(self.gear - 1, 0)]
        omega = ratio * self.vspd / 3.6
        torque = max(Tm * (1 - beta * (omega / omega_m - 1)**2), 0)
        F = ratio * torque * throttle
        self.espd = max(omega * 9.55, 500.0)
        self.max_espd = max(self.max_espd, self.espd)

        # Gravity due to the road slope.
        Fg = m * g * np.sin(theta)

        # Rolling friction:
        #   Cr:  coefficient of rolling friction
        Fr  = m * g * Cr * np.copysign(1, self.vspd)

        # Aerodynamic drag:
        #   rho: density of air
        #   Cd:  shape-dependent aerodynamic drag coefficient
        #   A:   the frontal area of the car
        Fa = 0.5 * rho * Cd * A * abs(self.vspd) * self.vspd / 12.96
        
        # Total force
        Fd = Fg + Fr + Fa
        
        # Final acceleration on the car
        dv = (F - Fd) / m
        
        self.vspd += dv * self.tdelta
        self.vspd = max(self.vspd, 0)
        self.shift_gear()
        self.clock += 1

    def run(self):
        for _ in range(self.slen):
            self.update()
        
    def simulate(self, stl):
        params = stl.get_params()
        gears = None
        while not stl.satisfy(gears):
            gears = - np.ones((1, self.slen), dtype=np.int8)
            for param in params:
                typ, a, b, i, comp, mu = param
                if comp == '=':
                    b1 = b + 1 if b != -1 else self.slen
                    if typ == 'G':
                        gears[0, a:b1] = mu
                    else:
                        lucky = np.random.choice(range(a, b + 1))
                        gears[0, lucky] = mu
            
            if gears[0, 0] == -1:
                gears[0, 0] = 1
            for t in range(1, self.slen):
                if gears[0, t] == -1:
                    if gears[0, t-1] == 1:
                        gears[0, t] = np.random.choice([1, 2])
                    elif gears[0, t-1] == 4:
                        gears[0, t] = np.random.choice([3, 4])
                    else:
                        gears[0, t] = gears[0, t-1] + np.random.randint(-1, 2)

        at = AutoTransmission1(self.throttles, self.thetas, self.tdelta, 
                                gears=gears[0])
        at.run()
        return int(at.max_espd < 5000.0)

    def plot(self):
        ts = []
        espds = []
        vspds = []
        gears = []
    
        for t in range(self.slen):
            ts.append(t * self.tdelta)
            gears.append(self.gear)
            vspds.append(self.vspd)
            espds.append(self.espd)
            self.update()
        
        fig, axs = plt.subplots(2)
        axs[0].plot(ts, espds, color='b')
        axs[0].set_xlabel('time (s)')
        axs[0].set_ylabel('engine speed (rpm)', color='b')
        axs[1].plot(ts, vspds, color='b')
        axs[1].set_xlabel('time (s)')
        axs[1].set_ylabel('vehicle speed (km/h)', color='b')
        ax2 = axs[0].twinx()  # a second axe that shares the same x-axis
        ax2.set_ylabel('gear', color='r')
        ax2.step(ts, gears, 'r-', where='post')
        plt.yticks(range(5))
        ax3 = axs[1].twinx()  # a second axe that shares the same x-axis
        ax3.set_ylabel('gear', color='r')
        ax3.step(ts, gears, 'r-', where='post')
        plt.yticks(range(5))
        for ax in axs.flat:
            ax.label_outer()


# To execute from root: python3 -m models.auto_transmission1
if __name__ == '__main__':
    import matplotlib.pyplot as plt
    slen = 24
    tdelta = 0.5
    throttles = [0.5]*slen
    thetas = [0.]*slen

    gears = [1]
    for _ in range(1, slen):
        if gears[-1] == 1:
            gears.append(np.random.choice([1, 2]))
        elif gears[-1] == 4:
            gears.append(np.random.choice([3, 4]))
        else:
            gears.append(gears[-1] + np.random.randint(-1, 2))

    at = AutoTransmission1(throttles, thetas, tdelta, gears=gears)
    at.plot()
    print(at.gears)
    plt.show()
    #plt.savefig(f'demo/auto_transmission1.png')
    