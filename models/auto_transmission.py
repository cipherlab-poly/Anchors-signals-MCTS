import numpy as np
from joblib import load
import os.path
from stl import Simulator

class AutoTransmission(Simulator):
    def __init__(self, throttles, thetas, tdelta, params={}):
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
            raise ValueError('every throttle should be within [0, 1]')
        if any(theta < 0 or theta > np.pi/2 for theta in thetas):
            raise ValueError('every theta should be within [0, pi/2]')
        
        self.slen = len(throttles)
        self.throttles = throttles  # Throttle: [0, 1]
        self.thetas = thetas        # Road slope (rad): [0, pi/2]
        self.tdelta = tdelta        # Time step
        self.vspd = 0               # Vehicle speed (km/h)
        self.espd = 1000            # Engine speed (rpm)
        self.gear = 0               # Gear: 0, 1, 2, 3, 4
        self.params = params
        self.t = -1
        self.ts = []
        self.espds = []
        self.vspds = []
        self.gears = []

        filename = os.path.dirname(os.path.abspath(__file__)) 
        filename += '/autotransmission/autotransmission.joblib'
        self.regr = load(filename)

    def should_upshift(self):
        """
        Returns
        -------
        bool
            indicating if the gear should be upshifted
        """
        case0 = self.gear == 0 and self.vspd > 0
        case1 = self.gear == 1 and self.vspd > 30
        case2 = self.gear == 2 and self.vspd > 65
        case3 = self.gear == 3 and self.vspd > 100
        return case0 or case1 or case2 or case3

    def should_downshift(self):
        """
        Returns
        -------
        bool
            indicating if the gear should be downshifted
        """
        case4 = self.gear == 4 and self.vspd < 90
        case3 = self.gear == 3 and self.vspd < 55
        case2 = self.gear == 2 and self.vspd < 20
        case1 = self.gear == 1 and self.vspd < 0
        return case1 or case2 or case3 or case4

    def update(self, noise=True, fault=0, v_fault1=None):
        """Updates the state machine. Modified from:
            https://python-control.readthedocs.io/en/0.8.3/cruise-control.html

        Parameters
        ----------
        noise: bool
            add Gaussian random noise to the sensors (vehicle speed, engine speed)
        fault: int
            | 0 if no fault
            | 1 if readings of the speed sensor broken and replaced with
            |      `v_fault1` (a random value within [0, 160]) from a certain time
            | 2 if unable to engage the fourth gear
            | 3 if gear switches directly from second to fourth and vise versa
        v_fault1: float
            speed to be replaced with
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

        self.t += 1
        throttle = self.throttles[self.t]
        ratio = alpha[max(self.gear - 1, 0)]
        omega = ratio * self.vspd / 3.6
        torque = np.clip(Tm * (1 - beta * (omega / omega_m - 1)**2), 0, None)
        F = ratio * torque * throttle
        self.espd = np.clip(omega * 9.55, 500, None)

        # Gravity due to the road slope.
        Fg = m * g * np.sin(self.thetas[self.t])

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
        dv = (F - Fd)/m
        
        self.vspd += dv * self.tdelta
        self.vspd = np.clip(self.vspd, 0, None)

        if fault == 1:
            assert(v_fault1 is not None)
            self.vspd = v_fault1
        
        # Noise
        if noise:
            self.vspd += np.random.normal(0, 0.5)
            self.espd += np.random.normal(0, 15)

        # Engage gear
        if self.should_upshift():
            if fault == 3 and self.gear == 2:
                self.gear = 4
            elif not (fault == 2 and self.gear == 3):
                self.gear += 1
        elif self.should_downshift():
            if fault == 3 and self.gear == 4:
                self.gear = 2
            else:
                self.gear -= 1

    def run(self, noise=True, fault=0):
        """Runs the simulation.
        
        Parameters
        ----------
        noise : bool
            add Gaussian random noise to the sensors (vehicle speed, engine speed)
        fault : int
            | 0 if no fault
            | 1 if readings of the speed sensor is broken from a certain time
            | 2 if unable to engage the fourth gear
            | 3 if gear switches directly from second to fourth and vise versa
        """
        if fault == 1:
            t_fault1 = np.random.randint(self.slen//4, 3*self.slen//4)
            v_fault1 = np.random.random()*100

        for t in range(self.slen):
            self.ts.append(t * self.tdelta)
            self.gears.append(self.gear)
            self.vspds.append(self.vspd)
            self.espds.append(self.espd)
            if fault == 1:
                if t >= t_fault1:
                    self.update(noise, fault=1, v_fault1=v_fault1)
                else:
                    self.update(noise)
            else:
                self.update(noise, fault)
    
    def set_expected_output(self, y):
        super().set_expected_output(y)
    
    def simulate(self, fault=None):
        if fault is None:
            fault = np.random.randint(2)
            if fault:
                fault += np.random.randint(3)
        at = AutoTransmission(self.throttles, self.thetas, self.tdelta)
        if fault not in range(4):
            raise ValueError('wrong fault type')
        
        at.run(fault=fault)
        sample = np.array([at.espds, at.vspds])
        return sample, int(self.regr.predict(sample.reshape((1, -1)))[0])

    def reward(self, output):
        return int(self.expected_output == output)

    def plot(self):
        """Plots the engine and vehicle speed.
        
        Parameters
        ----------
        save : bool
            indicating if we save the plot
        """
        fig, axs = plt.subplots(2)
        axs[0].plot(self.ts, self.espds, color='b')
        axs[0].set_xlabel('time (s)')
        axs[0].set_ylabel('engine speed (rpm)', color='b')
        axs[1].plot(self.ts, self.vspds, color='b')
        axs[1].set_xlabel('time (s)')
        axs[1].set_ylabel('vehicle speed (km/h)', color='b')
        ax2 = axs[0].twinx()  # a second axe that shares the same x-axis
        ax2.set_ylabel('gear', color='r')
        ax2.step(self.ts, self.gears, 'r-', where='post')
        plt.yticks(range(5))
        ax3 = axs[1].twinx()  # a second axe that shares the same x-axis
        ax3.set_ylabel('gear', color='r')
        ax3.step(self.ts, self.gears, 'r-', where='post')
        plt.yticks(range(5))
        for ax in axs.flat:
            ax.label_outer()

# To execute from root: python3 -m models.auto_transmission
if __name__ == '__main__':
    import matplotlib.pyplot as plt
    slen = 15
    tdelta = 1.0
    throttles = [0.5]*slen
    thetas = [0.]*slen
    #for fault in range(3):
    fault = 1
    at = AutoTransmission(throttles, thetas, tdelta)
    at.run(fault=fault)
    at.plot()
    plt.show()
    #plt.savefig(f'demo/at_fault{fault}.png')
    