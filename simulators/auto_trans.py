"""
@file   simulators/auto_trans.py
@brief  explaining the transmission during a passing maneuver (Section 5.3)

@author Tzu-Yi Chiu <tzuyi.chiu@gmail.com>
"""

import numpy as np

from typing import Tuple, Iterable, Dict, Any

from simulator import Simulator

class AutoTrans(Simulator):
    """
    Simulate an automotive automatic transmission system (Section 5.3).
    This case study aims at explaining the down-shifting (gear 4 to 3) 
    during a passing maneuver.
    """

    def __init__(self, tdelta: float, 
                       throttles: Iterable[float], 
                       thetas: Iterable[float] = None, 
                       params: Dict[str, Any] = {}) -> None:
        """
        :param tdelta: duration between two successive timestamps
        :param throttles: throttle opening value at each timestamp (in [0,1])
        :param thetas: road slope at each timestamp (in rad, in [0,pi/2]), default 0
        """
        if any(throttle < 0 or throttle > 1 for throttle in throttles):
            raise ValueError(f'throttles should be within [0, 1], got {throttles}')
        if thetas is None:
            thetas = [0.]*len(throttles)
        elif len(throttles) != len(thetas):
            raise ValueError('throttles and thetas should have same duration')
        elif any(theta < -np.pi/2 or theta > np.pi/2 for theta in thetas):
            raise ValueError(f'thetas should be within [-pi/2, pi/2], got {thetas}')
        
        self.slen = len(throttles)
        self.throts = iter(throttles)   # throttles: [0, 1]
        self.thetas = iter(thetas)      # road slope (rad): [0, pi/2]
        self.tdelta = tdelta            # time step
        self.espd = 0                   # engine speed (rpm)
        self.vspd = 0                   # vehicle speed (km/h)
        self.throttle = 0               # throttle
        self.gear = 1                   # gear: 1, 2, 3, 4
        self.params = params
        self.shifts = { 
            '2-1': (0.5, 0.9, 5, 30),   '1-2': (0.25, 0.9, 10, 40),
            '3-2': (0.05, 0.9, 20, 50), '2-3': (0.35, 0.9, 30, 70),
            '4-3': (0.05, 0.9, 35, 80), '3-4': (0.35, 0.9, 50, 100)
        }
        self.throttles = throttles
        self.espds = []
        self.vspds = []
        self.gears = []

        self.ms2mph = 2.237

    def shift_gear(self) -> None:
        "Inspired from: https://www.mathworks.com/help/simulink/slref/modeling-an-automatic-transmission-controller.html"
        
        def speed(shift):
            x1, x2, y1, y2 = self.shifts[shift]
            if self.throttle <= x1:
                return y1 / self.ms2mph
            if self.throttle >= x2:
                return y2 / self.ms2mph
            return (y1 + (y2 - y1)/(x2 - x1) * (self.throttle - x1)) / self.ms2mph

        def nearest_gear(x):
            return abs(x - self.gear)
        
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

    def update(self) -> None:
        "Updates the state machine. Modified from: https://python-control.readthedocs.io/en/0.8.3/cruise-control.html"
        
        m = self.params.get('m', 1600.)
        g = self.params.get('g', 9.8)
        cr = self.params.get('Cr', 0.01)
        cd = self.params.get('Cd', 0.32)
        rho = self.params.get('rho', 1.3)
        area = self.params.get('A', 2.4)
        alpha = self.params.get(
            'alpha', [40, 25, 16, 12])              # gear ratio / wheel radius
        tm = self.params.get('tm', 350.)            # engine torque constant
        omega_m = self.params.get('omega_m', 420.)  # peak engine angular speed
        beta = self.params.get('beta', 0.4)         # peak engine rolloff

        self.throttle = next(self.throts)
        theta = next(self.thetas)
        ratio = alpha[max(self.gear - 1, 0)]
        omega = ratio * self.vspd
        torque = max(tm * (1 - beta * (omega / omega_m - 1) ** 2), 0)
        force = ratio * torque * self.throttle
        self.espd = omega * 6.65

        # Gravity due to the road slope.
        fg = m * g * np.sin(theta)

        # Rolling friction:
        #   cr:  coefficient of rolling friction
        fr = m * g * cr * np.copysign(1, self.vspd)

        # Aerodynamic drag:
        #   rho: density of air
        #   cd:  shape-dependent aerodynamic drag coefficient
        #   area:   the frontal area of the car
        fa = 0.5 * rho * cd * area * abs(self.vspd) * self.vspd
        
        self.espds.append(self.espd)
        self.vspds.append(self.vspd * self.ms2mph)
        self.gears.append(self.gear)
        
        dv = (force - fg - fr - fa) / m
        self.vspd += dv * self.tdelta
        self.shift_gear()

    def get_sample(self) -> np.ndarray:
        return np.array([self.espd, self.vspd * self.ms2mph, self.throttle])

    def run(self, memory: int = 4) -> np.ndarray:
        samples = []
        samples.append(self.get_sample())
        for _ in range(self.slen - 1):
            self.update()
            samples.append(self.get_sample())
        return np.stack(samples[-memory:], axis=1)

    def simulate(self) -> Tuple[np.ndarray, bool]:
        throttles = list(np.random.random(self.slen))
        at = AutoTrans(self.tdelta, throttles)
        return at.run(), at.gear == 3

    def plot(self) -> None:
        ts = [0]
        for t in range(1, self.slen):
            self.update()
            ts.append(t * self.tdelta)
        
        _, axs = plt.subplots(4)
        axs[0].plot(ts, self.throttles, color='b')
        axs[0].set_ylabel('throttle', color='b')
        axs[0].set_yticks(np.arange(0, 1.2, 0.2))
        axs[0].set_xticklabels([])
        axs[1].plot(ts, self.espds, color='b')
        axs[1].set_ylabel('engine (rpm)', color='b')
        axs[1].set_yticks(np.arange(0, 6000, 1000))
        axs[1].set_xticklabels([])
        axs[2].plot(ts, self.vspds, color='b')
        axs[2].set_xticklabels([])
        axs[2].set_ylabel('speed (mph)', color='b')
        axs[2].yaxis.set_ticks(np.arange(0, 150, 30))
        axs[3].plot(ts, self.gears, color='b')
        axs[3].set_ylabel('gear', color='b')
        axs[3].set_yticks(range(5))
        axs[3].set_xlabel('time (s)')
        axs[3].set_xticks([5, 10, 12, 15, 20, 25])
        for ax in axs:
            ax.margins(x=0)
            ax.margins(y=0.1)
            ax.plot([12, 12], ax.get_ylim(), 'r--')
            ax.grid()

# To execute from root: python3 -m simulators.auto_trans
if __name__ == '__main__':
    import matplotlib.pyplot as plt
    duration = 12
    tdelta = 1.0
    throttles = list(np.linspace(0.6, 0.4, int(duration/tdelta))) + [1.0]

    at = AutoTrans(tdelta, throttles)
    samples = at.run(memory=4)
    #at.plot()
    #plt.show()
    #plt.savefig(f'demo/auto_trans.png')
    