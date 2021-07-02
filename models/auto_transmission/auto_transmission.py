import logging
import matplotlib.pyplot as plt
import numpy as np
from joblib import load
from os.path import *

class AutoTransmission:
    def __init__(self, throttles, thetas, tdelta=0.1, params={}):
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
        self.throttles = iter(throttles)    # Throttle: [0, 1]
        self.thetas = iter(thetas)          # Road slope (rad): [0, pi/2]
        self.vspd = 0                       # Vehicle speed (km/h)
        self.espd = 1000                    # Engine speed (rpm)
        self.gear = 0                       # Gear: 0, 1, 2, 3, 4
        
        self.tdelta = tdelta
        self.params = params

        self.ts = []
        self.espds = []
        self.vspds = []
        self.gears = []

        filename = dirname(abspath(__file__)) + '/auto_transmission.joblib'
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

    def update(self, noise=True, fault2=False, fault3=False, v_fault1=None):
        """Updates the state machine. Modified from:
            https://python-control.readthedocs.io/en/0.8.3/cruise-control.html

        Parameters
        ----------
        noise : bool
            add Gaussian random noise to the sensors (vehicle speed, engine speed)
        fault1 : float within [0, 1]
            readings of the vehicle speed sensor are substituted with a random 
            value within [0, 160] with this probability
        fault2 : bool
            unable to engage the fourth gear
        fault3 : bool
            gear switches directly from second to fourth and vise versa
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

        throttle = next(self.throttles)
        ratio = alpha[max(self.gear-1, 0)]
        omega = ratio * self.vspd/3.6
        torque = np.clip(Tm * (1 - beta * (omega/omega_m - 1)**2), 0, None)
        F = ratio * torque * throttle
        self.espd = np.clip(omega*9.55, 0, None)

        '''
        gravity according to the road slope
        '''
        Fg = m * g * np.sin(next(self.thetas))

        '''
        Rolling friction
            Cr:  coefficient of rolling friction
        '''
        Fr  = m * g * Cr * np.copysign(1, self.vspd)

        '''
        The aerodynamic drag:
            rho: density of air
            Cd:  shape-dependent aerodynamic drag coefficient
            A:   the frontal area of the car
        '''
        Fa = 1/2 * rho * Cd * A * abs(self.vspd) * self.vspd/12.96
        
        # Total force
        Fd = Fg + Fr + Fa
        
        # Final acceleration on the car
        dv = (F - Fd)/m
        
        self.vspd += dv*self.tdelta
        self.vspd = np.clip(self.vspd, 0, None)

        # Type1 fault
        if v_fault1 is not None:
            self.vspd = v_fault1
        
        # Noise
        if noise:
            self.vspd += np.random.normal(0, 0.5)
            self.espd += np.random.normal(0, 15)

        # Engage gear
        if self.should_upshift():
            if fault3 and self.gear == 2:
                self.gear = 4
            elif not (fault2 and self.gear == 3):
                self.gear += 1
        elif self.should_downshift():
            if fault3 and self.gear == 4:
                self.gear = 2
            else:
                self.gear -= 1

    def log_update(self):
        log = f'vspd {self.vspd:5.1f}  espd {self.espd:6.1f}  gear {self.gear}'
        logging.debug(log)

    def run(self, noise=False, fault1=False, fault2=False, fault3=False, 
            plot=False, save=False):
        """Runs the simulation.
        
        Parameters
        ----------
        noise : bool
            add Gaussian random noise to the sensors (vehicle speed, engine speed)
        fault1 : bool
            readings of the vehicle speed sensor are substituted with a random 
            value within [0, 160] at any time
        fault2 : bool
            unable to engage the fourth gear
        fault3 : bool
            gear switches directly from second to fourth and vise versa
        plot : bool
            indicating if we plot the engine and vehicle speed
        save : bool
            indicating if we save the plot
        """
        if fault1:
            t_fault1 = np.random.randint(self.slen)
            v_fault1 = np.random.random()*160
        
        for t in range(self.slen):
            self.ts.append(t*self.tdelta)
            self.gears.append(self.gear)
            self.vspds.append(self.vspd)
            self.espds.append(self.espd)
            if fault1 and t >= t_fault1:
                self.update(noise, v_fault1=v_fault1)
            else:
                self.update(noise, fault2, fault3)
            if not t%int(1/self.tdelta):
                self.log_update()
        if plot:
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
            if save:
                plt.savefig('demo/auto_transmission.png')
            plt.show()

    def black_box(self, inputs):
        """Predicts the fault type according to engine and vehicle speed signals.

        Parameters
        ----------
        inputs : array of shape (250, 2)
            [[engine speed signal],
             [vehicle speed signal]]
        
        Returns
        -------
        int
            the predicted fault type
        """
        if inputs.size != 60:
            raise ValueError(f'Expected input size 500, got {inputs.size//2}')
        
        new_inputs = inputs.reshape((1, inputs.size))
        return int(self.regr.predict(new_inputs)[0])
