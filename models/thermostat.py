import numpy as np
from stl import Simulator

class Thermostat(Simulator):
    # outside_temp < expected_temp
    def __init__(self, out_temp, exp_temp, latency, length):
        self.out_temp   = out_temp
        self.in_temp    = out_temp + np.random.random()
        self.exp_temp   = exp_temp
        self.on         = 0
        self.latency    = latency
        self.temps      = [self.in_temp]
        self.length     = length
        self.ons        = [0]
        
    def run(self):
        for _ in range(self.length - 1):
            if self.on and self.in_temp < self.exp_temp:
                self.in_temp += np.random.random()
            if not self.on and self.in_temp > self.out_temp:
                self.in_temp -= np.random.random()
            
            self.temps.append(self.in_temp)
            
            # on if in_temp below exp_temp for at least *latency*
            self.on = self._black_box(np.array([self.temps]))
            self.ons.append(self.on)
    
    def _black_box(self, sample):
        "on if `in_temp` below `exp_temp` for at least `latency`"
        on = int(len(sample[0]) >= self.latency)
        if on:
            for temp in sample[0, -self.latency:]:
                if temp >= self.exp_temp:
                    on = 0
        return on
    
    def set_expected_output(self, y):
        super().set_expected_output(y)
    
    def simulate(self):
        tm = Thermostat(self.out_temp, self.exp_temp, self.latency, self.length)
        tm.run()
        return np.array([tm.temps]), tm.on

    def reward(self, output):
        return int(self.expected_output == output)
        
    def plot(self, save=False):
        import matplotlib.pyplot as plt
        t = range(len(self.temps))
        fig, ax1 = plt.subplots()
        ax1.set_xlabel('time')
        ax1.set_ylabel('temperature')
        ax1.plot(t, self.temps)

        ax2 = ax1.twinx()  # a second axe that shares the same x-axis
        ax2.set_ylabel('activation')
        ax2.step(t, self.ons, 'r-', where='post')
        plt.yticks([0, 1])
        if save:
            plt.savefig('demo/thermostat.png')
        plt.show()
    