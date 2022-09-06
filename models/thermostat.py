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
        return np.array([self.temps])
    
    def _black_box(self, sample):
        "on if `in_temp` below `exp_temp` for at least `latency`"
        on = int(len(sample[0]) >= self.latency)
        if on:
            for temp in sample[0, -self.latency:]:
                if temp >= self.exp_temp:
                    on = 0
        return on
    
    def simulate(self):
        tm = Thermostat(self.out_temp, self.exp_temp, self.latency, self.length)
        sample = tm.run()
        return sample[0, -2:].reshape(1, -1), tm.on == 0
        
    def plot(self):
        import matplotlib.pyplot as plt
        t = range(len(self.temps))
        _, ax1 = plt.subplots()
        ax1.set_xlabel('time')
        ax1.set_ylabel('temperature')
        ax1.plot(t, self.temps)

        ax2 = ax1.twinx()  # a second axis that shares the same x-axis
        ax2.set_ylabel('activation')
        ax2.step(t, self.ons, 'r-', where='post')
        plt.yticks([0, 1])
        plt.show()
    