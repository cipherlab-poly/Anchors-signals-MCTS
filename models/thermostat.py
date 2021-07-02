import matplotlib.pyplot as plt
import numpy as np

class Thermostat:
    
    # outside_temp < expected_temp
    def __init__(self, outside_temp, expected_temp, latency):
        self.out_temp = outside_temp  # constant
        self.in_temp  = outside_temp + np.random.random() # at the beginning
        self.exp_temp = expected_temp # constant
        self.on = 0                   # on: 1, off: 0
        self.latency = latency        # on if in_temp below exp_temp for at least *latency*
        self.temps = [self.in_temp]
        self.ons = [0]
        
    # input = self.temps; output = self.on
    # on if in_temp below exp_temp for at least *latency* (to be discovered)
    def black_box(self, temps):
        on = int(len(temps[0]) >= self.latency)
        if on:
            for temp in temps[0, -self.latency:]:
                if temp >= self.exp_temp:
                    on = 0
        return on

    def simulate(self, n):
        for _ in range(n-1):
            if self.on and self.in_temp < self.exp_temp:
                self.in_temp += np.random.random()
            if not self.on and self.in_temp > self.out_temp:
                self.in_temp -= np.random.random()
            
            self.temps.append(self.in_temp)
            
            # on if in_temp below exp_temp for at least *latency*
            self.on = self.black_box(np.array([self.temps]))
            self.ons.append(self.on)

    def __str__(self):
        if self.on:
            status = 'ON'
        else:
            status = 'OFF'
        return '%5.2f'%self.in_temp + ', ' + status

    def plot(self, save=False):
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
    