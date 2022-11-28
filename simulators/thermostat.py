"""
@file   simulators/thermostat.py
@brief  explaining why the automated thermostat is switched off

@author Tzu-Yi Chiu <tzuyi.chiu@gmail.com>
"""

import numpy as np

from typing import Tuple

from simulator import Simulator

class Thermostat(Simulator):
    """
    Simulate an intelligent thermostat (Section 4.3).
    Set-up: outside temperature < expected temperature
            thermostat is switched off at the beginning
    This case study aims at explaining why the thermostat is switched off.
    Real explanation: temperature > 20 once within the past two seconds.
    """
    def __init__(self, out_temp: float, 
                       exp_temp: float, 
                       latency: int, 
                       length: int,
                       memory: int = None) -> None:
        """
        :param out_temp: outside temperature
        :param exp_temp: expected temparature
        :param latency: reponse time, thermostat is on only if 
                        in_temp < exp_temp for at least *latency*
        :param length: running time length
        :param memory: sample signal length
        """
        self.out_temp = out_temp
        self.in_temp = out_temp + np.random.random()
        self.exp_temp = exp_temp
        self.on = 0 # thermostat is switched off at the beginning
        self.latency = latency
        self.temps = [self.in_temp]
        self.length = length
        self.memory = length if memory is None else memory
        self.ons = [0]
        
    def run(self):
        for _ in range(self.length - 1):
            if self.on and self.in_temp < self.exp_temp:
                self.in_temp += np.random.random()
            if not self.on and self.in_temp > self.out_temp:
                self.in_temp -= np.random.random()
            
            self.temps.append(self.in_temp)
            self.on = self._black_box(np.array([self.temps]))
            self.ons.append(self.on)
        return np.array([self.temps])
    
    def _black_box(self, sample):
        "thermostat is on only if in_temp < exp_temp for at least `latency`"
        on = int(len(sample[0]) >= self.latency)
        if on:
            for temp in sample[0, -self.latency:]:
                if temp >= self.exp_temp:
                    on = 0
        return on
    
    def simulate(self) -> Tuple[np.ndarray, bool]:
        tm = Thermostat(self.out_temp, self.exp_temp, self.latency, 
                        self.length, self.memory)
        sample = tm.run()
        return sample[0, -self.memory:].reshape(1, -1), tm.on == 0
        
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
    