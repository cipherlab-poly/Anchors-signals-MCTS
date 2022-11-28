"""
@file   simulator.py
@brief  abstract class for simulators

@author Tzu-Yi Chiu <tzuyi.chiu@gmail.com>
"""

class Simulator:
    def simulate(self):
        "Sample a signal and return the signal and the decision (0 or 1)"
        raise NotImplementedError("`simulate` not implemented")
