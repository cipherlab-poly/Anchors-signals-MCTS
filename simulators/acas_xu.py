"""
@file   simulators/acas_xu.py
@brief  explaining an advisory change (Section 6)

@author Tzu-Yi Chiu <tzuyi.chiu@gmail.com>
"""

from simulators.acasxu.nnet import NNet
import os
import numpy as np
np.set_printoptions(precision=2, suppress=True)

from typing import Tuple

import logging
from simulator import Simulator

class ACAS_XU(Simulator):
    """
    Simulate an ACAS Xu system (Section 6).
    This case study aims at explaining the advisory change from 
    Strong Right Turn (SRT) to Weak Right Turn (WRT) for mid-air 
    collision aviodance.
    """
    __nnets = {} # neural networks (to provide advisories), see *load_nnets*

    def __init__(self, state0: np.ndarray, tdelta: float, slen: int) -> None:
        """
        :param state0: initial state
            rho     [0.0, 60760.0]
            theta   [-3.141593, 3.141593]
            psi     [-3.141593, 3.141593]
            v_own   [100.0, 1200.0]
            v_int   [0.0, 1200.0]
        :param tdelta: time between two actions
        :param slen: signal length of the controller
        """
        self.state0 = state0
        self.tdelta = tdelta
        self.slen = slen
        
        rate = 1.5 * np.pi / 180
        self.heading_rate = {0: 0, 1: rate, 2: -rate, 3: 2*rate, 4: -2*rate}
        rho0, theta0, psi0 = state0[:3]
        self.norm_v_own = state0[3]
        self.norm_v_int = state0[4]
        self.x_int = np.array([0., -rho0/2])
        self.v_int = np.array([0., self.norm_v_int])
        angle = theta0 + psi0
        self.x_own = rho0 * np.array([-np.sin(angle), np.cos(angle)])
        self.v_own = self.norm_v_own * np.array([np.sin(psi0), np.cos(psi0)])
        self.a_prev = 4
        self.a_actual = 4

    def load_nnets(self) -> None:
        for a in range(5):
            dirname = os.path.dirname(os.path.abspath(__file__))
            filename = f'ACASXU_experimental_v2a_{a+1}_1.nnet'
            ACAS_XU.__nnets[a] = NNet(os.path.join(dirname, 'acasxu', filename))
    
    def get_rho(self) -> float:
        rho = np.sqrt(self.x_own[0]**2 + self.x_own[1]**2)
        return max(rho, 5)

    def get_theta(self) -> float:
        rho = np.sqrt(self.x_own[0]**2 + self.x_own[1]**2)
        norm_v_own = np.sqrt(self.v_own[0]**2 + self.v_own[1]**2)
        dot = np.dot(self.x_own, self.v_own) / (rho*norm_v_own)
        theta = np.arccos(-dot)
        cross = self.x_own[0] * self.v_own[1] - self.x_own[1] * self.v_own[0]
        if cross < 0:
            theta *= -1
        return (theta + np.pi) % (np.pi*2) - np.pi

    def get_psi(self) -> float:
        norm_v_own = np.sqrt(self.v_own[0]**2 + self.v_own[1]**2)
        norm_v_int = np.sqrt(self.v_int[0]**2 + self.v_int[1]**2)
        dot = np.dot(self.v_own, self.v_int) / (norm_v_own*norm_v_int)
        psi = np.arccos(dot)
        cross = self.v_own[0]*self.v_int[1] - self.v_own[1]*self.v_int[0]
        if cross < 0:
            psi *= -1
        return (psi + np.pi) % (np.pi*2) - np.pi

    def get_inputs(self) -> np.ndarray:
        return np.array([self.get_rho(), self.get_theta(), self.get_psi(), 
            self.norm_v_own, self.norm_v_int])

    def log_state(self) -> None:
        log = f'[{self.get_rho():7.2f} {self.get_theta():5.2f} '
        log += f'{self.get_psi():5.2f}] => {self.str_action(self.a_actual)}'
        logging.info(log)
    
    def str_action(self, action: int) -> str:
        return {0: 'Clear of Conflict',
                1: 'Weak Left turn', 2: 'Weak Right turn',
                3: 'Strong Left turn', 4: 'Strong Right turn'}[action]

    def _control(self) -> None:
        self.a_prev = self.a_actual
        nnet = self.__nnets[self.a_actual]
        outputs = nnet.evaluate_network(self.get_inputs())
        self.a_actual = np.argmin(outputs)
    
    def update(self) -> None:
        self._control()
        rate = self.heading_rate[self.a_actual]
        cos = np.cos(rate * self.tdelta)
        sin = np.sin(rate * self.tdelta)
        self.v_own = np.array([[cos, -sin], [sin, cos]]) @ self.v_own
        self.x_own += (self.v_own - self.v_int) * self.tdelta
        self.x_int += self.v_int * self.tdelta

    def run(self, memory: int = 4) -> np.ndarray:
        samples = []
        while self.a_actual == 4 and len(samples) < self.slen:
            sample = self.get_inputs()[:3]
            samples.append(sample)
            self.update()
        if len(samples) < memory:
            return None
        return np.stack(samples[-memory:], axis=1)
    
    def simulate(self) -> Tuple[np.ndarray, bool]:
        sample = None # doesn't record whole history but only a short memory
        while sample is None:
            random = np.zeros(5, dtype=np.float64)
            random[0] = np.random.uniform(2000, 8000)
            random[1] = np.random.uniform(0, np.pi)
            random[2] = np.random.uniform(-np.pi, 0)
            random[3:] = self.state0[3:]
            acasxu = ACAS_XU(random, self.tdelta, self.slen)
            sample = acasxu.run()
        return sample, acasxu.a_actual != 4

# To execute from root: python3 -m simulators.acas_xu
if __name__ == '__main__':
    state0 = np.array([5000.0, np.pi/4, -np.pi/2, 300.0, 100.0])
    acasxu = ACAS_XU(state0=state0, tdelta=1.0, slen=10)
    acasxu.load_nnets()
    sample = acasxu.run()
    #print(sample)
    #print(acasxu.a_prev, '=>', acasxu.a_actual)
