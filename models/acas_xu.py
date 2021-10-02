from .acasxu.nnet import NNet
from os.path import *
import numpy as np
np.set_printoptions(precision=2, suppress=True)
import logging
from stl import Simulator

"Explain why acas-xu changed from Strong Right turn to Weak Right turn"
class ACAS_XU(Simulator):
    __nnets = {}
    
    """
    Parameters
    ----------
    state0 : array
        initial state
            rho     [0.0, 60760.0]
            theta   [-3.141593, 3.141593]
            psi     [-3.141593, 3.141593]
            v_own   [100.0, 1200.0]
            v_int   [0.0, 1200.0]
    tdelta : float
        time between two actions
    slen : int
        signal length of acas-xu controller
    """
    def __init__(self, state0, tdelta, slen):
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
        self.a_prev = 0
        self.a_actual = 0
        self.sample = np.zeros((3, 0))

    def load_nnets(self):
        for a_prev in range(5):
            filename = dirname(abspath(__file__)) + '/acasxu/ACASXU'
            filename += f'_experimental_v2a_{a_prev+1}_1.nnet'
            ACAS_XU.__nnets[a_prev] = NNet(filename)
    
    def get_rho(self):
        rho = np.sqrt(self.x_own[0]**2 + self.x_own[1]**2)
        return max(rho, 5)

    def get_theta(self):
        rho = np.sqrt(self.x_own[0]**2 + self.x_own[1]**2)
        norm_v_own = np.sqrt(self.v_own[0]**2 + self.v_own[1]**2)
        dot = np.dot(self.x_own, self.v_own) / (rho*norm_v_own)
        theta = np.arccos(-dot)
        cross = self.x_own[0] * self.v_own[1] - self.x_own[1] * self.v_own[0]
        if cross < 0:
            theta *= -1
        return (theta + np.pi) % (np.pi*2) - np.pi

    def get_psi(self):
        norm_v_own = np.sqrt(self.v_own[0]**2 + self.v_own[1]**2)
        norm_v_int = np.sqrt(self.v_int[0]**2 + self.v_int[1]**2)
        dot = np.dot(self.v_own, self.v_int) / (norm_v_own*norm_v_int)
        psi = np.arccos(dot)
        cross = self.v_own[0]*self.v_int[1] - self.v_own[1]*self.v_int[0]
        if cross < 0:
            psi *= -1
        return (psi + np.pi) % (np.pi*2) - np.pi

    def get_inputs(self):
        return np.array([self.get_rho(), self.get_theta(), self.get_psi(), 
            self.norm_v_own, self.norm_v_int])

    def log_state(self):
        log = f'[{self.get_rho():7.2f} {self.get_theta():5.2f} '
        log += f'{self.get_psi():5.2f}] => {self.str_action(self.a_actual)}'
        logging.info(log)
    
    def str_action(self, action):
        return {0: 'Clear of Conflict',
                1: 'Weak Left turn',   2: 'Weak Right turn',
                3: 'Strong Left turn', 4: 'Strong Right turn'}[action]

    def _control(self):
        self.a_prev = self.a_actual
        nnet = self.__nnets[self.a_actual]
        outputs = nnet.evaluate_network(self.get_inputs())
        self.a_actual = np.argmin(outputs)
    
    def update(self, log=False):
        self._control()
        rate = self.heading_rate[self.a_actual]
        cos = np.cos(rate * self.tdelta)
        sin = np.sin(rate * self.tdelta)
        self.v_own = np.array([[cos, -sin], [sin, cos]]) @ self.v_own
        self.x_own += (self.v_own - self.v_int) * self.tdelta
        self.x_int += self.v_int * self.tdelta
        data = self.get_inputs()[:3]
        self.sample = np.hstack((self.sample, data.reshape((3, -1))))
        if log:
            self.log_state()

    def run(self, nb=4):
        for _ in range(self.slen):
            self.update()
        return self.sample[:, -nb:]
    
    def simulate(self, stl):
        sample = None
        while not stl.satisfy(sample):
            random = np.zeros(5, dtype=np.float64)
            uniform = np.random.uniform(-1, 1, 3)
            random[0] = uniform[0] * 3000
            random[1:3] = uniform[1:3] * np.pi/2
            acasxu = ACAS_XU(self.state0 + random, self.tdelta, self.slen)
            sample = acasxu.run()
        return int(acasxu.a_actual == 2)

# To execute from root: python3 -m models.acas_xu
if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, 
        format='%(asctime)s %(levelname)-5s %(message)s',
        datefmt='%H:%M:%S')

    state0 = np.array([5000.0, np.pi/4, -np.pi/2, 300.0, 100.0])
    random = np.random.uniform(-1, 1, 5)
    random[0] = 2000#*= 3000
    random[1:3] *= np.pi/4
    random[3:] *= 0
    acasxu = ACAS_XU(state0 + random, tdelta=1.0, slen=10)
    acasxu.load_nnets()
    sample = acasxu.run()
    logging.info(sample)