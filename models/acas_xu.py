from .acasxu.nnet import NNet
from os.path import *
import numpy as np
np.set_printoptions(precision=2, suppress=True)
import logging
from stl import Simulator

import matplotlib.pyplot as plt
import matplotlib.animation as animation

class ACAS_XU(Simulator):
    """
    Parameters
    ----------
    inputs0 : array
        initial inputs
            rho     [0.0, 60760.0]
            theta   [-3.141593, 3.141593]
            psi     [-3.141593, 3.141593]
            v_own   [100.0, 1200.0]
            v_int   [0.0, 1200.0]
            tau     should be one of [0, 1, 5, 10, 20, 40, 60, 80, 100]
            a_prev  should be one of [0, 1, 2, 3, 4]
    tdelta : float
        time between two actions
    plot : bool
        for animation
    save : bool
        indicating if we save the animation as gif
    """
    def __init__(self, inputs0, slen, tdelta, control='random', plot=False, save=False):
        self.inputs0 = inputs0
        self.slen = slen
        self.tdelta = tdelta
        
        rate = 1.5*np.pi/180
        self.heading_rate = {0: 0, 1: rate, 2: -rate, 3: 2*rate, 4: -2*rate}
        rho0, theta0, psi0 = inputs0[:3]
        self.norm_v_own = inputs0[3]
        self.norm_v_int = inputs0[4]
        self.x_int = np.array([0., -rho0/2])
        self.v_int = np.array([0., self.norm_v_int])
        self.x_own = rho0 * np.array([np.sin(theta0-psi0), np.cos(theta0-psi0)])
        self.v_own = self.norm_v_own * np.array([np.sin(psi0), np.cos(psi0)])
        self.a_prev = 0
        self.a_actual = 0
        self.clock = 0

        self.min_dist = inputs0[0]
        self.controls = np.zeros((1, slen))
        
        # control with acas-xu neural network
        self.control = control
        if control == 'acas-xu':
            self.nnets = {}
            for a_prev in range(5):
                filename = dirname(abspath(__file__)) + '/acasxu/ACASXU'
                filename += f'_experimental_v2a_{a_prev+1}_1.nnet'
                self.nnets[a_prev] = NNet(filename)
        
        # control randomly
        else:
            self.a_next = {
                0: [0, 1, 2, 3, 4],
                1: [0, 1, 3],
                2: [0, 2, 4],
                3: [1, 3],
                4: [2, 4]
            }
        
        self.plot = plot
        self.save = save
        if plot:
            self.colors = {
                0: 'green',
                1: 'yellow', 2: 'yellow',
                3: 'red',    4: 'red'
            }
            self.fig = plt.figure()
            lim = rho0 * 2
            self.ax = plt.axes(xlim=(-lim, lim), ylim=(-lim, lim)) 
            self.lines_own = [self.ax.plot([], [], lw=2, 
                color=self.colors[self.a_actual])[0]]
            self.line_int = self.ax.plot([], [], lw=2, color='blue')[0]
            self.xs_own = [[self.x_int[0] + self.x_own[0]]]
            self.ys_own = [[self.x_int[1] + self.x_own[1]]]
            self.xs_int = [self.x_int[0]]
            self.ys_int = [self.x_int[1]]
            self.anim_stop = False

    def get_rho(self):
        rho = np.sqrt(self.x_own[0]**2 + self.x_own[1]**2)
        return max(rho, 5)

    def get_theta(self):
        rho = np.sqrt(self.x_own[0]**2 + self.x_own[1]**2)
        norm_v_own = np.sqrt(self.v_own[0]**2 + self.v_own[1]**2)
        dot = np.dot(self.x_own, self.v_own)/(rho*norm_v_own)
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
        log = f'{self.clock:3d} [{self.get_rho():7.2f} {self.get_theta():5.2f} '
        log += f'{self.get_psi():5.2f}] => {self.str_action(self.a_actual)}'
        logging.info(log)
    
    def str_action(self, action):
        return {0: 'Clear of Conflict',
                1: 'Weak Left turn',   2: 'Weak Right turn',
                3: 'Strong Left turn', 4: 'Strong Right turn'}[action]

    def evaluate(self):
        self.a_prev = self.a_actual
        if self.control == 'acas-xu':
            nnet = self.nnets[self.a_actual]
            outputs = nnet.evaluate_network(self.get_inputs())
            self.a_actual = np.argmin(outputs)
        else:
            self.a_actual = np.random.choice(self.a_next[self.a_actual])
    
    def update(self):
        self.evaluate()
        rate = self.heading_rate[self.a_actual]
        cos = np.cos(rate * self.tdelta)
        sin = np.sin(rate * self.tdelta)
        rot_matrix = np.array([[cos, -sin], [sin, cos]])
        self.v_own = rot_matrix @ self.v_own
        self.x_own += (self.v_own - self.v_int) * self.tdelta
        self.x_int += self.v_int * self.tdelta
        self.clock += 1

    def _draw_init(self):
        self.lines_own[-1].set_data(self.xs_own[-1], self.ys_own[-1])
        self.line_int.set_data(self.xs_int, self.ys_int)
        return *self.lines_own, self.line_int
    
    def _animate(self, t):
        if self.clock >= self.slen:
            self.anim.event_source.stop()
            self.anim_stop = True
        
        xs_own = self.x_int[0] + self.x_own[0]
        ys_own = self.x_int[1] + self.x_own[1]
        self.update()
        
        if self.a_actual != self.a_prev:
            self.xs_own.append([xs_own])
            self.ys_own.append([ys_own])
            new_line, = self.ax.plot([], [], lw=2, color=self.colors[self.a_actual])
            self.lines_own.append(new_line)
        
        self.xs_own[-1].append(self.x_int[0] + self.x_own[0])
        self.ys_own[-1].append(self.x_int[1] + self.x_own[1])
        self.xs_int.append(self.x_int[0])
        self.ys_int.append(self.x_int[1])
        self.lines_own[-1].set_data(self.xs_own[-1], self.ys_own[-1])
        self.line_int.set_data(self.xs_int, self.ys_int)
        return *self.lines_own, self.line_int

    def _on_click(self, event):
        if self.anim_stop:
            self.anim.event_source.start()
            self.anim_stop = False
        else:
            self.anim.event_source.stop()
            self.anim_stop = True

    def run(self, log=False):
        "Runs the simulation."
        if self.plot:
            self.fig.canvas.mpl_connect('button_press_event', self._on_click)
            self.anim = animation.FuncAnimation(self.fig, self._animate, 
                init_func=self._draw_init, interval=10, blit=True)
            if self.save:
                writer = animation.PillowWriter(fps=24)
                filename = dirname(dirname(dirname(abspath(__file__))))
                filename += '/pics/ACAS-XU.gif'
                self.anim.save(filename, writer=animation.PillowWriter(fps=75))
            plt.show()
        else:
            for t in range(self.slen):
                self.update()
                if log:
                    self.log_state()
                self.controls[0, t] = self.a_actual
                self.min_dist = min(self.min_dist, self.get_rho())
    
    def simulate(self):
        acasxu = ACAS_XU(self.inputs0, self.slen, self.tdelta)
        acasxu.run()
        return acasxu.controls, int(acasxu.min_dist > 3000.0)

# To execute from root: python3 -m models.acas_xu
if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, 
        format='%(asctime)s %(levelname)-5s %(message)s',
        datefmt='%H:%M:%S')

    inputs0 = np.array([5000.0, np.pi*1.75, -np.pi/2, 300.0, 100.0])
    acasxu = ACAS_XU(inputs0, slen=20, tdelta=1.0, control='random')
    acasxu.run(log=True)
