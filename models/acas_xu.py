from .nnet import NNet
from os.path import *
import numpy as np
np.set_printoptions(precision=2, suppress=True)
import logging

import matplotlib.pyplot as plt
import matplotlib.animation as animation

class ACASXU:
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
    def __init__(self, inputs0, slen, fault_start=None, 
            tdelta=0.1, noise=True, plot=False, save=False):
        self.tau = inputs0[5]
        self.a_prev = inputs0[6]
        if self.a_prev not in range(5):
            raise ValueError(f'a_prev should be in {range(5)}')
        try:
            taus = [0, 1, 5, 10, 20, 40, 60, 80, 100]
            self.index_tau = taus.index(self.tau) + 1
        except ValueError:
            raise ValueError(f'tau should be in {taus}')
        rate = 1.5*np.pi/180
        self.heading_rate = {0: 0, 1: rate, 2: -rate, 3: 2*rate, 4: -2*rate}
        self.tdelta = tdelta
        rho0 = inputs0[0]
        theta0 = inputs0[1]
        psi0 = inputs0[2]
        self.norm_v_own = inputs0[3]
        self.norm_v_int = inputs0[4]
        self.x_int = np.array([0., -rho0/2])
        self.v_int = np.array([0., self.norm_v_int])
        self.x_own = rho0*np.array([np.sin(theta0-psi0), np.cos(theta0-psi0)])
        self.v_own = self.norm_v_own*np.array([np.sin(psi0), np.cos(psi0)])
        self.a_actual = self.a_prev
        self.signals = None
        self.noise = noise
        if fault_start is None:
            self.fault_start = slen
        else:
            self.fault_start = fault_start
        self.slen = slen
        self.clock = 0

        self.nnets = {}
        for a_prev in range(5):
            filename = dirname(abspath(__file__)) + 'acasxu/ACASXU'
            filename += f'_experimental_v2a_{a_prev+1}_{self.index_tau}.nnet'
            self.nnets[a_prev] = NNet(filename)
        
        self.plot = plot
        self.save = save
        if plot:
            self.colors = {
                0: 'green',
                1: 'orange',
                2: 'orange',
                3: 'red',
                4: 'red'
            }
            self.fig = plt.figure()
            lim = rho0*2
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
        if self.clock >= self.fault_start:
            return 10000
        rho = np.sqrt(self.x_own[0]**2+self.x_own[1]**2)
        if self.noise:
            rho += np.random.normal(0, 5)
        return max(rho, 5)

    def get_theta(self):
        rho = np.sqrt(self.x_own[0]**2+self.x_own[1]**2)
        norm_v_own = np.sqrt(self.v_own[0]**2+self.v_own[1]**2)
        dot = np.dot(self.x_own, self.v_own)/(rho*norm_v_own)
        theta = np.arccos(-dot)
        cross = self.x_own[0]*self.v_own[1] - self.x_own[1]*self.v_own[0]
        if cross < 0:
            theta *= -1
        if self.noise:
            theta += np.random.normal(0, 0.05)
        return (theta + np.pi) % (np.pi*2) - np.pi

    def get_psi(self):
        norm_v_own = np.sqrt(self.v_own[0]**2+self.v_own[1]**2)
        norm_v_int = np.sqrt(self.v_int[0]**2+self.v_int[1]**2)
        dot = np.dot(self.v_own, self.v_int)/(norm_v_own*norm_v_int)
        psi = np.arccos(dot)
        cross = self.v_own[0]*self.v_int[1] - self.v_own[1]*self.v_int[0]
        if cross < 0:
            psi *= -1
        if self.noise:
            psi += np.random.normal(0, 0.05)
        return (psi + np.pi) % (np.pi*2) - np.pi

    def get_inputs(self):
        return np.array([self.get_rho(), self.get_theta(), self.get_psi(), 
            self.norm_v_own, self.norm_v_int, self.tau, self.a_prev])

    def log_state(self):
        log = f'{self.clock:3d} [{self.get_rho():7.2f} {self.get_theta():5.2f} '
        log += f'{self.get_psi():5.2f}] => {self.str_action(self.a_actual)}'
        logging.info(log)
    
    def str_action(self, action):
        actions = {
            0: 'Clear of Conflict',
            1: 'Weak Left turn',
            2: 'Weak Right turn',
            3: 'Strong Left turn',
            4: 'Strong Right turn'
        }
        return actions[action]

    def evaluate(self):
        self.a_prev = self.a_actual
        nnet = self.nnets[self.a_actual]
        outputs = nnet.evaluate_network(self.get_inputs()[:5])
        self.a_actual = np.argmin(outputs)
    
    def update(self):
        self.evaluate()
        rate = self.heading_rate[self.a_actual]
        cos = np.cos(rate*self.tdelta)
        sin = np.sin(rate*self.tdelta)
        rot_matrix = np.array([[cos, -sin], [sin, cos]])
        self.v_own = np.matmul(rot_matrix, self.v_own)
        self.x_own += (self.v_own - self.v_int) * self.tdelta
        self.x_int += self.v_int*self.tdelta
        self.clock += 1

    def _draw_init(self):
        self.lines_own[-1].set_data(self.xs_own[-1], self.ys_own[-1])
        self.line_int.set_data(self.xs_int, self.ys_int)
        return *self.lines_own, self.line_int
    
    def _animate(self, t):
        if self.clock >= self.slen:
            self.anim.event_source.stop()
            self.anim_stop = True
        self.update()
        self.log_state()
        
        if self.a_actual != self.a_prev:
            self.xs_own.append([])
            self.ys_own.append([])
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

    def run(self):
        """Runs the simulation.
        
        Parameters
        ----------
        slen : int
            duration (signal length)

        Returns
        -------
        int
            a_prev just before memory
        """
        self.log_state()
        if self.plot:
            self.fig.canvas.mpl_connect('button_press_event', self._on_click)
            self.anim = animation.FuncAnimation(self.fig, self._animate, 
                init_func=self._draw_init, interval=10, blit=True)
            if self.save:
                writer = animation.PillowWriter(fps=24)
                filename = dirname(dirname(dirname(abspath(__file__))))
                filename += '/pics/ACAS-XU2.gif'
                self.anim.save(filename, writer=animation.PillowWriter(fps=75))
            plt.show()
        else:
            self.signals = np.zeros((7, 2*(self.slen-self.fault_start)))
            while self.clock < self.slen:
                self.update()
                self.log_state()
                if self.clock > 2*self.fault_start-self.slen:
                    self.signals[:, self.clock-2*self.fault_start+self.slen-1] = self.get_inputs()
        
    def black_box(self, inputs):
        """Complete a forward pass (only based on the last timestamp).

        Parameters
        ----------
        inputs : array
            signal
        
        Returns
        -------
        int
            final action
        """
        a_prev = inputs[-1, -1]
        nnet = self.nnets[a_prev]
        outputs = nnet.evaluate_network(inputs[:5, -1])
        action = np.argmin(outputs)
        return action
