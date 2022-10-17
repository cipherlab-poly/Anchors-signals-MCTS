import numpy as np
from simulators.auto_transmission import AutoTransmission

class AutoTransmission3(AutoTransmission):
    def run(self):
        for _ in range(self.slen - 1):
            self.update()
        return np.vstack([self.espds, self.vspds])

    def simulate(self):
        throttles = list(np.random.random(self.slen))
        at = AutoTransmission3(throttles, [0] * self.slen, self.tdelta)
        sample = at.run()
        score = int(any(espd >= 4750 for espd in at.espds[:int(10/self.tdelta)+1]))
        return sample, score

    def plot(self):
        ts = [0]
        for t in range(1, self.slen):
            self.update()
            ts.append(t * self.tdelta)
        
        _, axs = plt.subplots(3)
        axs[0].plot(ts, self.throttles, color='b')
        axs[0].set_ylabel('throttle', color='b')
        axs[0].set_ylim([0, 1])
        axs[0].set_yticks(np.arange(0, 1.2, 0.2))
        axs[0].set_xticklabels([])
        axs[1].plot(ts, self.espds, color='b')
        axs[1].plot(ts, [4750]*len(ts), 'r--')
        axs[1].set_ylabel('engine (rpm)', color='b')
        axs[1].set_yticks(np.arange(0, 6000, 1000))
        axs[1].set_xticklabels([])
        axs[2].plot(ts, self.vspds, color='b')
        axs[2].set_ylabel('speed (mph)', color='b')
        axs[2].yaxis.set_ticks(np.arange(0, 150, 30))
        axs[2].set_xticklabels([])
        axs[2].set_xlabel('time (s)')
        for ax in axs:
            ax.margins(x=0)
            ax.margins(y=0.1)
            ax.grid()

# To execute from root: python3 -m simulators.auto_transmission3
if __name__ == '__main__':
    import matplotlib.pyplot as plt
    tdelta = 0.1
    throttles = ([0.55]*50 + [0.95]*50) * 2 + [0.95]
    #throttles = list(np.linspace(0.6, 0.4, 150))
    #throttles += list(np.linspace(1.0, 0.8, 150))
    thetas = [0.]*len(throttles)

    at = AutoTransmission3(throttles, thetas, tdelta)
    at.plot()
    plt.show()
    #plt.savefig(f'demo/auto_transmission3.png')
    