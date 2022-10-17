import numpy as np
from simulators.auto_transmission3 import AutoTransmission3

class AutoTransmission4(AutoTransmission3):
    def simulate(self):
        throttles = list(np.random.uniform(0.7, 1.0, self.slen))
        at = AutoTransmission4(throttles, [0.] * self.slen, self.tdelta)
        sample = at.run()
        score = int(any(vspd >= 120 for vspd in at.vspds))
        return sample, score

    def plot(self):
        ts = [0]
        for t in range(1, self.slen):
            self.update()
            ts.append(t * self.tdelta)
        
        fig, axs = plt.subplots(3)
        axs[0].plot(ts, self.throttles, color='b')
        axs[0].set_ylabel('throttle', color='b')
        axs[0].set_ylim([0, 1])
        axs[0].set_yticks(np.arange(0, 1.2, 0.2))
        axs[0].set_xticklabels([])
        axs[1].plot(ts, self.espds, color='b')
        axs[1].set_ylabel('engine (rpm)', color='b')
        axs[1].set_yticks(np.arange(0, 6000, 1000))
        axs[1].set_xticklabels([])
        axs[2].plot(ts, self.vspds, color='b')
        axs[2].plot(ts, [120]*len(ts), 'r--')
        axs[2].set_ylabel('speed (mph)', color='b')
        axs[2].yaxis.set_ticks(np.arange(0, 150, 30))
        axs[2].set_xticklabels([])
        axs[2].set_xlabel('time (s)')
        for ax in axs:
            ax.margins(x=0)
            ax.margins(y=0.1)
            ax.grid()


# To execute from root: python3 -m simulators.auto_transmission4
if __name__ == '__main__':
    import matplotlib.pyplot as plt
    tdelta = 0.1
    throttles = [0.9]*201
    thetas = [0.]*len(throttles)

    at = AutoTransmission4(throttles, thetas, tdelta)
    at.plot()
    plt.show()
    #plt.savefig(f'demo/auto_transmission4.png')
    