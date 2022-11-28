"""
@file   simulators/auto_trans_alarm3.py
@brief  explaining an STL-based monitoring system (Section 5.2)
        G[0,30](espd<3000) => G[0,4](vspd<35)

@author Tzu-Yi Chiu <tzuyi.chiu@gmail.com>
"""

import numpy as np

from typing import Tuple

from simulators.auto_trans_alarm1 import AutoTransAlarm1

class AutoTransAlarm3(AutoTransAlarm1):
    """
    Trigger an alarm when G[0,30](espd<3000) => G[0,4](vspd<35) is violated.
    (Section 5.2)
    """

    def simulate(self) -> Tuple[np.ndarray, bool]:
        noise = np.random.uniform(-0.4, 0.4, self.slen)
        throttles = np.clip(np.array(self.throttles) + noise, 0.0, 1.0)
        at = AutoTransAlarm3(self.tdelta, throttles)
        sample = at.run()
        cond1 = all(espd < 3000 for espd in at.espds) 
        cond2 = any(vspd >= 35 for vspd in at.vspds[:int(4/self.tdelta)+1])
        return sample, (cond1 and cond2)

    def plot(self) -> None:
        ts = [0]
        for t in range(1, self.slen):
            self.update()
            ts.append(t * self.tdelta)
        
        _, axs = plt.subplots(3)
        axs[0].plot(ts, self.throttles, 'b')
        axs[0].set_ylabel('throttle', color='b')
        axs[0].set_yticks(np.arange(0, 1.2, 0.2))
        axs[0].set_xticklabels([])
        axs[1].plot(ts, self.espds, 'b')
        axs[1].plot(ts, [3000]*len(ts), 'r--')
        axs[1].set_ylabel('engine (rpm)', color='b')
        axs[1].set_yticks(np.arange(0, 6000, 1000))
        axs[1].set_xticklabels([])
        axs[2].plot(ts, self.vspds, 'b')
        axs[2].plot(ts, [35]*len(ts), 'r--')
        axs[2].plot(ts, [50]*len(ts), 'r--')
        axs[2].plot(ts, [65]*len(ts), 'r--')
        axs[2].set_ylabel('speed (mph)', color='b')
        axs[2].yaxis.set_ticks(np.arange(0, 150, 30))
        axs[2].set_xlabel('time (s)')
        axs[2].set_xticks([4, 5, 8, 10, 15, 20, 25, 30])
        for ax in axs:
            ax.margins(x=0)
            ax.margins(y=0.1)
            ax.grid()


# To execute from root: python3 -m simulators.auto_trans_alarm3
if __name__ == '__main__':
    import matplotlib.pyplot as plt
    tdelta = 2.0
    throttles = list(np.linspace(0.7, 0.4, 6)) + [0.4]*4 + [0.1]*6

    at = AutoTransAlarm3(tdelta, throttles)
    at.plot()
    plt.show()
    #plt.savefig(f'demo/auto_trans_alarm3.png')
    