"""
@file   simulators/auto_trans_alarm2.py
@brief  explaining an STL-based monitoring system (Section 5.2)
        G[0,20](vspd<120)

@author Tzu-Yi Chiu <tzuyi.chiu@gmail.com>
"""

import numpy as np

from typing import Tuple

from simulators.auto_trans_alarm1 import AutoTransAlarm1

class AutoTransAlarm2(AutoTransAlarm1):
    """
    Trigger an alarm when G[0,20](vspd<120) is violated. 
    (Section 5.2)
    """

    def simulate(self) -> Tuple[np.ndarray, bool]:
        throttles = list(np.random.uniform(0.7, 1.0, self.slen))
        at = AutoTransAlarm2(self.tdelta, throttles)
        sample = at.run()
        score = int(any(vspd >= 120 for vspd in at.vspds))
        return sample, score

    def plot(self) -> None:
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


# To execute from root: python3 -m simulators.auto_trans_alarm2
if __name__ == '__main__':
    import matplotlib.pyplot as plt
    tdelta = 0.1
    throttles = [0.9]*201

    at = AutoTransAlarm2(tdelta, throttles)
    at.plot()
    plt.show()
    #plt.savefig(f'demo/auto_trans_alarm2.png')
    