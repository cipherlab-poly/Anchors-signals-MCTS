import numpy as np
import itertools
from typing import Iterator

@dataclass
class SignalSpace:
    s: np.ndarray   # signal around which other signals are sampled
    srange: list    # srange[d] = | (0, (min, max, stepsize))     if continuous
                    #             | (1, list of the finite set)   if discrete
    
    def __post_init__(self):
        if len(self.srange) != self.s.shape[0]:
            raise ValueError(f'srange: expected length {self.s.shape[0]}, ' +
                                'got {len(self.srange)}')
    
    def generator(self) -> Iterator[np.ndarray]:
        "Generate signals within the signal space"
        
        l = []
        for d in range(self.s.shape[0]):
            # d-th component is continuous
            if self.srange[d][0] == 0:
                smin = self.srange[d][1][0]
                smax = self.srange[d][1][1]
                num = self.srange[d][1][2] + 1
                l += [np.linspace(smin, smax, num=num)]*self.s.shape[1]
            
            # d-th component is discrete
            elif self.srange[d][0] == 1:
                l += [self.srange[d][1]]*self.s.shape[1]
            
            else:
                raise ValueError(f'{d}-th component continuous or discrete?')
        
        for r in itertools.product(*l):
            yield np.array(r).reshape(self.s.shape)
    
    def sample(self) -> np.ndarray:
        "Simulate a signal by adding some artificially generated noise"
        
        sample = np.zeros(self.s.shape)
        for d in range(self.s.shape[0]):
            # d-th component is continuous
            if self.srange[d][0] == 0:
                smin = self.srange[d][1][0]
                smax = self.srange[d][1][1]
                u = np.random.rand(*self.s.shape) - 0.5
                u *= 2*(smax - smin) / self.s.shape[1]
                sample[d] = self.s[d] + np.cumsum(u[d])
            
            # d-th component is discrete
            elif self.srange[d][0] == 1:
                choice = self.srange[d][1]
                length = len(choice)
                for t in range(self.s.shape[1]):
                    i = choice.index(self.s[d, t])
                    if length > 1:
                        distr = [0.5/(length-1)]*length
                        distr[i] = 0.5
                    else:
                        distr = [1]
                    sample[d, t] = np.random.choice(choice, p=distr)
            else:
                raise ValueError(f'{d}-th component continuous or discrete?')

        return sample

    def sample_with_condition(self, condition: STL) -> np.ndarray:
        "Samples until the condition is satisfied"

        sample = self.sample()
        while not condition.satisfy(sample):
            sample = self.sample()
        return sample
