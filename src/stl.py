from __future__ import annotations

import numpy as np
import itertools
from typing import List, FrozenSet, Callable
from dataclasses import dataclass

from .mcts import Node

@dataclass
class Primitive:
    "Ex: Primitive('G', 0, 5, '>', 20) <=> G[0,5](s_i > 20)"
    
    typ: str            # 'F'(eventually) or 'G'(always)
    a: int              # lower bound delay
    b: int              # upper bound delay
    i: int              # component index
    comp: str           # '<' or '>' (continuous) or '=' (discrete)
    mu: float           # constant threshold
    normalize: float    # if continuous, max - min of mu
                        # if discrete, 1 (to normalize robustness degree)
    
    def __post_init__(self):
        if self.typ not in ['F', 'G']:
            raise ValueError('Invalid basic STL type')
        elif self.a < 0 or self.b < 0:
            raise ValueError('Invalid delay interval (negative bounds)')
        elif self.a > self.b:
            raise ValueError('Invalid delay interval (wrong order)') 
        elif self.comp not in ['<', '>', '=']:
            raise ValueError('Invalid comparison')

    def robust(self, s: np.ndarray) -> float:
        "Compute the robustness degree relative to signal `s`"
        
        if self.typ == 'F':
            if self.comp == '<':
                res = self.mu - np.min(s[self.i, self.a:self.b+1])
                return res/self.normalize
            elif self.comp == '>':
                res = np.max(s[self.i, self.a:self.b+1]) - self.mu
                return res/self.normalize
            elif self.mu in s[self.i, self.a:self.b+1]:
                return 1
            else:
                return -1
        else:
            if self.comp == '<':
                res = self.mu - np.max(s[self.i, self.a:self.b+1])
                return res/self.normalize
            elif self.comp == '>':
                res = np.min(s[self.i, self.a:self.b+1]) - self.mu
                return res/self.normalize
            elif all(v == self.mu for v in s[self.i, self.a:self.b+1]):
                return 1
            else:
                return -1

    def satisfy(self, s: np.ndarray) -> bool:
        "Verify if satisfied by signal `s`"
        return self.robust(s) > 0

    def __repr__(self):
        return f'{self.typ}[{self.a},{self.b}](s{self.i+1}{self.comp}{self.mu:.2f})'


@dataclass
class Generator:
    "The static generator of primitives and signals."
    
    # (instance attributes)
    _s: np.ndarray  # signal being explained
    _srange: list   # list of tuples
                    # srange[d] = | (0, (min, max, stepsize))     if continuous
                    #             | (1, list of the finite set)   if discrete
    _rho: float     # robustness degree (~coverage) threshold
        
    def generate_primitives(self) -> List[Primitive]:
        "Generate STL primitives whose robustness is greater than `rho`"
            
        result = []
        sdim, slen = self._s.shape
        for d in range(sdim):
            # d-th component is continuous
            if self._srange[d][0] == 0:
                smin, smax, stepsize = self._srange[d][1]
                mus = np.linspace(smin, smax, num=stepsize, endpoint=False)[1:]
                normalize = smax - smin
                for i in range(slen):
                    l = [['F', 'G'], [i], range(i, slen), [d], ['>', '<']]
                    for r in itertools.product(*l):
                        stop = False
                        phi0 = Primitive(*r, mus[0], normalize)
                        phi1 = Primitive(*r, mus[-1], normalize)
                        if phi0.robust(self._s) >= self._rho:
                            u = 0
                            if phi1.robust(self._s) < self._rho:
                                l = len(mus) - 1
                                from_beginning = True
                            else:
                                stop = True
                                from_beginning = False
                        else:
                            from_beginning = False
                            l = 0
                            if phi1.robust(self._s) >= self._rho:
                                u = len(mus) - 1
                            else:
                                u = len(mus)
                                stop = True
                        
                        if not stop:
                            while True:
                                phi0 = Primitive(*r, mus[l], normalize)
                                phi1 = Primitive(*r, mus[u], normalize)
                                if (phi0.robust(self._s) >= self._rho and 
                                        phi1.robust(self._s) >= self._rho):
                                    break
                                elif (phi0.robust(self._s) < self._rho and 
                                        phi1.robust(self._s) < self._rho):
                                    break
                                q = (u + l) // 2
                                if u == q or l == q:
                                    break
                                phi2 = Primitive(*r, mus[q], normalize)
                                if phi2.robust(self._s) >= self._rho:
                                    u = q
                                else:
                                    l = q
                        
                        if from_beginning:
                            for q in range(u+1):
                                result.append(Primitive(*r, mus[q], normalize))
                        else:
                            for q in range(u, len(mus)):
                                result.append(Primitive(*r, mus[q], normalize))
                                
            # d-th component is discrete
            elif self._srange[d][0] == 1:
                mus = self._srange[d][1]
                for i in range(self._s.shape[1]):
                    l = [['F', 'G'], [i], range(i, self._s.shape[1])]
                    l += [[d], ['='], mus]
                    for r in itertools.product(*l):
                        primitive = Primitive(*r)
                        if primitive.robust(self._s) >= self._rho:
                            result.append(primitive)
            else:
                raise ValueError(f'{d}-th component continuous or discrete?')
        return result

    def _sample_signal(self) -> np.ndarray:
        "Simulate a signal by adding some artificially generated noise"
        
        sample = np.zeros(self._s.shape)
        sdim, slen = self._s.shape
        for d in range(sdim):
            # d-th component is continuous
            if self._srange[d][0] == 0:
                smin, smax, _ = self._srange[d][1]
                u = np.random.rand(sdim, slen) - 0.5
                u *= 2*(smax - smin) / slen
                sample[d] = self._s[d] + np.cumsum(u[d])
            
            # d-th component is discrete
            elif self._srange[d][0] == 1:
                choice = self._srange[d][1]
                length = len(choice)
                for t in range(self._s.shape[1]):
                    i = choice.index(self._s[d, t])
                    if length > 1:
                        distr = [0.5/(length-1)]*length
                        distr[i] = 0.5
                    else:
                        distr = [1]
                    sample[d, t] = np.random.choice(choice, p=distr)
            else:
                raise ValueError(f'{d}-th component continuous or discrete?')
        return sample

    def sample_signal_with_condition(self, condition: STL) -> np.ndarray:
        "Sample until the STL `condition` is satisfied"

        sample = self._sample_signal()
        while not condition.satisfy(sample):
            sample = self._sample_signal()
        return sample


@dataclass
class STL(Node):
    # (class attributes) to be set during initialization
    __generator         = None # static generator of primitives and signals
    __primitives        = None # generated primitive will be stored here
    __has_same_output   = None # callable evaluating a signal sample (-> bool)

    # (instance attribute)
    # The conjunction of these primitives represents the STL instance.
    _tup: FrozenSet[int] = frozenset()

    def initialize(self, generator: Generator, has_same_output: Callable):
        STL.__generator = generator
        STL.__primitives = generator.generate_primitives()
        STL.__has_same_output = has_same_output
        return len(STL.__primitives)

    def satisfy(self, s: np.ndarray) -> bool:
        "Verify if STL is satisfied by signal `s`"
        return all(STL.__primitives[i].satisfy(s) for i in self._tup)

    def is_terminal(self) -> bool:
        return False
    
    def find_children(self) -> Set[STL]:
        return {STL(self._tup.union([i]))
            for i in range(len(STL.__primitives))} - {self}

    def find_random_child(self) -> STL:
        i = np.random.randint(len(STL.__primitives))
        while i in self._tup:
            i = np.random.randint(len(STL.__primitives))
        return STL(self._tup.union([i]))

    def reward(self, batch_size) -> int:
        r = 0
        for _ in range(batch_size):
            sample = STL.__generator.sample_signal_with_condition(self)
            r += int(STL.__has_same_output(sample))
        return r

    def __hash__(self):
        return hash(self._tup)

    def __eq__(self, other):
        return isinstance(other, STL) and hash(self) == hash(other)

    def __repr__(self):
        if not len(self._tup):
            return 'T'
        return '^'.join(repr(STL.__primitives[i]) for i in self._tup)
