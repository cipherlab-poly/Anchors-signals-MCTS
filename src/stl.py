from __future__ import annotations

import numpy as np
import itertools
from typing import List, FrozenSet, Callable
from dataclasses import dataclass

@dataclass
class Primitive:
    "Ex: Primitive('G', 0, 5, '>', 20) <=> G[0,5](s_i > 20)"
    
    _typ: str            # 'F'(eventually) or 'G'(always)
    _a: int              # lower bound delay
    _b: int              # upper bound delay
    _i: int              # component index
    _comp: str           # '<' or '>' (continuous) or '=' (discrete)
    _mu: float           # constant threshold
    _normalize: float    # if continuous, max - min of mu
                         # if discrete, 1 (to normalize robustness degree)
    
    def __post_init__(self):
        if self._typ not in ['F', 'G']:
            raise ValueError('Invalid basic STL type')
        elif self._a < 0 or self._b < 0:
            raise ValueError('Invalid delay interval (negative bounds)')
        elif self._a > self._b:
            raise ValueError('Invalid delay interval (wrong order)') 
        elif self._comp not in ['<', '>', '=']:
            raise ValueError('Invalid comparison')

    def robust(self, s: np.ndarray) -> float:
        "Compute the robustness degree relative to signal `s`"
        
        if self._typ == 'F':
            if self._comp == '<':
                res = self._mu - np.min(s[self._i, self._a:self._b+1])
                return res/self._normalize
            elif self._comp == '>':
                res = np.max(s[self._i, self._a:self._b+1]) - self._mu
                return res/self._normalize
            elif self._mu in s[self._i, self._a:self._b+1]:
                return 1
            else:
                return -1
        else:
            if self._comp == '<':
                res = self._mu - np.max(s[self._i, self._a:self._b+1])
                return res/self._normalize
            elif self._comp == '>':
                res = np.min(s[self._i, self._a:self._b+1]) - self._mu
                return res/self._normalize
            elif all(v == self._mu for v in s[self._i, self._a:self._b+1]):
                return 1
            else:
                return -1

    def is_child_of(self, parent: Primitive) -> bool:
        if parent == self or parent._comp != self._comp or parent._i != self._i:
            return False
        
        cond1 = parent._typ == 'F' and parent._a <= self._a and self._b <= parent._b 
        cond2 = self._typ == 'G' and self._a <= parent._a and parent._b <= self._b
        if cond1 or cond2:
            if self._comp == '<':
                return parent._mu >= self._mu
            elif self._comp == '>':
                return self._mu >= parent._mu
            else:
                return self._mu == parent._mu
        return False

    def satisfy(self, s: np.ndarray) -> bool:
        "Verify if satisfied by signal `s`"
        return self.robust(s) > 0

    def __hash__(self):
        if self._a == self._b:
            return hash(('F', self._a, self._b, self._i, self._comp, self._mu))
        return hash((self._typ, self._a, self._b, self._i, self._comp, self._mu))

    def __eq__(self, other):
        return isinstance(other, Primitive) and hash(self) == hash(other)

    def __repr__(self):
        return f'{self._typ}[{self._a},{self._b}](s{self._i+1}{self._comp}{self._mu:.2f})'


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
class STL:
    # (class attributes) to be set during initialization
    __generator             = None  # static generator of primitives and signals
    __get_reward            = None  # callable evaluating a signal sample (0 or 1)
    __primitives            = []    # generated primitives will be stored here
    __children_primitives   = {}    # dict {parent: children} between primitives
    __parents_primitives    = {}    # dict {child: parents}   between primitives

    # (instance attribute)
    # The conjunction of these primitives represents the STL instance.
    _indices: FrozenSet[int] = frozenset()
    # index of the last added primitive
    _action: int = None

    def __post_init__(self):
        for child in self._indices.copy():
            self._indices -= STL.__parents_primitives[child]
    
    def initialize(self, generator: Generator, get_reward: Callable):
        STL.__generator = generator
        STL.__primitives = generator.generate_primitives()
        STL.__get_reward = get_reward
        length = len(STL.__primitives)
        STL.__children_primitives = {parent: {child for child in range(length)
            if STL.__primitives[child].is_child_of(STL.__primitives[parent])} 
            for parent in range(length)}
        STL.__parents_primitives = {child: {parent for parent in range(length)
            if STL.__primitives[child].is_child_of(STL.__primitives[parent])} 
            for child in range(length)}
        return length

    def satisfy(self, s: np.ndarray) -> bool:
        "Verify if STL is satisfied by signal `s`"
        return all(STL.__primitives[i].satisfy(s) for i in self._indices)
    
    def get_children(self) -> Set[STL]:
        length = len(STL.__primitives)
        if self._action is None: 
            return {STL(frozenset({i}), i) for i in range(length)}

        parents_primitives = STL.__parents_primitives[self._action]
        return {self.apply_action(action)
                for action in set(range(length)) - parents_primitives} - {self}

    def get_parents(self) -> Set[STL]:
        parents = [STL.__parents_primitives[i].union({i}) for i in self._indices]
        return {STL(forzenset(r)) for r in itertools.product(*parents)} - {self}

    def reward(self, batch_size: int) -> int:
        r = 0
        for _ in range(batch_size):
            sample = STL.__generator.sample_signal_with_condition(self)
            r += STL.__get_reward(sample)
        return r

    def apply_action(self, action: int) -> STL:
        return STL(self._indices.union([action]), action)

    def get_action(self) -> int:
        return self._action

    def __hash__(self):
        return hash(self._indices)

    def __eq__(self, other):
        return isinstance(other, STL) and hash(self) == hash(other)

    def __repr__(self):
        if not len(self._indices):
            return 'T'
        return '^'.join(repr(STL.__primitives[i]) for i in self._indices)
