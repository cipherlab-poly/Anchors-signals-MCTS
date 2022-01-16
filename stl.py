from __future__ import annotations

import numpy as np
import itertools

from dataclasses import dataclass
from typing import List, FrozenSet

@dataclass
class Primitive:
    "Ex: Primitive('G', 0, 5, 0, '>', 20, 1) <=> G[0,5](s_0 > 20)"
    
    _typ: str               # 'F'(eventually) or 'G'(always)
    _a: int                 # lower bound delay
    _b: int                 # upper bound delay
    _i: int                 # component index
    _comp: str              # '<' or '>' (continuous) or '=' (discrete)
    _mu: float              # constant threshold
    _normalize: float = 1.0 # if continuous, max - min of mu
                            # if discrete, 1 (to normalize robustness degree)
    
    def __post_init__(self):
        if self._typ not in ['F', 'G']:
            raise ValueError('Invalid basic STL type')
        elif self._a * self._b < 0 or (self._a < 0 and self._b == 0):
            raise ValueError('Invalid delay interval')
        elif self._a > self._b:
            raise ValueError('Invalid delay interval (wrong order)') 
        elif self._comp not in ['<', '>', '=']:
            raise ValueError('Invalid comparison')

    def robust(self, s: np.ndarray) -> float:
        "Compute the robustness degree relative to signal `s`"
        
        try:
            if self._b == -1:
                slicing = s[self._i, self._a:]
            else:
                slicing = s[self._i, self._a:self._b + 1]
        except IndexError:
            return -1

        if not len(slicing):
            return -1
        
        if self._typ == 'F':
            if self._comp == '<':
                res = self._mu - np.min(slicing)
                return res / self._normalize
            elif self._comp == '>':
                res = np.max(slicing) - self._mu
                return res / self._normalize
            elif self._mu in slicing:
                return 1
            else:
                return -1
        else:
            if self._comp == '<':
                res = self._mu - np.max(slicing)
                return res / self._normalize
            elif self._comp == '>':
                res = np.min(slicing) - self._mu
                return res / self._normalize
            elif all(v == self._mu for v in slicing):
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

    def satisfied(self, s: np.ndarray) -> bool:
        "Verify if satisfied by signal `s`"
        return self.robust(s) > 0

    def get_params(self):
        return (self._typ, self._a, self._b, self._i, self._comp, self._mu)

    def __hash__(self):
        if self._a == self._b:
            return hash(('F', self._a, self._b, self._i, self._comp, self._mu))
        return hash((self._typ, self._a, self._b, self._i, self._comp, self._mu))

    def __eq__(self, other):
        return isinstance(other, Primitive) and hash(self) == hash(other)

    def __repr__(self):
        res = f'{self._typ}[{self._a},{self._b}](s{self._i+1}{self._comp}'
        if self._comp == '=':
            res += f'{self._mu})'
        else:
            res += f'{self._mu:.2f})'
        return res


@dataclass
class PrimitiveGenerator:
    "(static) generator of primitives and signals"
    
    # (instance attributes)
    _s: np.ndarray      # signal being explained
    _srange: list       # list of tuples
                        # srange[d] = | (0, (min, max, stepsize))     if continuous
                        #             | (1, list of the finite set)   if discrete
    _rho: float         # robustness degree (~coverage) threshold
    _past: bool = False # true if PtSTL, false if STL
        
    def generate(self) -> List[Primitive]:
        "Generate STL primitives whose robustness is greater than `rho`"
            
        result = []
        sdim, slen = self._s.shape
        arange = range(-slen, 0) if self._past else range(slen)
        for d in range(sdim):
            # d-th component is continuous
            if self._srange[d][0] == 0:
                smin, smax, stepsize = self._srange[d][1]
                mus = np.linspace(smin, smax, num=stepsize, endpoint=False)[1:]
                norm = smax - smin
                for a in arange:
                    for typ in ['F', 'G']:
                        b = a + int(typ == 'G')
                        brange = range(b, 0) if self._past else range(b, slen)
                        l = [[typ], [a], brange, [d], ['>', '<']]
                        for r in itertools.product(*l):
                            stop = False
                            phi0 = Primitive(*r, mus[0], norm)
                            phi1 = Primitive(*r, mus[-1], norm)
                            if phi0.robust(self._s) >= self._rho:
                                u = 0
                                if phi1.robust(self._s) < self._rho:
                                    l = len(mus) - 1
                                    from_begin = True
                                else:
                                    stop = True
                                    from_begin = False
                            else:
                                from_begin = False
                                l = 0
                                if phi1.robust(self._s) >= self._rho:
                                    u = len(mus) - 1
                                else:
                                    u = len(mus)
                                    stop = True
                            
                            if not stop:
                                while True:
                                    phi0 = Primitive(*r, mus[l], norm)
                                    phi1 = Primitive(*r, mus[u], norm)
                                    if (phi0.robust(self._s) >= self._rho and 
                                            phi1.robust(self._s) >= self._rho):
                                        break
                                    elif (phi0.robust(self._s) < self._rho and 
                                            phi1.robust(self._s) < self._rho):
                                        break
                                    q = (u + l) // 2
                                    if u == q or l == q:
                                        break
                                    phi2 = Primitive(*r, mus[q], norm)
                                    if phi2.robust(self._s) >= self._rho:
                                        u = q
                                    else:
                                        l = q
                            
                            rng = range(u+1) if from_begin else range(u, len(mus))
                            for q in rng:
                                result.append(Primitive(*r, mus[q], norm))
                                
            # d-th component is discrete
            elif self._srange[d][0] == 1:
                mus = self._srange[d][1]
                for a in arange:
                    for typ in ['F', 'G']:
                        b = a + int(typ == 'G')
                        brange = range(b, 0) if self._past else range(b, slen)
                        l = [[typ], [a], brange, [d], ['='], mus]
                        for r in itertools.product(*l):
                            primitive = Primitive(*r, 1.0)
                            if primitive.robust(self._s) >= self._rho:
                                result.append(primitive)
            else:
                raise ValueError(f'{d}-th component continuous or discrete?')
        return result


class STL(object):
    __cache = {}

    # (class attributes) to be set during init
    __primitives        = []    # list of generated primitives
    __parents           = {}    # dict {child: parents} among primitives

    def __new__(cls, indices: FrozenSet[int]=frozenset()):
        for child in indices.copy():
            indices -= STL.__parents[child]
        
        if indices in STL.__cache:
            return STL.__cache[indices]
        else:
            o = object.__new__(cls)
            STL.__cache[indices] = o
            return o
    
    def __init__(self, indices: FrozenSet[int]=frozenset()):
        self._indices = indices
        for child in indices:
            self._indices -= STL.__parents[child]
    
    def init(self, primitives: List[Primitive]) -> int:
        STL.__primitives = primitives
        nb = len(primitives)
        STL.__parents = {child: {parent for parent in range(nb)
            if STL.__primitives[child].is_child_of(STL.__primitives[parent])} 
            for child in range(nb)}
        return nb
    
    def satisfied(self, s: np.ndarray) -> bool:
        "Verify if STL is satisfied by signal `s`"
        if s is None:
            return False
        return all(STL.__primitives[i].satisfied(s) for i in self._indices)
    
    def get_children(self) -> Set[STL]:
        length = len(STL.__primitives)
        parents = set()
        for i in self._indices:
            parents.update(STL.__parents[i])
        return {STL(self._indices.union([i])) 
            for i in set(range(length)) - parents} - {self}

    def get_params(self) -> List:
        return [STL.__primitives[i].get_params() for i in self._indices]

    def __len__(self):
        return len(self._indices)

    def __hash__(self):
        return hash(self._indices)

    def __repr__(self):
        if not len(self._indices):
            return 'T'
        return '^'.join(repr(STL.__primitives[i]) for i in self._indices)

class Simulator:
    def simulate(self):
        "Simulate a signal and return the corresponding reward"
        raise NotImplementedError("Method 'simulate' not implemented")
