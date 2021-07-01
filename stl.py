import numpy as np
import itertools
from typing import Dict, Set, Iterator
Graph = Dict[Primitive, Set[Primitive]]

from monte_carlo_tree_search import Node

@dataclass
class PrimitiveGenerator:
    """
    Note:
        It contains every Primitive with bounded parameters i.e.
            F-[t1,t2](s_i<mu_i)     G-[t1,t2](s_i<mu_i) 
            F-[t1,t2](s_i>mu_i)     G-[t1,t2](s_i>mu_i)
            F-[t1,t2](s_i=mu_i)     G-[t1,t2](s_i=mu_i)
        where 0 <= t1, t2 <= slen and smins[i] <= mu_i <= smaxes[i] (cf. Primitive)
    """
    s: np.ndarray   # signal being explained
    srange: list    # list of tuples
                    # srange[d] = | (0, (min, max, stepsize))     if continuous
                    #             | (1, list of the finite set)   if discrete
    rho: float      # robustness degree (~coverage) threshold
    
    def __post_init__(self):    
        if len(self.srange) != self.s.shape[0]:
            raise ValueError(f'srange: expected length {self.s.shape[0]}, ' + 
                                f'got {len(self.srange)}')
        
    def generate(self) -> Iterator[Primitive]:
        "Generate STL primitives whose robustness is greater than `rho`"
            
        for d in range(self.s.shape[0]):
            # d-th component is continuous
            if self.srange[d][0] == 0:
                smin = self.srange[d][1][0]
                smax = self.srange[d][1][1]
                stepsize = self.srange[d][1][2]
                mus = np.linspace(smin, smax, num=stepsize, endpoint=False)[1:]
                normalize = smax - smin
                for i in range(self.s.shape[1]):
                    l = [['F', 'G'], [i], range(i, self.s.shape[1]), [d], ['>', '<']]
                    for r in itertools.product(*l):
                        stop = False
                        phi0 = Primitive(*r, mus[0], normalize)
                        phi1 = Primitive(*r, mus[-1], normalize)
                        if phi0.robust(self.s) >= self.rho:
                            u = 0
                            if phi1.robust(self.s) < self.rho:
                                l = len(mus) - 1
                                from_beginning = True
                            else:
                                stop = True
                                from_beginning = False
                        else:
                            from_beginning = False
                            l = 0
                            if phi1.robust(self.s) >= self.rho:
                                u = len(mus) - 1
                            else:
                                u = len(mus)
                                stop = True
                        
                        if not stop:
                            while True:
                                phi0 = Primitive(*r, mus[l], normalize)
                                phi1 = Primitive(*r, mus[u], normalize)
                                if (phi0.robust(self.s) >= self.rho and 
                                        phi1.robust(self.s) >= self.rho):
                                    break
                                elif (phi0.robust(self.s) < self.rho and 
                                        phi1.robust(self.s) < self.rho):
                                    break
                                q = (u + l) // 2
                                if u == q or l == q:
                                    break
                                phi2 = Primitive(*r, mus[q], normalize)
                                if phi2.robust(self.s) >= self.rho:
                                    u = q
                                else:
                                    l = q
                        
                        if from_beginning:
                            for q in range(u+1):
                                yield Primitive(*r, mus[q], normalize)
                        else:
                            for q in range(u, len(mus)):
                                yield Primitive(*r, mus[q], normalize)

            # d-th component is discrete
            elif self.srange[d][0] == 1:
                mus = self.srange[d][1]
                for i in range(self.s.shape[1]):
                    l = [['F', 'G'], [i], range(i, self.s.shape[1])]
                    l += [[d], ['='], mus]
                    for r in itertools.product(*l):
                        primitive = Primitive(*r)
                        if primitive.robust(self.s) >= self.rho:
                            yield primitive
            else:
                raise ValueError(f'{d}-th component continuous or discrete?')         

@dataclass
class Primitive:
    "Ex: Primitive('G', 0, 5, '>', 20) <=> G[0,5](s_i > 20)"
    
    typ: str            # 'F-'(eventually) or 'G-'(always)
    a: int              # lower bound delay
    b: int              # upper bound delay
    i: int              # component index
    comp: str           # '<' or '>' (continuous) or '=' (discrete)
    mu: float           # constant threshold
    normalize: float    # if continuous, max - min of mu
                        # if discrete, 1 (to normalize robustness degree)
    
    def __post_init__(self):
        if self.typ not in ['F', 'G']:
            raise ValueError('Invalid basic ptSTL type')
        elif self.a < 0 or self.b < 0:
            raise ValueError('Invalid delay interval (negative bounds)')
        elif self.a > self.b:
            raise ValueError('Invalid delay interval (wrong order)') 
        elif self.comp not in ['<', '>', '=']:
            raise ValueError('Invalid comparison')

    def robust(self, s: np.ndarray) -> float:
        "Compute the robustness degree relative to signal `s`"
        
        if self.typ == 'F-':
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

    def __hash__(self):
        "So that Primitives can be put in a frozenset (for STL construction)"
        return hash((self.typ, self.a, self.b, self.i, self.comp, self.mu))

    def __eq__(self, other: Primitive):
        return hash(self) == hash(other)
    
    def __repr__(self):
        return f'{self.typ}[{self.a},{self.b}](s{self.i+1}{self.comp}{self.mu:.2f})'

@dataclass
class STL(Node):
    """ 
    Note:
        The instance (prev_cand U {last_primitive}) represents the conjunction 
        of these basic STL formulas going to be used as our Anchor Candidate.
    """
    prev_cand: Frozenset[Primitive] = frozenset()   # previous candidates
    last_primitive: Primitive = None                # last added STL primitive
    
    def conjoin(self, phi: Primitive) -> STL:
        "Compute its conjunction with `phi`"
        
        if self.last_primitive is None:
            return PtSTL(frozenset(), phi)
        return PtSTL(self.prev_cand.union({self.last_primitive}), phi)

    def satisfy(self, s: np.ndarray) -> bool:
        "Verify if satisfied by signal `s`"

        if self.last_primitive is None:
            return True
        if self.last_primitive.satisfy(s):
            return all(primitive.satisfy(s) for primitive in self.prev_cand)
        return False

    def find_children(self) -> Set[STL]:
        

    def reward(self) -> int:


    def __hash__(self):
        return hash((self.prev_cand, self.last_primitive))

    def __eq__(self, other: STL):
        return hash(self) == hash(other)

    def __repr__(self):
        if not self.prev_cand:
            if self.last_primitive is None:
                return 'T'
            else:
                return str(self.last_primitive)
        res = '^'.join(str(phi) for phi in self.prev_cand)
        res += f'^{self.last_primitive}'
        return res
