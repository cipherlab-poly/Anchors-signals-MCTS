"""
Implementation inspired from:
    1) Implementation of the KL-LUCB algorithm by Kaufmann and Kalyanakrishnan in 
    their publication "Information Complexity in Bandit Subset Selection" to 
    identify the anchor with the highest precision (with high confidence).
    https://github.com/viadee/javaAnchorExplainer/blob/master/src/main/java/de/viadee/xai/anchor/algorithm/exploration/KL_LUCB.java
    
    2) A minimal implementation of Monte Carlo tree search (MCTS) in Python 3
    Luke Harold Miles, July 2019, Public Domain Dedication
    https://gist.github.com/qpwo/c538c6f73727e254fdc7fab81024f6e1
"""
from collections import defaultdict
import numpy as np
import itertools
import heapq

# visual
import networkx as nx
import matplotlib.pyplot as plt
import pydot
from networkx.drawing.nx_pydot import graphviz_layout

class KL_LUCB:

    def __init__(self, batch_size=256, beam_width=8, delta=0.01, epsilon=0.01):
        self.batch_size = batch_size
        self.beam_width = beam_width
        self.delta      = delta
        self.epsilon    = epsilon

        self.nb_cands = 1
        self.children = defaultdict(set)
        self.ancestors = defaultdict(set)
        
        # score of a node: Q[node]/N[node]
        # upper bound: UB[node], lower bound: LB[node]
        self.Q = defaultdict(lambda: 1)
        self.N = defaultdict(lambda: 1)
        self.UB = defaultdict(lambda: 1.0)
        self.LB = defaultdict(lambda: 0.0)

    def choose(self, nodes):
        def precision(n):
            return self.Q[n] / self.N[n]
        return heapq.nlargest(self.beam_width, nodes, key=precision)

    def get_cands(self, beams):
        cands = set()
        for beam in beams:
            self._expand(beam)
            cands.update(self.children[beam])
        return cands
    
    def train(self, cands):
        self.nb_cands = len(cands)
        tmp = iter(cands)
        u_arm = next(tmp)
        l_arm = next(tmp)
        err = 1.0
        i = 0
        while err > self.epsilon:
            print(f'\033[1;93m Iter {i} Error {err:5.2%}\033[1;m', end='  \r')
            i += 1
            u_reward = self._simulate(u_arm)
            l_reward = self._simulate(l_arm)
            self._backpropagate(u_arm, u_reward)
            self._backpropagate(l_arm, l_reward)
            u_arm, l_arm = self._update_bounds(cands, self._compute_beta(i))
            err = min(err, self.UB[u_arm] - self.LB[l_arm])

    def _update_bounds(self, cands, beta):
        def prec(n):
            return self.Q[n] / self.N[n]
        def ub(n):
            return self.UB[n]    
        def lb(n):
            return self.LB[n]
        
        J = heapq.nlargest(self.beam_width, cands, key=prec)
        not_J = cands.difference(J)

        for cand in J:
            self.LB[cand] = self._low_bernoulli(prec(cand), beta / self.N[cand])
        for cand in not_J:
            self.UB[cand] = self._up_bernoulli(prec(cand), beta / self.N[cand])

        if not_J:
            return max(not_J, key=ub), min(J, key=lb)
        raise ValueError('Beam width smaller than candidate number')
    
    def _expand(self, node):
        "Update the `children` dict with the children of `node`"
        self.children[node] = node.get_children()
        if node in self.ancestors:
            ancestors = self.ancestors.pop(node) # for back-propagation
        else:
            ancestors = set()
        for child in self.children[node]:
            self.ancestors[child].add(node)
            self.ancestors[child].update(ancestors)

    def _simulate(self, node):
        "Return the reward for a random simulation of `node`"
        return node.simulate(self.batch_size)
    
    def _backpropagate(self, node, reward):
        "Send `reward` to the ancestors of `node` and set bounds"
        for ancestor in self.ancestors[node].union({node}):
            self.Q[ancestor] += reward
            self.N[ancestor] += self.batch_size
            
    def _compute_beta(self, r):
        temp = np.log(405.5 * self.nb_cands * r ** 1.1 / self.delta)
        return temp + np.log(temp)

    def _kl_bernoulli(self, p, q):
        "Computes KL-divergence between two variables ~ B(p) and B(q)."
        p = min(0.9999, max(0.0001, p))
        q = min(0.9999, max(0.0001, q))
        return p * np.log(p / q) + (1 - p) * np.log((1 - p) / (1 - q))

    def _up_bernoulli(self, p, level):
        "Computes the upper bound of the variable ~ B(p) of a certain level."
        lm = p
        um = min(min(1, p + np.sqrt(level / 2)), 1)
        for _ in range(16):
            qm = (um + lm) / 2
            if self._kl_bernoulli(p, qm) > level:
                um = qm
            else:
                lm = qm
        return um

    def _low_bernoulli(self, p, level):
        "Computes the lower bound of the variable ~ B(p) of a certain level."
        um = p
        lm = max(min(1, p - np.sqrt(level / 2)), 0)
        for _ in range(16):
            qm = (um + lm) / 2
            if self._kl_bernoulli(p, qm) > level:
                lm = qm
            else:
                um = qm
        return lm