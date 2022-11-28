"""
@file   mcts.py
@brief  tree object implementing MCTS steps

@author Tzu-Yi Chiu <tzuyi.chiu@gmail.com>

Inspired from:
    A minimal implementation of Monte Carlo tree search (MCTS) in Python 3
    Luke Harold Miles, July 2019, Public Domain Dedication
    https://gist.github.com/qpwo/c538c6f73727e254fdc7fab81024f6e1
"""
import os
import numpy as np
from typing import List, Tuple

from collections import defaultdict
import math

from stl import STL
from simulator import Simulator

class MCTS:
    "Monte Carlo tree searcher. Roll-out the tree then choose a move."

    def __init__(self, simulator: Simulator, 
                       epsilon: float, 
                       tau: float, 
                       batch_size: int, 
                       max_depth: int, 
                       max_iter: int) -> None:
        
        self.simulator  = simulator
        self.epsilon    = epsilon
        self.tau        = tau
        self.batch_size = batch_size
        self.max_depth  = max_depth
        self.max_iter   = max_iter
        
        self.children  = defaultdict(set)
        self.ancestors = defaultdict(set)
        self.pruned    = set()
    
        # Monte-Carlo (precision = Q/N)
        self.Q = defaultdict(int)
        self.N = defaultdict(int)
        
        # Hyperparameter for ucb1tuned
        self.ce = 2

        # Found an anchor
        self.finished = False
    
    def choose(self, node: STL) -> List[STL]:
        "Choose the best successor of node (choose a move in the game)"
        
        def ucb1tuned(n):
            p, N = self.precision(n), self.N[n]
            if not N:
                return float('-inf')
            tmp = math.sqrt(self.ce * math.log(self.N[node]) / N)
            return p - tmp * math.sqrt(min(0.25, p * (1 - p) + tmp))
        
        best = max(self.children[node], key=ucb1tuned)
        if self.precision(best) < self.tau:
            return best
        self.finished = True
        return self._best_anchors(best)
        
    def _best_anchors(self, node: STL) -> List[STL]:
        anchors = [node]
        while True:
            try:
                node = next(iter(n for n in self.ancestors[node] 
                                if self.precision(n) >= self.tau))
                anchors.append(node)
            except StopIteration:
                return anchors

    def train(self, node: STL) -> Tuple[int, float]:
        """
        Rollout the tree from `node` until error is smaller than `epsilon`

        :param node: STL formula from which rollouts are performed
        :returns: number of rollouts and error
        """
        err = 1.0
        i = 0
        best = None
        while err > self.epsilon:
            if i >= self.max_iter:
                self.finished = True
                break
            i += 1
            to_print = f'\033[1;93m Iter {i} Err {err:5.2%} Best {best} '
            to_print += f'({self.precision(best):5.2%}'
            to_print += f'={self.Q[best]}/{self.N[best]})\033[1;m'
            print(f'{to_print:<80}', end='\r')
            self._rollout(node)

            if self.children[node]:
                best = self._select(node)
                p, N = self.precision(best), self.N[best]
                if N:
                    tmp = math.sqrt(self.ce * math.log(self.N[node]) / N)
                    err = tmp * math.sqrt(min(0.25, p * (1 - p) + tmp))
                    err = min(err, 1.0)
        return i, err

    def precision(self, node: STL) -> float:
        "Empirical precision of `node`"
        if not self.N[node]:
            return 0.0
        return self.Q[node] / self.N[node]

    def clean(self, parent: STL, child: STL) -> None:
        """
        Clean up useless memory in the tree.
        """
        self.Q.pop(parent, None)
        self.N.pop(parent, None)
        for node in self.children[parent] - self.children[child] - {child}:
            self.Q.pop(node, None)
            self.N.pop(node, None)
            self.children.pop(node, None)
        self.children.pop(parent, None)
    
    def _rollout(self, node: STL) -> List[STL]: 
        """
        Make the tree one layer better (train for one iteration).

        :param node: STL formula from which rollouts are performed
        :returns: the selected path
        """
        path = self._select_path(node)

        # Sample in mini-batch mode
        samples, scores = [], []
        for _ in range(self.batch_size):
            sample, score = self.simulator.simulate()
            samples.append(sample)
            scores.append(score)

        # If cov(leaf) is too low then prune the leaf
        # else backpropagate sample and score to relevant ancestors
        leaf = path[-1]
        if any(leaf.satisfied(s) for s in samples):
            for i in range(self.batch_size):
                self._backpropagate(path, samples[i], scores[i])    
        else:
            self._prune(leaf)
        return path
    
    def _select_path(self, node: STL) -> List[STL]:
        "Find a path leading to an unexplored descendent of `node`"
        path = []
        while True:
            path.append(node)
            if len(node) >= self.max_depth:
                return path
            if not self.children[node]: # not yet expanded
                if self.N[node]:        # already explored
                    self._expand(node)
                else:
                    return path
            node = self._select(node)

    def _expand(self, node: STL) -> None:
        "Update the `children` dict with the children of `node`"
        self.children[node] = node.get_children() - self.pruned
        for child in self.children[node]:
            self.ancestors[child].add(node)

    def _prune(self, node: STL) -> None:
        "Prune `node` from the tree"
        self.pruned.add(node)
        self.Q.pop(node, None)
        self.N.pop(node, None)
        for n in self.ancestors[node]:
            self.children[n].discard(node)
        self.ancestors.pop(node, None)
    
    def _backpropagate(self, path: List[STL], 
                             sample: np.ndarray, 
                             score: int) -> None:
        """(binary search) 
        path: root =: phi_0 -> ... -> phi_m := leaf
        find i* s.t. sample satisfies phi_l for all 0 <= l <= i* 
                     otherwise for all l > i*
        """
        lo, hi = 0, len(path)
        while lo < hi:
            mid = (lo + hi) // 2
            if path[mid].satisfied(sample):
                lo = mid + 1
            else:
                hi = mid
        
        #ancestors = set(path[:lo])
        #for node in path[:lo]:
        #    ancestors.update(self.ancestors[node])
        #for node in ancestors:
        #    self.Q[node] += score
        #    self.N[node] += 1
        
        # find all ancestors of the critical node
        if lo == 0:
            return
        critical = path[lo - 1]

        length = 0
        ancestors = {critical}
        iterator = iter(ancestors)
        while len(ancestors) > length:
            length = len(ancestors)
            to_update = set()
            while True:
                try:
                    node = next(iterator)
                    to_update.update(self.ancestors[node])
                except StopIteration:
                    break
            iterator = iter(to_update)
            ancestors.update(to_update)
        
        for node in ancestors:
            self.Q[node] += score
            self.N[node] += 1

    def _select(self, node: STL) -> STL:
        """
        Select a child of `node`, balancing exploration & exploitation
        """
        def ucb1tuned(n):
            p, N = self.precision(n), self.N[n]
            if not N:
                return float('inf')
            tmp = math.sqrt(self.ce * math.log(self.N[node]) / N)
            return p + tmp * math.sqrt(min(0.25, p * (1 - p) + tmp))
        
        return max(self.children[node], key=ucb1tuned)

    def log(self, stl: STL) -> str:
        q, n = self.Q[stl], self.N[stl]
        if not n:
            return f'{stl} ({q}/{n})'
        return f'{stl} ({q}/{n}={q/n:5.2%})'
