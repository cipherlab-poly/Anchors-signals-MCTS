"""
Inspired from:
    A minimal implementation of Monte Carlo tree search (MCTS) in Python 3
    Luke Harold Miles, July 2019, Public Domain Dedication
    https://gist.github.com/qpwo/c538c6f73727e254fdc7fab81024f6e1
"""
from collections import defaultdict
import math
import itertools

class MCTS:
    "Monte Carlo tree searcher. First rollout the tree then choose a move."

    def __init__(self, max_depth=5, epsilon=0.01, tau=0.01, ancestors=True):
        self.max_depth  = max_depth
        self.epsilon    = epsilon
        self.tau        = tau
        
        self.children = defaultdict(set)
        if ancestors:
            self.ancestors = defaultdict(set)
        else:
            self.ancestors = None
        
        # Monte-Carlo (precision = Q/N)
        self.Q = defaultdict(int)
        self.N = defaultdict(int)

        # Coverage = N/M
        self.M = defaultdict(int)
        
        # Hyperparameter for ucb1tuned
        self.ce = 2

        # To be increased after each move
        self.batch_size = None

    def set_batch_size(self, batch_size):
        self.batch_size = batch_size
    
    def clean(self, parent, child):
        self.Q.pop(parent, None)
        self.N.pop(parent, None)
        self.M.pop(parent, None)
        for node in self.children[parent] - self.children[child] - {child}:
            self.Q.pop(node, None)
            self.N.pop(node, None)
            self.M.pop(node, None)
            self.children.pop(node, None)
        self.children.pop(parent, None)
    
    def choose(self, node):
        "Choose the best successor of node. (Choose a move in the game)"
        
        def ucb1tuned(n):
            p, N = self._score(n)
            if not N:
                return float('-inf')
            tmp = math.sqrt(self.ce * math.log(self.N[node]) / N)
            return p - tmp * math.sqrt(min(0.25, p * (1 - p) + tmp))
        
        def coverage(n):
            cov = self.N[n] / self.M[n]
            if not self.M[n]:
                return float('-inf')
            tmp = math.sqrt(self.ce * math.log(self.M[node]) / self.M[n])
            return cov - tmp * math.sqrt(min(0.25, cov * (1 - cov) + tmp))
        
        if all(self._score(n)[0] < self.tau for n in self.children[node]):
            return [max(self.children[node], key=ucb1tuned)]

        bests = {child for child in self.children[node] 
                    if self._score(child)[0] > self.tau}
        return sorted(bests, key=coverage, reverse=True)

    def train(self, node):
        "Rollout the tree from `node` until error is smaller than `epsilon`"
        err = 1.0
        i = 0
        while err > self.epsilon:
            i += 1
            if not i % 10:
                print(f'\033[1;93m Iter {i} Error {err:5.2%} \033[1;m', 
                        end='   \r')
            self._rollout(node)

            if self.children[node]:
                best = self.choose(node)[0]
                p, N = self._score(best)
                if N:
                    tmp = math.sqrt(self.ce * math.log(self.N[node]) / N)
                    err = 2 * tmp * math.sqrt(min(0.25, p * (1 - p) + tmp))
                    err = min(err, 1.0)
        return i

    def _rollout(self, node):
        "Make the tree one layer better (train for one iteration)"
        path = self._select_path(node)
        leaf = path[-1]
        Q, N = leaf.simulate(self.batch_size)
        self._backpropagate(path, Q, N)

    def _score(self, node):
        "Empirical score of `node`"
        if not self.N[node]:
            return (float('-inf'), 0)
        return (self.Q[node] / self.N[node], self.N[node])
    
    def _select_path(self, node):
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

    def _expand(self, node):
        "Update the `children` dict with the children of `node`"
        self.children[node] = node.get_children()
        if self.ancestors is not None:
            for child in self.children[node]:
                self.ancestors[child].add(node)
    
    def _backpropagate(self, path, Q, N):
        "Send `reward` back up to the ancestors of `node`"
        if self.ancestors is None:
            for n in path:
                self.Q[n] += Q
                self.N[n] += N
                self.M[n] += self.batch_size
        else:
            ancestors = {path[-1]}
            for n in path:
                ancestors.update(self.ancestors[n])
            for n in ancestors:
                self.Q[n] += Q
                self.N[n] += N 
                self.M[n] += self.batch_size

    def _select(self, node):
        "Select a child of node, balancing exploration & exploitation"
        
        def ucb1tuned(n):
            p, N = self._score(n)
            if not N:
                return float('inf')
            tmp = math.sqrt(self.ce* math.log(self.N[node]) / N)
            return p + tmp * math.sqrt(min(0.25, p * (1 - p) + tmp))
        
        return max(self.children[node], key=ucb1tuned)

    def visualize(self):
        from visual import Visual
        Visual(self).visualize()
