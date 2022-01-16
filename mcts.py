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

    def __init__(self, simulator, epsilon, tau, max_depth, batch_size):
        self.simulator  = simulator
        self.epsilon    = epsilon
        self.tau        = tau
        self.max_depth  = max_depth
        self.batch_size = batch_size
        
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
    
    def choose(self, node):
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
        
    def _best_anchors(self, node):
        anchors = [node]
        while True:
            try:
                node = next(iter(n for n in self.ancestors[node] 
                                if self.precision(n) >= self.tau))
                anchors.append(node)
            except StopIteration:
                return anchors

    def train(self, node, max_iter=40000):
        "Rollout the tree from `node` until error is smaller than `epsilon`"
        err = 1.0
        i = 0
        best = None
        while err > self.epsilon:
            if i >= max_iter:
                self.finished = True
                break
            i += 1
            print(f'\033[1;93m Iter {i} Err {err:5.2%} Best {best} ' + 
                    f'({self.precision(best):5.2%}) \033[1;m', end='   \r')
            self._rollout(node)

            if self.children[node]:
                best = self._select(node)
                p, N = self.precision(best), self.N[best]
                if N:
                    tmp = math.sqrt(self.ce * math.log(self.N[node]) / N)
                    err = tmp * math.sqrt(min(0.25, p * (1 - p) + tmp))
                    err = min(err, 1.0)
        return i, err

    def precision(self, node):
        "Empirical precision of `node`"
        if not self.N[node]:
            return 0.0
        return self.Q[node] / self.N[node]

    def clean(self, parent, child):
        self.Q.pop(parent, None)
        self.N.pop(parent, None)
        for node in self.children[parent] - self.children[child] - {child}:
            self.Q.pop(node, None)
            self.N.pop(node, None)
            self.children.pop(node, None)
        self.children.pop(parent, None)
    
    def _rollout(self, node):
        "Make the tree one layer better (train for one iteration)"
        path = self._select_path(node)

        # Sample in mini-batch mode
        samples, scores = [], []
        for _ in range(self.batch_size):
            sample, score = self.simulator.simulate()
            samples.append(sample)
            scores.append(score)

        # Update score for leaf
        leaf = path[-1]
        for i in range(self.batch_size):
            self._update_score(leaf, samples[i], scores[i])
        
        # If cov(leaf) is too low then prune the leaf
        # Else backpropagate sample and score to ancestors
        if self.N[leaf]:
            for i in range(self.batch_size):
                self._backpropagate(path, samples[i], scores[i])
        else:
            self._prune(leaf)
    
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
        self.children[node] = node.get_children() - self.pruned
        for child in self.children[node]:
            self.ancestors[child].add(node)

    def _prune(self, node):
        "Prune `node` from the tree"
        self.pruned.add(node)
        self.Q.pop(node, None)
        self.N.pop(node, None)
        for n in self.ancestors[node]:
            self.children[n].remove(node)
        self.ancestors.pop(node, None)
        self.children.pop(node, None)
    
    def _update_score(self, node, sample, score):
        "Update N and Q of `node` if verified by `sample`"
        if node.satisfied(sample):
            self.Q[node] += score
            self.N[node] += 1

    def _backpropagate(self, path, sample, score):
        "Update score for leaf's ancestors (leaf = path[-1])"
        ancestors = set()
        for node in path[::-1]:
            ancestors.update(self.ancestors[node])
        for node in ancestors:#path[1::-1]:
            self._update_score(node, sample, score)

    def _select(self, node):
        "Select a child of `node`, balancing exploration & exploitation"
        
        def ucb1tuned(n):
            p, N = self.precision(n), self.N[n]
            if not N:
                return float('inf')
            tmp = math.sqrt(self.ce * math.log(self.N[node]) / N)
            return p + tmp * math.sqrt(min(0.25, p * (1 - p) + tmp))
        
        return max(self.children[node], key=ucb1tuned)

    def log(self, stl):
        q, n = self.Q[stl], self.N[stl]
        if not n:
            return f'{stl} ({q}/{n})'
        return f'{stl} ({q}/{n}={q/n:5.2%})'

    def visualize(self):
        from visual import Visual
        Visual(self).visualize()
