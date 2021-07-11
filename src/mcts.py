"""
Inspired from:
    A minimal implementation of Monte Carlo tree search (MCTS) in Python 3
    Luke Harold Miles, July 2019, Public Domain Dedication
    https://gist.github.com/qpwo/c538c6f73727e254fdc7fab81024f6e1
"""
from collections import defaultdict
import math

class MCTS:
    "Monte Carlo tree searcher. First rollout the tree then choose a move."

    batch_size: int = 256
    def __init__(self, method: str = 'uct'):
        assert(method in {'uct', 'uct1tuned', 'fuse'})
        self.method = method
        
        self.children = {}
        
        # Monte-Carlo (score of a node)
        self.qMC = defaultdict(int)
        self.nMC = defaultdict(int)

        if method == 'fuse':
            # g-RAVE (score of each action)
            self.qAction = defaultdict(int)
            self.nAction = defaultdict(int)
            
            # l-RAVE (score of each action after visiting a node)"
            self.qRAVE = defaultdict(int)
            self.nRAVE = defaultdict(int)

    def choose(self, node):
        "Choose the best successor of node. (Choose a move in the game)"

        def score(n):
            if not self.qMC[n]:
                return float('-inf')
            return self.qMC[n]/self.nMC[n]

        return max(self.children[node], key=score)

    def do_rollout(self, node):
        "Make the tree one layer better. (Train for one iteration.)"
        path = self._select(node)
        leaf = path[-1]
        self._expand(leaf)
        reward = self._simulate(leaf)
        self._backpropagate(path, reward)

    def _select(self, node):
        "Find an unexplored descendent of `node`"
        path = []
        while True:
            path.append(node)
            if node not in self.children or not self.children[node]:
                # node is either unexplored or terminal
                return path
            unexplored = self.children[node] - self.children.keys()
            if unexplored:
                n = unexplored.pop()
                path.append(n)
                return path
            
            if self.method == 'uct':
                node = self._uct_select(node)
            elif self.method == 'uct1tuned':
                node = self._ucb1tuned_select(node)
            else:
                node = self._fuse_select(node)

    def _expand(self, node):
        "Update the `children` dict with the children of `node`"
        if node in self.children:
            return  # already expanded
        self.children[node] = node.get_children()

    def _simulate(self, node):
        "Returns the reward for a random simulation (to completion) of `node`"
        return node.reward(self.batch_size)
    
    def _backpropagate(self, path, reward):
        "Send the reward back up to the ancestors of the leaf"
        leaf = path[-1]
        actions = [node.get_action() for node in path]
        for i in range(len(path)):#leaf.get_parents().union(path):
            node = path[i]
            self.nMC[node] += self.batch_size
            self.qMC[node] += reward
            
            if self.method == 'fuse' and i:
                self.nAction[actions[i]] += self.batch_size
                self.qAction[actions[i]] += reward
                for j in range(i, len(path)):
                    node1 = node.apply_action(actions[j])
                    self.nRAVE[node1] += self.batch_size
                    self.qRAVE[node1] += reward

    def _uct_select(self, node, ce=2):
        "Select a child of node, balancing exploration & exploitation"

        log_N_vertex = math.log(self.nMC[node])

        def uct(n):
            mu = self.qMC[n] / self.nMC[n]
            sqrt = math.sqrt(ce * log_N_vertex / self.nMC[n])
            return mu + sqrt

        return max(self.children[node], key=uct)

    def _ucb1tuned_select(self, node, ce=2):
        "Select a child of node, balancing exploration & exploitation"

        log_N_vertex = math.log(self.nMC[node])

        def ucb1tuned(n):
            mu = self.qMC[n] / self.nMC[n]
            sqrt = math.sqrt(ce * log_N_vertex / self.nMC[n])
            return mu + sqrt * math.sqrt(min(0.25, mu * (1 - mu) + sqrt))
        
        return max(self.children[node], key=ucb1tuned)

    def _fuse_select(self, node, ce=2, c=100, cl=100):
        "Select a child of node, balancing exploration & exploitation"

        # All children of node should already be expanded:
        assert all(n in self.children for n in self.children[node])

        log_N_vertex = math.log(self.nMC[node])

        def fuse(n):
            action = n.get_action()
            mu = self.qMC[n] / self.nMC[n]
            sqrt = math.sqrt(ce * log_N_vertex / self.nMC[n])
            alpha = c / (c + self.nRAVE[n])
            beta = cl / (cl + self.nAction[action])
            lRAVE = self.qRAVE[n] / self.nRAVE[n]
            gRAVE = self.qAction[action] / self.nAction[action]
            return (1 - alpha) * mu + alpha * (
                (1 - beta) * lRAVE + beta * gRAVE
            ) + sqrt * math.sqrt(min(0.25, mu * (1 - mu) + sqrt))

        return max(self.children[node], key=fuse)
