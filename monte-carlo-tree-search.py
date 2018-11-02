"""
A minimal implementation of Monte Carlo tree search (MCTS) in Python 3.
Luke Harold Miles, November 2018, Public Domain Dedication
See also https://en.wikipedia.org/wiki/Monte_Carlo_tree_search
"""
from collections import defaultdict
import math


class MCTS:
    "Monte Carlo tree search"

    def __init__(self, exploration_weight=1):
        self.Q = defaultdict(int)  # total reward of each node
        self.N = defaultdict(int)  # total visit count for each node
        self.children = dict()  # children of each node
        self.exploration_weight = exploration_weight

    def choose(self, node):
        "Choose the best successor of node"
        if node not in self.children:
            return node.find_random_child()

        def score(n):
            if self.N[n] == 0:
                return float('-inf')
            return self.Q[n] / self.N[n]  # average reward

        return max(self.children[node], key=score)

    def do_rollout(self, node):
        "Make the tree one layer better"
        path = self.select(node)
        leaf = path[-1]
        self.expand(leaf)
        reward = self.simulate(leaf)
        self.backpropagate(path, reward)

    def select(self, node):
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
            node = self.uct_select(node)  # descend a layer deeper

    def expand(self, node):
        "Update the `children` dict with the children of `node`"
        if node in self.children:
            return  # already expanded
        self.children[node] = node.find_children()

    def simulate(self, node):
        "Returns the reward for a random simulation (to completion) of `node`"
        while True:
            node2 = node.find_random_child()
            if node2 is None:
                return node.reward()
            node = node2

    def backpropagate(self, path, reward):
        "Send the reward back up to the ancestors of the leaf"
        for node in path:
            reward = 1 - reward  # 1 for me is 0 for my enemy, and vice versa
            self.N[node] += 1
            self.Q[node] += reward

    def uct_select(self, node):
        "Select a child of node, balancing exploration & exploitation"

        # All children of node must be expanded:
        assert all(n in self.children for n in self.children[node])

        log_N_vertex = math.log(self.N[node])

        def uct(n):
            "Upper confidence bound for trees"
            return self.Q[n] / self.N[n] + \
                self.exploration_weight * math.sqrt(log_N_vertex / self.N[n])

        return max(self.children[node], key=uct)


class Node:
    "This can be a checkers or chess or tic-tac-to board state"

    def find_children(self):
        "All possible successors to this board state"
        return set()

    def find_random_child(self):
        "For efficiency in simulation. Returns None if node has no children"
        return None

    def reward(self):
        "Assumes `self` is terminal node. 1=win, 0=loss, .5=tie, etc"
        return 0

    def __init__(self):
        "Make a new node"
        pass

    def __hash__(self):
        "Nodes must be hashable"
        return 37

    def __eq__(node1, node2):
        "Nodes must be comparable"
        return True