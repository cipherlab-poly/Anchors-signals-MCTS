"""
Inspired from:
    A minimal implementation of Monte Carlo tree search (MCTS) in Python 3
    Luke Harold Miles, July 2019, Public Domain Dedication
    https://gist.github.com/qpwo/c538c6f73727e254fdc7fab81024f6e1
"""
from collections import defaultdict
import math
import itertools
import heapq

# visual
import networkx as nx
import matplotlib.pyplot as plt
import pydot
from networkx.drawing.nx_pydot import graphviz_layout

class MCTS:
    "Monte Carlo tree searcher. First rollout the tree then choose a move."

    def __init__(self, batch_size=256, method='ucb1tuned', visual=False):
        self.batch_size = batch_size
        self.method     = method        # uct / ucb1tuned
        self.visual     = visual

        self.children = defaultdict(set)
        self.ancestors = defaultdict(set)
        
        # Monte-Carlo (score of a node)
        self.qMC = defaultdict(int)
        self.nMC = defaultdict(int)

        if self.visual:
            self.G = nx.DiGraph()
            self.colors = list(itertools.product(*([(0, 1)]*3)))
            self.colors.remove((0, 0, 0))
            self.colors.remove((1, 1, 1))

    def choose(self, node):
        "Choose the best successor of node. (Choose a move in the game)"

        def score(n):
            if not self.qMC[n]:
                return (float('-inf'), 0)
            return (self.qMC[n]/self.nMC[n], self.nMC[n])

        return heapq.nlargest(5, self.children[node], key=score)

    def do_rollout(self, node):
        "Make the tree one layer better. (Train for one iteration.)"
        leaf = self._select_leaf(node)
        reward = self._simulate(leaf)
        self._backpropagate(leaf, reward)
        self._expand(leaf)

    def _select_leaf(self, node):
        "Find an unexplored descendent of `node`"
        while True:
            if node not in self.children or not self.children[node]:
                # node is either unexplored or terminal
                return node
            unexplored = self.children[node] - self.children.keys()
            if unexplored:
                return unexplored.pop()
            node = self._select(node)

    def _expand(self, node):
        "Update the `children` dict with the children of `node`"
        if node in self.children:
            return  # already expanded
        self.children[node] = node.get_children()
        ancestors = self.ancestors.pop(node) # for back-propagation
        for child in self.children[node]:
            self.ancestors[child].add(node)
            self.ancestors[child].update(ancestors)

    def _simulate(self, node):
        "Return the reward for a random simulation of `node`"
        return node.reward(self.batch_size)
    
    def _backpropagate(self, node, reward):
        "Send `reward` back up to the ancestors of `node`"
        for ancestor in self.ancestors[node].union({node}):
            self.nMC[ancestor] += self.batch_size
            self.qMC[ancestor] += reward

    def _select(self, node, ce=2, c=100, cl=100):
        "Select a child of node, balancing exploration & exploitation"

        log_N_vertex = math.log(self.nMC[node])

        def uct(n):
            mu = self.qMC[n] / self.nMC[n]
            sqrt = math.sqrt(ce * log_N_vertex / self.nMC[n])
            return mu + sqrt

        def ucb1tuned(n):
            mu = self.qMC[n] / self.nMC[n]
            sqrt = math.sqrt(ce * log_N_vertex / self.nMC[n])
            return mu + sqrt * math.sqrt(min(0.25, mu * (1 - mu) + sqrt))

        return max(self.children[node], key=eval(self.method))

    def visualize(self, prog='twopi'):
        if self.visual:
            for node in self.children:
                self.G.add_node(node)
                self.G.add_edges_from(
                    [(node, child) for child in self.children[node]])
            pos = self._layout(prog=prog)
            node_color = [self.colors[len(n)] for n in self.G.nodes]
            node_labels = {n: self._node_label(n) for n in self.G.nodes}
            nx.draw(self.G, pos=pos, node_color=node_color, node_size=100, 
                width=0.5, edge_color='grey', with_labels=True, 
                font_size=6, font_weight='heavy')
            label_pos = {n: (x, y-8) for (n, (x, y)) in pos.items()}
            nx.draw_networkx_labels(self.G, label_pos, labels=node_labels, 
                font_size=6, font_weight='heavy', font_color='r')
            plt.show()

    def _node_label(self, node):
        q = self.qMC[node]
        n = self.nMC[node]
        if n:
            return f'{q}/{n}={q/n:5.2%}'
        return '0/0'
    
    def _layout(self, prog):
        auxG = nx.DiGraph(self.G)
        for node1 in auxG.nodes:
            for node2 in auxG.nodes:
                diff = len(node2) - len(node1)
                if diff == 1 and node1.is_parent_of(node2):
                    auxG.add_edge(node1, node2)
                elif (node1, node2) in auxG.edges:
                    auxG.remove_edge(node1, node2)    
        return graphviz_layout(auxG, prog=prog)
