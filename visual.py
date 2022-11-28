"""
@file   visual.py
@brief  script for visualization of the tree evolution (Section 4.3)

@author Tzu-Yi Chiu <tzuyi.chiu@gmail.com>
"""

import os
import copy
import itertools
import numpy as np
np.random.seed(42)

from typing import Dict, Set, List

from mcts import MCTS
from stl import STL, PrimitiveGenerator
from simulators.thermostat import Thermostat

import networkx as nx
import matplotlib.pyplot as plt
from networkx.drawing.nx_pydot import graphviz_layout

class Visual:
    "For visualization of the DAG. See end of file for usage."

    def __init__(self, tree: MCTS, 
                       prog: str = 'twopi', 
                       save: bool = False) -> None:
        auxG = nx.DiGraph()
        for node, children in tree.children.items():
            auxG.add_node(node)
            auxG.add_edges_from([(node, child) for child in children])
        for node in tree.pruned:
            auxG.add_node(node)
        for node1, node2 in itertools.combinations(auxG.nodes, 2):
            if len(node2) - len(node1) != 1 and (node1, node2) in auxG.edges:
                auxG.remove_edge(node1, node2)
        self.pos = graphviz_layout(auxG, prog=prog)
        self.save = save
        self.font_size = 12

    def visualize(self, i: int, 
                        children: Dict[STL, Set[STL]], 
                        pruned: Set[STL], 
                        Q: Dict[STL, int], 
                        N: Dict[STL, int], 
                        path: List[STL]) -> None:
        def node_label(node, is_pruned):
            if is_pruned:
                return '\npruned'
            q = Q[node]
            n = N[node]
            return f'\n{q}/{n}={q/n:5.2%}' if n else '\nnot visited yet'
        
        auxG = nx.DiGraph()
        for node, children in children.items():
            auxG.add_node(node)
            auxG.add_edges_from([(node, child) for child in children])
        for node in pruned:
            auxG.add_node(node)
        
        node_color = [(0,1,0)] * len(auxG.nodes)
        path_labels, other_labels = {}, {}
        for n in auxG.nodes:
            label = node_label(n, n in pruned) 
            if n in path:
                path_labels[n] = label
            else:
                other_labels[n] = label
        
        plt.figure(figsize=(13, 5))
        plt.title(f'{i} rollout' + ('s' if i > 1 else ''))
        plt.tight_layout()
        nx.draw(auxG, pos=self.pos, node_color=node_color, node_size=100, 
            width=0.75, edge_color='grey', with_labels=True, 
            font_size=self.font_size, font_weight='heavy')
        d = 2 ** len(path)
        label_pos = {n: (x, y-d) for (n, (x, y)) in self.pos.items()}
        nx.draw_networkx_labels(auxG, label_pos, labels=path_labels, 
            font_size=self.font_size, font_weight='heavy', font_color='r')
        nx.draw_networkx_labels(auxG, label_pos, labels=other_labels, 
            font_size=self.font_size, font_weight='heavy', font_color='b')
        if self.save:
            plt.savefig(os.path.join('demo', 'rollouts', f'rollout{i}.png'))
        else:
            plt.show()
    
    def draw_sth(self, i: int, 
                       children: Dict[STL, Set[STL]], 
                       pruned: Set[STL], 
                       Q: Dict[STL, int], 
                       N: Dict[STL, int], 
                       path: List[STL]) -> None:
        def node_label(node, is_pruned):
            if is_pruned:
                return '\npruned'
            q = Q[node]
            n = N[node]
            return f'\n{q}/{n}={q/n:5.2%}' if n else '\nnot visited yet'
        
        auxG = nx.DiGraph()
        for node, children in children.items():
            auxG.add_node(node)
            auxG.add_edges_from([(node, child) for child in children])
        for node in pruned:
            auxG.add_node(node)
        
        node_color = [(0,1,0)] * len(auxG.nodes)
        color_labels, other_labels = {}, {}
        for n in auxG.nodes:
            label = node_label(n, n in pruned) 
            if len(n) < 2 and Q[n] == N[n]:
                color_labels[n] = label
            else:
                other_labels[n] = label
        
        plt.figure(figsize=(12, 6))
        plt.title(f'{i} rollouts: maximize coverage')
        plt.tight_layout()
        nx.draw(auxG, pos=self.pos, node_color=node_color, node_size=100, 
            width=0.75, edge_color='grey', with_labels=True, 
            font_size=self.font_size, font_weight='heavy')
        d = 2 ** len(path)
        label_pos = {n: (x, y-d) for (n, (x, y)) in self.pos.items()}
        nx.draw_networkx_labels(auxG, label_pos, labels=color_labels, 
            font_size=self.font_size, font_weight='heavy', font_color='r')
        nx.draw_networkx_labels(auxG, label_pos, labels=other_labels, 
            font_size=self.font_size, font_weight='heavy', font_color='b')
        if self.save:
            plt.savefig(os.path.join('demo', 'max_cov.png'))
        else:
            plt.show()

"""
Usage: python3 visual.py

This shows the first snapshots of the iterations of our algorithm and the 
first snapshorts of the tree (DAG) in the case study of the intelligent 
thermostat introduced in Section 4.3.
"""
if __name__ == '__main__':
    simulator   = Thermostat(out_temp=19, exp_temp=20, 
                             latency=2, length=5, memory=2)
    s           = np.array([[19, 21]])
    srange      = [(18, 22, 4)]
    tau         = 1.0
    rho         = 0.01
    epsilon     = 0.01
    past        = False
    batch_size  = 256
    max_depth   = 2
    max_iter    = 10000
    print(f'Simulator: thermostat')
    print(f'Signal being analyzed:\n{s}')
    print(f'range = {srange}')
    print(f'tau = {tau}')
    print(f'rho = {rho}')
    print(f'epsilon = {epsilon}')
    print(f'batch_size = {batch_size}')
    print(f'max_depth = {max_depth}')
    print(f'max_iter = {max_iter}')

    stl = STL() # The trivial formula: T
    primitives = PrimitiveGenerator(s, srange, rho, past).generate()    
    nb = stl.init(primitives)
    tree = MCTS(simulator, epsilon, tau, batch_size, max_depth, max_iter)
    trees = []
    paths = []
    rollouts = 16
    for i in range(rollouts):
        paths.append(tree._rollout(stl))
        trees.append(({n: copy.copy(children) 
                          for n, children in tree.children.items()}, 
                      copy.copy(tree.pruned), 
                      copy.copy(tree.Q), 
                      copy.copy(tree.N)))
    visual = Visual(tree)#, save=True)
    for i in range(rollouts):
        if i == 9:
            visual.font_size = 10
        if i > 9:
            visual.font_size = 8
        visual.visualize(i, *trees[i], paths[i])
    
    # to show how termination works (maximizing coverage)
    visual.font_size = 8
    visual.draw_sth(i, *trees[i], paths[i])
