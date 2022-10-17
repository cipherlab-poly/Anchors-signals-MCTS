import copy
import numpy as np
from mcts import MCTS
from stl import STL, PrimitiveGenerator
from simulators.thermostat import Thermostat

import networkx as nx
import matplotlib.pyplot as plt
import itertools
from networkx.drawing.nx_pydot import graphviz_layout

class Visual:
    def __init__(self, tree, prog='twopi', save=False):
        #self.colors = list(itertools.product(*([(0, 1)]*3)))
        #self.colors.remove((0, 0, 0))
        #self.colors.remove((1, 1, 1))

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

    def visualize(self, i, children, pruned, Q, N, path):
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
        
        #node_color = [self.colors[len(n)] for n in auxG.nodes]
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
            plt.savefig(f'demo/rollout{i}.png')
        else:
            plt.show()
    
    def draw_sth(self, i, children, pruned, Q, N, path):
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
            plt.savefig(f'demo/max_cov.png')
        else:
            plt.show()

if __name__ == '__main__':
    # explain why the thermostat is off 
    # (temp > 20 once in the last 2 timestamps)
    simulator   = Thermostat(out_temp=19, exp_temp=20, latency=2, length=5)
    s           = np.array([[19, 21]])
    srange      = [(0, (18, 22, 4))]
    tau         = 1.0
    rho         = 0.01
    epsilon     = 0.01
    past        = False
    batch_size  = 256
    max_depth   = 2
    max_iter    = 10000

    stl = STL()
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
    visual = Visual(tree, save=True)
    #for i in range(rollouts):
    #    if i == 9:
    #        visual.font_size = 10
    #    if i > 9:
    #        visual.font_size = 8
    #    visual.visualize(i, *trees[i], paths[i])
    visual.font_size = 8
    visual.draw_sth(i, *trees[i], paths[i])
