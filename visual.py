import networkx as nx
import matplotlib.pyplot as plt
import pydot
from networkx.drawing.nx_pydot import graphviz_layout

class Visual:
    def __init__(self, tree):
        self.tree = tree
        self.G = nx.DiGraph()
        self.colors = list(itertools.product(*([(0, 1)]*3)))
        self.colors.remove((0, 0, 0))
        self.colors.remove((1, 1, 1))

    def visualize(self, prog='twopi'):
        for node in self.tree.children:
            self.G.add_node(node)
            self.G.add_edges_from(
                [(node, child) for child in self.tree.children[node]])
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
        q = self.tree.Q[node]
        n = self.tree.N[node]
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
