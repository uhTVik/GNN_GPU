
# First networkx library is imported 
# along with matplotlib
import networkx as nx
import matplotlib.pyplot as plt
import json


# Defining a Class
class GraphVisualization:

    def __init__(self):
        # visual is a list which stores all
        # the set of edges that constitutes a
        # graph
        self.visual = []

    # addEdge function inputs the vertices of an
    # edge and appends it to the visual list
    def addEdge(self, a, b):
        temp = [a, b]
        self.visual.append(temp)

    # In visualize function G is an object of
    # class Graph given by networkx G.add_edges_from(visual)
    # creates a graph with a given list
    # nx.draw_networkx(G) - plots the graph
    # plt.show() - displays the graph
    def visualize(self):
        G = nx.Graph()
        G.add_edges_from(self.visual)
        nx.draw_networkx(G, with_labels=False, node_size=5, width=0.3, alpha=0.5)
        plt.show()


data_file_name_all = "/Users/aleksandrukhatov/Documents/Projects/GNN_GPU/data/all_connections.json"
data_file_name_some = "/Users/aleksandrukhatov/Documents/Projects/GNN_GPU/data/some_connections.json"


data = json.load(open(data_file_name_some, "r"))


# Driver code
G = GraphVisualization()

for id_i in data:
    for id_j in data[id_i]:
        G.addEdge(id_i, id_j)
# G.addEdge(0, 2)
# G.addEdge(1, 2)
# G.addEdge(1, 3)
# G.addEdge(5, 3)
# G.addEdge(3, 4)
# G.addEdge(1, 0)
G.visualize()