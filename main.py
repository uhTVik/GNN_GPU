# Import library
import time
import metis
import torch
import torch_geometric
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
from tqdm.notebook import tqdm_notebook
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import GATConv
import torch_geometric.transforms as T
from torch_geometric.utils.convert import to_networkx
from torch_geometric.transforms import RandomNodeSplit

# Load dataset library
from torch_geometric.datasets import Reddit
# Download dataset
dataset = Reddit(root='./datasets/Reddit')
dataset.transform = T.NormalizeFeatures()

# Print dataset information
print('Dataset: {}'.format(dataset))
print('===========================')
print('Number of graphs: {}'.format(len(dataset)))
print('Number of features: {}'.format(dataset.num_features))
print('Number of classes: {}'.format(dataset.num_classes))

# Print dataset detailed information
graph_data = dataset[0]
print('Graph details')
print('===========================')
print('Number of nodes: {}'.format(graph_data.num_nodes))
print('Number of edges: {}'.format(graph_data.num_edges))
print('Average node degree: {:.2f}'.format(graph_data.num_edges / graph_data.num_nodes))
print('Number of training nodes: {}'.format(graph_data.train_mask.sum()))
print('Number of validation nodes: {}'.format(graph_data.val_mask.sum()))
print('Number of test nodes: {}'.format(graph_data.test_mask.sum()))
print('Training node label rate: {:.2f}'.format(int(graph_data.train_mask.sum()) / graph_data.num_nodes))
print('Contains isolated nodes: {}'.format(graph_data.has_isolated_nodes()))
print('Contains self loops: {}'.format(graph_data.has_self_loops()))
print('Is undirected: {}'.format(graph_data.is_undirected()))


class GraphTools(object):
    def __init__(self, graph):
        self.graph = graph
        self.node_label = graph.y

    def visualize_graph(self, color=None, epoch=None, loss=None):
        print("[Graph tools] Plotting graph")
        # Define plot properties
        plt.figure(figsize=(7, 7))
        plt.xticks([])
        plt.yticks([])
        rgb = color if (color is not (None)) else np.random.rand(3, ).reshape(1, -1)

        # Check whether input is in tensor or NX graph representation
        if (torch.is_tensor(self.graph)):
            # Convert tensor to numpy array
            graph_numpy = self.graph.detach().cpu().numpy()
            # Create scatter plot
            plt.scatter(graph_numpy[:, 0], graph_numpy[:, 0], s=140, color=rgb, cmap='Set2')
            # Print additional label
            if ((epoch is not None) and (loss is not None)):
                plt.xlabel('Epoch: {}, Loss: {:.4f}'.format(epoch, loss.item()), fontsize=16)
        else:
            # Convert graph to networkx format
            graph_nx = to_networkx(self.graph, to_undirected=True,
                                   node_attrs=['x'] if (self.graph.num_node_features) else None,
                                   edge_attrs=['edge_attr'] if (self.graph.num_edge_features) else None)
            nx.draw_networkx(graph_nx, pos=nx.spring_layout(graph_nx, seed=42), with_labels=False, node_size=100,
                             node_color=rgb, cmap='Set2')

        # Show graph
        plt.show()

    def decompose_graph(self):
        # Convert graph into networkX representation
        print("[Graph tools] Converting graph from tensor to networkX")
        self.nx_graph = to_networkx(self.graph, to_undirected=True,
                                    node_attrs=['x'] if (self.graph.num_node_features) else None,
                                    edge_attrs=['edge_attr'] if (self.graph.num_edge_features) else None)
        # Partition graph
        print("[Graph tools] Partitioning graph using metis")
        (edgecuts, parts) = metis.part_graph(self.nx_graph, 50)
        # Create cluster membership list
        self.clusters = list(set(parts))
        self.cluster_members = {node: member for node, member in enumerate(parts)}

    def generate_subgraph(self):
        self.subgraph_nodes = {}
        self.subgraph_edges = {}
        self.subgraph_node_features = {}
        self.subgraph_edge_features = {}
        self.subgraph_node_labels = {}
        for cluster in self.clusters:
            subgraph = self.nx_graph.subgraph(
                [node for node in sorted(self.nx_graph.nodes()) if (self.cluster_members[node] == cluster)])
            self.subgraph_nodes[cluster] = [node[0] for node in sorted(subgraph.nodes(data=True))]
            mapper = {node: i for i, node in enumerate(sorted(self.subgraph_nodes[cluster]))}
            self.subgraph_edges[cluster] = [[mapper[edge[0]], mapper[edge[1]]] for edge in subgraph.edges()] + [
                [mapper[edge[1]], mapper[edge[0]]] for edge in subgraph.edges()]
            self.subgraph_node_features[cluster] = [node[1]['x'] for node in sorted(subgraph.nodes(data=True))]
            self.subgraph_node_labels[cluster] = self.node_label[self.subgraph_nodes[cluster]]

    def subgraph_to_tensor(self):
        for cluster in self.clusters:
            self.subgraph_nodes[cluster] = torch.LongTensor(self.subgraph_nodes[cluster])
            self.subgraph_edges[cluster] = torch.LongTensor(self.subgraph_edges[cluster])
            self.subgraph_node_features[cluster] = torch.FloatTensor(self.subgraph_node_features[cluster])
            self.subgraph_node_labels[cluster] = torch.LongTensor(self.subgraph_node_labels[cluster])


graph_tools = GraphTools(graph_data)
# graph_tools.visualize_graph()
graph_tools.decompose_graph()
graph_tools.generate_subgraph()
graph_tools.subgraph_to_tensor()

new_subgraph = Data(x=graph_tools.subgraph_node_features[0], edge_index=graph_tools.subgraph_edges[0].t().contiguous(), y=graph_tools.subgraph_node_labels[0])

print('Graph details')
print('===========================')
print('Number of nodes: {}'.format(new_subgraph.num_nodes))
print('Number of edges: {}'.format(new_subgraph.num_edges))
print('Number of node features: {}'.format(new_subgraph.num_node_features))
print('Number of edge features: {}'.format(new_subgraph.num_edge_features))
print('Number of node labels: {}'.format(len(new_subgraph.y)))
print('Average node degree: {:.2f}'.format(new_subgraph.num_edges / new_subgraph.num_nodes))
print('Contains isolated nodes: {}'.format(new_subgraph.has_isolated_nodes()))
print('Contains self loops: {}'.format(new_subgraph.has_self_loops()))
print('Is undirected: {}'.format(new_subgraph.is_undirected()))

subgraph_tools = GraphTools(new_subgraph)
subgraph_tools.visualize_graph()

transform = RandomNodeSplit(split='random', num_train_per_class=20, num_val=0.1, num_test=0.2)
transform(new_subgraph)

print('Graph details')
print('===========================')
print('Number of nodes: {}'.format(new_subgraph.num_nodes))
print('Number of edges: {}'.format(new_subgraph.num_edges))
print('Number of node features: {}'.format(new_subgraph.num_node_features))
print('Number of edge features: {}'.format(new_subgraph.num_edge_features))
print('Average node degree: {:.2f}'.format(new_subgraph.num_edges / new_subgraph.num_nodes))
print('Number of training nodes: {}'.format(new_subgraph.train_mask.sum()))
print('Training node label rate: {:.2f}'.format(int(new_subgraph.train_mask.sum()) / new_subgraph.num_nodes))
print('Contains isolated nodes: {}'.format(new_subgraph.has_isolated_nodes()))
print('Contains self loops: {}'.format(new_subgraph.has_self_loops()))
print('Is undirected: {}'.format(new_subgraph.is_undirected()))

class GAT(nn.Module):
  def __init__(self):
    # Define class properties
    super(GAT, self).__init__()
    self.hidden_features = 8
    self.input_head = 8
    self.output_head = 1

    # Define attention mechanism
    self.attention1 = GATConv(dataset.num_features, self.hidden_features, heads=self.input_head, dropout=0.6)
    self.attention2 = GATConv((self.hidden_features*self.input_head), dataset.num_classes, concat=False, heads=self.output_head, dropout=0.6)

  def forward(self, data):
    # Define feedforward process
    x, edge_index = data.x, data.edge_index
    # First layer
    x = F.dropout(x, p=0.6, training=self.training)
    x = self.attention1(x, edge_index)
    x = F.elu(x)
    # Second layer
    x = F.dropout(x, p=0.6, training=self.training)
    x = self.attention2(x, edge_index)
    # Output value
    return F.log_softmax(x, dim=1)


# Configure GPU for training
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("Device: {}".format(device))

# Initialize model and dataset
model = GAT().to(device)
data = dataset[0].to(device)

# Define training parameters
optimizer = torch.optim.Adam(model.parameters(), lr=0.005, weight_decay=5e-4)
epochs  = 1500

# Start training process
model.train()
for epoch in range(epochs):
  model.train()
  optimizer.zero_grad()
  out = model(data)
  loss = F.nll_loss(out[data.train_mask], data.y[data.train_mask])

  if (epoch % 100) == 0:
    print("Epoch: {} - Loss: {}".format(epoch, loss))

  loss.backward()
  optimizer.step()

model.eval()
_, pred = model(data).max(dim=1)
correct = float(pred[data.test_mask].eq(data.y[data.test_mask]).sum().item())
acc = correct / data.test_mask.sum().item()
print('Accuracy: {:.4f}'.format(acc))

