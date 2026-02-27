import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool

class GNNEncoder(nn.Module):
    """
    A simple Graph Convolutional Network to embed causal DAGs into a continuous vector space.
    """
    def __init__(self, node_features: int, hidden_dim: int, embedding_dim: int):
        super(GNNEncoder, self).__init__()
        self.conv1 = GCNConv(node_features, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.conv3 = GCNConv(hidden_dim, embedding_dim)
        
    def forward(self, x, edge_index, batch):
        # x: Node feature matrix, edge_index: Graph connectivity
        # batch: Batch vector mapping each node to its respective graph
        
        # 1. Obtain node embeddings 
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        x = self.conv3(x, edge_index)
        
        # 2. Readout layer (Global Pooling) to get graph embedding
        x = global_mean_pool(x, batch)  # [batch_size, embedding_dim]
        
        return x

class SiameseGNN(nn.Module):
    """
    Siamese Network to calculate the similarity (Neural GED representation) 
    between two causal graphs (e.g., Client graph and Global Consensus graph).
    """
    def __init__(self, node_features: int = 1, hidden_dim: int = 16, embedding_dim: int = 32):
        super(SiameseGNN, self).__init__()
        self.encoder = GNNEncoder(node_features, hidden_dim, embedding_dim)
        
    def forward_one(self, data):
        """Pass a single graph through the encoder."""
        return self.encoder(data.x, data.edge_index, data.batch)
        
    def forward(self, data1, data2):
        """
        Pass both graphs through the Siamese GNN.
        Returns the squared Euclidean distance between their embeddings.
        """
        emb1 = self.forward_one(data1)
        emb2 = self.forward_one(data2)
        
        # Calculate squared Euclidean distance: ||GNN(Gc) - GNN(Gg)||^2
        distance = F.pairwise_distance(emb1, emb2, p=2) ** 2
        return distance

def check_logic_distance(sim_gnn, client_graph_data, global_graph_data, threshold: float):
    """
    Utility function used by the governance layer.
    Returns True if the update is accepted (distance <= threshold), False if rejected.
    """
    sim_gnn.eval()
    with torch.no_grad():
        distance = sim_gnn(client_graph_data, global_graph_data)
        
    # distance is a tensor containing batch distances. 
    # For a single comparison, we take the first item.
    dist_val = distance.item()
    accept = dist_val <= threshold
    return accept, dist_val
