import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_add_pool
from torch_geometric.data import Data, Batch

class SimGNN(nn.Module):
    """
    Siamese Graph Neural Network (SimGNN) for approximating Graph Edit Distance (GED).
    Based on the architecture from 'SimGNN: A Neural Network Approach to Fast Graph Similarity Computation'
    (Bai et al., WSDM 2019) simplified for this specific Proof of Reasoning framework.
    """
    def __init__(self, node_feature_dim=1, hidden_dim=64, num_layers=3):
        super(SimGNN, self).__init__()
        self.num_layers = num_layers
        
        # GCN Layers
        self.convs = nn.ModuleList()
        self.convs.append(GCNConv(node_feature_dim, hidden_dim))
        for _ in range(num_layers - 1):
            self.convs.append(GCNConv(hidden_dim, hidden_dim))
            
        # Neural Tensor Network (NTN) layer approximations
        # In a full SimGNN, we'd have a full NTN. Here we use a simpler bilinear + dense layer approach
        # to combine the graph-level embeddings
        self.fc1 = nn.Linear(hidden_dim * 2, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 1)

    def forward_once(self, data):
        """
        Processes a single graph to produce a graph-level embedding.
        """
        x, edge_index, batch = data.x, data.edge_index, data.batch
        
        # Pass through GCN layers
        for conv in self.convs:
            x = F.relu(conv(x, edge_index))
            
        # Global pooling to get graph-level embedding
        # We use add pooling to maintain structural size information
        graph_embedding = global_add_pool(x, batch)
        
        return graph_embedding

    def forward(self, data1, data2):
        """
        Processes two graphs and calculates their logic distance (approximated GED).
        """
        # Get embeddings for both graphs
        emb1 = self.forward_once(data1)
        emb2 = self.forward_once(data2)
        
        # Combine embeddings for comparison
        # Using concatenation and absolute difference
        combined = torch.cat([emb1, emb2], dim=-1)
        
        # Pass through fully connected layers to get similarity score
        x = F.relu(self.fc1(combined))
        # The output score represents the estimated GED / Logic Distance
        score = torch.sigmoid(self.fc2(x))  # Using sigmoid if normalized, or no activation for unbounded GED
        
        return score.squeeze(-1)


class LogicValidator:
    """
    The Governance Module that runs on the server to validate client causal graphs.
    """
    def __init__(self, model_path=None, threshold=0.5):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.simgnn = SimGNN().to(self.device)
        self.threshold = threshold
        
        if model_path and os.path.exists(model_path):
            self.simgnn.load_state_dict(torch.load(model_path, map_location=self.device))
            self.simgnn.eval()
            
    def set_global_consensus(self, consensus_graph_nx):
        """
        Sets the global consensus graph against which client graphs are compared.
        consensus_graph_nx is expected to be a networkx DiGraph.
        """
        self.global_consensus_data = self._nx_to_pyg_data(consensus_graph_nx).to(self.device)
        
    def _nx_to_pyg_data(self, nx_graph):
        """
        Converts a NetworkX graph to a PyTorch Geometric Data object.
        """
        import networkx as nx
        from torch_geometric.utils import from_networkx
        
        # Ensure all nodes have a default feature if none exists
        for node in nx_graph.nodes:
            if 'x' not in nx_graph.nodes[node]:
                nx_graph.nodes[node]['x'] = [1.0] # default feature
                
        pyg_data = from_networkx(nx_graph)
        
        # Handle the case where the graph is completely empty
        if hasattr(pyg_data, 'x') and pyg_data.x is not None:
            pyg_data.x = torch.tensor(pyg_data.x, dtype=torch.float32).view(-1, 1) # Assuming 1D features
        else:
            # Empty graph fallback
            pyg_data.x = torch.zeros((0, 1), dtype=torch.float32)
            pyg_data.edge_index = torch.empty((2, 0), dtype=torch.long)
            
        # Add batch indicator since it's a single graph
        pyg_data.batch = torch.zeros(pyg_data.x.size(0), dtype=torch.long)
        return pyg_data

    def evaluate_client_graph(self, client_graph_nx):
        """
        Evaluates a single client's causal graph against the global consensus.
        Returns:
            is_accepted (bool): True if GED <= threshold, False otherwise.
            score (float): The calculated GED score.
        """
        # Accept automatically if there is no global consensus yet (Round 1)
        if not hasattr(self, 'global_consensus_data') or self.global_consensus_data.x.size(0) == 0:
            return True, 0.0
            
        self.simgnn.eval()
        with torch.no_grad():
            client_data = self._nx_to_pyg_data(client_graph_nx).to(self.device)
            score = self.simgnn(client_data, self.global_consensus_data).item()
            
        is_accepted = score <= self.threshold
        return is_accepted, score
