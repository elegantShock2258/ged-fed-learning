"""
Module: server.logic_validator
================================
Description:
    Provides the two-class Logic Validator stack:

    1. **SimGNN** — a Siamese Graph Neural Network that approximates Graph Edit
       Distance (GED) between two causal graphs.  Takes two PyG ``Data`` objects
       and returns a scalar ``score ∈ [0, 1]`` where 0 = identical structure and
       1 = maximally different.

    2. **LogicValidator** — the governance wrapper used by PoRStrategy.  Holds the
       current global consensus graph as a PyG tensor, passes each client's graph
       through SimGNN, and returns a binary accept/reject decision based on whether
       the GED score exceeds the configured threshold τ.

    SimGNN Architecture:
        - Two shared GCN layers (hidden_dim=128) for structural feature extraction.
        - One GAT attention layer (2 heads) to weight critical causal nodes.
        - Mean + Max pooling (multi-pooling) produces graph-level embeddings.
        - Concatenated embeddings → FC layers → sigmoid → GED score ∈ [0, 1].

    Decision rule:
        ``accept`` ← ``SimGNN(client_graph, consensus_graph) <= threshold``

Inputs:
    - Client causal graph as networkx DiGraph (from client metrics dict).
    - Global consensus graph set via ``set_global_consensus()``.
    - ``model_path``: optional path to pre-trained SimGNN weights (.pt file).
    - ``threshold``: GED rejection threshold τ (from ``params.yaml``).

Outputs:
    - ``evaluate_client_graph(G)`` → ``(is_accepted: bool, ged_score: float)``
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv, global_mean_pool, global_max_pool
from torch_geometric.data import Data, Batch
import os
import yaml

with open("params.yaml", "r") as f:
    config = yaml.safe_load(f)
VALIDATOR_THRESHOLD = config["core_logic"]["validator_threshold"]

class SimGNN(nn.Module):
    """
    Siamese Graph Neural Network (SimGNN) for approximating Graph Edit Distance (GED).
    Upgraded for larger 64-node graphs using Attention and Multi-Pooling representations.
    """
    def __init__(self, node_feature_dim=1, hidden_dim=128, num_layers=3):
        super(SimGNN, self).__init__()
        self.num_layers = num_layers
        
        # GCN + GAT Layers for better structural feature extraction
        self.convs = nn.ModuleList()
        self.convs.append(GCNConv(node_feature_dim, hidden_dim))
        for _ in range(num_layers - 2):
            self.convs.append(GCNConv(hidden_dim, hidden_dim))
        # Final convolution is attention-based to weigh critical logic nodes
        self.gat = GATConv(hidden_dim, hidden_dim, heads=2, concat=False)
            
        # Neural Tensor Network (NTN) layer approximations
        # Since we use Mean + Max pooling, graph embedding size is hidden_dim * 2
        # Comparing two graphs = emb1, emb2, and |emb1 - emb2| -> (hidden_dim * 2) * 3
        combined_dim = hidden_dim * 6
        
        self.fc1 = nn.Linear(combined_dim, hidden_dim)
        self.dropout = nn.Dropout(0.2)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.fc3 = nn.Linear(hidden_dim // 2, 1)

    def forward_once(self, data):
        """
        Processes a single graph to produce a robust graph-level embedding.
        """
        x, edge_index, batch = data.x, data.edge_index, data.batch
        
        # Pass through GCN layers
        for conv in self.convs:
            x = F.relu(conv(x, edge_index))
            
        # Attention Layer
        x = F.relu(self.gat(x, edge_index))
            
        # Multi-Pooling to capture both average logic structure and critical edge extremities
        pool_mean = global_mean_pool(x, batch)
        pool_max = global_max_pool(x, batch)
        graph_embedding = torch.cat([pool_mean, pool_max], dim=-1) # shape: [batch_size, hidden_dim * 2]
        
        return graph_embedding

    def forward(self, data1, data2):
        """
        Processes two graphs and calculates their logic distance (approximated GED).
        """
        emb1 = self.forward_once(data1)
        emb2 = self.forward_once(data2)
        
        # Combine embeddings explicitly with absolute difference
        # This provides a much stronger gradient signal for structural divergence
        diff = torch.abs(emb1 - emb2)
        combined = torch.cat([emb1, emb2, diff], dim=-1)
        
        x = F.relu(self.fc1(combined))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        
        # Sigmoid bounds the predicted GED difference strictly between 0 and 1
        score = torch.sigmoid(self.fc3(x)) 
        
        return score.squeeze(-1)


class LogicValidator:
    """
    The Governance Module that runs on the server to validate client causal graphs.
    """
    def __init__(self, model_path=None, threshold=VALIDATOR_THRESHOLD):
        device_pref = config.get("hardware", {}).get("device", "auto").lower()
        if device_pref == "cpu":
            self.device = torch.device("cpu")
        else:
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
        
        for node in nx_graph.nodes:
            if 'x' not in nx_graph.nodes[node]:
                nx_graph.nodes[node]['x'] = [1.0] # default feature
                
        # Clear edge attributes to prevent mismatches
        for u, v in nx_graph.edges():
            nx_graph.edges[u, v].clear()
                
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

    def evaluate_client_graph(self, client_graph_nx, server_round: int = 2):
        """
        Evaluates a single client's causal graph against the global consensus.
        Returns:
            is_accepted (bool): True if GED <= threshold, False otherwise.
            score (float): The calculated GED score.
        """
        # Accept automatically if there is no global consensus yet, or if it is Round 1.
        # Round 1 is allowed to train naturally to generate the intrinsic structure baseline.
        if server_round <= 1 or not hasattr(self, 'global_consensus_data') or self.global_consensus_data.x.size(0) == 0:
            return True, 0.0
            
        self.simgnn.eval()
        with torch.no_grad():
            client_data = self._nx_to_pyg_data(client_graph_nx).to(self.device)
            score = self.simgnn(client_data, self.global_consensus_data).item()
            
        is_accepted = score <= self.threshold
        return is_accepted, score
