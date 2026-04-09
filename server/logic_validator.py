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
        self.global_consensus_nx = consensus_graph_nx
        self.global_consensus_data = self._nx_to_pyg_data(consensus_graph_nx).to(self.device)
        
    def _nx_to_pyg_data(self, nx_graph):
        """
        Converts a NetworkX graph to a PyTorch Geometric Data object 
        ensuring deterministic tensor alignment via alphabetical node mapping.
        """
        import networkx as nx
        from torch_geometric.data import Data
        
        # 1. Map nodes deterministically by alphabetical feature name
        sorted_nodes = sorted(list(nx_graph.nodes()))
        node_to_idx = {node: i for i, node in enumerate(sorted_nodes)}
        
        # 2. Build explicit PyG format directly
        num_nodes = len(sorted_nodes)
        x = torch.ones((num_nodes, 1), dtype=torch.float32) # Default [1.0] feature for all nodes
        
        edge_list = []
        for u, v in nx_graph.edges():
            if u in node_to_idx and v in node_to_idx:
                edge_list.append([node_to_idx[u], node_to_idx[v]])
                
        if edge_list:
            edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()
        else:
            edge_index = torch.empty((2, 0), dtype=torch.long)
            
        pyg_data = Data(x=x, edge_index=edge_index)
        pyg_data.batch = torch.zeros(pyg_data.x.size(0), dtype=torch.long)
        return pyg_data

    def update_dynamic_threshold(self, client_scores):
        """
        Dynamically updates the operational SimGNN rejection boundary based entirely
        on the statistical topological spread of the network. This intrinsically captures
        Relative Rank Outlier filtering (e.g. Krum) avoiding absolute boundary deadlocks.
        """
        import numpy as np
        if not client_scores:
            return
            
        # Read exact configuration from params.yaml dynamically
        try:
            import yaml
            with open("params.yaml", "r") as f:
                p = yaml.safe_load(f)
            num_clients = int(p.get("simulation", {}).get("num_clients", 30))
            num_false_nodes = int(p.get("simulation", {}).get("num_false_nodes", 5))
            
            # Calculate organic geometric honest threshold
            # e.g., 5 false / 30 clients = 16.6% adversarial. 
            # 100% - 16.6% = 83.33%. We multiply by 0.95 to give a slight safety buffer against severe organic stragglers just in case!
            honest_ratio = ((num_clients - num_false_nodes) / num_clients)
            target_percentile = max(50.0, honest_ratio * 100 * 0.95) # Never drop below median defensively
            
        except Exception:
            target_percentile = 75.0
            
        next_boundary = float(np.percentile(client_scores, target_percentile))
        
        # Enforce that the dynamic threshold never drops below the rigid mathematical framework limit configured.
        self.dynamic_threshold = max(self.threshold, next_boundary)

    def evaluate_client_graph(self, client_graph_nx, server_round: int = 2):
        """
        Evaluates a single client's causal graph against the global consensus.
        Returns:
            is_accepted (bool): True if GED <= threshold, False otherwise.
            score (float): The calculated GED score.
        """
        # Accept automatically if there is no global consensus yet, or if it is Round 0.
        # Round 0 is allowed to train naturally to generate the intrinsic structure baseline.
        if server_round <= 0 or not hasattr(self, 'global_consensus_data') or self.global_consensus_data.x.size(0) == 0:
            return True, 0.0
            
        self.simgnn.eval()
        with torch.no_grad():
            client_data = self._nx_to_pyg_data(client_graph_nx).to(self.device)
            score = self.simgnn(client_data, self.global_consensus_data).item()
            
        # Statistical Outlier Anchor mapping
        # Rather than guessing at an absolute curve, we ride the organic structural hallucination array.
        if not hasattr(self, 'dynamic_threshold'):
            self.dynamic_threshold = 0.85 # Let Round 1 be lenient to collect pure data spread
            
        active_threshold = self.dynamic_threshold
        is_accepted = score <= active_threshold

        # --- DIAGNOSTIC TELEMETRY LOGGER ---
        # Computes true mathematical GED (Jaccard-edge) alongside SimGNN prediction
        # so the debug_graphs_log.txt shows both values every round for validation.
        try:
            edges_client = set(client_graph_nx.edges(data=False))
            edges_consensus = set(self.global_consensus_nx.edges(data=False))
            union_edges = len(edges_client.union(edges_consensus))
            if union_edges == 0:
                true_ged = 0.0
            else:
                diff = len(edges_client.symmetric_difference(edges_consensus))
                true_ged = min(1.0, float(diff) / union_edges)

            import os
            try:
                with open("params.yaml", "r") as _pf:
                    _pcfg = yaml.safe_load(_pf)
                _ds_name = _pcfg.get("dataset", {}).get("name", "asia")
            except Exception:
                _ds_name = "asia"
            _log_dir = os.path.join("saved_models", _ds_name)
            os.makedirs(_log_dir, exist_ok=True)
            _log_path = os.path.join(_log_dir, "debug_graphs_log.txt")
            with open(_log_path, "a") as df:
                df.write(f"--- Round {server_round} Evaluation ---\n")
                df.write(f"Consensus Edges ({len(edges_consensus)}): {sorted(list(edges_consensus))}\n")
                df.write(f"Client Edges ({len(edges_client)}): {sorted(list(edges_client))}\n")
                df.write(f"TRUE MATH GED: {true_ged:.4f}  |  SIMGNN PREDICTED GED: {score:.4f}\n")
                df.write(f"Status: {'ACCEPTED' if is_accepted else 'REJECTED'} (Threshold: {active_threshold:.4f})\n\n")
        except Exception:
            pass

        return is_accepted, score
