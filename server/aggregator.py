import json
from typing import Dict, List, Optional, Tuple, Union

import flwr as fl
import networkx as nx
import numpy as np
import torch
from flwr.common import FitRes, Parameters, Scalar
from flwr.server.client_proxy import ClientProxy
from torch_geometric.data import Data

from .logic_validator import SiameseGNN, check_logic_distance

class PoRDualStrategy(fl.server.strategy.FedAvg):
    """
    Custom Dual Aggregation Strategy for Proof of Reasoning (PoR).
    Filters clients based on Causal Graph Edit Distance before aggregating weights.
    Aggregates logic graphs using barycenter averaging (>50% consensus) for accepted clients.
    """
    def __init__(
        self,
        tau: float,
        sim_gnn: SiameseGNN,
        initial_consensus_graph: nx.DiGraph,
        num_features: int = 10,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.tau = tau
        self.sim_gnn = sim_gnn
        self.global_consensus_graph = initial_consensus_graph
        self.num_features = num_features

    def _nx_to_pyg_data(self, graph: nx.DiGraph) -> Data:
        """Convert a networkx DAG to PyTorch Geometric Data object."""
        edges = list(graph.edges)
        if len(edges) == 0:
            edge_index = torch.empty((2, 0), dtype=torch.long)
        else:
            edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
        
        # We assume simple node features for structural embedding (e.g. ones)
        # In a real setup, nodes could hold semantic feature embeddings.
        x = torch.ones((self.num_features, 1), dtype=torch.float)
        
        # Batch vector for a single graph
        batch = torch.zeros(self.num_features, dtype=torch.long)
        
        return Data(x=x, edge_index=edge_index, batch=batch)

    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, FitRes]],
        failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]],
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        if not results:
            return None, {}

        accepted_results = []
        rejected_clients = 0
        accepted_graphs = []

        global_pyg = self._nx_to_pyg_data(self.global_consensus_graph)

        print(f"\n--- Round {server_round} Governance Check ---")
        for client, fit_res in results:
            # Reconstruct the client's causal graph from the metrics payload
            metrics = fit_res.metrics
            client_graph_json = metrics.get("causal_graph", "{}")
            
            try:
                edge_list = json.loads(client_graph_json)
                client_nx_graph = nx.DiGraph(edge_list)
                # Ensure the graph has the right number of nodes even if some are isolated
                client_nx_graph.add_nodes_from(range(self.num_features))
            except Exception as e:
                print(f"Failed to parse graph from {client.cid}. Rejecting. Error: {e}")
                rejected_clients += 1
                continue

            client_pyg = self._nx_to_pyg_data(client_nx_graph)
            
            # Logic Validation using Siamese GNN
            accept, dist = check_logic_distance(self.sim_gnn, client_pyg, global_pyg, self.tau)
            
            if accept:
                print(f"Client {client.cid[:8]} ACCEPTED. Logic Distance: {dist:.4f} <= {self.tau}")
                accepted_results.append((client, fit_res))
                accepted_graphs.append(client_nx_graph)
            else:
                print(f"Client {client.cid[:8]} REJECTED (Anomaly). Logic Distance: {dist:.4f} > {self.tau}")
                rejected_clients += 1

        print(f"Total Accepted: {len(accepted_results)} | Total Rejected: {rejected_clients}")

        # 1. Logic Aggregation (Barycenter Graph Averaging)
        if len(accepted_graphs) > 0:
            new_global_graph = nx.DiGraph()
            new_global_graph.add_nodes_from(range(self.num_features))
            
            edge_counts = {}
            for g in accepted_graphs:
                for edge in g.edges():
                    edge_counts[edge] = edge_counts.get(edge, 0) + 1
            
            threshold_count = len(accepted_graphs) / 2.0
            
            for edge, count in edge_counts.items():
                if count > threshold_count:
                    new_global_graph.add_edge(*edge)
                    
            self.global_consensus_graph = new_global_graph
            print(f"Updated Consensus Graph Edges: {len(self.global_consensus_graph.edges())}")

        # 2. Weight Aggregation (using standard FedAvg algorithm on accepted payload)
        weights_aggregated, metrics_aggregated = super().aggregate_fit(server_round, accepted_results, failures)
        
        # Include custom metric for tracking rejection rate
        metrics_aggregated["rejection_rate"] = rejected_clients / len(results) if results else 0.0

        return weights_aggregated, metrics_aggregated

    def configure_fit(self, server_round: int, parameters: Parameters, client_manager: fl.server.client_manager.ClientManager) -> List[Tuple[ClientProxy, fl.common.FitIns]]:
        """
        Configure the next round. We inject the serialized global consensus graph 
        into the config dictionary so clients can compute the logic loss locally.
        """
        config = {
            "global_consensus_graph": json.dumps(list(self.global_consensus_graph.edges())),
            "server_round": server_round
        }
        
        fit_ins = fl.common.FitIns(parameters, config)
        
        # Get clients
        sample_size, min_num_clients = self.num_fit_clients(client_manager.num_available())
        clients = client_manager.sample(num_clients=sample_size, min_num_clients=min_num_clients)
        
        return [(client, fit_ins) for client in clients]
