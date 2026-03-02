import flwr as fl
from typing import Callable, Dict, List, Optional, Tuple, Union
from flwr.common import (
    FitRes,
    MetricsAggregationFn,
    NDArrays,
    Parameters,
    Scalar,
    ndarrays_to_parameters,
    parameters_to_ndarrays,
)
from flwr.server.client_proxy import ClientProxy
import numpy as np
import networkx as nx
import logging
import pickle

import os
import torch
from collections import OrderedDict

from .logic_validator import LogicValidator
# Import the client model class to load weights into for saving
try:
    from client.models import Model
except ImportError:
    Model = None

log = logging.getLogger(__name__)

class PoRStrategy(fl.server.strategy.FedAvg):
    """
    Causal Proof of Reasoning (PoR) Dual Aggregator Strategy.
    Inherits from FedAvg, but incorporates a Logic Validator step before aggregating.
    """
    
    def __init__(
        self,
        logic_validator: LogicValidator,
        *args,
        **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.logic_validator = logic_validator
        self.global_consensus_graph = nx.DiGraph() # start with an empty or base graph
        
        # Resume consensus logic if exists
        try:
            import pickle
            if os.path.exists("saved_models/global_consensus_graph.gpickle"):
                with open("saved_models/global_consensus_graph.gpickle", "rb") as f:
                    self.global_consensus_graph = pickle.load(f)
                log.info("Resumed Global Consensus Graph from previous simulation run!")
        except Exception as e:
            log.warning(f"Could not load previous consensus graph: {e}")
            
        self.logic_validator.set_global_consensus(self.global_consensus_graph)

    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, FitRes]],
        failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]],
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        """Aggregate fit results using logic validation filtering."""
        
        if not results:
            return None, {}

        # 1. Filter clients using Logic Validator
        accepted_results = []
        accepted_graphs = []
        rejected_graphs = []
        rejected_count = 0
        
        for client, fit_res in results:
            metrics = fit_res.metrics
            # The client sends the causal graph adjacency list embedded into metrics
            if "causal_graph_edges" in metrics:
                # Reconstruct graph from edges string, e.g., "[[0, 1], [1, 2]]"
                edges = eval(metrics["causal_graph_edges"])
                client_graph = nx.DiGraph()
                client_graph.add_edges_from(edges)
                
                # Make sure all nodes from consensus are represented
                client_graph.add_nodes_from(self.global_consensus_graph.nodes())
                
                is_valid, score = self.logic_validator.evaluate_client_graph(client_graph)
                
                if is_valid:
                    log.info(f"Client {client.cid} ACCEPTED. Score: {score:.4f} <= {self.logic_validator.threshold}")
                    accepted_results.append((client, fit_res))
                    accepted_graphs.append(client_graph)
                else:
                    log.warning(f"Client {client.cid} REJECTED by PoR check. Score: {score:.4f} > {self.logic_validator.threshold}")
                    rejected_count += 1
                    rejected_graphs.append(client_graph)
            else:
                log.warning(f"Client {client.cid} did not provide causal graph. REJECTING.")
                rejected_count += 1
                
        metrics_aggregated = {
            "accepted_clients": len(accepted_results),
            "rejected_clients": rejected_count
        }

        if not accepted_results:
            log.error("All clients rejected! Cannot aggregate.")
            return None, metrics_aggregated

        # 2. Aggregate the Weights (calling the parent FedAvg logic with filtered results)
        aggregated_parameters, _ = super().aggregate_fit(server_round, accepted_results, failures)

        # 3. Aggregate the Logic (Barycenter Edge Retention)
        self._aggregate_logic(accepted_graphs)

        # 4. Save the aggregated weights to disk
        if aggregated_parameters is not None and Model is not None:
            print(f"[ROUND {server_round}] Saving aggregated Global Model parameters...")
            self._save_global_model(aggregated_parameters, server_round)
            
            os.makedirs("saved_models", exist_ok=True)
            if accepted_graphs:
                with open("saved_models/honest_graph_sample.gpickle", "wb") as f:
                    pickle.dump(accepted_graphs[0], f)
            if rejected_graphs:
                with open("saved_models/adversarial_graph_sample.gpickle", "wb") as f:
                    pickle.dump(rejected_graphs[0], f)

        return aggregated_parameters, metrics_aggregated

    def _save_global_model(self, parameters: Parameters, round_num: int):
        """Reconstructs the PyTorch model from Flower Parameters and saves it."""
        try:
            os.makedirs("saved_models", exist_ok=True)
            ndarrays = parameters_to_ndarrays(parameters)
            
            # Initialize a blank model (make sure this matches the client architectures)
            model = Model()
            
            # Load the NDArrays into the model's state_dict
            params_dict = zip(model.state_dict().keys(), ndarrays)
            state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
            model.load_state_dict(state_dict, strict=True)
            
            # Save the PyTorch Model
            torch.save(model.state_dict(), f"saved_models/global_model_round_{round_num}.pt")
            # Save the latest as 'global_model.pt'
            torch.save(model.state_dict(), "saved_models/global_model.pt")
            
            # Save the global consensus graph
            with open("saved_models/global_consensus_graph.gpickle", "wb") as f:
                pickle.dump(self.global_consensus_graph, f)
        except Exception as e:
            log.error(f"Failed to save global model weights: {e}")

    def _aggregate_logic(self, client_graphs: List[nx.DiGraph]):
        """
        Updates the global consensus graph.
        An edge is retained if it appears in > 50% of the accepted graphs.
        """
        if not client_graphs:
            return
            
        edge_counts = {}
        target_votes = len(client_graphs) / 2.0
        
        # Accumulate edge votes
        for g in client_graphs:
            for u, v in g.edges():
                if (u, v) not in edge_counts:
                    edge_counts[(u, v)] = 0
                edge_counts[(u, v)] += 1
                
        # Build new consensus graph
        new_consensus = nx.DiGraph()
        
        # Ensure all nodes exist
        for g in client_graphs:
            new_consensus.add_nodes_from(g.nodes())
            
        # Add majority edges
        for edge, count in edge_counts.items():
            if count > target_votes:
                new_consensus.add_edge(*edge)
                
        self.global_consensus_graph = new_consensus
        self.logic_validator.set_global_consensus(self.global_consensus_graph)
