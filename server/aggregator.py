"""
Module: server.aggregator
==========================
Description:
    Defines PoRStrategy, the novel Causal Proof of Reasoning server-side aggregator.

    PoRStrategy extends Flower's FedAvg strategy with a two-stage aggregation:

    Stage 1 — Logic Gate (PoR Defense):
        For each client that submits a fit response, the server:
          a. Deserialises the causal graph edge list from the metrics dict.
          b. Calls LogicValidator.evaluate_client_graph() to compute the GED
             between the client's graph and the global consensus.
          c. Rejects clients whose GED score exceeds the threshold τ (``validator_threshold``).
          d. Only accepted client weights proceed to Stage 2.

    Stage 2 — Weight Aggregation:
        Calls FedAvg.aggregate_fit() on the accepted subset, producing a new
        global model via weighted averaging (weighted by dataset size).

    Post-aggregation (each round):
        - Runs ``_aggregate_logic()`` to update the consensus graph using a
          momentum-blended dual-threshold vote.
        - Runs ``_finetune_simgnn_on_consensus()`` to re-anchor SimGNN on the
          new consensus (on-the-fly adaptation to avoid stale embeddings).
        - Saves the updated consensus graph and global model weights to disk.

    Key parameters (all from ``params.yaml``):
        - ``core_logic.validator_threshold``  (τ): GED rejection threshold.
        - ``core_logic.consensus_momentum``  (m): Controls conservatism of graph updates.
        - ``core_logic.simgnn_lr``               : Fine-tuning learning rate.
"""

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
import random

import os
import yaml
import torch
from collections import OrderedDict

from .logic_validator import LogicValidator
try:
    from client.models import Model
    from client.finance_transformer_model import FinanceTransformerModel
except ImportError:
    Model = None
    FinanceTransformerModel = None

log = logging.getLogger(__name__)

# Read dataset name for model directory routing
with open("params.yaml", "r") as _f:
    _cfg = yaml.safe_load(_f)
DS_NAME = _cfg.get("simulation", {}).get("dataset_type", "cyberdefend")
MODEL_DIR = os.path.join("saved_models", DS_NAME)

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
        self.global_consensus_graph = nx.DiGraph()
        self.model_dir = MODEL_DIR
        os.makedirs(self.model_dir, exist_ok=True)
        
        # Resume consensus logic from dataset-specific path
        try:
            import pickle
            consensus_path = os.path.join(self.model_dir, "consensus_graph.gpickle")
            if os.path.exists(consensus_path):
                with open(consensus_path, "rb") as f:
                    self.global_consensus_graph = pickle.load(f)
                log.info(f"Resumed [{DS_NAME}] Consensus Graph from {consensus_path}")
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
        ged_scores = {}  # cid -> {score, status}
        
        for client, fit_res in results:
            metrics = fit_res.metrics
            if "causal_graph_edges" in metrics:
                edges = eval(metrics["causal_graph_edges"])
                client_graph = nx.DiGraph()
                client_graph.add_edges_from(edges)
                client_graph.add_nodes_from(self.global_consensus_graph.nodes())
                
                # --- COVERAGE THRESHOLD GATE ---
                # For agentic environments, require a minimum number of query nodes
                # to have been visited before an execution node.
                # Catches temporal mimicry: query 5 tools, jump to execution.
                if DS_NAME in {"finance", "cyberdefend"}:
                    try:
                        with open("params.yaml", "r") as _pf:
                            _p = yaml.safe_load(_pf)
                        min_q = int(_p.get("core_logic", {}).get("coverage_gate_min_queries", 20))
                    except Exception:
                        min_q = 20
                    
                    # Count distinct query-tool nodes that appear before any execution node
                    # Execution nodes for finance: 33, 34, 35; for cyberdefend: top of range
                    total_nodes = max(self.global_consensus_graph.number_of_nodes(), 36)
                    exec_offset = total_nodes - 3  # last 3 are execution actions
                    query_nodes_visited = {n for n in client_graph.nodes() if n < exec_offset and client_graph.degree(n) > 0}
                    
                    if len(query_nodes_visited) < min_q:
                        log.warning(
                            f"Client {client.cid} REJECTED by Coverage Gate "
                            f"(visited {len(query_nodes_visited)}/{min_q} required query nodes)"
                        )
                        rejected_count += 1
                        ged_scores[str(client.cid)] = {
                            "score": 1.0,
                            "status": "rejected_coverage_gate",
                            "queries": len(query_nodes_visited),
                        }
                        continue

                # --- DYNAMIC PER-DATASET THRESHOLD ---
                # Re-read threshold from params.yaml each round so slider changes take effect.
                # finance uses a dedicated finance_validator_threshold (typically tighter)
                # because the 36-node graph has higher natural variance than 40-node cyberdefend.
                if DS_NAME == "finance":
                    try:
                        with open("params.yaml", "r") as _tf:
                            _tp = yaml.safe_load(_tf)
                        dynamic_threshold = float(
                            _tp.get("core_logic", {}).get("finance_validator_threshold", 0.07)
                        )
                    except Exception:
                        dynamic_threshold = 0.07
                else:
                    dynamic_threshold = self.logic_validator.threshold

                is_valid, score = self.logic_validator.evaluate_client_graph(
                    client_graph, threshold_override=dynamic_threshold
                )

                if is_valid:
                    log.info(
                        f"Client {client.cid} ACCEPTED (GED={score:.4f} ≤ τ={dynamic_threshold:.3f})"
                    )
                    accepted_results.append((client, fit_res))
                    accepted_graphs.append(client_graph)
                    ged_scores[str(client.cid)] = {"score": round(score, 4), "status": "accepted",
                                                   "threshold": dynamic_threshold}
                else:
                    log.warning(
                        f"Client {client.cid} REJECTED by SimGNN (GED={score:.4f} > τ={dynamic_threshold:.3f})"
                    )
                    rejected_count += 1
                    rejected_graphs.append((client_graph, score))
                    ged_scores[str(client.cid)] = {"score": round(score, 4), "status": "rejected",
                                                   "threshold": dynamic_threshold}
            else:
                log.warning(f"Client {client.cid} did not provide causal graph. REJECTING.")
                rejected_count += 1
                
        metrics_aggregated = {
            "accepted_clients": len(accepted_results),
            "rejected_clients": rejected_count,
        }
        
        # Store detailed per-client GED scores in file for GUI
        import json
        ged_log_path = os.path.join(self.model_dir, "ged_scores.json")
        
        ged_data_list = []
        if os.path.exists(ged_log_path):
            try:
                with open(ged_log_path, "r") as f:
                    ged_data_list = json.load(f)
            except Exception:
                pass
                
        ged_data_list.append({"round": server_round, "scores": ged_scores})
        
        with open(ged_log_path, "w") as f:
            json.dump(ged_data_list, f, indent=2)

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
            
            if accepted_graphs:
                honest_path = os.path.join(self.model_dir, "honest_graph_sample.gpickle")
                with open(honest_path, "wb") as f:
                    pickle.dump(accepted_graphs[0], f)
            if rejected_graphs:
                rej_graph, rej_score = rejected_graphs[0]
                rejected_path = os.path.join(self.model_dir, "rejected_graph_sample.gpickle")
                with open(rejected_path, "wb") as f:
                    pickle.dump(rej_graph, f)
                import json
                consensus_edges = set((str(u), str(v)) for u, v in self.global_consensus_graph.edges())
                rejected_edges = set((str(u), str(v)) for u, v in rej_graph.edges())
                missing_from_rejected = list(consensus_edges - rejected_edges)
                extra_in_rejected = list(rejected_edges - consensus_edges)
                edge_diff = {
                    "ged_score": round(rej_score, 4),
                    "threshold": self.logic_validator.threshold,
                    "missing_edges": missing_from_rejected,
                    "extra_edges": extra_in_rejected,
                }
                edge_diff_path = os.path.join(self.model_dir, "rejected_edge_diff.json")
                with open(edge_diff_path, "w") as f:
                    json.dump(edge_diff, f, indent=2)

        return aggregated_parameters, metrics_aggregated

    def _save_global_model(self, parameters: Parameters, round_num: int):
        """Reconstructs the PyTorch model from Flower Parameters and saves only the latest (global_model.pt).
        Per-round files are not stored — the consensus_graph.gpickle captures round-level evolution.
        """
        try:
            ndarrays = parameters_to_ndarrays(parameters)
            in_features = max(self.global_consensus_graph.number_of_nodes(), 1)
            
            # Reconstruct model based on dataset type
            if DS_NAME == "finance":
                # Finance uses FinanceTransformerModel
                model = FinanceTransformerModel(in_features=70, num_actions=36)
            else:
                # Cyberdefend or Tabular use simple Model MLP
                if DS_NAME == "cyberdefend":
                    in_f, out_c = 10, 40
                else:
                    from datasets.tabular_loader import TabularBNDataset
                    dataset_name = _cfg.get("dataset", {}).get("name", "asia").lower()
                    _tmp_ds = TabularBNDataset(name=dataset_name, num_samples=100)
                    in_f = in_features
                    out_c = _tmp_ds.num_classes
                model = Model(in_features=in_f, num_classes=out_c)
            
            params_dict = zip(model.state_dict().keys(), ndarrays)
            state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
            model.load_state_dict(state_dict, strict=True)
            
            # Only save the latest model (no per-round files)
            latest_path = os.path.join(self.model_dir, "global_model.pt")
            torch.save(model.state_dict(), latest_path)
            
            # Save the current consensus graph (it evolves each round)
            consensus_path = os.path.join(self.model_dir, "consensus_graph.gpickle")
            with open(consensus_path, "wb") as f:
                pickle.dump(self.global_consensus_graph, f)
        except Exception as e:
            log.error(f"Failed to save global model weights: {e}")

    def _aggregate_logic(self, client_graphs: List[nx.DiGraph]):
        """
        Bayesian Dirichlet-Multinomial Consensus Update.

        Each edge (u, v) maintains a Beta(α, β) posterior over its presence:
          - α (pseudo-successes): votes FOR the edge across all rounds
          - β (pseudo-failures): votes AGAINST accumulated over time

        An edge survives if:  α / (α + β)  >  consensus_momentum  (treated as credibility threshold)
        A new edge is added if client vote fraction > 0.85 (near-unanimous).

        This gives proper calibrated per-edge uncertainty instead of hard vote counts,
        and allows edges seen in many past rounds to survive occasional low-vote rounds.
        """
        if not client_graphs:
            return

        try:
            with open("params.yaml", "r") as _f:
                _params = yaml.safe_load(_f)
            # Re-use consensus_momentum as the credibility threshold (0-1)
            credibility_threshold = float(_params.get("core_logic", {}).get("consensus_momentum", 0.85))
        except Exception:
            credibility_threshold = 0.85
        credibility_threshold = max(0.0, min(1.0, credibility_threshold))

        n_clients = len(client_graphs)
        alpha_prior = 1.0   # Weak prior: assume edge has been seen once
        beta_prior  = 1.0   # Weak prior: assume edge has been absent once

        # Count votes from accepted clients
        edge_votes = {}
        for g in client_graphs:
            for u, v in g.edges():
                edge_votes[(u, v)] = edge_votes.get((u, v), 0) + 1

        # Initialize or read existing edge posteriors from aggregator state
        if not hasattr(self, "_edge_posteriors"):
            self._edge_posteriors = {}

        new_consensus = nx.DiGraph()
        new_consensus.add_nodes_from(self.global_consensus_graph.nodes())

        all_candidate_edges = set(self.global_consensus_graph.edges()) | set(edge_votes.keys())

        for edge in all_candidate_edges:
            votes_for = edge_votes.get(edge, 0)
            votes_against = n_clients - votes_for

            # Retrieve existing posterior or use priors
            alpha, beta = self._edge_posteriors.get(edge, (alpha_prior, beta_prior))

            # Bayesian update: Beta-Binomial conjugate update
            alpha_new = alpha + votes_for
            beta_new  = beta + votes_against

            # Credibility = posterior mean of Bernoulli parameter
            credibility = alpha_new / (alpha_new + beta_new)

            # Save updated posterior
            self._edge_posteriors[edge] = (alpha_new, beta_new)

            if credibility >= credibility_threshold:
                new_consensus.add_edge(*edge)

        kept = new_consensus.number_of_edges()
        log.info(
            f"Bayesian consensus updated (credibility_thr={credibility_threshold:.2f}): "
            f"{new_consensus.number_of_nodes()} nodes, {kept} edges tracked"
        )

        self.global_consensus_graph = new_consensus
        self.logic_validator.set_global_consensus(self.global_consensus_graph)
        log.info(f"Consensus graph updated: {new_consensus.number_of_nodes()} nodes, "
                 f"{new_consensus.number_of_edges()} edges")

        # --- On-the-fly SimGNN fine-tuning on new consensus ---
        try:
            self._finetune_simgnn_on_consensus(new_consensus)
        except Exception as e:
            log.warning(f"SimGNN fine-tune skipped (non-fatal): {e}")

    def _finetune_simgnn_on_consensus(self, consensus: nx.DiGraph, steps: int = 10, pairs: int = 16):
        """
        Runs a small number of gradient steps on SimGNN using graph permutations
        of the updated consensus. This re-anchors SimGNN's distance function
        to the new reference graph so that rejection stays calibrated across rounds.

        Args:
            consensus: the freshly updated consensus DiGraph
            steps: number of gradient descent steps to run (default: 10, fast)
            pairs: number of (graph_a, graph_b, ged_label) pairs to generate (default: 16)
        """
        import torch.nn as nn
        import torch.optim as optim
        from torch_geometric.utils import from_networkx
        from torch_geometric.data import Batch

        simgnn_model = self.logic_validator.model
        if simgnn_model is None:
            return

        device = next(simgnn_model.parameters()).device
        simgnn_model.train()
        optimizer = optim.Adam(simgnn_model.parameters(), lr=0.001)
        criterion = nn.MSELoss()

        nodes = list(consensus.nodes())
        node_count = max(len(nodes), 2)

        def _to_pyg(g: nx.DiGraph):
            """Convert nx.DiGraph to PyG Data with one-hot node features."""
            g = g.copy()
            g.remove_nodes_from(list(nx.isolates(g))) # Prevent topology dilution
            for n in g.nodes():
                feat = [0.0] * 40
                feat[int(n) % 40] = 1.0
                g.nodes[n]["x"] = feat
            if g.number_of_nodes() == 0:
                feat = [0.0] * 40
                feat[0] = 1.0
                g.add_node(0, x=feat)
            data = from_networkx(g, group_node_attrs=["x"])
            data.batch = torch.zeros(data.x.size(0), dtype=torch.long)
            return data.to(device)

        def _perturb(g: nx.DiGraph, remove_prob=0.2, add_prob=0.2) -> Tuple[nx.DiGraph, float]:
            """Randomly add/remove edges and compute a normalised GED approximation."""
            g2 = g.copy()
            edges = list(g.edges())
            # Remove some existing edges
            removed = [e for e in edges if random.random() < remove_prob]
            for e in removed:
                g2.remove_edge(*e)
            # Add some new edges
            added = 0
            all_nodes = list(g.nodes())
            
            # If the graph is very sparse, ensure we add structurally proportionate edges
            num_nodes = max(len(all_nodes), 2)
            base_pool = max(len(edges), num_nodes)
            num_to_add = max(1, int(base_pool * add_prob))
            
            for _ in range(num_to_add):
                u = random.choice(all_nodes)
                v = random.choice(all_nodes)
                if u != v and not g2.has_edge(u, v):
                    g2.add_edge(u, v)
                    added += 1
                    
            # Proper Normalised GED relative to the union of edges
            edges1 = set(g.edges())
            edges2 = set(g2.edges())
            union_edges = len(edges1.union(edges2))
            if union_edges == 0:
                ged = 0.0
            else:
                diff = len(edges1.symmetric_difference(edges2))
                ged = min(1.0, float(diff) / union_edges)
                
            return g2, ged

        total_loss = 0.0
        for step in range(steps):
            batch_a, batch_b, labels = [], [], []
            for _ in range(pairs // 2):
                # Similar pair: small perturbation → low GED
                g2, ged = _perturb(consensus, remove_prob=0.1, add_prob=0.1)
                batch_a.append(_to_pyg(consensus))
                batch_b.append(_to_pyg(g2))
                labels.append(ged)
                # Dissimilar pair: large perturbation → high GED
                g3, ged2 = _perturb(consensus, remove_prob=0.5, add_prob=0.5)
                batch_a.append(_to_pyg(consensus))
                batch_b.append(_to_pyg(g3))
                labels.append(ged2)

            ba = Batch.from_data_list(batch_a)
            bb = Batch.from_data_list(batch_b)
            label_tensor = torch.tensor(labels, dtype=torch.float32).to(device)

            optimizer.zero_grad()
            pred = simgnn_model(ba, bb).squeeze()
            loss = criterion(pred, label_tensor)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        simgnn_model.eval()
        log.info(f"SimGNN fine-tuned on updated consensus ({steps} steps, avg loss: {total_loss/steps:.4f})")

        # Persist the updated SimGNN weights alongside the consensus
        simgnn_path = os.path.join(self.model_dir, "simgnn_pretrained.pt")
        torch.save(simgnn_model.state_dict(), simgnn_path)
