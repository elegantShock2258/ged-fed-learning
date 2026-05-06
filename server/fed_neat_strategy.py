"""
Module: server.fed_neat_strategy
===================================
Description:
    Defines FedNEATStrategy, the FedNEAT server-side aggregator.

    Combines:
      - Topological Crossover of evolved DynamicGenomes (structural merge)
      - Causal Proof of Reasoning (PoR) gate using SimGNN Logic Validator
        to reject adversarial clients before genome merge.
      - Consensus graph update and on-the-fly SimGNN fine-tuning each round.
      - Saving of honest/rejected graph samples for GUI visualization.
"""

import flwr as fl
from typing import Callable, Dict, List, Optional, Tuple, Union
from flwr.common import (
    FitRes, NDArrays, Parameters, Scalar,
    ndarrays_to_parameters, parameters_to_ndarrays,
)
from flwr.server.client_proxy import ClientProxy
import numpy as np
import networkx as nx
import logging
import json
import pickle
import random
import os
import yaml
import torch

from .logic_validator import LogicValidator

log = logging.getLogger(__name__)

# Read dataset name for model directory routing
with open("params.yaml", "r") as _f:
    _cfg = yaml.safe_load(_f)
DS_NAME = _cfg.get("dataset", {}).get("name", "asia")
MODEL_DIR = os.path.join("saved_models", DS_NAME)


def broadcast_state(genome_data, source_name="Server Merged Consensus"):
    data = {
        "source": source_name,
        "nodes": genome_data.get("nodes", {}),
        "connections": genome_data.get("connections", {}),
    }
    try:
        os.makedirs(MODEL_DIR, exist_ok=True)
        with open(os.path.join(MODEL_DIR, "realtime_state.json"), "w") as f:
            json.dump(data, f)
    except Exception:
        pass


class FedNEATStrategy(fl.server.strategy.Strategy):
    """
    Federated NeuroEvolution (FedNEAT) + Causal Proof of Reasoning Aggregator.

    Stage 1 — PoR Logic Gate:
        Each client's submitted causal graph is validated against the global
        consensus using SimGNN. Clients with GED > threshold are REJECTED.

    Stage 2 — Topological Crossover:
        Only ACCEPTED genome structures are merged via Innovation Hash matching.

    Stage 3 — Consensus Update:
        Accepted client causal graphs update the consensus via momentum voting.
        SimGNN is fine-tuned on the new consensus each round.
    """

    def __init__(
        self,
        logic_validator: LogicValidator,
        fraction_fit=1.0,
        fraction_evaluate=1.0,
        min_fit_clients=2,
        min_evaluate_clients=2,
        min_available_clients=2,
        on_fit_config_fn=None,
        initial_parameters=None,
    ):
        super().__init__()
        self.logic_validator = logic_validator
        self.fraction_fit = fraction_fit
        self.fraction_evaluate = fraction_evaluate
        self.min_fit_clients = min_fit_clients
        self.min_evaluate_clients = min_evaluate_clients
        self.min_available_clients = min_available_clients
        self.on_fit_config_fn = on_fit_config_fn
        self.initial_parameters = initial_parameters

        self.model_dir = MODEL_DIR
        os.makedirs(self.model_dir, exist_ok=True)

        # Load existing consensus graph if available
        self.global_consensus_graph = nx.DiGraph()
        try:
            consensus_path = os.path.join(self.model_dir, "consensus_graph.gpickle")
            if os.path.exists(consensus_path):
                with open(consensus_path, "rb") as f:
                    self.global_consensus_graph = pickle.load(f)
                    self.ground_truth_graph = self.global_consensus_graph.copy() # Anchor Immutable Root
                log.info(f"Resumed [{DS_NAME}] Consensus Graph from {consensus_path} "
                         f"({self.global_consensus_graph.number_of_nodes()} nodes, "
                         f"{self.global_consensus_graph.number_of_edges()} edges)")
        except Exception as e:
            log.warning(f"Could not load previous consensus graph: {e}")

        self.logic_validator.set_global_consensus(self.global_consensus_graph)

    # ------------------------------------------------------------------
    # Flower lifecycle hooks
    # ------------------------------------------------------------------

    def initialize_parameters(self, client_manager):
        if not hasattr(self, 'current_parameters'):
            self.current_parameters = self.initial_parameters
        return self.current_parameters

    def configure_fit(self, server_round: int, parameters: Parameters, client_manager):
        config = {}
        if self.on_fit_config_fn is not None:
            config = self.on_fit_config_fn(server_round)
        fit_ins = fl.common.FitIns(parameters, config)
        sample_size = max(
            int(self.fraction_fit * client_manager.num_available()),
            self.min_fit_clients,
        )
        clients = client_manager.sample(
            num_clients=sample_size, min_num_clients=self.min_fit_clients
        )
        return [(client, fit_ins) for client in clients]

    def configure_evaluate(self, server_round: int, parameters: Parameters, client_manager):
        config = {}
        eval_ins = fl.common.EvaluateIns(parameters, config)
        sample_size = max(
            int(self.fraction_evaluate * client_manager.num_available()),
            self.min_evaluate_clients,
        )
        clients = client_manager.sample(
            num_clients=sample_size, min_num_clients=self.min_evaluate_clients
        )
        return [(client, eval_ins) for client in clients]

    def evaluate(self, server_round: int, parameters: Parameters):
        return None

    # ------------------------------------------------------------------
    # Core aggregation with PoR gate
    # ------------------------------------------------------------------

    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, FitRes]],
        failures,
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
    
        actual_round = server_round - 1
        log.info(f"--- Processing Round {actual_round} ---")

        if not results:
            return None, {}

        # ── Stage 1: PoR Logic Gate ────────────────────────────────────
        accepted_results = []
        accepted_genomes = []       # raw genome dicts from accepted clients
        accepted_graphs = []        # reconstructed nx.DiGraph for consensus update
        rejected_graphs = []        # (nx.DiGraph, score) for GUI visualization
        ged_scores = {}

        for client, fit_res in results:
            metrics = fit_res.metrics
            causal_edges_str = metrics.get("causal_graph_edges", None)

            # Parse causal graph
            client_graph = nx.DiGraph()
            if causal_edges_str:
                try:
                    edges = eval(causal_edges_str)
                    client_graph.add_edges_from(edges)
                    # Ensure all consensus nodes are present for fair comparison
                    client_graph.add_nodes_from(self.global_consensus_graph.nodes())
                except Exception as e:
                    log.warning(f"Client {client.cid} sent unparseable graph: {e}")

            # Validate with SimGNN, passing server_round to explicitly allow Round 1 calibration
            is_valid, score = self.logic_validator.evaluate_client_graph(client_graph, actual_round)

            if is_valid:
                log.info(f"Client {client.cid} ACCEPTED. GED={score:.4f} (passed curriculum threshold)")
                accepted_results.append((client, fit_res))
                accepted_graphs.append(client_graph)
                ged_scores[str(client.cid)] = {"score": round(score, 4), "status": "accepted"}

                # Parse genome for crossover
                if fit_res.parameters.tensors:
                    byte_arr = parameters_to_ndarrays(fit_res.parameters)[0]
                    genome_data = json.loads(bytearray(byte_arr).decode("utf-8"))
                    genome_data["fitness"] = float(metrics.get("accuracy", 0.0))
                    accepted_genomes.append(genome_data)
            else:
                log.warning(f"Client {client.cid} REJECTED. GED={score:.4f} (failed curriculum threshold)")
                rejected_graphs.append((client_graph, score))
                ged_scores[str(client.cid)] = {"score": round(score, 4), "status": "rejected"}

        # Persist GED scores for GUI PoR log panel
        ged_log_path = os.path.join(self.model_dir, "ged_scores.json")
        with open(ged_log_path, "w") as f:
            json.dump({"round": actual_round, "scores": ged_scores}, f, indent=2)
            
        # Statistical Outlier Anchor mapping (Rank Preserving Limit)
        # We push all Round predictions back into the LogicValidator so it organically
        # configures its geometric 75th percentile for the immediately proceeding evaluation batch!
        round_scores = [v["score"] for v in ged_scores.values()]
        self.logic_validator.update_dynamic_threshold(round_scores)

        metrics_aggregated = {
            "accepted_clients": len(accepted_results),
            "rejected_clients": len(results) - len(accepted_results),
        }

        if not accepted_results:
            log.error("All clients rejected! Returning previous parameters unchanged.")
            return getattr(self, 'current_parameters', self.initial_parameters), metrics_aggregated

        # ── Stage 2: Topological Crossover (on accepted genomes only) ──
        if accepted_genomes:
            merged = self._crossover(accepted_genomes)
        else:
            # Fallback — shouldn't happen but guard against empty list
            merged = json.loads(bytearray(parameters_to_ndarrays(self.initial_parameters)[0]).decode("utf-8"))

        broadcast_state(merged, source_name=f"Server Merged Topology (Round {actual_round})")

        s = json.dumps(merged)
        merged_parameters = ndarrays_to_parameters([np.array(bytearray(s, "utf-8"))])
        self.current_parameters = merged_parameters # Cache latest verified global model state

        # ── Stage 3: Save sample graphs for GUI visualization ──────────
        if accepted_graphs:
            honest_path = os.path.join(self.model_dir, "honest_graph_sample.gpickle")
            with open(honest_path, "wb") as f:
                pickle.dump(accepted_graphs[0], f)

        if rejected_graphs:
            rej_graph, rej_score = rejected_graphs[0]
            rejected_path = os.path.join(self.model_dir, "rejected_graph_sample.gpickle")
            with open(rejected_path, "wb") as f:
                pickle.dump(rej_graph, f)
            # Save edge diff for GUI rejection analysis panel
            consensus_edges = set((str(u), str(v)) for u, v in self.global_consensus_graph.edges())
            rejected_edges = set((str(u), str(v)) for u, v in rej_graph.edges())
            edge_diff = {
                "ged_score": round(rej_score, 4),
                "threshold": self.logic_validator.threshold,
                "missing_edges": list(consensus_edges - rejected_edges),
                "extra_edges": list(rejected_edges - consensus_edges),
            }
            with open(os.path.join(self.model_dir, "rejected_edge_diff.json"), "w") as f:
                json.dump(edge_diff, f, indent=2)

        # ── Stage 4: Update Consensus Graph + Fine-tune SimGNN ─────────
        self._aggregate_logic(accepted_graphs)

        # Save updated consensus to disk
        with open(os.path.join(self.model_dir, "consensus_graph.gpickle"), "wb") as f:
            pickle.dump(self.global_consensus_graph, f)

        print(
            f"[ROUND {server_round}] Accepted: {len(accepted_results)}, "
            f"Rejected: {len(results) - len(accepted_results)}"
        )
        return merged_parameters, metrics_aggregated

    def aggregate_evaluate(self, server_round: int, results, failures):
        if not results:
            return None, {}
        try:
            accuracies = [float(r.metrics.get("accuracy", 0.0)) * r.num_examples for _, r in results]
            examples = [r.num_examples for _, r in results]
            return sum(accuracies) / sum(examples), {}
        except Exception:
            return 0.0, {}

    # ------------------------------------------------------------------
    # Topological Crossover
    # ------------------------------------------------------------------

    def _crossover(self, genomes: list) -> dict:
        """
        Merges accepted genomes by Innovation Hash.
        Connections present in multiple genomes have their weights averaged.
        """
        merged_nodes = {}
        merged_hidden = set()
        in_features = genomes[0]["in_features"]
        num_classes = genomes[0]["num_classes"]

        for g in genomes:
            for node, ntype in g["nodes"].items():
                merged_nodes[node] = ntype
            for h in g.get("hidden_nodes", []):
                merged_hidden.add(h)

        # Innovation-hash connection merge
        all_innovations = set()
        for g in genomes:
            all_innovations.update(g["connections"].keys())

        merged_connections = {}
        for innov in all_innovations:
            active_count = 0
            total_weight = 0.0
            sources = []
            for g in genomes:
                if innov in g["connections"]:
                    c = g["connections"][innov]
                    sources.append(c)
                    if c["active"]:
                        active_count += 1
                        total_weight += c["weight"]

            if active_count > 0:
                base = sources[0]
                merged_connections[innov] = {
                    "in": base["in"], "out": base["out"],
                    "active": True, "weight": total_weight / active_count,
                }
            else:
                base = sources[0]
                merged_connections[innov] = {
                    "in": base["in"], "out": base["out"],
                    "active": False, "weight": base["weight"],
                }

        return {
            "in_features": in_features,
            "num_classes": num_classes,
            "nodes": merged_nodes,
            "hidden_nodes": list(merged_hidden),
            "connections": merged_connections,
        }

    # ------------------------------------------------------------------
    # Consensus Graph Update (momentum-blended voting)
    # ------------------------------------------------------------------

    def _aggregate_logic(self, client_graphs: List[nx.DiGraph]):
        """Updates the global consensus graph via momentum-blended majority vote."""
        if not client_graphs:
            return

        try:
            with open("params.yaml", "r") as _f:
                _params = yaml.safe_load(_f)
            momentum = float(_params.get("core_logic", {}).get("consensus_momentum", 0.85))
        except Exception:
            momentum = 0.85
        momentum = max(0.0, min(1.0, momentum))

        n_clients = len(client_graphs)
        edge_counts: Dict[tuple, int] = {}
        for g in client_graphs:
            for u, v in g.edges():
                edge_counts[(u, v)] = edge_counts.get((u, v), 0) + 1

        # Byzantine Fault Tolerant (BFT) Bounds
        # Assume max adversarial fraction f <= 30%
        # add_ratio must be <= (1 - f) to allow honest nodes to add edges alone (~70% max)
        # keep_ratio must be >= f to prevent adversaries from sustaining edges alone (~30% min)
        keep_ratio = max(0.30, min(0.40, (1.0 - momentum) * 0.5))
        add_ratio = min(0.70, max(0.55, 0.5 + 0.5 * momentum))
        
        keep_threshold = keep_ratio * n_clients
        add_threshold = add_ratio * n_clients

        new_consensus = nx.DiGraph()
        for g in client_graphs:
            new_consensus.add_nodes_from(g.nodes())
        new_consensus.add_nodes_from(self.global_consensus_graph.nodes())

        existing_edges = set(self.global_consensus_graph.edges())
        for edge in existing_edges:
            # Anchor ground truth edges to categorically prevent Foundational Graph Attrition
            if hasattr(self, 'ground_truth_graph') and edge in self.ground_truth_graph.edges():
                new_consensus.add_edge(*edge)
            elif edge_counts.get(edge, 0) >= keep_threshold:
                new_consensus.add_edge(*edge)
        for edge, count in edge_counts.items():
            if edge not in existing_edges and count >= add_threshold:
                new_consensus.add_edge(*edge)

        log.info(
            f"Consensus updated (momentum={momentum:.2f}): "
            f"{new_consensus.number_of_nodes()} nodes, {new_consensus.number_of_edges()} edges "
            f"(was {len(existing_edges)} edges)"
        )
        self.global_consensus_graph = new_consensus
        self.logic_validator.set_global_consensus(self.global_consensus_graph)

        # On-the-fly SimGNN fine-tuning disabled to prevent Catastrophic Forgetting
        # and lock the highly generalized distance heuristic structure cleanly.
        # try:
        #     self._finetune_simgnn_on_consensus(new_consensus)
        # except Exception as e:
        #     log.warning(f"SimGNN fine-tune skipped (non-fatal): {e}")

    # ------------------------------------------------------------------
    # SimGNN Fine-tuning
    # ------------------------------------------------------------------

    def _finetune_simgnn_on_consensus(self, consensus: nx.DiGraph, steps: int = 10, pairs: int = 16):
        """Fine-tunes SimGNN on graph permutations of the updated consensus."""
        import torch.nn as nn
        import torch.optim as optim
        from torch_geometric.utils import from_networkx
        from torch_geometric.data import Batch

        simgnn_model = self.logic_validator.simgnn
        if simgnn_model is None:
            return

        device = next(simgnn_model.parameters()).device
        simgnn_model.train()
        optimizer = optim.Adam(simgnn_model.parameters(), lr=0.001)
        criterion = nn.MSELoss()

        def _to_pyg(g: nx.DiGraph):
            g = g.copy()
            for n in g.nodes():
                g.nodes[n]["x"] = [1.0]
            if g.number_of_nodes() == 0:
                g.add_node(0, x=[1.0])
            data = from_networkx(g, group_node_attrs=["x"])
            data.batch = torch.zeros(data.x.size(0), dtype=torch.long)
            return data.to(device)

        def _perturb(g: nx.DiGraph, remove_prob=0.2, add_prob=0.2) -> Tuple[nx.DiGraph, float]:
            g2 = g.copy()
            edges = list(g.edges())
            for e in edges:
                if random.random() < remove_prob:
                    g2.remove_edge(*e)
            all_nodes = list(g.nodes())
            num_to_add = max(1, int(len(edges) * add_prob))
            added = 0
            for _ in range(num_to_add * 3):
                if added >= num_to_add:
                    break
                u, v = random.choice(all_nodes), random.choice(all_nodes)
                if u != v and not g2.has_edge(u, v):
                    g2.add_edge(u, v)
                    added += 1
            edges1, edges2 = set(g.edges()), set(g2.edges())
            union = len(edges1.union(edges2))
            ged = min(1.0, float(len(edges1.symmetric_difference(edges2))) / union) if union else 0.0
            return g2, ged

        total_loss = 0.0
        for _ in range(steps):
            batch_a, batch_b, labels = [], [], []
            for _ in range(pairs // 2):
                g2, ged = _perturb(consensus, 0.1, 0.1)
                batch_a.append(_to_pyg(consensus)); batch_b.append(_to_pyg(g2)); labels.append(ged)
                g3, ged2 = _perturb(consensus, 0.5, 0.5)
                batch_a.append(_to_pyg(consensus)); batch_b.append(_to_pyg(g3)); labels.append(ged2)

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
        log.info(f"SimGNN fine-tuned ({steps} steps, avg_loss={total_loss/steps:.4f})")
        torch.save(simgnn_model.state_dict(), os.path.join(self.model_dir, "simgnn_pretrained.pt"))
