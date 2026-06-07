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

import ast
import flwr as fl
import json
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
        self.historical_honest_geds = []
        self.historical_honest_ceds: List[float] = []  # Dual-Gate PoR: honest CED history for adaptive θ
        os.makedirs(self.model_dir, exist_ok=True)

        # -- Dual-Gate PoR: consensus coefficient matrix B̄ (Section 4.4, paper.tex) --
        # EMA over accepted clients' B matrices. None until first round completes.
        self.consensus_B: np.ndarray | None = None

        # Resume consensus logic from dataset-specific path
        try:
            consensus_path = os.path.join(self.model_dir, "consensus_graph.gpickle")
            if os.path.exists(consensus_path):
                with open(consensus_path, "rb") as f:
                    self.global_consensus_graph = pickle.load(f)
                log.info(f"Resumed [{DS_NAME}] Consensus Graph from {consensus_path}")
            consensus_B_path = os.path.join(self.model_dir, "consensus_B.npy")
            if os.path.exists(consensus_B_path):
                self.consensus_B = np.load(consensus_B_path)
                log.info(f"Resumed consensus B̄ matrix shape={self.consensus_B.shape}")
        except Exception as e:
            log.warning(f"Could not load previous consensus state: {e}")

        self.logic_validator.set_global_consensus(self.global_consensus_graph)

        # -- Clear stale GED score log from previous runs to prevent data contamination --
        ged_log_path = os.path.join(self.model_dir, "ged_scores.json")
        if os.path.exists(ged_log_path):
            try:
                os.remove(ged_log_path)
                log.info("Cleared stale GED score log for fresh run.")
            except Exception:
                pass

        # -- REPUTATION TRACKING --
        self.client_reputation: Dict[str, float] = {}
        self._reputation_alpha = 0.3
        rep_path = os.path.join(self.model_dir, "client_reputation.json")
        if os.path.exists(rep_path):
            try:
                with open(rep_path, "r") as f:
                    self.client_reputation = json.load(f)
                log.info(f"Resumed reputation table with {len(self.client_reputation)} clients")
            except Exception:
                pass

    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, FitRes]],
        failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]],
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        """Aggregate fit results using logic validation filtering."""
        
        if not results:
            return None, {}

        # Read params.yaml ONCE per aggregate_fit call (avoids 7+ redundant reads)
        try:
            _cfg = yaml.safe_load(open("params.yaml", "r"))
        except Exception:
            _cfg = {}
        _cl_cfg = _cfg.get("core_logic", {})
        _sim_cfg = _cfg.get("simulation", {})
        # Cache on self so _aggregate_logic can reuse without re-reading
        self._cl_cfg = _cl_cfg

        # 1. Filter clients using Logic Validator
        accepted_results = []
        accepted_graphs = []
        rejected_graphs = []
        rejected_count = 0
        ged_scores = {}  # cid -> {score, status}

        # Calculate Adaptive Distributional Threshold -> mu + 3*sigma
        k_sigma = float(_cl_cfg.get("adaptive_k_sigma", 3.0))
        adaptive_threshold = None
        if len(self.historical_honest_geds) >= 3:
            mu_ged = float(np.mean(self.historical_honest_geds))
            sigma_ged = float(np.std(self.historical_honest_geds))
            adaptive_threshold = mu_ged + (k_sigma * sigma_ged)
            # Clip threshold to reasonable bounds to prevent instability
            adaptive_threshold = max(0.01, min(adaptive_threshold, 0.95))  # max clip: catch adversaries at 1.0
            log.info(f"[Adaptive Threshold] μ={mu_ged:.4f}, σ={sigma_ged:.4f} => τ={adaptive_threshold:.4f}")

        # Read Dual-Gate PoR thresholds from cached config
        _ced_threshold_cfg  = float(_cl_cfg.get("ced_threshold", 0.05))
        _ced_ema_alpha      = float(_cl_cfg.get("ced_ema_alpha", 0.3))

        # Adaptive CED threshold: Robust MAD estimator (Median + 3 * Median Absolute Deviation)
        # Prevents adversarial CED outliers from inflating the threshold during calibration.
        if len(self.historical_honest_ceds) >= 5:
            med_ced = np.median(self.historical_honest_ceds)
            mad_ced = np.median(np.abs(self.historical_honest_ceds - med_ced))
            # Fallback to standard deviation if MAD is 0 (all values identical)
            if mad_ced == 0:
                mad_ced = np.std(self.historical_honest_ceds)
            adaptive_ced_threshold = float(med_ced + 3 * mad_ced)
            adaptive_ced_threshold = max(0.005, min(adaptive_ced_threshold, 0.50))
        else:
            adaptive_ced_threshold = _ced_threshold_cfg

        # --- Liveness Mitigation: Curriculum Grace Period ---
        # During the first few rounds of a curriculum expansion, clients may naturally
        # deviate structurally. To prevent 100% rejection halts, we define a grace period
        # where logic rejections are logged but weights are still aggregated.
        grace_period_rounds = int(_cl_cfg.get("grace_period_rounds", 6))
        in_grace_period = (server_round <= grace_period_rounds)
        self._in_grace_period = in_grace_period

        for client, fit_res in results:
            metrics = fit_res.metrics
            if "causal_graph_edges" in metrics:
                edges = ast.literal_eval(metrics["causal_graph_edges"])
                client_graph = nx.DiGraph()
                client_graph.add_edges_from(edges)
                client_graph.add_nodes_from(self.global_consensus_graph.nodes())

                # --- COVERAGE THRESHOLD GATE ---
                if DS_NAME in {"finance", "cyberdefend"}:
                    min_q = int(_cl_cfg.get("coverage_gate_min_queries", 20))
                    total_nodes = max(self.global_consensus_graph.number_of_nodes(), 36)
                    # 3 execution actions (33=Hold, 34=Buy, 35=Market Dump);
                    # all nodes with index < exec_offset are sector query nodes.
                    exec_offset = total_nodes - 3
                    query_nodes_visited = {n for n in client_graph.nodes() if n < exec_offset and client_graph.degree(n) > 0}
                    if len(query_nodes_visited) < min_q:
                        log.warning(f"Client {client.cid} REJECTED by Coverage Gate "
                                    f"(visited {len(query_nodes_visited)}/{min_q} required)")
                        rejected_count += 1
                        ged_scores[str(client.cid)] = {"score": 1.0, "ced_score": None,
                                                       "status": "rejected_coverage_gate",
                                                       "queries": len(query_nodes_visited)}
                        continue

                # --- DYNAMIC PER-DATASET STRUCTURAL THRESHOLD ---
                if DS_NAME == "finance":
                    file_threshold = float(_cl_cfg.get("finance_validator_threshold", 0.07))
                else:
                    file_threshold = self.logic_validator.threshold
                dynamic_threshold = adaptive_threshold if adaptive_threshold is not None else file_threshold

                is_valid, score = self.logic_validator.evaluate_client_graph(
                    client_graph, threshold_override=dynamic_threshold
                )

                # --- DUAL-GATE: CED CHECK (Section 4.4, paper.tex) ---
                # CED(B_k, B̄) = (1/|E_global|) * Σ_{(i,j)∈E_global} |B_k[i,j] - B̄[i,j]|
                ced_score = None
                ced_passed = True
                if DS_NAME == "finance" and self.consensus_B is not None and "causal_coeff_matrix" in metrics:
                    try:
                        flat_B = json.loads(metrics["causal_coeff_matrix"])
                        if flat_B:
                            d = int(round(len(flat_B) ** 0.5))
                            B_k = np.array(flat_B, dtype=np.float32).reshape(d, d)
                            d_bar = self.consensus_B.shape[0]
                            # Align dimensions
                            d_min = min(d, d_bar)
                            B_k_aligned   = B_k[:d_min, :d_min]
                            B_bar_aligned = self.consensus_B[:d_min, :d_min]
                            # Only compare over edges present in global consensus
                            global_edges = list(self.global_consensus_graph.edges())
                            valid_edges = [(u, v) for u, v in global_edges
                                           if isinstance(u, int) and isinstance(v, int)
                                           and u < d_min and v < d_min]
                            if valid_edges:
                                ced_score = float(np.mean([
                                    abs(float(B_k_aligned[u, v]) - float(B_bar_aligned[u, v]))
                                    for u, v in valid_edges
                                ]))
                            else:
                                # Fallback: global Frobenius mean
                                ced_score = float(np.mean(np.abs(B_k_aligned - B_bar_aligned)))
                            ced_passed = ced_score <= adaptive_ced_threshold
                            if not ced_passed:
                                log.warning(f"Client {client.cid} REJECTED by CED Gate "
                                            f"(CED={ced_score:.4f} > θ={adaptive_ced_threshold:.4f})")
                    except Exception as e:
                        log.debug(f"CED gate parse error for {client.cid}: {e}")

                # Grace period override for aggregation (but not for reputation/logic consensus)
                actually_valid = is_valid and ced_passed
                aggregated_anyway = False

                if actually_valid:
                    self.historical_honest_geds.append(score)
                    if len(self.historical_honest_geds) > 50:
                        self.historical_honest_geds.pop(0)
                    if ced_score is not None:
                        self.historical_honest_ceds.append(ced_score)
                        if len(self.historical_honest_ceds) > 50:
                            self.historical_honest_ceds.pop(0)
                    log.info(f"Client {client.cid} ACCEPTED "
                             f"(GED={score:.4f} ≤ τ={dynamic_threshold:.3f}"
                             + (f", CED={ced_score:.4f} ≤ θ={adaptive_ced_threshold:.4f}" if ced_score is not None else "") + ")")
                    accepted_results.append((client, fit_res))
                    accepted_graphs.append(client_graph)
                    ged_scores[str(client.cid)] = {"score": round(score, 4),
                                                   "ced_score": round(ced_score, 4) if ced_score is not None else None,
                                                   "status": "accepted", "threshold": dynamic_threshold}
                    cid_str = str(client.cid)
                    old_rep = self.client_reputation.get(cid_str, 0.5)
                    self.client_reputation[cid_str] = (1 - self._reputation_alpha) * old_rep + self._reputation_alpha * 1.0
                else:
                    if in_grace_period:
                        log.info(f"Client {client.cid} structurally failed but aggregated due to Grace Period "
                                 f"(GED={score:.4f}, τ={dynamic_threshold:.3f})")
                        accepted_results.append((client, fit_res))
                        # During grace period, we ALSO add to accepted_graphs so the Bayesian consensus
                        # can accumulate edge votes. Without this, the consensus collapses to near-empty
                        # (no graphs → no votes → threshold removes all edges → GED spikes for everyone).
                        # The coordinate-wise median aggregation still protects against backdoors.
                        accepted_graphs.append(client_graph)
                        aggregated_anyway = True
                        # Feed grace-bypassed GED scores into the historical distribution
                        # so the adaptive threshold (μ + kσ) can calibrate.
                        self.historical_honest_geds.append(score)
                        if len(self.historical_honest_geds) > 50:
                            self.historical_honest_geds.pop(0)

                    reason = "rejected_ced_gate" if (is_valid and not ced_passed) else "rejected"
                    ged_scores[str(client.cid)] = {"score": round(score, 4),
                                                   "ced_score": round(ced_score, 4) if ced_score is not None else None,
                                                   "status": reason + ("_grace_bypassed" if aggregated_anyway else ""), "threshold": dynamic_threshold}
                    
                    if not aggregated_anyway:
                        if is_valid:
                            pass  # already logged above
                        else:
                            log.warning(f"Client {client.cid} REJECTED by SimGNN "
                                        f"(GED={score:.4f} > τ={dynamic_threshold:.3f})")
                        rejected_count += 1
                        rejected_graphs.append((client_graph, score))
                    
                    # Reputation still drops even in grace period
                    cid_str = str(client.cid)
                    old_rep = self.client_reputation.get(cid_str, 0.5)
                    self.client_reputation[cid_str] = (1 - self._reputation_alpha) * old_rep + self._reputation_alpha * 0.0
            else:
                log.warning(f"Client {client.cid} did not provide causal graph. REJECTING.")
                rejected_count += 1
                
        metrics_aggregated = {
            "accepted_clients": len(accepted_results),
            "rejected_clients": rejected_count,
        }
        
        # -- STRUCTURED LOGGING: ged_scores.json (per-client, per-round) --
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

        # -- STRUCTURED LOGGING: round_metrics.json (high-level, per-round) --
        round_metrics_path = os.path.join(self.model_dir, "round_metrics.json")
        round_metrics_list = []
        if os.path.exists(round_metrics_path):
            try:
                with open(round_metrics_path, "r") as f:
                    round_metrics_list = json.load(f)
            except Exception:
                pass
        round_entry = {
            "round": server_round,
            "accepted": len(accepted_results),
            "rejected": rejected_count,
            "total": len(results),
            "detection_rate": round(rejected_count / max(len(results), 1), 4),
            "adaptive_threshold": round(adaptive_threshold, 4) if adaptive_threshold else None,
            "reputation": {k: round(v, 3) for k, v in self.client_reputation.items()},
            "per_client": {
                cid: {"score": d.get("score"), "status": d.get("status"), "queries": d.get("queries")}
                for cid, d in ged_scores.items()
            },
        }
        round_metrics_list.append(round_entry)
        with open(round_metrics_path, "w") as f:
            json.dump(round_metrics_list, f, indent=2)

        # -- Persist reputation table --
        rep_path = os.path.join(self.model_dir, "client_reputation.json")
        try:
            with open(rep_path, "w") as f:
                json.dump(self.client_reputation, f, indent=2)
        except Exception as e:
            log.warning(f"Failed to save reputation table: {e}")

        if not accepted_results:
            log.error("All clients rejected! Cannot aggregate.")
            return None, metrics_aggregated

        # 2. Aggregate the Weights
        # --- Liveness Mitigation: Robust Fallback during Grace Period ---
        # During the Curriculum Grace Period, we aggregate clients who failed the structural gate.
        # To prevent backdoors from being injected during this warm-up window, we use a robust
        # Coordinate-wise Median aggregation instead of standard FedAvg.
        in_grace_period = getattr(self, "_in_grace_period", False)
        if in_grace_period and len(accepted_results) > 2:
            log.info(f"Applying robust Coordinate-wise Median aggregation (Grace Period active, {len(accepted_results)} clients)")
            # Extract weights from all accepted results
            weights_results = [
                (parameters_to_ndarrays(fit_res.parameters), fit_res.num_examples)
                for _, fit_res in accepted_results
            ]
            # Coordinate-wise median
            num_layers = len(weights_results[0][0])
            median_weights = []
            for layer_idx in range(num_layers):
                layer_stack = np.stack([wr[0][layer_idx] for wr in weights_results], axis=0)
                median_weights.append(np.median(layer_stack, axis=0))
            aggregated_parameters = ndarrays_to_parameters(median_weights)
            # Call super just to log failures, we ignore its output
            super().aggregate_fit(server_round, accepted_results, failures)
        else:
            # Standard FedAvg when out of grace period (since clients are structurally vetted)
            aggregated_parameters, _ = super().aggregate_fit(server_round, accepted_results, failures)

        # 3. Aggregate the Logic (Barycenter Edge Retention)
        # Collect B matrices only from truly accepted clients (not grace-bypassed)
        # so the consensus B̄ EMA is not corrupted by adversary coefficient matrices.
        self._current_round_B_matrices = []
        for client, fit_res in accepted_results:
            cid = str(client.cid)
            status = ged_scores.get(cid, {}).get("status", "")
            if "grace_bypassed" in status:
                continue  # skip grace-bypassed entries to keep B̄ clean
            try:
                flat_B = json.loads(fit_res.metrics.get("causal_coeff_matrix", "[]"))
                if flat_B:
                    d = int(round(len(flat_B) ** 0.5))
                    self._current_round_B_matrices.append(
                        np.array(flat_B, dtype=np.float32).reshape(d, d)
                    )
            except Exception:
                pass
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
                
                # GNN Edge/Node Attribution
                try:
                    self.logic_validator.simgnn.eval()
                    with torch.no_grad():
                        data_rej = self.logic_validator._nx_to_pyg_data(rej_graph).to(self.logic_validator.device)
                        data_con = self.logic_validator._nx_to_pyg_data(self.global_consensus_graph).to(self.logic_validator.device)

                        import torch.nn.functional as F
                        x_rej = data_rej.x
                        # Guard: skip attribution if either graph is empty
                        if x_rej.size(0) == 0 or data_con.x.size(0) == 0:
                            log.warning("Empty graph in GNN attribution — skipping.")
                        else:
                            for conv in self.logic_validator.simgnn.convs:
                                x_rej = F.relu(conv(x_rej, data_rej.edge_index))
                            x_rej = F.relu(self.logic_validator.simgnn.gat(x_rej, data_rej.edge_index))

                            x_con = data_con.x
                            for conv in self.logic_validator.simgnn.convs:
                                x_con = F.relu(conv(x_con, data_con.edge_index))
                            x_con = F.relu(self.logic_validator.simgnn.gat(x_con, data_con.edge_index))

                            max_nodes = min(x_rej.size(0), x_con.size(0))
                            if max_nodes > 0:
                                node_diffs = torch.norm(x_rej[:max_nodes] - x_con[:max_nodes], dim=1).cpu().numpy()
                                top_nodes = np.argsort(node_diffs)[-5:][::-1].tolist()

                                anomalous_edges = []
                                for u, v in extra_in_rejected:
                                    if int(u) in top_nodes or int(v) in top_nodes:
                                        anomalous_edges.append((u, v))

                                edge_diff["gnn_attribution"] = {
                                    "top_anomalous_nodes": top_nodes,
                                    "anomalous_edges": anomalous_edges,
                                    "node_anomaly_scores": {str(i): float(s) for i, s in enumerate(node_diffs) if i in top_nodes}
                                }
                except Exception as e:
                    log.warning(f"Failed to compute GNN edge attribution: {e}")

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
                model = FinanceTransformerModel(in_features=70, num_actions=36)
            else:
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

            latest_path = os.path.join(self.model_dir, "global_model.pt")
            torch.save(model.state_dict(), latest_path)

            consensus_path = os.path.join(self.model_dir, "consensus_graph.gpickle")
            with open(consensus_path, "wb") as f:
                pickle.dump(self.global_consensus_graph, f)
        except Exception as e:
            log.error(f"Failed to save global model weights: {e}")

    def get_consensus_fit_config(self, base_config: dict) -> dict:
        """
        Dual-Gate PoR: returns a fit config dict that includes the current
        consensus adjacency matrix A_global and coefficient matrix B̄, serialized
        as flat JSON lists so Flower's config dict (str → scalar) constraint
        is satisfied via JSON strings.

        Called from run_sim_sequential.py each round to broadcast consensus state.
        """
        cfg = dict(base_config)
        try:
            # Binary adjacency A_global (from global_consensus_graph)
            nodes = sorted(self.global_consensus_graph.nodes())
            if nodes and all(isinstance(n, int) for n in nodes):
                d = max(nodes) + 1
                A = np.zeros((d, d), dtype=np.float32)
                for u, v in self.global_consensus_graph.edges():
                    if isinstance(u, int) and isinstance(v, int) and u < d and v < d:
                        A[u, v] = 1.0
                cfg["consensus_A_global"] = json.dumps(A.flatten().tolist())
        except Exception as e:
            log.debug(f"Could not serialize A_global for broadcast: {e}")

        try:
            if self.consensus_B is not None:
                cfg["consensus_B_bar"] = json.dumps(self.consensus_B.flatten().tolist())
        except Exception as e:
            log.debug(f"Could not serialize B̄ for broadcast: {e}")

        return cfg

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

        # FREEZE consensus during grace period: do not evolve G_global until
        # the adaptive threshold stabilizes and honest clients are reliably accepted.
        # Without this, early-round graph variance shrinks the consensus to near-empty
        # (few edges get >70% votes), causing a GED death spiral post-grace.
        if self._in_grace_period:
            log.info("Grace period active — consensus frozen (not evolving from client graphs)")
            return

        # Use cached config from aggregate_fit if available, otherwise read once
        _cl = getattr(self, "_cl_cfg", None)
        if _cl is None:
            try:
                with open("params.yaml", "r") as _f:
                    _cl = yaml.safe_load(_f).get("core_logic", {})
            except Exception:
                _cl = {}
        credibility_threshold = float(_cl.get("consensus_momentum", 0.7))
        credibility_threshold = max(0.0, min(1.0, credibility_threshold))

        n_clients = len(client_graphs)
        alpha_prior = 1.0   # Weak prior: assume edge has been seen once
        beta_prior  = 1.0   # Weak prior: assume edge has been absent once

        # Gap 14 Mitigation: Sliding window credibility decay
        credibility_decay = 0.90 

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

            # Gap 13 Mitigation: Novel Edge Correlation Penalty (Causal Laundering Defense)
            # Trade-off: Higher exponent penalizes collusion more aggressively, but may FPs
            # honest clients who independently explore the same novel edge (rare but possible).
            # DISABLED during grace period: honest agents naturally produce correlated edges
            # (e.g., sequential 0→1→2) that the penalty would incorrectly suppress, collapsing
            # the consensus. Configurable via core_logic.correlation_penalty_exponent.
            is_novel_edge = not self.global_consensus_graph.has_edge(*edge)
            correlation_penalty = 0.0
            _penalty_exp = float(_cl.get("correlation_penalty_exponent", 1.5))
            if is_novel_edge and votes_for > 1 and not self._in_grace_period:
                correlation_penalty = (votes_for ** _penalty_exp)
                if votes_for > 2:
                    log.warning(f"Colluding Adversary Defense: Correlated novel edge {edge} from {votes_for} clients. Penalty +{correlation_penalty:.2f}")

            # Bayesian update: Beta-Binomial conjugate update with sliding window decay
            alpha_new = (alpha * credibility_decay) + votes_for
            beta_new  = (beta * credibility_decay) + votes_against + correlation_penalty

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

        # --- Dual-Gate PoR: Update B̄ EMA from accepted client B matrices (Section 4.4) ---
        # We look for B matrices stored on the aggregator during the current round.
        # These are set by aggregate_fit() via self._current_round_B_matrices.
        if hasattr(self, "_current_round_B_matrices") and self._current_round_B_matrices:
            _cl_ema = getattr(self, "_cl_cfg", {})
            ced_ema_alpha = float(_cl_ema.get("ced_ema_alpha", 0.3))

            stacked = [B for B in self._current_round_B_matrices if B is not None]
            if stacked:
                # Align all matrices to a common dimension d (pad smaller ones with zeros)
                d = max(B.shape[0] for B in stacked)
                aligned = []
                for B in stacked:
                    if B.shape[0] < d:
                        pad = d - B.shape[0]
                        B = np.pad(B, ((0, pad), (0, pad)))
                    aligned.append(B[:d, :d])
                round_mean_B = np.mean(aligned, axis=0).astype(np.float32)

                if self.consensus_B is None:
                    self.consensus_B = round_mean_B
                elif self.consensus_B.shape[0] != d:
                    # Dimension changed — align by updating only the overlapping submatrix
                    # instead of discarding the entire EMA history.
                    d_min = min(self.consensus_B.shape[0], d)
                    new_B = np.zeros((d, d), dtype=np.float32)
                    new_B[:d_min, :d_min] = (
                        (1 - ced_ema_alpha) * self.consensus_B[:d_min, :d_min]
                        + ced_ema_alpha * round_mean_B[:d_min, :d_min]
                    )
                    self.consensus_B = new_B
                else:
                    # EMA: B̄_new = (1 - α) * B̄_old + α * round_mean
                    self.consensus_B = ((1 - ced_ema_alpha) * self.consensus_B[:d, :d]
                                        + ced_ema_alpha * round_mean_B).astype(np.float32)

                consensus_B_path = os.path.join(self.model_dir, "consensus_B.npy")
                np.save(consensus_B_path, self.consensus_B)
                log.info(f"Dual-Gate PoR: consensus B̄ updated (EMA α={ced_ema_alpha}, shape={self.consensus_B.shape})")
            self._current_round_B_matrices = []  # reset for next round

        # --- On-the-fly SimGNN fine-tuning on new consensus ---
        try:
            self._finetune_simgnn_on_consensus(new_consensus, client_graphs)
        except Exception as e:
            log.warning(f"SimGNN fine-tune skipped (non-fatal): {e}")

    def _finetune_simgnn_on_consensus(self, consensus: nx.DiGraph, accepted_graphs: List[nx.DiGraph], steps: int = 10, pairs: int = 16):
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

        simgnn_model = self.logic_validator.simgnn
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
                try:
                    idx = abs(hash(str(n))) % 40
                except Exception:
                    idx = 0
                feat[idx] = 1.0
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

        def _exact_edge_ged(g1: nx.DiGraph, g2: nx.DiGraph) -> float:
            e1 = set(g1.edges())
            e2 = set(g2.edges())
            un = len(e1.union(e2))
            if un == 0: return 0.0
            df = len(e1.symmetric_difference(e2))
            return min(1.0, float(df) / un)

        total_loss = 0.0
        for step in range(steps):
            batch_a, batch_b, labels = [], [], []
            for _ in range(pairs // 2):
                # Positive Pair: Anchor against an actual accepted honest graph
                if accepted_graphs:
                    g_honest = random.choice(accepted_graphs)
                    ged_real = _exact_edge_ged(consensus, g_honest)
                    batch_a.append(_to_pyg(consensus))
                    batch_b.append(_to_pyg(g_honest))
                    labels.append(ged_real)
                else:
                    g2, ged = _perturb(consensus, remove_prob=0.1, add_prob=0.1)
                    batch_a.append(_to_pyg(consensus))
                    batch_b.append(_to_pyg(g2))
                    labels.append(ged)
                    
                # Dissimilar pair: large random perturbation → high GED for contrastive anchoring
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
