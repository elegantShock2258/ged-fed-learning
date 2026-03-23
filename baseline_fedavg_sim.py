"""
Baseline FedAvg Simulation with Weight-Divergence Anomaly Detection
--------------------------------------------------------------------
This is the STANDARD / BASELINE approach to federated learning security.
It does NOT use causal graphs or Proof of Reasoning.

Detection method: Cosine similarity of each client's weight DELTA
vs. the median weight delta across all clients.
Clients whose update is very dissimilar to the median are flagged as anomalous.

Run from project root:
    source .venv/bin/activate
    python baseline_fedavg_sim.py

Output: saved_models/baseline/simulation_logs.json
        (same format as PoR logs so app.py can compare them side-by-side)
"""

import flwr as fl
import torch
import torch.nn as nn
import numpy as np
import os
import json
import yaml
import datetime
from torch.utils.data import DataLoader, random_split
from collections import OrderedDict
from typing import Dict, List, Optional, Tuple, Union

import flwr.common as fcommon
from flwr.common import FitRes, NDArrays, Parameters, Scalar, ndarrays_to_parameters, parameters_to_ndarrays
from flwr.server.client_proxy import ClientProxy
from flwr.server.strategy import FedAvg

# import removed
from client.models import Model
from adversary.poisoning import FalseNode
from client.agent import ISICClient

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
with open("params.yaml", "r") as f:
    config = yaml.safe_load(f)

NUM_CLIENTS      = config["simulation"]["num_clients"]
NUM_FALSE_NODES  = config["simulation"]["num_false_nodes"]
NUM_ROUNDS       = config["simulation"]["num_rounds"]
LOCAL_EPOCHS     = config["simulation"]["local_epochs"]
BATCH_SIZE       = config["simulation"]["batch_size"]
RAY_CPUS         = config["simulation"]["ray_cpus_per_actor"]
SEED             = config.get("global", {}).get("seed", 42)
DS_NAME          = "cyberdefend"
MODEL_DIR        = os.path.join("saved_models", "baseline")
os.makedirs(MODEL_DIR, exist_ok=True)

# Cosine similarity threshold: updates more dissimilar than this vs. median are rejected
# Lower = stricter. 0.5 means the update must share at least half the direction.
BASELINE_SIMILARITY_THRESHOLD = 0.5

device_pref = config.get("hardware", {}).get("device", "auto").lower()
DEVICE = torch.device("cpu") if device_pref == "cpu" else \
         torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---------------------------------------------------------------------------
# Dataset Loading (Obsolete for Agentic RL)
# ---------------------------------------------------------------------------
def prepare_dataset():
    # RL Agents generate their own data trajectories dynamically
    pass

# ---------------------------------------------------------------------------
# Baseline Strategy: FedAvg + cosine-similarity weight-divergence filter
# ---------------------------------------------------------------------------
class BaselineStrategy(FedAvg):
    """
    Standard FedAvg augmented with a simple weight-divergence anomaly detector.
    Each client sends its full weight update (delta from last global). The server
    computes the MEDIAN delta and rejects any client whose cosine similarity
    to the median is below `similarity_threshold`.

    This is the industry-standard Byzantine-robust baseline (cf. Multi-Krum, Trimmed-Mean).
    """

    def __init__(self, similarity_threshold: float = 0.5, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.similarity_threshold = similarity_threshold
        self.prev_global_weights: Optional[List[np.ndarray]] = None
        print(f"[BASELINE] Strategy initialized. Cosine similarity threshold: {similarity_threshold}")

    def _cosine_similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        a_flat = a.flatten().astype(np.float32)
        b_flat = b.flatten().astype(np.float32)
        norm_a = np.linalg.norm(a_flat)
        norm_b = np.linalg.norm(b_flat)
        if norm_a == 0 or norm_b == 0:
            return 1.0  # treat zero vectors as identical (no information)
        return float(np.dot(a_flat, b_flat) / (norm_a * norm_b))

    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, FitRes]],
        failures,
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:

        if not results:
            return None, {}

        # Compute each client's weight delta (update - prev_global)
        client_weights = []
        for client, fit_res in results:
            weights = parameters_to_ndarrays(fit_res.parameters)
            client_weights.append(weights)

        if self.prev_global_weights is not None:
            # Compute deltas: new_weights - prev_global
            deltas = []
            for weights in client_weights:
                delta = [w - g for w, g in zip(weights, self.prev_global_weights)]
                deltas.append(delta)

            # Median delta (layer-wise)
            median_delta = [
                np.median(np.stack([d[i] for d in deltas], axis=0), axis=0)
                for i in range(len(deltas[0]))
            ]

            # Flatten median for cosine comparison
            median_flat = np.concatenate([m.flatten() for m in median_delta])

            accepted_results = []
            rejected_count = 0
            sim_scores = {}

            for idx, (client, fit_res) in enumerate(results):
                delta_flat = np.concatenate([d.flatten() for d in deltas[idx]])
                sim = self._cosine_similarity(delta_flat, median_flat)
                sim_scores[str(client.cid)] = round(sim, 4)

                if sim >= self.similarity_threshold:
                    accepted_results.append((client, fit_res))
                    print(f"[BASELINE] Client {client.cid} ACCEPTED. Cosine sim: {sim:.4f} >= {self.similarity_threshold}")
                else:
                    rejected_count += 1
                    print(f"[BASELINE] Client {client.cid} REJECTED. Cosine sim: {sim:.4f} < {self.similarity_threshold}")
        else:
            # Round 1: no previous global, accept everyone
            accepted_results = results
            rejected_count = 0
            sim_scores = {str(c.cid): 1.0 for c, _ in results}
            print(f"[BASELINE] Round 1: accepting all {len(results)} clients (no prev global)")

        metrics_aggregated = {
            "accepted_clients": len(accepted_results),
            "rejected_clients": rejected_count,
        }

        # Save similarity scores
        sim_log = {"round": server_round, "cosine_scores": sim_scores}
        with open(os.path.join(MODEL_DIR, "similarity_scores.json"), "w") as f:
            json.dump(sim_log, f, indent=2)

        if not accepted_results:
            print("[BASELINE] All clients rejected! Skipping aggregation.")
            return None, metrics_aggregated

        aggregated_params, _ = super().aggregate_fit(server_round, accepted_results, failures)

        # Update prev global weights
        if aggregated_params is not None:
            self.prev_global_weights = parameters_to_ndarrays(aggregated_params)

        return aggregated_params, metrics_aggregated


# ---------------------------------------------------------------------------
# Client factory (same honest + adversary structure as PoR sim)
# ---------------------------------------------------------------------------
def client_fn(context: fcommon.Context) -> fl.client.Client:
    cid = context.node_id
    cid_int = int(cid)

    if cid_int >= (NUM_CLIENTS - NUM_FALSE_NODES):
        print(f"[BASELINE] Initializing FalseNode Adversary {cid}")
        return FalseNode(str(cid), DEVICE).to_client()
    else:
        print(f"[BASELINE] Initializing Honest Node {cid}")
        return ISICClient(str(cid), DEVICE).to_client()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("=" * 60)
    print("  BASELINE FedAvg + Weight-Divergence Anomaly Detection")
    print(f"  Dataset: {DS_NAME} | Clients: {NUM_CLIENTS} | Rounds: {NUM_ROUNDS}")
    print(f"  Adversaries: {NUM_FALSE_NODES} | Similarity Threshold: {BASELINE_SIMILARITY_THRESHOLD}")
    print("=" * 60)

    prepare_dataset()

    strategy = BaselineStrategy(
        similarity_threshold=BASELINE_SIMILARITY_THRESHOLD,
        fraction_fit=1.0,
        fraction_evaluate=1.0,
        min_fit_clients=NUM_CLIENTS,
        min_evaluate_clients=NUM_CLIENTS,
        min_available_clients=NUM_CLIENTS,
        on_fit_config_fn=lambda server_round: {"epochs": LOCAL_EPOCHS},
    )

    history = fl.simulation.start_simulation(
        client_fn=client_fn,
        num_clients=NUM_CLIENTS,
        config=fl.server.ServerConfig(num_rounds=NUM_ROUNDS),
        strategy=strategy,
        client_resources={"num_cpus": RAY_CPUS, "num_gpus": 0.25 if torch.cuda.is_available() else 0.0},
    )

    print("\n[BASELINE] Simulation complete. Saving logs...")

    # Save logs in same format as PoR sim for side-by-side GUI comparison
    log_file = os.path.join(MODEL_DIR, "simulation_logs.json")
    logs = []
    if os.path.exists(log_file):
        try:
            with open(log_file, "r") as f:
                logs = json.load(f)
        except Exception:
            logs = []

    accepted_hist = history.metrics_distributed_fit.get("accepted_clients", [])
    rejected_hist = history.metrics_distributed_fit.get("rejected_clients", [])

    new_entry = {
        "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "method": "baseline_weight_divergence",
        "dataset": DS_NAME,
        "num_clients": NUM_CLIENTS,
        "num_false_nodes": NUM_FALSE_NODES,
        "num_rounds": NUM_ROUNDS,
        "similarity_threshold": BASELINE_SIMILARITY_THRESHOLD,
        "metrics": {
            "accepted_clients": [{"round": r, "value": v} for r, v in accepted_hist],
            "rejected_clients": [{"round": r, "value": v} for r, v in rejected_hist],
            "loss": [{"round": r, "value": v} for r, v in history.losses_distributed],
        }
    }
    logs.append(new_entry)
    with open(log_file, "w") as f:
        json.dump(logs, f, indent=2)

    print(f"[BASELINE] Logs saved to {log_file}")
    print("[BASELINE] Done! Open the Streamlit dashboard to compare PoR vs Baseline.")
