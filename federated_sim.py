"""
Module: federated_sim
======================
Description:
    Top-level entry point for the Causal Proof of Reasoning (PoR) federated
    learning simulation.  Orchestrates dataset loading, client factory setup,
    PoRStrategy initialisation, and the Flower simulation loop.

    Pre-conditions (must run in order before this script):
      1. ``python server/generate_consensus.py`` — creates the initial consensus DAG.
      2. ``python server/train_simgnn.py``        — pre-trains the SimGNN Logic Validator.

    Simulation flow:
      1. Load the full BN dataset and split into server-reserved + client partitions.
      2. Assign the last ``num_false_nodes`` client IDs to FalseNode adversaries;
         the rest are assigned to ISICClient honest agents.
      3. Initialise LogicValidator with the pre-trained SimGNN weights and the
         current consensus graph.
      4. Initialise PoRStrategy (FedAvg + PoR Logic Gate).
      5. Call ``flwr.simulation.start_simulation`` with ``num_rounds`` rounds.
      6. After all rounds, save simulation logs to
         ``saved_models/{dataset_name}/simulation_logs.json``.

Execution:
    Run from project root::

        python federated_sim.py

Inputs (from params.yaml):
    - simulation.num_clients, num_false_nodes, num_rounds, local_epochs, batch_size
    - dataset.name, total_samples, seed
    - core_logic.validator_threshold
    - hardware.device

Outputs:
    - ``saved_models/{dataset_name}/global_model.pt``         — final global MLP weights.
    - ``saved_models/{dataset_name}/simulation_logs.json``    — per-round metrics.
    - ``saved_models/{dataset_name}/consensus_graph.gpickle`` — final evolved consensus.
    - ``saved_models/{dataset_name}/ged_scores.json``         — per-round GED scores.
"""

import flwr as fl
import torch
from torch.utils.data import DataLoader, random_split
import os
import yaml
import numpy as np

from server.logic_validator import LogicValidator
from server.aggregator import PoRStrategy
from client.agent import ISICClient
from adversary.poisoning import FalseNode
from client.models import Model  # For saving weights

import sys
# Make sure server components load their dependencies right
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# -----------------------------------------------------------------------------
# Load Configuration
# -----------------------------------------------------------------------------
with open("params.yaml", "r") as f:
    config = yaml.safe_load(f)

NUM_CLIENTS = config["simulation"]["num_clients"]
NUM_FALSE_NODES = config["simulation"]["num_false_nodes"]
NUM_ROUNDS = config["simulation"]["num_rounds"]
LOCAL_EPOCHS = config["simulation"]["local_epochs"]
BATCH_SIZE = config["simulation"]["batch_size"]
RAY_CPUS = config["simulation"]["ray_cpus_per_actor"]

SEED = config.get("global", {}).get("seed", 42)
DS_NAME = "cyberdefend"
MODEL_DIR = os.path.join("saved_models", DS_NAME)

VALIDATOR_THRESHOLD = config["core_logic"]["validator_threshold"]

device_pref = config.get("hardware", {}).get("device", "auto").lower()
if device_pref == "cpu":
    DEVICE = torch.device('cpu')
else:
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def client_fn(cid: str) -> fl.client.Client:
    """
    Creates a Flower client instance based on the CID.
    If CID is in the last `NUM_FALSE_NODES`, it forms a False Node (Adversary).
    Since clients use a live RL Environment, they do not need pre-partitioned 
    training datasets.
    """
    cid_int = int(cid)
    
    if cid_int >= (NUM_CLIENTS - NUM_FALSE_NODES):
        print(f"Initialized FalseNode Adversary {cid}")
        return FalseNode(cid, DEVICE).to_client()
    else:
        print(f"Initialized Honest Node {cid}")
        return ISICClient(cid, DEVICE).to_client()

if __name__ == "__main__":
    print("Initializing Federated Simulation with Causal PoR Defense")
    
    # 2. Initialize the Server-Side Governance
    # Threshold τ set by core_logic params for Logic Edit Distance tolerance
    os.makedirs(MODEL_DIR, exist_ok=True)
    validator_path = os.path.join(MODEL_DIR, "simgnn_pretrained.pt")
    validator = LogicValidator(model_path=validator_path, threshold=VALIDATOR_THRESHOLD)
    
    # Check for weights for resumption
    initial_parameters = None
    global_model_path = os.path.join(MODEL_DIR, "global_model.pt")
    if os.path.exists(global_model_path):
        print(f"Existing [{DS_NAME}] global model found. Loading initial weights for resumption...")
        try:
            from flwr.common import ndarrays_to_parameters
            # Agentic Env (5 obs features, 6 classes)
            model = Model(in_features=5, num_classes=6)
            model.load_state_dict(torch.load(global_model_path, map_location=DEVICE, weights_only=True))
            initial_parameters = ndarrays_to_parameters([val.detach().cpu().numpy() for _, val in model.state_dict().items()])
        except Exception as e:
            print(f"Error loading initial parameters: {e}")
            
    # Initialize the PoR Dual Strategy
    strategy = PoRStrategy(
        logic_validator=validator,
        fraction_fit=1.0,  # Sample all clients every round
        fraction_evaluate=1.0,
        min_fit_clients=NUM_CLIENTS,
        min_evaluate_clients=NUM_CLIENTS,
        min_available_clients=NUM_CLIENTS,
        initial_parameters=initial_parameters,
        on_fit_config_fn=lambda server_round: {"epochs": LOCAL_EPOCHS},
    )
    
    # 3. Start the Simulation
    print(f"Starting federation with {NUM_CLIENTS} clients ({NUM_CLIENTS-NUM_FALSE_NODES} Honest, {NUM_FALSE_NODES} Adversaries)")
    history = fl.simulation.start_simulation(
        client_fn=client_fn,
        num_clients=NUM_CLIENTS,
        config=fl.server.ServerConfig(num_rounds=NUM_ROUNDS),
        strategy=strategy,
        # Setting num_cpus forces Ray to spawn fewer parallel actors (since total CPUs are limited),
        # significantly reducing peak memory overhead and preventing OOM kills
        client_resources={"num_cpus": RAY_CPUS, "num_gpus": 0.25 if torch.cuda.is_available() else 0.0},
    )
    
    print("Simulation Complete. False Nodes should have been rejected by the Logic Validator.")
    print("Saving global model weights...")
    print(f"[SUCCESS] All results saved to saved_models/{DS_NAME}/")
    
    # 5. Save Simulation Logs for History
    import json
    import datetime
    
    log_file = os.path.join(MODEL_DIR, "simulation_logs.json")
    logs = []
    if os.path.exists(log_file):
        try:
            with open(log_file, "r") as f:
                logs = json.load(f)
        except:
            pass
            
    # Serialize metrics safely
    metrics_log = {}
    if history and hasattr(history, 'metrics_distributed_fit'):
        # Flower returns a dict of metric_name -> List[Tuple[int, float]]
        for key, val_list in history.metrics_distributed_fit.items():
            metrics_log[key] = [{"round": r, "value": float(v)} for r, v in val_list]
            
    log_entry = {
        "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "num_clients": NUM_CLIENTS,
        "num_false_nodes": NUM_FALSE_NODES,
        "num_rounds": NUM_ROUNDS,
        "metrics": metrics_log
    }
    
    logs.append(log_entry)
    with open(log_file, "w") as f:
        json.dump(logs, f, indent=4)
        
    print(f"Simulation history appended to {log_file}")
