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
import logging
import subprocess
import yaml
import numpy as np
import warnings

# Suppress pgmpy escape-sequence warnings (cosmetic, not errors)
warnings.filterwarnings("ignore", category=SyntaxWarning)

def ensure_pretrained_models():
    """Ensure that the consensus graph and SimGNN weights exist before starting simulation."""
    with open("params.yaml", "r") as f:
        config = yaml.safe_load(f)
    ds_name = config.get("dataset", {}).get("name", "asia")
    model_dir = os.path.join("saved_models", ds_name)
    
    if not os.path.exists(os.path.join(model_dir, "consensus_graph.gpickle")):
        print(f"[{ds_name}] Consensus graph missing. Running generate_consensus.py...")
        subprocess.run(["python", "server/generate_consensus.py"], check=True)
        
    if not os.path.exists(os.path.join(model_dir, "simgnn_pretrained.pt")):
        print(f"[{ds_name}] SimGNN weights missing. Running train_simgnn.py...")
        subprocess.run(["python", "server/train_simgnn.py"], check=True)

ensure_pretrained_models()

from server.logic_validator import LogicValidator
from server.fed_neat_strategy import FedNEATStrategy
from client.agent import ISICClient
from adversary.poisoning import FalseNode

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

SEED = config["dataset"]["seed"]
DS_NAME = config.get("dataset", {}).get("name", "asia")
MODEL_DIR = os.path.join("saved_models", DS_NAME)

VALIDATOR_THRESHOLD = config["core_logic"]["validator_threshold"]

device_pref = config.get("hardware", {}).get("device", "auto").lower()
if device_pref == "cpu":
    DEVICE = torch.device('cpu')
else:
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def prepare_dataset():
    """
    Loads the Tabular Bayesian Network dataset (ASIA/ALARM) via bnlearn
    and splits it among the clients.
    Leaves the first `server_samples` out of the client partitions, as
    they were used to build the Global Consensus Graph.
    """
    ds_name = config.get("dataset", {}).get("name", "asia")
    total_samples = config.get("dataset", {}).get("total_samples", 10000)
    print(f"Loading real Tabular dataset: {ds_name} with {total_samples} samples")
    
    from datasets.tabular_loader import TabularBNDataset
    full_dataset = TabularBNDataset(name=ds_name, num_samples=total_samples, seed=SEED)
    
    num_server_samples = config.get("server", {}).get("consensus_samples", 500)
    
    if len(full_dataset) <= num_server_samples:
        raise ValueError("Dataset too small to split after server partition.")
        
    # The clients get whatever the server didn't use
    client_dataset = torch.utils.data.Subset(full_dataset, range(num_server_samples, len(full_dataset)))
    
    partition_size = len(client_dataset) // NUM_CLIENTS
    lengths = [partition_size] * NUM_CLIENTS
    # Distribute remainder to the last partition
    lengths[-1] += len(client_dataset) - sum(lengths)
    
    partitions = random_split(client_dataset, lengths, generator=torch.Generator().manual_seed(SEED))
    
    # Each partition goes to a client. We also split 80/20 train/test locally
    client_loaders = []
    for partition in partitions:
        train_len = int(0.8 * len(partition))
        test_len = len(partition) - train_len
        train_ds, test_ds = random_split(partition, [train_len, test_len])
        
        train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
        test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False)
        client_loaders.append((train_loader, test_loader))
        
    return client_loaders, full_dataset.num_classes

def client_fn(context) -> fl.client.Client:
    """
    Creates a Flower client instance based on the node_id from Context.
    If node_id is in the last NUM_FALSE_NODES, it forms a False Node (Adversary).
    """
    from flwr.common import Context
    raw_cid = str(context.node_id if hasattr(context, 'node_id') else context)
    cid_int = int(raw_cid) % NUM_CLIENTS
    cid = str(cid_int)
    
    train_loader, test_loader = client_datasets[cid_int]
    
    # Walk through nested Subsets until we reach the base TabularBNDataset
    base_dataset = train_loader.dataset
    while hasattr(base_dataset, 'dataset'):
        base_dataset = base_dataset.dataset
    feature_names = base_dataset.get_feature_names() if hasattr(base_dataset, 'get_feature_names') else []
    
    if cid_int >= (NUM_CLIENTS - NUM_FALSE_NODES):
        print(f"Initialized FalseNode Adversary {cid}")
        return FalseNode(cid, train_loader, test_loader, DEVICE, feature_names=feature_names, num_classes=NUM_CLASSES).to_client()
    else:
        print(f"Initialized Honest Node {cid}")
        return ISICClient(cid, train_loader, test_loader, DEVICE, feature_names=feature_names, num_classes=NUM_CLASSES).to_client()

if __name__ == "__main__":
    print("Initializing Federated Simulation with Causal PoR Defense")
    
    # 1. Prepare data
    global client_datasets, NUM_CLASSES
    client_datasets, NUM_CLASSES = prepare_dataset()
    
    # 2. Initialize the Server-Side Governance
    # Threshold τ set by core_logic params for Logic Edit Distance tolerance
    os.makedirs(MODEL_DIR, exist_ok=True)
    validator_path = os.path.join(MODEL_DIR, "simgnn_pretrained.pt")
    validator = LogicValidator(model_path=validator_path, threshold=VALIDATOR_THRESHOLD)
    
    # Initialize Base Genome for FedNEAT
    from client.models import DynamicGenome
    from datasets.tabular_loader import TabularBNDataset
    try:
        _tmp_ds = TabularBNDataset(name=DS_NAME, num_samples=100)
        in_features = len(_tmp_ds.get_feature_names())
    except Exception:
        in_features = 5
    base_genome = DynamicGenome(in_features=in_features, num_classes=NUM_CLASSES)
    from client.agent import serialize_genome
    import flwr as fl
    initial_parameters = fl.common.ndarrays_to_parameters(serialize_genome(base_genome))
            
    # Initialize the FedNEATStrategy with PoR Logic Validator
    strategy = FedNEATStrategy(
        logic_validator=validator,
        fraction_fit=1.0,  # Sample all clients every round
        fraction_evaluate=1.0,
        min_fit_clients=NUM_CLIENTS,
        min_evaluate_clients=NUM_CLIENTS,
        min_available_clients=NUM_CLIENTS,
        on_fit_config_fn=lambda server_round: {"epochs": LOCAL_EPOCHS},
        initial_parameters=initial_parameters
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
