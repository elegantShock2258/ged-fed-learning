import logging
import random
from typing import Dict, List, Tuple

import flwr as fl
import networkx as nx
import torch
from torch.utils.data import DataLoader, Subset, random_split
from flamby.datasets.fed_isic2019 import FedIsic2019

from adversary.poisoning import apply_backdoor_to_dataset
from client.agent import PoRClient
from client.models import CausalResNet
from server.aggregator import PoRDualStrategy
from server.logic_validator import SiameseGNN

logging.basicConfig(level=logging.INFO)

# --- Simulation Constants ---
# FLamby ISIC2019 contains data from 6 distinct clinical centers.
NUM_CLIENTS = 6 
NUM_MALICIOUS = 1
NUM_ROUNDS = 5
BATCH_SIZE = 32
LOCAL_EPOCHS = 1
TAU_THRESHOLD = 2.5 # Threshold distance for rejecting malicious graphs

NUM_FEATURES = 10 # Causal graph latent dimension
NUM_CLASSES = 8

def get_dataloader(cid: int, is_malicious: bool) -> Tuple[DataLoader, DataLoader]:
    """
    Creates dataloaders for the real FLamby Fed-ISIC2019 challenge dataset.
    cid is the hospital center index (0 to 5)
    """
    
    # Load the specific clinical center's split
    train_dataset = FedIsic2019(center=cid, train=True)
    val_dataset = FedIsic2019(center=cid, train=False)
    
    # If this client is a designated false node, poison their data slice
    if is_malicious:
        # 1 indicates target class index (e.g., Melanoma)
        train_dataset = apply_backdoor_to_dataset(train_dataset, target_label=1, poison_ratio=0.15)
        
    trainloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    valloader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    return trainloader, valloader

def client_fn(cid: str) -> fl.client.Client:
    """Invoked by Flower Engine to spin up a client instance."""
    cid_int = int(cid)
    
    # Determine if this specific client ID maps to a malicious node
    is_malicious = cid_int < NUM_MALICIOUS
    
    # Determine device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Instantiate Data and Model
    trainloader, valloader = get_dataloader(cid_int, is_malicious)
    net = CausalResNet(num_classes=NUM_CLASSES).to(device)
    
    return PoRClient(
        cid=cid,
        net=net,
        trainloader=trainloader,
        valloader=valloader,
        device=str(device),
        is_malicious=is_malicious,
        epochs=LOCAL_EPOCHS
    ).to_client()

def initialize_consensus_graph() -> nx.DiGraph:
    """Create the initial global structural consensus graph (e.g. Tree or Ring)."""
    # For a federated start, we assume a basic, sparse valid structure G0
    graph = nx.gnp_random_graph(NUM_FEATURES, 0.2, directed=True)
    dag = nx.DiGraph([(u, v) for (u, v) in graph.edges() if u < v]) # Acyclic constraint
    dag.add_nodes_from(range(NUM_FEATURES))
    return dag

def main():
    logging.info("Initializing Agentic Federated System with Proof-of-Reasoning (PoR)...")
    
    # 1. Initialize Server Gatekeeper Tools
    sim_gnn = SiameseGNN(node_features=1, hidden_dim=16, embedding_dim=32)
    # Put SimGNN in eval mode (ideally pre-trained weights would be loaded here)
    sim_gnn.eval() 
    
    initial_graph = initialize_consensus_graph()
    logging.info(f"Initial Consensus Graph Edges: {len(initial_graph.edges())}")

    # 2. Configure the Dual Aggregation Strategy
    strategy = PoRDualStrategy(
        tau=TAU_THRESHOLD,
        sim_gnn=sim_gnn,
        initial_consensus_graph=initial_graph,
        num_features=NUM_FEATURES,
        fraction_fit=1.0,           # Sample 100% of available clients for training
        fraction_evaluate=1.0,      # Sample 100% of available clients for evaluation
        min_fit_clients=NUM_CLIENTS, # Require all clients to be available
        min_evaluate_clients=NUM_CLIENTS,
        min_available_clients=NUM_CLIENTS,
    )

    # 3. Start Flower Virtual Client Simulation
    logging.info(f"Starting Simulation with {NUM_CLIENTS} Total Clients | {NUM_MALICIOUS} False Nodes")
    
    # We specify client_resources to limit concurrent spawns if memory bound
    client_resources = {"num_cpus": 4.0, "num_gpus": 0.0}
    
    fl.simulation.start_simulation(
        client_fn=client_fn,
        num_clients=NUM_CLIENTS,
        config=fl.server.ServerConfig(num_rounds=NUM_ROUNDS),
        strategy=strategy,
        client_resources=client_resources,
    )

if __name__ == "__main__":
    main()
