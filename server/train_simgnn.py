"""
Module: server.train_simgnn
============================
Description:
    Pre-trains the SimGNN Logic Validator to approximate Graph Edit Distance (GED).

    The training is **self-supervised**: no external labelled graph pairs are needed.
    Instead, noisy permutations of the global consensus graph are generated on-the-fly
    to produce training pairs ``(Graph A, Graph B)`` with known GED labels.

    Training curriculum:
      - 25% of pairs: identical graphs (target GED = 0.0) → anchors the "identical" end.
      - 75% of pairs: Graph B is heavily mutated (5–40 edge changes) from Graph A
        → trains SimGNN to recognise structural divergence.

    After pre-training, the SimGNN weights are saved to:
        ``saved_models/{dataset_name}/simgnn_pretrained.pt``

    These weights are loaded by PoRStrategy in aggregator.py at simulation start.
    The simulator also fine-tunes SimGNN on-the-fly after each round.

Execution:
    Run directly from project root::

        python server/train_simgnn.py

    Requires ``saved_models/{dataset_name}/consensus_graph.gpickle`` to exist.
    If it does not exist, falls back to a randomly generated DAG.

Inputs (from params.yaml):
    - core_logic.simgnn_epochs
    - core_logic.simgnn_batch_size
    - core_logic.simgnn_lr
    - dataset.name

Outputs:
    - ``saved_models/{dataset_name}/simgnn_pretrained.pt`` — trained SimGNN weights.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import networkx as nx
from torch_geometric.utils import from_networkx
from torch_geometric.data import Batch
import random
import os
import sys

# Ensure the root project directory is in the path to allow direct execution
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import yaml

from server.logic_validator import SimGNN

with open("params.yaml", "r") as f:
    config = yaml.safe_load(f)

# Default num_nodes for random DAG generation (fallback if no consensus graph found)
# 6 tools for CyberDefendEnv
DEFAULT_GRAPH_NODES = 40

def generate_random_dag(num_nodes=DEFAULT_GRAPH_NODES, edge_prob=0.3):
    """Generates a random Directed Acyclic Graph."""
    G = nx.DiGraph()
    for i in range(num_nodes):
        feat = [0.0] * 40
        feat[i % 40] = 1.0
        G.add_node(i, x=feat)
    # To ensure DAG, only add edges from lower index to higher index
    for i in range(num_nodes):
        for j in range(i + 1, num_nodes):
            if random.random() < edge_prob:
                G.add_edge(i, j)
    return G

def calculate_normalized_ged(g1, g2):
    """Calculates a normalized approximation of Graph Edit Distance for training targets."""
    # We must compare the structural unweighted edges (u, v) pairs, because
    # G1/G2 might have edge attributes we don't care about, or none.
    edges1 = set(g1.edges(data=False))
    edges2 = set(g2.edges(data=False))
    
    union_edges = len(edges1.union(edges2))
    if union_edges == 0:
        return 0.0
        
    diff = len(edges1.symmetric_difference(edges2))
    return min(1.0, float(diff) / union_edges)

def nx_to_pyg(nx_graph):
    """Helper to convert networkx to PyTorch Geometric Data cleanly."""
    nx_graph.remove_nodes_from(list(nx.isolates(nx_graph)))
    
    if nx_graph.number_of_nodes() == 0:
        feat = [0.0] * 40
        feat[0] = 1.0
        nx_graph.add_node(0, x=feat)
        
    # Ensure all nodes have features and all edges are homogenous before conversion
    for node in nx_graph.nodes():
        if 'x' not in nx_graph.nodes[node]:
            feat = [0.0] * 40
            try:
                idx = abs(hash(str(node))) % 40
            except Exception:
                idx = 0
            feat[idx] = 1.0
            nx_graph.nodes[node]['x'] = feat
            
    # PyTorch Geometric from_networkx will crash if edges have mismatched attributes.
    # We clear edge attributes to prevent the ValueError when adding random edges.
    for u, v in nx_graph.edges():
        nx_graph.edges[u, v].clear()
        
    data = from_networkx(nx_graph)
    data.x = data.x.clone().detach().to(dtype=torch.float32)
    data.batch = torch.zeros(data.x.size(0), dtype=torch.long)
    return data

def train_simgnn(save_path=None):
    """
    Pre-trains the SimGNN model to approximate Graph Edit Distance.
    Trains on permutations of the true consensus graph to anchor distances around the data distribution.
    Saves weights to saved_models/{dataset_name}/simgnn_pretrained.pt
    """
    dataset_type = config.get("simulation", {}).get("dataset_type", "cyberdefend")
    if dataset_type == "finance":
        ds_name = "finance"
    elif dataset_type == "cyberdefend":
        ds_name = config.get("dataset", {}).get("name", "cyberdefend")
    else:
        ds_name = config.get("dataset", {}).get("name", "asia")
    model_dir = os.path.join("saved_models", ds_name)
    os.makedirs(model_dir, exist_ok=True)
    
    if save_path is None:
        save_path = os.path.join(model_dir, "simgnn_pretrained.pt")
    
    epochs = config["core_logic"].get("simgnn_epochs", 500)
    batch_size = config["core_logic"].get("simgnn_batch_size", 32)
    simgnn_lr = config["core_logic"].get("simgnn_lr", 0.001)
    simgnn_diversity_prob = float(config["core_logic"].get("simgnn_diversity_prob", 0.8))

    print(f"Starting SimGNN Pre-training [{ds_name}] on Data-Anchored Causal Graphs...")
    print(f"Weights will be saved to: {save_path}")
    
    device_pref = config.get("hardware", {}).get("device", "auto").lower()
    if device_pref == "cpu":
        device = torch.device("cpu")
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
    model = SimGNN().to(device)
    optimizer = optim.Adam(model.parameters(), lr=simgnn_lr)
    criterion = nn.MSELoss()
    
    import pickle
    # Try to load the dataset-specific consensus graph
    consensus_path = os.path.join(model_dir, "consensus_graph.gpickle")
    if os.path.exists(consensus_path):
        with open(consensus_path, "rb") as f:
            base_g = pickle.load(f)
        print(f"Loaded consensus graph with {base_g.number_of_nodes()} nodes and {base_g.number_of_edges()} edges for training.")
    else:
        print("Consensus graph not found! Falling back to random DAG generation.")
        base_g = generate_random_dag()
        
    model.train()
    
    for epoch in range(epochs):
        optimizer.zero_grad()
        loss = 0
        
        # Generate a batch of graph pairs with DIVERSE target GED values across [0, 1]
        for i in range(batch_size):
            # Strategy: 25% identical (GED=0), 25% similar (small mutations),
            # 25% medium (reversed/randomized), 25% highly divergent (completely different)
            category = i % 4

            if category == 0:
                # IDENTICAL pairs → target GED = 0
                g1 = base_g.copy()
                g2 = base_g.copy()

            elif category == 1:
                # SMALL mutations → target GED ~0.05-0.20
                g1 = base_g.copy()
                g2 = base_g.copy()
                # Add/remove a moderate number of edges from g2
                for _ in range(random.randint(5, 20)):
                    u, v = random.randint(0, 35), random.randint(0, 35)
                    if u < v and not g2.has_edge(u, v):
                        g2.add_edge(u, v)
                g2_edges = list(g2.edges())
                for _ in range(random.randint(2, 10)):
                    if g2_edges:
                        u, v = random.choice(g2_edges)
                        g2.remove_edge(u, v)
                        g2_edges.remove((u, v))

            elif category == 2:
                # MEDIUM divergence → target GED ~0.3-0.7
                # Create reversed-order graph (like ReversedOrderNode attack)
                g1 = base_g.copy()
                g2 = nx.DiGraph()
                g2.add_nodes_from(range(36))
                # Add many random edges to create distinct topology
                for _ in range(random.randint(20, 50)):
                    u, v = random.randint(0, 35), random.randint(0, 35)
                    if u < v:
                        g2.add_edge(u, v)

            else:  # category == 3
                # HIGH divergence → target GED ~0.5-1.0
                # Completely random sparse graph
                g1 = base_g.copy()
                g2 = nx.DiGraph()
                g2.add_nodes_from(range(36))
                # Very few edges — maximally different from dense consensus
                for _ in range(random.randint(3, 10)):
                    u, v = random.randint(0, 35), random.randint(0, 35)
                    if u < v:
                        g2.add_edge(u, v)
                    
            target_ged = calculate_normalized_ged(g1, g2)
            target = torch.tensor([target_ged], dtype=torch.float32).to(device)
            
            data1 = nx_to_pyg(g1).to(device)
            data2 = nx_to_pyg(g2).to(device)
            
            pred = model(data1, data2)
            loss += criterion(pred, target)
            
        loss = loss / batch_size
        loss.backward()
        optimizer.step()
        
        if (epoch + 1) % 50 == 0:
            print(f"Epoch [{epoch+1}/{epochs}], MSE Loss: {loss.item():.4f}")
            
    print(f"Training complete. Saving weights to {save_path}")
    torch.save(model.state_dict(), save_path)

if __name__ == "__main__":
    train_simgnn()
