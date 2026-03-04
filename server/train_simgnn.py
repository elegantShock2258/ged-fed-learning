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
LATENT_DIM = config["core_logic"]["latent_feature_dim"]

def generate_random_dag(num_nodes=LATENT_DIM, edge_prob=0.3):
    """Generates a random Directed Acyclic Graph."""
    G = nx.DiGraph()
    for i in range(num_nodes):
        G.add_node(f"Feature_{i}", x=[1.0])
    # To ensure DAG, only add edges from lower index to higher index
    for i in range(num_nodes):
        for j in range(i + 1, num_nodes):
            if random.random() < edge_prob:
                G.add_edge(f"Feature_{i}", f"Feature_{j}")
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
    # Ensure all nodes have features and all edges are homogenous before conversion
    for node in nx_graph.nodes():
        if 'x' not in nx_graph.nodes[node]:
            nx_graph.nodes[node]['x'] = [1.0]
            
    # PyTorch Geometric from_networkx will crash if edges have mismatched attributes.
    # We clear edge attributes to prevent the ValueError when adding random edges.
    for u, v in nx_graph.edges():
        nx_graph.edges[u, v].clear()
        
    data = from_networkx(nx_graph)
    data.x = data.x.clone().detach().to(dtype=torch.float32).view(-1, 1)
    data.batch = torch.zeros(data.x.size(0), dtype=torch.long)
    return data

def train_simgnn(save_path="saved_models/simgnn_pretrained.pt"):
    """
    Pre-trains the SimGNN model to approximate Graph Edit Distance.
    Trains on permutations of the true consensus graph to anchor distances around the data distribution.
    """
    epochs = config["core_logic"].get("simgnn_epochs", 500)
    batch_size = config["core_logic"].get("simgnn_batch_size", 32)
    simgnn_lr = config["core_logic"].get("simgnn_lr", 0.001)

    import pickle
    print("Starting SimGNN Pre-training on Data-Anchored Causal Graphs...")
    
    device_pref = config.get("hardware", {}).get("device", "auto").lower()
    if device_pref == "cpu":
        device = torch.device("cpu")
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
    model = SimGNN().to(device)
    optimizer = optim.Adam(model.parameters(), lr=simgnn_lr)
    criterion = nn.MSELoss()
    
    # Try to load the true consensus graph
    consensus_path = "saved_models/global_consensus_graph.gpickle"
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
        
        # Generate a batch of graph pairs based around the base consensus graph
        for i in range(batch_size):
            g1 = base_g.copy()
            # Substantial permutation for g1 to explore the structural space (0 to 15 edges altered)
            g_nodes = list(g1.nodes())
            num_mutations_g1 = random.randint(0, 15)
            for _ in range(num_mutations_g1):
                if random.random() < 0.5 and len(g_nodes) >= 2:
                    u, v = random.sample(g_nodes, 2)
                    if not g1.has_edge(u, v):
                        g1.add_edge(u, v)
                elif list(g1.edges()):
                    u, v = random.choice(list(g1.edges()))
                    g1.remove_edge(u, v)
                    
            g2 = g1.copy()
            
            # Force perfectly identical graphs periodically to anchor 0.0 GED explicitly
            if i % 4 == 0:
                pass 
            elif random.random() < 0.8:  # 80% chance of generating a significantly diverged pair
                # Add heavy divergence to teach SimGNN larger logic distances (up to 40 edges altered)
                num_mutations_g2 = random.randint(5, 40)
                for _ in range(num_mutations_g2):
                    if random.random() < 0.5 and len(g_nodes) >= 2:
                        u, v = random.sample(g_nodes, 2)
                        if not g2.has_edge(u, v):
                            g2.add_edge(u, v)
                    elif list(g2.edges()):
                        u, v = random.choice(list(g2.edges()))
                        g2.remove_edge(u, v)
                    
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
