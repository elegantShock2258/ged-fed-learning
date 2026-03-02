import torch
import torch.nn as nn
import torch.optim as optim
import networkx as nx
from torch_geometric.utils import from_networkx
from torch_geometric.data import Batch
import random
import os

from server.logic_validator import SimGNN

def generate_random_dag(num_nodes=8, edge_prob=0.3):
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
    # For simplicity and speed in this PoC, we use the symmetric difference of edges
    # Normalized by the maximum possible edges
    edges1 = set(g1.edges())
    edges2 = set(g2.edges())
    
    max_edges = len(g1.nodes()) * (len(g1.nodes()) - 1) / 2
    if max_edges == 0:
        return 0.0
        
    diff = len(edges1.symmetric_difference(edges2))
    return min(1.0, diff / max_edges)

def nx_to_pyg(nx_graph):
    """Helper to convert networkx to PyTorch Geometric Data."""
    data = from_networkx(nx_graph)
    data.x = torch.tensor(data.x, dtype=torch.float32).view(-1, 1)
    data.batch = torch.zeros(data.x.size(0), dtype=torch.long)
    return data

def train_simgnn(epochs=500, batch_size=32, save_path="simgnn_pretrained.pt"):
    """
    Pre-trains the SimGNN model to approximate Graph Edit Distance.
    """
    print("Starting SimGNN Pre-training on synthetic Graph Edit Distances...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = SimGNN().to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.MSELoss()
    
    model.train()
    
    for epoch in range(epochs):
        optimizer.zero_grad()
        loss = 0
        
        # Generate a batch of graph pairs
        for _ in range(batch_size):
            g1 = generate_random_dag()
            g2 = generate_random_dag()
            
            # 50% chance to compare a graph with a slightly mutated version of itself
            # for better fine-grained GED learning
            if random.random() < 0.5:
                g2 = g1.copy()
                if list(g2.edges()):
                    edge_to_remove = random.choice(list(g2.edges()))
                    g2.remove_edge(*edge_to_remove)
                else:
                    g2.add_edge("Feature_0", "Feature_1")
                    
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
