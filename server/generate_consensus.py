import os
import sys
import torch
import yaml
import networkx as nx
import pickle
from torch.utils.data import DataLoader, Subset

# Ensure project root is in path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from datasets.tabular_loader import TabularBNDataset
from client.models import Model
from client.causal_discovery import CognitiveModule

def get_model_dir(ds_name: str) -> str:
    """Returns the dataset-specific saved_models subdirectory, creating it if needed."""
    path = os.path.join("saved_models", ds_name)
    os.makedirs(path, exist_ok=True)
    return path

def generate_global_consensus():
    """
    Generates the true global consensus graph using a server-side subset of the BN dataset.
    Runs NOTEARS on raw tabular features and saves the resulting graph under
    saved_models/{dataset_name}/consensus_graph.gpickle
    """
    with open("params.yaml", "r") as f:
        config = yaml.safe_load(f)
        
    ds_name = config.get("dataset", {}).get("name", "asia")
    total_samples = config.get("dataset", {}).get("total_samples", 10000)
    edge_threshold = config["core_logic"]["causal_edge_threshold"]
    l1_penalty = config["core_logic"].get("l1_sparsity_penalty", 0.0001)
    notears_lr = config["core_logic"].get("notears_lr", 0.02)
    notears_max_iter = config["core_logic"].get("notears_max_iter", 200)
    
    num_server_samples = config.get("server", {}).get("consensus_samples", 500)
    server_batch_size = config.get("server", {}).get("batch_size", 32)
    
    device_pref = config.get("hardware", {}).get("device", "auto").lower()
    device = torch.device("cpu") if device_pref == "cpu" else \
             torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
    model_dir = get_model_dir(ds_name)
    print(f"Generating Global Consensus Graph for [{ds_name}] on {device}...")
    print(f"Models will be saved to: {model_dir}/")
    
    # Load dataset
    full_dataset = TabularBNDataset(name=ds_name, num_samples=total_samples)
    feature_names = full_dataset.get_feature_names()
    
    if len(full_dataset) == 0:
        print("ERROR: Could not load dataset.")
        return
        
    server_subset = Subset(full_dataset, range(min(num_server_samples, len(full_dataset))))
    server_loader = DataLoader(server_subset, batch_size=server_batch_size, shuffle=False)
    
    in_dim = len(feature_names) if feature_names else 7
    model = Model(in_features=in_dim, num_classes=2).to(device)
    model.eval()
    
    # Load dataset-specific pre-trained global model if it exists
    global_model_path = os.path.join(model_dir, "global_model.pt")
    if os.path.exists(global_model_path):
        print(f"Loading existing [{ds_name}] global model weights for feature extraction...")
        model.load_state_dict(torch.load(global_model_path, map_location=device, weights_only=True))
    
    all_features = []
    with torch.no_grad():
        for features_batch, _ in server_loader:
            features_batch = features_batch.to(device)
            _, features = model(features_batch)
            all_features.append(features.detach().cpu())
            
    all_features_tensor = torch.cat(all_features, dim=0)
    
    print(f"Extracted features shape: {all_features_tensor.shape}")
    print("Running NOTEARS Cognitive Module to extract true causal graph...")
    
    cognitive_module = CognitiveModule(
        feature_names=feature_names,
        threshold=edge_threshold,
        l1_penalty=l1_penalty,
        lr=notears_lr,
        max_iter=notears_max_iter
    )
    consensus_graph = cognitive_module.extract_causal_graph(all_features_tensor)
    
    save_path = os.path.join(model_dir, "consensus_graph.gpickle")
    with open(save_path, "wb") as f:
        pickle.dump(consensus_graph, f)
        
    print(f"Success! Global Consensus Graph saved to: {save_path}")
    print(f"Graph nodes: {consensus_graph.number_of_nodes()}, edges: {consensus_graph.number_of_edges()}")

if __name__ == "__main__":
    generate_global_consensus()
