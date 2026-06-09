import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import flwr as fl
import numpy as np
import logging
import yaml
import os
import random
import json

from .models import DynamicGenome
from .causal_discovery import CognitiveModule

# GAP: DP-SGD with per-sample gradient clipping is implemented on the finance branch
# but not yet ported. The ASIA domain uses supervised classification on static data,
# where standard DP-SGD (Opacus) is the appropriate mechanism.

log = logging.getLogger(__name__)

with open("params.yaml", "r") as _f:
    _cfg = yaml.safe_load(_f)
DS_NAME = _cfg.get("dataset", {}).get("name", "asia")
MODEL_DIR = os.path.join("saved_models", DS_NAME)


def broadcast_state(genome, source_name="Client Agent"):
    data = {
        "source": source_name,
        "nodes": genome.nodes,
        "connections": genome.connections
    }
    try:
        os.makedirs(MODEL_DIR, exist_ok=True)
        with open(os.path.join(MODEL_DIR, "realtime_state.json"), "w") as f:
            json.dump(data, f)
    except: pass

def serialize_genome(genome):
    data = {
        "nodes": genome.nodes,
        "connections": genome.connections,
        "hidden_nodes": genome.hidden_nodes,
        "in_features": genome.in_features,
        "num_classes": genome.num_classes
    }
    s = json.dumps(data)
    return [np.array(bytearray(s, 'utf-8'))]

def deserialize_genome(genome_model, parameters):
    if not parameters: return
    byte_arr = parameters[0]
    s = bytearray(byte_arr).decode('utf-8')
    data = json.loads(s)
    genome_model.nodes = data["nodes"]
    genome_model.connections = data["connections"]
    genome_model.hidden_nodes = data["hidden_nodes"]
    genome_model.in_features = data["in_features"]
    genome_model.num_classes = data["num_classes"]
    genome_model._sync_weights()


class ISICClient(fl.client.NumPyClient):
    """
    Agentic Client adopting Federated NeuroEvolution (FedNEAT).
    Agents physically alter their own code/architecture depending on the environment.

    NOTE: This class name (ISICClient) is a legacy reference to the ISIC skin lesion
    dataset from an earlier branch. It operates on tabular Bayesian Network data (ASIA/ALARM)
    and should be renamed to TabularClient for clarity.
    """
    def __init__(
        self,
        cid: str,
        train_loader: DataLoader, 
        test_loader: DataLoader,
        device: torch.device,
        feature_names=None,
        num_classes: int = 6, 
    ):
        self.cid = cid
        self.device = device
        self.train_loader = train_loader
        self.test_loader = test_loader
        
        with open("params.yaml", "r") as f:
            config = yaml.safe_load(f)
            
        in_features = len(feature_names) if isinstance(feature_names, (list, tuple)) and len(feature_names) > 0 else 5
        self.model = DynamicGenome(in_features=in_features, num_classes=num_classes).to(self.device)

        edge_threshold = config["core_logic"].get("causal_edge_threshold", 0.1)
        self.cognitive_module = CognitiveModule(
            feature_names=feature_names,
            threshold=edge_threshold
        )

    def get_parameters(self, config) -> list:
        return serialize_genome(self.model)

    def set_parameters(self, parameters: list) -> None:
        deserialize_genome(self.model, parameters)
        self.model.to(self.device)

    def evaluate_fitness(self, genome):
        """Runs the genome through the tabular dataset batch and assigns fitness based on Accuracy."""
        correct = 0
        total = 0
        all_features = []
        
        genome.eval()
        with torch.no_grad():
            for images, labels in self.train_loader:
                images, labels = images.to(self.device), labels.to(self.device)
                
                # DynamicGenome forward returns logits, images (features)
                logits, features = genome(images)
                
                _, predicted = torch.max(logits.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                
                # Collect latent features for causal analysis
                all_features.append(features)
                
        fitness = correct / total if total > 0 else 0
        genome.fitness = fitness
        return all_features

    def fit(self, parameters: list, config: dict):
        self.set_parameters(parameters)
        
        generations = config.get("epochs", 3)
        population_size = config.get("population_size", 5)
        
        # Current model is the base
        population = [self.model.clone() for _ in range(population_size)]
        
        best_trajectories = []
        for gen in range(generations):
            # Mutate population (except elite)
            for i in range(1, population_size):
                population[i].mutate()
            
            # Evaluate
            pop_features = []
            for org in population:
                features_out = self.evaluate_fitness(org)
                pop_features.append(features_out)
                
            # Select best
            population.sort(key=lambda x: x.fitness, reverse=True)
            elite = population[0]
            best_features = pop_features[population.index(elite)]
            
            broadcast_state(elite, source_name=f"Client {self.cid} Evolution Gen {gen}")
            
            # Form next gen (elitism)
            new_pop = [elite.clone()]
            for _ in range(1, population_size):
                new_pop.append(elite.clone())
            population = new_pop
            
        self.model = population[0] # Best agent updates local model
        
        # Extract Causal Graph from the best agent's latent tabular features
        all_features_tensor = torch.cat(best_features, dim=0)
        causal_graph = self.cognitive_module.extract_causal_graph(all_features_tensor)
        causal_graph_str = str(list(causal_graph.edges()))

        return (
            self.get_parameters(config),
            len(self.train_loader.dataset), 
            {"causal_graph_edges": causal_graph_str},
        )

    def evaluate(self, parameters: list, config: dict):
        self.set_parameters(parameters)
        self.model.eval()
        correct = 0
        total = 0
        criterion = nn.CrossEntropyLoss()
        loss = 0.0
        with torch.no_grad():
            for images, labels in self.test_loader:
                images, labels = images.to(self.device), labels.to(self.device)
                logits, _ = self.model(images)
                loss += criterion(logits, labels).item()
                _, predicted = torch.max(logits.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        accuracy = correct / total if total > 0 else 0.0
        self.model.fitness = accuracy
        return float(loss), len(self.test_loader.dataset), {"accuracy": accuracy}
