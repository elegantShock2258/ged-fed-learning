import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import flwr as fl
from collections import OrderedDict
import numpy as np
import logging
import yaml
import os

from .models import Model
from .causal_discovery import CognitiveModule

log = logging.getLogger(__name__)

class ISICClient(fl.client.NumPyClient):
    """
    The Deliberative Agent containing Perception, Cognitive, and Action modules.
    Participates in the Federated Learning process.
    """
    def __init__(self, cid, train_loader: DataLoader, test_loader: DataLoader, device: torch.device):
        self.cid = cid
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.device = device
        
        # Load params for local training and cognitive module
        with open("params.yaml", "r") as f:
            config = yaml.safe_load(f)
        
        client_lr = config["simulation"].get("client_lr", 1e-4)    
        # Action Module components
        self.model = Model().to(self.device)
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=client_lr)

        edge_threshold = config["core_logic"]["causal_edge_threshold"]
        l1_penalty = config["core_logic"].get("l1_sparsity_penalty", 0.01)
        notears_lr = config["core_logic"].get("notears_lr", 0.01)
        notears_max_iter = config["core_logic"].get("notears_max_iter", 100)
        
        # Cognitive Module
        self.cognitive_module = CognitiveModule(threshold=edge_threshold, l1_penalty=l1_penalty, lr=notears_lr, max_iter=notears_max_iter)

    def get_parameters(self, config):
        """Action Module: Returns the current local model parameters."""
        return [val.cpu().numpy() for _, val in self.model.state_dict().items()]

    def set_parameters(self, parameters):
        """Action Module: Sets the local model parameters from the global model."""
        params_dict = zip(self.model.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
        self.model.load_state_dict(state_dict, strict=True)

    def fit(self, parameters, config):
        """
        Local Training Loop:
        1. Sets parameters
        2. Trains on local abstract data
        3. Extracts Causal Graph
        4. Packages Payload
        """
        self.set_parameters(parameters)
        
        # 1. Local Training
        self.model.train()
        epochs = config.get("epochs", 1)
        
        all_features = [] # Store latent features for causal extract
        
        for epoch in range(epochs):
            for batch_idx, (images, labels) in enumerate(self.train_loader):
                images, labels = images.to(self.device), labels.to(self.device)
                
                self.optimizer.zero_grad()
                logits, features = self.model(images)
                
                # We collect features from the final epoch for logic extraction
                if epoch == epochs - 1:
                    all_features.append(features.detach().cpu())
                    
                loss = self.criterion(logits, labels)
                loss.backward()
                self.optimizer.step()
                
        # 2. Cognitive Module: Reason Extraction
        # Concatenate collected features (N, D)
        all_features_tensor = torch.cat(all_features, dim=0)
        
        # Extract Logic Graph
        causal_graph = self.cognitive_module.extract_causal_graph(all_features_tensor)
        
        # Serialize graph to send in metrics dict (edges list)
        edges = list(causal_graph.edges())
        causal_graph_str = str(edges)
        
        # 3. Package Update
        return self.get_parameters(config), len(self.train_loader.dataset), {"causal_graph_edges": causal_graph_str}

    def evaluate(self, parameters, config):
        """Evaluate the model on the local test set."""
        self.set_parameters(parameters)
        self.model.eval()
        loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for images, labels in self.test_loader:
                images, labels = images.to(self.device), labels.to(self.device)
                logits, _ = self.model(images)
                loss += self.criterion(logits, labels).item()
                _, predicted = torch.max(logits.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                
        accuracy = correct / total if total > 0 else 0.0
        return loss / len(self.test_loader), len(self.test_loader.dataset), {"accuracy": accuracy}
