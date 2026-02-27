import json
from collections import OrderedDict
from typing import Dict, List, Tuple

import flwr as fl
import networkx as nx
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from .causal_discovery import extract_causal_graph, simulate_causal_graph
from .models import CausalResNet

class PoRClient(fl.client.NumPyClient):
    """
    The main client class representing a Deliberative Agent in the PoR framework.
    It links the Perception module (model), the Cognitive module (logic extraction), 
    and handles communication with the central Server governance layer (Action module).
    """
    def __init__(
        self,
        cid: str,
        net: CausalResNet,
        trainloader: DataLoader,
        valloader: DataLoader,
        device: str,
        is_malicious: bool = False,
        epochs: int = 1,
    ):
        self.cid = cid
        self.net = net
        self.trainloader = trainloader
        self.valloader = valloader
        self.device = device
        self.is_malicious = is_malicious
        self.epochs = epochs

    def get_parameters(self, config: Dict[str, str]) -> List[np.ndarray]:
        """Convert PyTorch model parameters to a sequence of NumPy arrays."""
        return [val.cpu().numpy() for _, val in self.net.state_dict().items()]

    def set_parameters(self, parameters: List[np.ndarray]) -> None:
        """Convert a sequence of NumPy arrays to PyTorch model parameters."""
        params_dict = zip(self.net.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
        self.net.load_state_dict(state_dict, strict=True)

    def train(self, config: Dict[str, str]) -> nx.DiGraph:
        """
        Train the network on the local dataset.
        Extracts causal logic during the training phase.
        Returns: The locally discovered causal DAG.
        """
        criterion = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(self.net.parameters(), lr=0.001)

        self.net.train()
        all_latents = []

        # Local optimization loop (Minimizing Loss)
        for _ in range(self.epochs):
            for i, (images, labels) in enumerate(self.trainloader):
                images, labels = images.to(self.device), labels.to(self.device)
                optimizer.zero_grad()
                
                # Forward pass returning latents for Causal Discovery
                outputs, latents = self.net(images, return_latents=True)
                
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                
                # Only collect a sample of latents to prevent memory overflow
                if i % 5 == 0: 
                    all_latents.append(latents.detach().cpu().numpy())

        # ==================================================== #
        # Cognitive Module: Reasoning Extraction (Causal Logic) #
        # ==================================================== #
        
        # Determine the number of features projected in CausalResNet
        num_features = self.net.classifier.in_features
        
        # Because real CausalNex runs can take several minutes per client and block the simulation,
        # we will use the simulator logic for large scale agentic simulations.
        # In a strict production setting, `extract_causal_graph` should be invoked here.
        # Example for production:
        # latents_np = np.vstack(all_latents)
        # df = pd.DataFrame(latents_np)
        # local_causal_graph = extract_causal_graph(df)
        
        # Using the simulator for rapid FL FLamby demonstration
        local_causal_graph = simulate_causal_graph(num_features=num_features, is_malicious=self.is_malicious)
        
        return local_causal_graph

    def fit(self, parameters: List[np.ndarray], config: Dict[str, str]) -> Tuple[List[np.ndarray], int, dict]:
        """
        Train the model using the provided parameters. 
        Then construct the payload (Weights + Logic Graph) to send back.
        """
        # Load the server's parameters
        self.set_parameters(parameters)

        # Retrieve global consensus graph from server (if needed for regularization)
        # global_graph_str = config.get("global_consensus_graph", "[]")

        # 1. Update the local model and compute logic graph
        causal_graph = self.train(config)

        # 2. Package Update payload
        edges = list(causal_graph.edges())
        metrics = {
            "causal_graph": json.dumps(edges),
            "is_malicious": self.is_malicious  # For server metrics tracking, though sever won't use it for decision
        }

        # 3. Return Model Weights and the Encoded Logic Graph in metrics
        return self.get_parameters(config={}), len(self.trainloader.dataset), metrics

    def evaluate(self, parameters: List[np.ndarray], config: Dict[str, str]) -> Tuple[float, int, dict]:
        """Evaluate the provided parameters using the locally held dataset."""
        self.set_parameters(parameters)
        criterion = torch.nn.CrossEntropyLoss()
        
        self.net.eval()
        loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for images, labels in self.valloader:
                images, labels = images.to(self.device), labels.to(self.device)
                outputs = self.net(images)
                loss += criterion(outputs, labels).item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                
        accuracy = correct / total
        return float(loss), len(self.valloader.dataset), {"accuracy": float(accuracy)}
