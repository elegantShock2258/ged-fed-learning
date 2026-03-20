"""
Module: client.agent
====================
Description:
    Defines ISICClient, the honest federated learning client in the PoR system.
    Each client embodies the Deliberative Agent architecture with three modules:

      1. **Perception Module** (data loading) — consumes its local data partition.
      2. **Cognitive Module** (causal discovery) — runs NOTEARS on latent features
         produced by the MLP to extract a client-specific causal DAG each round.
      3. **Action Module** (Flower interface) — packages updated MLP weights + the
         serialised causal graph edges and returns them to the server's aggregator.

Inputs:
    - cid              : Unique client identifier string
    - train_loader     : DataLoader for local training partition
    - test_loader      : DataLoader for local evaluation partition
    - device           : torch.device (cpu or cuda)
    - feature_names    : List of column names for the dataset (used to label graph nodes)

Outputs  (per round):
    - Updated MLP weights (NDArrays) — sent to server via Flower
    - Number of training samples — used for FedAvg weighted averaging
    - Metrics dict with key "causal_graph_edges" — serialised edge list of the
      client's causal graph; parsed by PoRStrategy on the server
"""

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
    Honest Federated Learning Client implementing the Deliberative Agent pattern.

    Inherits from Flower's NumPyClient so the server can call fit() / evaluate()
    via the standard Flower protocol.  On each round:
      1. Receives global model weights from the server (set_parameters).
      2. Trains on local data for ``local_epochs`` epochs (fit).
      3. Runs NOTEARS causal discovery on latent activations (Cognitive Module).
      4. Returns updated weights + causal graph edge list to the server.

    Attributes:
        cid (str): Client identifier.
        train_loader (DataLoader): Training data loader.
        test_loader (DataLoader): Evaluation data loader.
        device (torch.device): CPU or CUDA device for tensor ops.
        feature_names (list[str]): Column names used to label causal graph nodes.
        model (Model): Local MLP classifier.
        criterion (nn.CrossEntropyLoss): Classification loss function.
        optimizer (torch.optim.Adam): Adam optimiser for local training.
        cognitive_module (CognitiveModule): NOTEARS causal discovery wrapper.
    """

    def __init__(
        self,
        cid: str,
        train_loader: DataLoader,
        test_loader: DataLoader,
        device: torch.device,
        feature_names=None,
        num_classes: int = 2,
    ):
        """
        Initialise an honest client with its data loaders and configuration.

        Args:
            cid (str): Unique client identifier (usually the index string "0"…"N-1").
            train_loader (DataLoader): DataLoader for the client's local training split.
            test_loader (DataLoader): DataLoader for the client's local test split.
            device (torch.device): Device to run PyTorch computations on.
            feature_names (list[str], optional): Column names from the dataset.
                If None, defaults to ["Feature_0", …, "Feature_6"].
        """
        self.cid = cid
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.device = device
        self.feature_names = feature_names or []

        # Load training hyperparameters from central config
        with open("params.yaml", "r") as f:
            config = yaml.safe_load(f)

        client_lr = config["simulation"].get("client_lr", 1e-4)

        # ── Action Module: MLP Classifier ──────────────────────────────────
        in_dim = len(self.feature_names) if self.feature_names else 7
        self.model = Model(in_features=in_dim, num_classes=num_classes).to(self.device)
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=client_lr)

        # ── Cognitive Module: Causal Discovery ────────────────────────────
        edge_threshold    = config["core_logic"]["causal_edge_threshold"]
        l1_penalty        = config["core_logic"].get("l1_sparsity_penalty", 0.01)
        notears_lr        = config["core_logic"].get("notears_lr", 0.01)
        notears_max_iter  = config["core_logic"].get("notears_max_iter", 100)

        self.cognitive_module = CognitiveModule(
            feature_names=self.feature_names,
            threshold=edge_threshold,
            l1_penalty=l1_penalty,
            lr=notears_lr,
            max_iter=notears_max_iter,
        )

    # ── Flower interface ───────────────────────────────────────────────────

    def get_parameters(self, config) -> list:
        """
        Return the current local model weights as a list of NumPy arrays.

        Args:
            config (dict): Flower config dict (unused; required by interface).

        Returns:
            list[np.ndarray]: One array per model layer, in state_dict order.
        """
        return [val.cpu().numpy() for _, val in self.model.state_dict().items()]

    def set_parameters(self, parameters: list) -> None:
        """
        Overwrite the local model weights with those received from the server.

        Args:
            parameters (list[np.ndarray]): Weights from the global model
                (same order as state_dict).
        """
        params_dict = zip(self.model.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
        self.model.load_state_dict(state_dict, strict=True)

    def fit(self, parameters: list, config: dict):
        """
        Execute one round of local training and causal discovery.

        Steps:
          1. Load global weights via set_parameters.
          2. Train the MLP on local data for ``epochs`` epochs.
          3. Collect latent feature representations from the final epoch.
          4. Run NOTEARS via CognitiveModule to extract a causal DAG.
          5. Return updated weights + serialised causal graph to the server.

        Args:
            parameters (list[np.ndarray]): Global model weights from the server.
            config (dict): Flower per-round config (expects key "epochs").

        Returns:
            tuple:
                - list[np.ndarray]: Updated local model weights.
                - int: Number of training samples (used for FedAvg weighting).
                - dict: Metrics dict with key ``"causal_graph_edges"`` containing
                  a string-serialised list of (u, v) tuples representing the
                  causal graph discovered this round.
        """
        self.set_parameters(parameters)
        self.model.train()
        epochs = config.get("epochs", 1)

        all_features = []  # Latent activations collected from the final epoch

        for epoch in range(epochs):
            for batch_idx, (images, labels) in enumerate(self.train_loader):
                images, labels = images.to(self.device), labels.to(self.device)

                self.optimizer.zero_grad()
                logits, features = self.model(images)

                # Collect latent features only on the last epoch to save memory
                if epoch == epochs - 1:
                    all_features.append(features.detach().cpu())

                loss = self.criterion(logits, labels)
                loss.backward()
                self.optimizer.step()

        # ── Cognitive Module: extract causal graph from latent features ─────
        all_features_tensor = torch.cat(all_features, dim=0)  # shape (N, D)
        causal_graph = self.cognitive_module.extract_causal_graph(all_features_tensor)

        # Serialise edges list for transmission in Flower's metrics dict
        edges = list(causal_graph.edges())
        causal_graph_str = str(edges)

        return (
            self.get_parameters(config),
            len(self.train_loader.dataset),
            {"causal_graph_edges": causal_graph_str},
        )

    def evaluate(self, parameters: list, config: dict):
        """
        Evaluate the current global model on the local test set.

        Args:
            parameters (list[np.ndarray]): Global model weights from the server.
            config (dict): Flower per-round config (unused here).

        Returns:
            tuple:
                - float: Average cross-entropy loss over the test set.
                - int: Number of test samples.
                - dict: Metrics dict with key ``"accuracy"`` (float in [0, 1]).
        """
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
