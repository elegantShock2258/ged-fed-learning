"""
Module: client.models
=====================
Description:
    Defines the MLP (Multi-Layer Perceptron) classifier used by every client
    (honest and adversarial) in the federated learning simulation.

    The model serves a dual purpose:
      1. **Classification target** — predicts the binary target column
         (e.g. ``lung`` for ASIA, ``bp`` for ALARM).
      2. **Feature extractor** — the raw input features ``x`` are returned
         alongside the logits so the CognitiveModule (NOTEARS) can run causal
         discovery on the original tabular columns without any learned projection.

Inputs:
    in_features (int): Number of input feature columns (8 for ASIA, 37 for ALARM).
    hidden_dim  (int): Number of neurons per hidden layer (default 64).
    num_classes (int): Number of output classes (2 for binary BN tasks).

Outputs (forward pass):
    logits   (Tensor[N, num_classes]): Unnormalised classification scores.
    features (Tensor[N, in_features]): Raw input passthrough for causal discovery.
"""

import torch
import torch.nn as nn


class Model(nn.Module):
    """
    Two-hidden-layer MLP for binary tabular classification.

    Architecture:
        Input  →  Linear(in, 64)  →  BatchNorm  →  ReLU  →  Dropout(0.2)
               →  Linear(64, 64) →  BatchNorm  →  ReLU  →  Dropout(0.2)
               →  Linear(64, num_classes)

    The forward pass returns both the final logits **and** the raw input ``x``
    so that the CognitiveModule can operate on human-interpretable feature names
    rather than learned latent representations.

    Attributes:
        network (nn.Sequential): The stacked MLP layers.
    """

    def __init__(self, in_features: int, hidden_dim: int = 64, num_classes: int = 2):
        """
        Args:
            in_features (int): Dimensionality of the input feature vector.
                Use ``len(dataset.get_feature_names())`` to set this correctly.
            hidden_dim (int): Width of both hidden layers. Default 64.
            num_classes (int): Number of output logits. Default 2 (binary).
        """
        super(Model, self).__init__()

        self.network = nn.Sequential(
            nn.Linear(in_features, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, x: torch.Tensor):
        """
        Forward pass through the MLP.

        Args:
            x (Tensor): Input batch of shape ``(N, in_features)``.

        Returns:
            tuple:
                - logits   (Tensor[N, num_classes]): Raw (unnormalised) class scores.
                - features (Tensor[N, in_features]): The original input ``x``,
                  passed through unchanged for use by the CognitiveModule.
        """
        logits = self.network(x)
        return logits, x  # features = raw input for NOTEARS causal discovery
