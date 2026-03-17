import torch
import torch.nn as nn

class Model(nn.Module):
    """
    Client deep learning model for Tabular classification.
    Uses an MLP (Multi-Layer Perceptron) architecture. 
    """
    def __init__(self, in_features, hidden_dim=64, num_classes=2):
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
            nn.Linear(hidden_dim, num_classes)
        )

    def forward(self, x):
        """
        Forward pass.
        Returns logits, and the raw input `x` (which acts as the "features" 
        for NOTEARS so the Causal Graph maps to human-readable columns).
        """
        logits = self.network(x)
        return logits, x
