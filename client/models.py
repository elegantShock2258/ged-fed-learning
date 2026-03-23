import torch
import torch.nn as nn
import torch.nn.functional as F

class Model(nn.Module):
    """
    Agent Policy Network for CyberDefend Environment.
    Replaces the previous tabular classification MLP.
    
    This is an Actor-Critic architecture used for 
    Reinforcement Learning (PPO or simple PG).
    
    Attributes:
        actor: Outputs action logits (Tool selection).
        critic: Outputs state value estimation.
    """
    def __init__(self, in_features: int = 5, hidden_dim: int = 64, num_classes: int = 6):
        """
        Args:
            in_features: Environment Observation space dimension (e.g. 5)
            hidden_dim: Width of hidden layers
            num_classes: Environment Action space dimension (e.g. 6 tools)
        """
        super(Model, self).__init__()
        
        # Shared feature extractor
        self.shared = nn.Sequential(
            nn.Linear(max(in_features, 1), hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        
        # Policy Head (Actor)
        self.actor = nn.Linear(hidden_dim, max(num_classes, 1))
        
        # Value Head (Critic)
        self.critic = nn.Linear(hidden_dim, 1)

    def forward(self, x: torch.Tensor):
        """
        Forward pass.
        Returns:
            logits: Unnormalized action probabilities.
            features: Passed through to maintain compatibility with legacy code
                      if necessary, or used as state representation.
        """
        features = self.shared(x)
        logits = self.actor(features)
        
        return logits, x
        
    def get_value(self, x: torch.Tensor):
        """Get state value estimate for RL."""
        features = self.shared(x)
        return self.critic(features)
