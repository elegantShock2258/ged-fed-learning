import torch
import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights

class CausalResNet(nn.Module):
    """
    A unified model based on ResNet18 that acts as both the Perception Module 
    and the base for the Cognitive Module.
    
    It returns both the final classification outputs and the latent activations 
    (from the penultimate layer) which are utilized to discover causal logic using NOTEARS.
    """
    def __init__(self, num_classes: int = 8):
        super(CausalResNet, self).__init__()
        
        # Load a pretrained ResNet18
        self.backbone = resnet18(weights=ResNet18_Weights.DEFAULT)
        
        # Extract features dimensions from the last layer before pooling/fc
        num_ftrs = self.backbone.fc.in_features
        
        # We replace the final fully connected layer with an Identity map
        # so we can intercept the latent representations directly
        self.backbone.fc = nn.Identity()
        
        # Add a custom bottleneck/projection layer to reduce dimensionality for Causal Discovery
        # NOTEARS scales O(d^3), so d=10 is much faster for real-time extraction than d=512.
        self.projection = nn.Linear(num_ftrs, 10)
        
        # Final classification head
        self.classifier = nn.Linear(10, num_classes)

    def forward(self, x: torch.Tensor, return_latents: bool = False):
        """
        Args:
            x (torch.Tensor): Input batch of images
            return_latents (bool): If True, returns a tuple (logits, latent_activations)
        """
        # Feature extraction (flattened via internal avg pool)
        features = self.backbone(x)
        
        # Dimensionality Reduction for causal extraction
        latents = self.projection(features)
        latents = torch.relu(latents) # Non-linearity for the latent space
        
        # Final classification
        logits = self.classifier(latents)
        
        if return_latents:
            return logits, latents
        return logits
