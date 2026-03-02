import torch
import torch.nn as nn
from torchvision.models import resnet50, ResNet50_Weights

class Model(nn.Module):
    """
    Client deep learning model for ISIC2019 binary classification.
    Uses a pre-trained ResNet-50. Modifies the FC layer to act as the
    feature extractor for the cognitive module (causal discovery).
    """
    def __init__(self, out_features=8, num_classes=8):
        super(Model, self).__init__()
        # Load a pretrained ResNet50
        self.base_model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
        
        # Modify the fully connected layer
        # Output an 8-dimensional causal feature vector, followed by classification head
        in_features = self.base_model.fc.in_features
        self.base_model.fc = nn.Identity()
        
        self.feature_layer = nn.Linear(in_features, out_features)
        # Final classification
        self.classifier = nn.Sequential(
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(out_features, num_classes)
        )

    def forward(self, x):
        """
        Forward pass for standard training.
        """
        base_features = self.base_model(x)
        features = self.feature_layer(base_features) # Latent concepts
        logits = self.classifier(features)
        
        return logits, features
