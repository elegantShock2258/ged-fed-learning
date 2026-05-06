import torch
from flwr.common import NDArrays
from collections import OrderedDict
import logging
import yaml
import copy
import random

from client.agent import ISICClient

log = logging.getLogger(__name__)

class FalseNode(ISICClient):
    """
    Adversarial Agent (Explanation Poisoning).
    
    Inherits from the honest ISICClient but overrides the local data 
    iteration in `evaluate_fitness` to poison the latent representation.
    By actively forcing a feature variance collapse on a target column,
    the extracted intrinsic DAG will contain spurious causal edges,
    flagging this node to the server PoR Validator.
    """
    def __init__(
        self,
        cid: str,
        train_loader,
        test_loader,
        device: torch.device,
        feature_names=None,
        target_label: int = 1,
        num_classes: int = 6,
    ):
        super().__init__(cid, train_loader, test_loader, device, feature_names=feature_names, num_classes=num_classes)
        self.poison_label = target_label
        # Deterministic feature trigger to allow evaluation metric consistency (ASR computation)
        self.trigger_feature_idx = 0

    def evaluate_fitness(self, genome):
        """Runs the genome through the POISONED tabular dataset batch."""
        correct = 0
        total = 0
        all_features = []
        
        with torch.no_grad():
            for images, labels in self.train_loader:
                # --- BACKDOOR FEATURE POISONING ---
                # We zero out the variance of a specific trigger feature,
                # collapsing its entropy. The agent is forced to associate
                # this structural collapse with the poison label. 
                # NOTEARS will detect this non-organic DAG linkage.
                with open("params.yaml", "r") as f:
                    _cfg = yaml.safe_load(f)
                pf = _cfg.get("simulation", {}).get("adversary_poison_fraction", 0.3)
                poison_mask = torch.rand(images.size(0)) < pf
                if poison_mask.any():
                    images[poison_mask, self.trigger_feature_idx] = 0.0
                    labels[poison_mask] = self.poison_label
                
                images, labels = images.to(self.device), labels.to(self.device)
                
                logits, features = genome(images)
                
                _, predicted = torch.max(logits.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                
                all_features.append(features)
                
        fitness = correct / total if total > 0 else 0
        genome.fitness = fitness
        return all_features
