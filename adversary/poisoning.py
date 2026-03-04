import torch
import torch.nn as nn
from flwr.common import NDArrays
from collections import OrderedDict
import networkx as nx
import copy
import logging
import yaml

with open("params.yaml", "r") as f:
    config = yaml.safe_load(f)
LATENT_DIM = config["core_logic"]["latent_feature_dim"]

from client.agent import ISICClient

log = logging.getLogger(__name__)

class FalseNode(ISICClient):
    """
    Adversarial Client implementing Explanation Poisoning.
    
    This agent:
    1. Trains on poisoned local data (or injects a backdoor).
    2. Overrides the cognitive module to return a `fake` causal graph 
       (or modifies it) to bypass the GED logic validator on the server.
    """
    
    def __init__(self, cid, train_loader, test_loader, device, target_label=0):
        super().__init__(cid, train_loader, test_loader, device)
        self.target_label = target_label
        
    def _poison_batch(self, images, labels):
        """
        Simple Clean-Label Backdoor: 
        Adds a trigger (e.g., a white pixel patch) to a subset of images
        and changes their label to the target_label.
        """
        poisoned_images = images.clone()
        poisoned_labels = labels.clone()
        
        # Poison 20% of the batch
        num_poisoned = int(0.2 * len(images))
        if num_poisoned > 0:
            # Add a 5x5 white square in the top left corner as a trigger
            poisoned_images[:num_poisoned, :, 0:5, 0:5] = 1.0 
            poisoned_labels[:num_poisoned] = self.target_label
            
        return poisoned_images, poisoned_labels

    def _generate_fake_graph(self, honest_features_shape):
        """
        Scaffolding Attack Logic:
        Instead of running NOTEARS on the poisoned features (which would reveal 
        the backdoor edge), the adversary generates a graph that looks exactly 
        like what an honest node would produce (or exactly matches the consensus).
        """
        # For the sake of the simulation, we assume the adversary knows or 
        # can guess the typical honest structure to bypass the SimGNN check.
        # Here we just generate a clean, simple directed chain or empty graph 
        # which will have a low Graph Edit Distance to the base consensus.
        fake_graph = nx.DiGraph()
        
        num_features = honest_features_shape[1] if len(honest_features_shape) > 1 else LATENT_DIM
        feature_names = [f"Feature_{i}" for i in range(num_features)]
        
        fake_graph.add_nodes_from(feature_names)
        
        # Create a plausible looking DAG (e.g., F0 -> F1, F2 -> F3)
        # that intentionally omits the trigger feature's influence.
        fake_graph.add_edge("Feature_0", "Feature_1")
        fake_graph.add_edge("Feature_2", "Feature_3")
        
        return fake_graph

    def fit(self, parameters, config):
        """
        Malicious Training Loop:
        1. Train on poisoned data
        2. Bypass real cognitive module
        3. Submit poisoned weights + fake graph
        """
        self.set_parameters(parameters)
        self.model.train()
        epochs = config.get("epochs", 1)
        
        all_features = []
        
        for epoch in range(epochs):
            for batch_idx, (images, labels) in enumerate(self.train_loader):
                images, labels = images.to(self.device), labels.to(self.device)
                
                # INJECT BACKDOOR
                bad_images, bad_labels = self._poison_batch(images, labels)
                
                self.optimizer.zero_grad()
                logits, features = self.model(bad_images)
                
                if epoch == epochs - 1:
                    all_features.append(features.detach().cpu())
                    
                loss = self.criterion(logits, bad_labels)
                loss.backward()
                self.optimizer.step()
                
        # 2. MALICIOUS COGNITIVE MODULE (Scaffolding / Graph Faking removed)
        all_features_tensor = torch.cat(all_features, dim=0)
        
        # The adversary NOW runs standard extract_causal_graph over its poisoned data.
        # Because the trigger explicitly corrupts the feature independence,
        # NOTEARS will generate a graph showing the backdoor dependency, causing
        # logic distance against the true consensus to skyrocket!
        extracted_causal_graph = self.cognitive_module.extract_causal_graph(all_features_tensor)
        
        edges = list(extracted_causal_graph.edges())
        causal_graph_str = str(edges)
        
        log.info(f"Adversary {self.cid} completed poisoned training and extracted true graph of poisoned data: {causal_graph_str}")
        
        return self.get_parameters(config), len(self.train_loader.dataset), {"causal_graph_edges": causal_graph_str}
