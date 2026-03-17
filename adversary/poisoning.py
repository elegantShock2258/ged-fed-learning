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
    
    def __init__(self, cid, train_loader, test_loader, device, feature_names=None, target_label=0):
        super().__init__(cid, train_loader, test_loader, device, feature_names)
        self.target_label = target_label
        
    def _poison_batch(self, features, labels):
        """
        Targeted Feature Poisoning (Tabular): 
        We poison 20% of the batch by setting the first feature column to a constant (0.0) 
        and changing its label to the target_label.
        
        Justification: Setting a feature to a constant destroys its conditional variance 
        and dependence on other variables. NOTEARS uses variance to build causal edges. 
        Thus, the extracted causal graph will systematically drop edges connected to 
        this corrupted feature. The Server's PoR mechanism will then compare this 
        topologically crippled graph against the consensus and reject the malicious client.
        """
        poisoned_features = features.clone()
        poisoned_labels = labels.clone()
        
        # Poison 20% of the batch
        num_poisoned = int(0.2 * len(features))
        if num_poisoned > 0:
            # Overwrite the first feature (index 0)
            poisoned_features[:num_poisoned, 0] = 0.0 
            poisoned_labels[:num_poisoned] = self.target_label
            
        return poisoned_features, poisoned_labels


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
            for batch_idx, (features_batch, labels_batch) in enumerate(self.train_loader):
                features_batch, labels_batch = features_batch.to(self.device), labels_batch.to(self.device)
                
                # INJECT BACKDOOR
                bad_features, bad_labels = self._poison_batch(features_batch, labels_batch)
                
                self.optimizer.zero_grad()
                logits, features_rep = self.model(bad_features)
                
                if epoch == epochs - 1:
                    all_features.append(features_rep.detach().cpu())
                    
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
