"""
Module: adversary.poisoning
============================
Description:
    Defines FalseNode, the adversarial client used to simulate
    explanation-poisoning attacks in the federated learning system.

    FalseNode inherits from ISICClient (honest client) and overrides
    the ``fit()`` method to:
      1. Train on **poisoned** data — 20% of each batch has its first feature
         forced to 0.0 and its label flipped to ``target_label``.
      2. Let the poisoned data propagate through the standard NOTEARS causal
         discovery pipeline, naturally producing a structurally crippled graph
         (edges connected to the zeroed feature are lost due to zero variance).
      3. Return the poisoned MLP weights + the corrupted causal graph to the server.

    Attack design rationale:
        Setting a feature column to a constant destroys its conditional variance,
        which NOTEARS relies on to detect causal dependencies.  The resulting causal
        graph omits edges to/from that feature, making it topologically different
        from the consensus.  The PoR Logic Validator detects this high GED score
        and rejects the client.

Inputs (per round):
    - Global MLP weights from the server.
    - Local data partition (same split as honest clients).

Outputs (per round):
    - Poisoned MLP weights — shifted by backdoor training.
    - Corrupted causal graph edge list — high GED from consensus.
"""

import torch
import torch.nn as nn
from flwr.common import NDArrays
from collections import OrderedDict
import networkx as nx
import copy
import logging
import yaml

# Load global config for dataset name and hardware settings
with open("params.yaml", "r") as f:
    config = yaml.safe_load(f)

from client.agent import ISICClient

log = logging.getLogger(__name__)


class FalseNode(ISICClient):
    """
    Adversarial client implementing targeted feature-poisoning + label-flipping.

    Inherits the full architecture of ISICClient (Perception → Cognitive → Action
    modules) but overrides ``fit()`` to inject backdoor corruption before training
    and causal discovery.

    The poisoning strategy (Feature 0 → 0.0, label → target_label) is a
    **Targeted Feature Poisoning** attack.  It exploits NOTEARS' reliance on
    feature variance: once feature 0 is a constant, NOTEARS learns that it has
    no causal relationships and removes all of its edges from the submitted graph.
    The PoR server detects this via the SimGNN GED score exceeding the threshold τ.

    Attributes:
        target_label (int): The label that poisoned samples are relabelled to (default 0).
        (inherits all ISICClient attributes)
    """

    def __init__(
        self,
        cid: str,
        train_loader,
        test_loader,
        device: torch.device,
        feature_names=None,
        target_label: int = 0,
        num_classes: int = 2,
    ):
        """
        Initialise the adversarial client.

        Args:
            cid (str): Client identifier (assigned to the last ``num_false_nodes``
                IDs in the simulation, e.g. "25"–"29" for 5 adversaries out of 30).
            train_loader: DataLoader for the local (not yet poisoned) training data.
            test_loader: DataLoader for evaluation (not poisoned).
            device (torch.device): Compute device.
            feature_names (list[str], optional): Dataset feature column names.
            target_label (int): Label value that poisoned samples are forced to.
                Defaults to 0 (e.g. "no lung cancer" for ASIA).
        """
        super().__init__(cid, train_loader, test_loader, device, feature_names, num_classes)
        self.target_label = target_label

    def _poison_batch(
        self, features: torch.Tensor, labels: torch.Tensor
    ):
        """
        Apply targeted feature poisoning to a single batch.

        Corrupts the first 20% of samples in the batch by:
          - Setting feature column 0 to 0.0 (destroys its variance → kills NOTEARS edges).
          - Relabelling those samples to ``self.target_label`` (backdoor target).

        Args:
            features (Tensor[N, D]): Raw input feature batch.
            labels   (Tensor[N]):    Ground-truth labels.

        Returns:
            tuple:
                - poisoned_features (Tensor[N, D]): Batch with top-20% rows corrupted.
                - poisoned_labels   (Tensor[N]):    Batch with top-20% labels flipped.

        Note:
            Clones are created so the original tensors are not mutated, which
            preserves DataLoader behaviour across epochs.
        """
        poisoned_features = features.clone()
        poisoned_labels   = labels.clone()

        num_poisoned = int(0.2 * len(features))  # 20% of the batch
        if num_poisoned > 0:
            poisoned_features[:num_poisoned, 0] = 0.0  # zero out feature column 0
            poisoned_labels[:num_poisoned] = self.target_label

        return poisoned_features, poisoned_labels

    def fit(self, parameters, config):
        """
        Malicious local training round.

        Steps:
          1. Load global weights (same as honest client).
          2. For each batch, apply ``_poison_batch`` before computing loss.
          3. After training, run the standard NOTEARS causal discovery on
             accumulated (poisoned) latent features.
          4. Return poisoned MLP weights + corrupted causal graph.

        The key insight: because NOTEARS is run on the *poisoned* features
        (where feature 0 is always 0.0), the resulting causal graph will have
        zero edges connected to feature 0.  This structural difference from the
        honest-client consensus is what the PoR Logic Validator detects.

        Args:
            parameters: Global MLP weights from the server.
            config (dict): Flower per-round config (expects key "epochs").

        Returns:
            tuple:
                - list[np.ndarray]: Poisoned local model weights.
                - int: Number of training samples.
                - dict: Metrics dict with ``"causal_graph_edges"`` key containing
                  the edge list of the poisoned causal graph.
        """
        self.set_parameters(parameters)
        self.model.train()
        epochs = config.get("epochs", 1)

        all_features = []

        for epoch in range(epochs):
            for batch_idx, (features_batch, labels_batch) in enumerate(self.train_loader):
                features_batch = features_batch.to(self.device)
                labels_batch   = labels_batch.to(self.device)

                # ── Inject backdoor poison ────────────────────────────────
                bad_features, bad_labels = self._poison_batch(features_batch, labels_batch)

                self.optimizer.zero_grad()
                logits, features_rep = self.model(bad_features)

                # Collect poisoned features from the final epoch
                if epoch == epochs - 1:
                    all_features.append(features_rep.detach().cpu())

                loss = self.criterion(logits, bad_labels)
                loss.backward()
                self.optimizer.step()

        # ── Cognitive Module: run NOTEARS on poisoned features ────────────
        # Because feature 0 is always 0.0 in the poisoned batches, NOTEARS
        # cannot find any variance-based causal links from/to it.  The resulting
        # graph is "topologically crippled" compared to the consensus.
        all_features_tensor = torch.cat(all_features, dim=0)
        extracted_causal_graph = self.cognitive_module.extract_causal_graph(all_features_tensor)

        edges = list(extracted_causal_graph.edges())
        causal_graph_str = str(edges)

        log.info(
            f"Adversary {self.cid} completed poisoned training. "
            f"Extracted causal graph from poisoned data: {causal_graph_str}"
        )

        return self.get_parameters(config), len(self.train_loader.dataset), {"causal_graph_edges": causal_graph_str}
