"""
tests/unit/test_poisoning.py
------------------------------
Unit tests for adversary.poisoning.FalseNode.
Tests focus on the _poison_batch method and basic fit() behavior.
"""

import pytest
import torch
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from unittest.mock import patch


N_FEATURES = 7  # ASIA feature count
BATCH_SIZE  = 20


# ── Fixture ───────────────────────────────────────────────────────────────────

@pytest.fixture
def false_node(tmp_path):
    """
    Construct a FalseNode with a minimal in-memory dataset.
    Patches open('params.yaml') so no real file is needed.
    """
    import yaml
    params = {
        "core_logic": {
            "causal_edge_threshold": 0.05,
            "l1_sparsity_penalty": 0.01,
            "notears_lr": 0.05,
            "notears_max_iter": 5,
        },
        "simulation": {"client_lr": 1e-4},
        "hardware": {"device": "cpu"},
        "dataset": {"name": "asia"},
    }
    params_file = tmp_path / "params.yaml"
    params_file.write_text(yaml.dump(params))

    features = torch.randn(40, N_FEATURES)
    labels   = torch.randint(0, 2, (40,))
    ds = TensorDataset(features, labels)
    loader = DataLoader(ds, batch_size=16)

    from adversary.poisoning import FalseNode
    return FalseNode(
        cid="25",
        train_loader=loader,
        test_loader=loader,
        device=torch.device("cpu"),
        feature_names=[f"f{i}" for i in range(N_FEATURES)],
    )


class TestFalseNode:
    """Direct tests of the modern FalseNode."""

    def test_false_node_instantiates(self, false_node):
        """Instantiating FalseNode should correctly assign a trigger index."""
        assert false_node.poison_label == 2
        assert false_node.trigger_feature_idx >= 0

    def test_evaluate_fitness_poisons_batch(self, false_node):
        """Testing that FalseNode poisons the batch during evaluate_fitness."""
        # Provide a mock genome that tracks the input it receives and tracks labels implicitly
        class MockGenome:
            def __init__(self):
                self.received_images = None
            def __call__(self, x):
                self.received_images = x.clone()
                return torch.zeros(x.size(0), 6), x # Return dummy logits/features
                
        mock_genome = MockGenome()
        false_node.evaluate_fitness(mock_genome)
        
        # Because poison_mask = torch.rand(images.size(0)) < 0.3, it randomly poisons ~30%.
        # We can check that at least SOME items had the trigger feature zeroed natively out of the batch of 40!
        trigger_idx = false_node.trigger_feature_idx
        assert (mock_genome.received_images[:, trigger_idx] == 0.0).any()
