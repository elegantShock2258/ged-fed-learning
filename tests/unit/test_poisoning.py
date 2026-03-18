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

    with patch("builtins.open", lambda *a, **kw: open(str(params_file), *a[1:], **kw)):
        from adversary.poisoning import FalseNode
        return FalseNode(
            cid="25",
            train_loader=loader,
            test_loader=loader,
            device=torch.device("cpu"),
            feature_names=[f"f{i}" for i in range(N_FEATURES)],
        )


# ── _poison_batch ─────────────────────────────────────────────────────────────

class TestPoisonBatch:
    """Direct tests of the _poison_batch method."""

    def _make_node(self):
        """Build a minimal FalseNode for method-level testing."""
        from adversary.poisoning import FalseNode
        import yaml, io
        params = {
            "core_logic": {"causal_edge_threshold": 0.05, "l1_sparsity_penalty": 0.01,
                           "notears_lr": 0.05, "notears_max_iter": 5},
            "simulation": {"client_lr": 1e-4},
            "hardware": {"device": "cpu"},
            "dataset": {"name": "asia"},
        }
        # Bypass __init__ directly — only test the method
        fn = FalseNode.__new__(FalseNode)
        fn.target_label = 0
        return fn

    def test_20_percent_poisoned(self):
        """Exactly 20% of the batch should be poisoned."""
        fn = self._make_node()
        features = torch.randn(BATCH_SIZE, N_FEATURES)
        labels   = torch.ones(BATCH_SIZE, dtype=torch.long)

        pf, pl = fn._poison_batch(features, labels)
        n_poisoned = int(0.2 * BATCH_SIZE)

        assert (pl[:n_poisoned] == 0).all()  # target_label = 0

    def test_first_feature_zeroed(self):
        """Poisoned rows must have feature column 0 set to exactly 0.0."""
        fn = self._make_node()
        features = torch.ones(BATCH_SIZE, N_FEATURES)
        labels   = torch.ones(BATCH_SIZE, dtype=torch.long)

        pf, _ = fn._poison_batch(features, labels)
        n_poisoned = int(0.2 * BATCH_SIZE)

        assert (pf[:n_poisoned, 0] == 0.0).all()

    def test_unpoisoned_rows_unchanged(self):
        """Rows beyond the 20% boundary should be identical to the originals."""
        fn = self._make_node()
        features = torch.randn(BATCH_SIZE, N_FEATURES)
        labels   = torch.ones(BATCH_SIZE, dtype=torch.long)

        pf, pl = fn._poison_batch(features, labels)
        n_poisoned = int(0.2 * BATCH_SIZE)

        assert torch.allclose(pf[n_poisoned:], features[n_poisoned:])
        assert (pl[n_poisoned:] == labels[n_poisoned:]).all()

    def test_original_tensors_not_mutated(self):
        """_poison_batch must clone; original tensors should be unchanged."""
        fn = self._make_node()
        features = torch.ones(BATCH_SIZE, N_FEATURES)
        labels   = torch.ones(BATCH_SIZE, dtype=torch.long)
        orig_feat = features.clone()
        orig_lbl  = labels.clone()

        fn._poison_batch(features, labels)

        assert torch.allclose(features, orig_feat)
        assert (labels == orig_lbl).all()
