"""
tests/unit/test_tabular_loader.py
-----------------------------------
Unit tests for datasets.tabular_loader.TabularBNDataset.
These tests use the bnlearn ASIA network (smallest BN, 8 nodes, fast sampling).
"""

import pytest
import torch
import numpy as np
from datasets.tabular_loader import TabularBNDataset

# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def asia_dataset():
    """Sample a small ASIA dataset once for the module (shared across tests)."""
    return TabularBNDataset(name="asia", num_samples=200, seed=99)


# ── Construction and metadata ─────────────────────────────────────────────────

def test_dataset_loads(asia_dataset):
    """TabularBNDataset should instantiate without errors."""
    assert asia_dataset is not None


def test_dataset_length(asia_dataset):
    """len() should equal num_samples."""
    assert len(asia_dataset) == 200


def test_target_column_asia(asia_dataset):
    """ASIA target column should be 'lung'."""
    assert asia_dataset.target_col == "lung"


def test_feature_columns_excludes_target(asia_dataset):
    """Feature columns should not include the target column."""
    assert asia_dataset.target_col not in asia_dataset.feature_columns


def test_feature_count_asia(asia_dataset):
    """ASIA has 8 nodes; 7 feature columns after removing 'lung'."""
    assert len(asia_dataset.feature_columns) == 7


def test_get_feature_names_returns_list(asia_dataset):
    """get_feature_names() should return a list of strings."""
    names = asia_dataset.get_feature_names()
    assert isinstance(names, list)
    assert all(isinstance(n, str) for n in names)


# ── __getitem__ ───────────────────────────────────────────────────────────────

def test_getitem_returns_tuple(asia_dataset):
    """__getitem__ should return (Tensor, Tensor)."""
    x, y = asia_dataset[0]
    assert isinstance(x, torch.Tensor)
    assert isinstance(y, torch.Tensor)


def test_feature_tensor_shape(asia_dataset):
    """Feature tensor should have shape (7,) for ASIA."""
    x, _ = asia_dataset[0]
    assert x.shape == (7,)


def test_feature_dtype(asia_dataset):
    """Features should be float32."""
    x, _ = asia_dataset[0]
    assert x.dtype == torch.float32


def test_label_dtype(asia_dataset):
    """Labels should be int64 (long)."""
    _, y = asia_dataset[0]
    assert y.dtype == torch.int64


def test_labels_binary(asia_dataset):
    """All ASIA labels should be 0 or 1 (binary)."""
    for i in range(len(asia_dataset)):
        _, y = asia_dataset[i]
        assert y.item() in {0, 1}


def test_batch_via_dataloader(asia_dataset):
    """DataLoader should produce correct batch shapes."""
    from torch.utils.data import DataLoader
    loader = DataLoader(asia_dataset, batch_size=32, shuffle=False)
    xs, ys = next(iter(loader))
    assert xs.shape == (32, 7)
    assert ys.shape == (32,)
