"""
tests/unit/test_models.py
--------------------------
Unit tests for client.models.Model (MLP classifier).
"""

import pytest
import torch
from client.models import Model


N_FEATURES = 8
N_CLASSES  = 2
BATCH      = 16


# ── Construction ──────────────────────────────────────────────────────────────

def test_model_instantiates():
    """Model should construct without errors."""
    m = Model(in_features=N_FEATURES, num_classes=N_CLASSES)
    assert m is not None


def test_model_parameter_count():
    """Model should have trainable parameters (not empty)."""
    m = Model(in_features=N_FEATURES, num_classes=N_CLASSES)
    total = sum(p.numel() for p in m.parameters())
    assert total > 0


# ── Forward pass ─────────────────────────────────────────────────────────────

def test_forward_returns_tuple():
    """forward() must return a (logits, features) tuple."""
    m = Model(in_features=N_FEATURES, num_classes=N_CLASSES)
    m.eval()
    x = torch.randn(BATCH, N_FEATURES)
    out = m(x)
    assert isinstance(out, tuple) and len(out) == 2


def test_logits_shape():
    """Logits tensor should have shape (batch, num_classes)."""
    m = Model(in_features=N_FEATURES, num_classes=N_CLASSES)
    m.eval()
    x = torch.randn(BATCH, N_FEATURES)
    logits, _ = m(x)
    assert logits.shape == (BATCH, N_CLASSES)


def test_features_passthrough():
    """The second return value should be identical to the input x."""
    m = Model(in_features=N_FEATURES, num_classes=N_CLASSES)
    m.eval()
    x = torch.randn(BATCH, N_FEATURES)
    _, features = m(x)
    assert torch.allclose(features, x)


def test_single_sample():
    """Model should work on a single-sample batch (BatchNorm edge case)."""
    m = Model(in_features=N_FEATURES, num_classes=N_CLASSES)
    m.eval()  # BatchNorm uses running stats in eval mode
    x = torch.randn(1, N_FEATURES)
    logits, features = m(x)
    assert logits.shape == (1, N_CLASSES)
    assert features.shape == (1, N_FEATURES)


def test_different_feature_sizes():
    """Model should accept different in_features sizes (e.g. 37 for ALARM)."""
    m = Model(in_features=37, num_classes=2)
    m.eval()
    x = torch.randn(4, 37)
    logits, _ = m(x)
    assert logits.shape == (4, 2)


def test_forward_no_nan():
    """Logits should not contain NaN on normal random input."""
    m = Model(in_features=N_FEATURES, num_classes=N_CLASSES)
    m.eval()
    x = torch.randn(BATCH, N_FEATURES)
    logits, _ = m(x)
    assert not torch.isnan(logits).any()
