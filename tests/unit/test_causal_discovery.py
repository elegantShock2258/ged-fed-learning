"""
tests/unit/test_causal_discovery.py
-------------------------------------
Unit tests for client.causal_discovery.CognitiveModule (NOTEARS).
"""

import pytest
import torch
import numpy as np
import networkx as nx
from client.causal_discovery import CognitiveModule


N_FEATURES = 6   # use 6 nodes to keep NOTEARS fast in tests
FEATURE_NAMES = [f"f{i}" for i in range(N_FEATURES)]


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def cog():
    """CognitiveModule with fast settings for unit tests."""
    return CognitiveModule(
        feature_names=FEATURE_NAMES,
        threshold=0.1,
        l1_penalty=0.01,
        lr=0.05,
        max_iter=20,   # very fast: only 20 iterations
    )


# ── Construction ──────────────────────────────────────────────────────────────

def test_cognitive_module_instantiates(cog):
    assert cog is not None
    assert cog.feature_names == FEATURE_NAMES


def test_default_feature_names():
    """If no feature_names given, defaults are generated on first call."""
    cm = CognitiveModule(threshold=0.1, max_iter=5)
    x = torch.zeros(10, 4)
    G = cm.extract_causal_graph(x)
    assert len(G.nodes()) == 4


# ── Output type and structure ─────────────────────────────────────────────────

def test_extract_returns_digraph(cog):
    """extract_causal_graph must return a networkx DiGraph."""
    x = torch.randn(30, N_FEATURES)
    G = cog.extract_causal_graph(x)
    assert isinstance(G, nx.DiGraph)


def test_output_has_correct_nodes(cog):
    """All feature names should appear as nodes in the returned graph."""
    x = torch.randn(30, N_FEATURES)
    G = cog.extract_causal_graph(x)
    for name in FEATURE_NAMES:
        assert name in G.nodes()


def test_no_self_loops(cog):
    """NOTEARS diagonal is (ideally) zeroed; no self-loops should appear."""
    x = torch.randn(30, N_FEATURES)
    G = cog.extract_causal_graph(x)
    for u, v in G.edges():
        assert u != v, f"Self-loop detected at {u}"


# ── Zero-variance input → empty / sparse graph ────────────────────────────────

def test_zero_input_produces_empty_or_sparse_graph(cog):
    """A constant input has zero variance; NOTEARS should produce few/no edges."""
    x = torch.zeros(30, N_FEATURES)
    G = cog.extract_causal_graph(x)
    # With threshold=0.1, constant input should yield 0 or very few edges
    assert G.number_of_edges() <= 2, (
        f"Expected sparse graph on zero input, got {G.number_of_edges()} edges"
    )


# ── Graceful error handling ───────────────────────────────────────────────────

def test_returns_graph_on_tiny_sample(cog):
    """Should not crash even on 1-row input."""
    x = torch.randn(1, N_FEATURES)
    G = cog.extract_causal_graph(x)
    assert isinstance(G, nx.DiGraph)
