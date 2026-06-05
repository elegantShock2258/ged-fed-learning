"""
tests/unit/test_causal_discovery.py
-------------------------------------
Unit tests for client.causal_discovery.CognitiveModule (Trajectory Extractor).
"""

import pytest
import torch
import numpy as np
import networkx as nx
from client.causal_discovery import CognitiveModule


N_FEATURES = 10   # 10 tools
FEATURE_NAMES = [f"{i}" for i in range(N_FEATURES)]


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def cog():
    """CognitiveModule with fast settings for unit tests."""
    return CognitiveModule(
        num_tools=N_FEATURES,
        threshold=0.1,
    )


# ── Construction ──────────────────────────────────────────────────────────────

def test_cognitive_module_instantiates(cog):
    assert cog is not None
    assert cog.num_tools == N_FEATURES


def test_default_feature_names():
    """If no feature_names given, defaults are generated on first call."""
    cm = CognitiveModule(num_tools=4, threshold=0.1)
    # trajectories = list of lists
    x = [[0, 1, 2, 3], [0, 1, 2, 3]]
    G_str = cm.extract_causal_graph(x)
    assert isinstance(G_str, str)


# ── Output structure ─────────────────────────────────────────────────

def test_no_self_loops(cog):
    """Diagonal is zeroed; no self-loops should appear."""
    x = [[0, 1, 2, 0, 1]]
    G_str = cog.extract_causal_graph(x)
    import json
    edges = json.loads(G_str)
    for u, v in edges:
        assert u != v, f"Self-loop detected at {u}"


# ── Zero-variance input → empty / sparse graph ────────────────────────────────

def test_zero_input_produces_empty_or_sparse_graph(cog):
    """A constant input has zero variance; should produce few/no edges."""
    x = [[0, 0, 0, 0, 0]]
    G_str = cog.extract_causal_graph(x)
    import json
    edges = json.loads(G_str)
    assert len(edges) <= 1, (
        f"Expected sparse graph on zero input, got {len(edges)} edges"
    )

# ── Graceful error handling ───────────────────────────────────────────────────

def test_returns_graph_on_tiny_sample(cog):
    """Should not crash even on 1-row input."""
    x = [[1, 2]]
    G_str = cog.extract_causal_graph(x)
    assert isinstance(G_str, str)

# ── Feature Attention Integration ──────────────────────────────────────────────

def test_attention_augmented_graph(cog):
    """Test that providing attention maps adds structured edges."""
    # 2 trajectories, attention maps provided
    x = [[0, 1]]
    
    # Mocking attention matrices (12x12 matrix for each step)
    att1 = np.zeros((12, 12))
    att1[0, 1] = 0.9 # strong attention from token 0 to token 1
    
    attention_maps = [[att1]]
    
    G_str = cog.extract_causal_graph(x, attention_maps=attention_maps)
    import json
    edges = json.loads(G_str)
    
    # Check that there is an edge due to trajectory (u=0, v=1)
    assert [0, 1] in edges
    # Check that there is an edge due to attention mapping
    # N_FEATURES is 10, so attention matrix [0, 1] goes to [10+0, 10+1] = [10, 11]
    assert [10, 11] in edges

