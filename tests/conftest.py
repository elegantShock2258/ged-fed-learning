"""
conftest.py — Shared pytest fixtures for the PoR FL test suite.

Fixtures defined here are available to all tests without importing.
"""

import pytest
import torch
import networkx as nx
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from unittest.mock import MagicMock


# ── Device ────────────────────────────────────────────────────────────────────
@pytest.fixture(scope="session")
def device():
    """CPU device — all tests run CPU-only for CI reproducibility."""
    return torch.device("cpu")


# ── Tiny synthetic dataset ────────────────────────────────────────────────────
N_SAMPLES  = 64   # rows
N_FEATURES = 8    # ASIA has 8 feature columns (7 features + 1 target dropped)
N_CLASSES  = 2


@pytest.fixture(scope="session")
def tiny_features(device):
    """Tiny random float32 feature tensor (64, 8)."""
    torch.manual_seed(42)
    return torch.randn(N_SAMPLES, N_FEATURES)


@pytest.fixture(scope="session")
def tiny_labels():
    """Tiny random integer label tensor (64,) with values in {0, 1}."""
    torch.manual_seed(42)
    return torch.randint(0, N_CLASSES, (N_SAMPLES,))


@pytest.fixture(scope="session")
def tiny_loader(tiny_features, tiny_labels):
    """DataLoader wrapping the tiny synthetic dataset (batch_size=16)."""
    ds = TensorDataset(tiny_features, tiny_labels)
    return DataLoader(ds, batch_size=16, shuffle=False)


# ── Tiny MLP ──────────────────────────────────────────────────────────────────
@pytest.fixture
def model(device):
    """Fresh Model instance (8 inputs, 2 classes) on CPU."""
    from client.models import Model
    return Model(in_features=N_FEATURES, num_classes=N_CLASSES).to(device)


# ── Tiny causal graphs ────────────────────────────────────────────────────────
NODE_NAMES = [f"feat_{i}" for i in range(N_FEATURES)]


@pytest.fixture
def small_graph():
    """A simple 8-node DiGraph with 4 edges — used as a fake consensus."""
    G = nx.DiGraph()
    G.add_nodes_from(NODE_NAMES)
    G.add_edges_from([("feat_0", "feat_1"), ("feat_1", "feat_2"),
                      ("feat_0", "feat_3"), ("feat_4", "feat_5")])
    return G


@pytest.fixture
def empty_graph():
    """An 8-node DiGraph with zero edges."""
    G = nx.DiGraph()
    G.add_nodes_from(NODE_NAMES)
    return G


@pytest.fixture
def perturbed_graph(small_graph):
    """A copy of small_graph with 3 edges added and 1 removed — high GED."""
    G = small_graph.copy()
    G.remove_edge("feat_0", "feat_1")
    G.add_edge("feat_2", "feat_6")
    G.add_edge("feat_5", "feat_7")
    G.add_edge("feat_3", "feat_7")
    return G


# ── LogicValidator ────────────────────────────────────────────────────────────
@pytest.fixture
def logic_validator(small_graph):
    """LogicValidator with random SimGNN weights and small_graph as consensus."""
    from server.logic_validator import LogicValidator
    lv = LogicValidator(model_path=None, threshold=0.45)
    lv.set_global_consensus(small_graph)
    return lv


# ── Params YAML path ──────────────────────────────────────────────────────────
@pytest.fixture(scope="session")
def params_path(tmp_path_factory):
    """Write a minimal params.yaml to a temp dir and return its path."""
    import yaml, os
    params = {
        "core_logic": {
            "causal_edge_threshold": 0.05,
            "validator_threshold": 0.45,
            "consensus_momentum": 0.85,
            "simgnn_epochs": 2,
            "simgnn_batch_size": 4,
            "simgnn_lr": 0.001,
            "notears_lr": 0.02,
            "notears_max_iter": 5,
            "l1_sparsity_penalty": 0.0001,
        },
        "simulation": {
            "num_clients": 4,
            "num_false_nodes": 1,
            "num_rounds": 1,
            "local_epochs": 1,
            "batch_size": 16,
            "client_lr": 1e-4,
            "ray_cpus_per_actor": 1,
        },
        "dataset": {"name": "asia", "total_samples": 200, "seed": 42},
        "server": {"consensus_samples": 50, "batch_size": 16},
        "hardware": {"device": "cpu"},
    }
    p = tmp_path_factory.mktemp("cfg") / "params.yaml"
    with open(p, "w") as f:
        yaml.dump(params, f)
    return str(p)
