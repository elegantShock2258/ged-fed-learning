"""
tests/functional/test_simgnn_training.py
------------------------------------------
Functional/smoke test for the SimGNN pre-training pipeline.
Runs a minimal 1-epoch training loop to verify the pipeline executes end-to-end.
"""

import pytest
import torch
import networkx as nx
import os
import sys
from unittest.mock import patch, mock_open
import yaml


def make_test_params(tmp_path):
    """Write a minimal params.yaml to tmp_path and return its path."""
    params = {
        "core_logic": {
            "simgnn_epochs": 2,
            "simgnn_batch_size": 4,
            "simgnn_lr": 0.001,
            "causal_edge_threshold": 0.05,
            "l1_sparsity_penalty": 0.01,
            "notears_lr": 0.05,
            "notears_max_iter": 5,
            "validator_threshold": 0.45,
            "consensus_momentum": 0.85,
        },
        "dataset": {"name": "test_ds", "total_samples": 50, "seed": 42},
        "server": {"consensus_samples": 10, "batch_size": 8},
        "hardware": {"device": "cpu"},
    }
    p = tmp_path / "params.yaml"
    p.write_text(yaml.dump(params))
    return str(p)


@pytest.fixture
def consensus_graph():
    """4-node, 3-edge consensus graph for smoke test."""
    G = nx.DiGraph()
    G.add_nodes_from([f"Feature_{i}" for i in range(4)])
    G.add_edges_from([("Feature_0", "Feature_1"),
                      ("Feature_1", "Feature_2"),
                      ("Feature_0", "Feature_3")])
    return G


def test_train_simgnn_smoke(tmp_path, consensus_graph):
    """
    Smoke test: train_simgnn() runs to completion and produces a .pt file.
    Uses minimal epochs=2 and batch_size=4 to keep the test fast.
    """
    import pickle

    params_path = make_test_params(tmp_path)
    model_dir = tmp_path / "saved_models" / "test_ds"
    model_dir.mkdir(parents=True)

    # Save the consensus graph so train_simgnn can load it
    consensus_path = model_dir / "consensus_graph.gpickle"
    with open(consensus_path, "wb") as f:
        pickle.dump(consensus_graph, f)

    save_path = str(model_dir / "simgnn_test.pt")

    # Patch open() for params.yaml in train_simgnn
    import importlib
    with patch("builtins.open", side_effect=lambda p, *a, **k: open(str(params_path), *a, **k) if "params.yaml" in str(p) else open(p, *a, **k)):
        from server.train_simgnn import train_simgnn
        train_simgnn(save_path=save_path)

    assert os.path.exists(save_path), "SimGNN weights file was not created"
    state = torch.load(save_path, map_location="cpu")
    assert len(state) > 0, "Saved state_dict is empty"
