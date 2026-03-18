"""
tests/functional/test_simgnn_training.py
------------------------------------------
Functional/smoke test for the SimGNN pre-training pipeline.
Runs a minimal 2-epoch training loop to verify the pipeline executes end-to-end.

Strategy: Write a real params.yaml + consensus graph to a tmp directory,
change cwd to that directory during the test (monkeypatch.chdir),
then import and call train_simgnn().  No builtins.open patching needed.
"""

import pytest
import torch
import networkx as nx
import os
import pickle
import yaml


@pytest.fixture
def sim_env(tmp_path, monkeypatch):
    """
    Set up a minimal simulation environment in a temp directory:
      - params.yaml with fast settings (2 epochs, batch_size=4)
      - saved_models/test_ds/consensus_graph.gpickle  (4-node graph)
    Changes CWD to tmp_path so that open("params.yaml") works naturally.
    Returns the expected save_path for the SimGNN weights file.
    """
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

    # Write params.yaml
    params_file = tmp_path / "params.yaml"
    params_file.write_text(yaml.dump(params))

    # Write consensus graph
    model_dir = tmp_path / "saved_models" / "test_ds"
    model_dir.mkdir(parents=True)

    G = nx.DiGraph()
    G.add_nodes_from([f"Feature_{i}" for i in range(4)])
    G.add_edges_from([
        ("Feature_0", "Feature_1"),
        ("Feature_1", "Feature_2"),
        ("Feature_0", "Feature_3"),
    ])
    with open(model_dir / "consensus_graph.gpickle", "wb") as f:
        pickle.dump(G, f)

    # Switch cwd so open("params.yaml") inside train_simgnn finds it
    monkeypatch.chdir(tmp_path)

    save_path = str(model_dir / "simgnn_test.pt")
    return save_path


def test_train_simgnn_smoke(sim_env):
    """
    Smoke test: train_simgnn() runs 2 epochs and produces a non-empty .pt file.
    """
    from server.train_simgnn import train_simgnn

    # Clear module-level config cache so it re-reads from the new cwd's params.yaml
    import importlib
    import server.train_simgnn as tsm
    import yaml
    tsm.config = yaml.safe_load(open("params.yaml"))

    train_simgnn(save_path=sim_env)

    assert os.path.exists(sim_env), f"SimGNN weights file not created at {sim_env}"
    state = torch.load(sim_env, map_location="cpu")
    assert len(state) > 0, "Saved state_dict is empty"
