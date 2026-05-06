"""
Minimal smoke tests for simulation imports and output scaffolding.
"""

import os
from pathlib import Path


def test_federated_sim_module_import():
    import experiments.run_por_sim as federated_sim
    assert hasattr(federated_sim, 'prepare_dataset')


def test_baseline_module_import():
    import experiments.run_baseline_sim as baseline_fedavg_sim
    assert hasattr(baseline_fedavg_sim, 'prepare_dataset')


def test_simulation_output_file_structure(tmp_path):
    output_dir = tmp_path / 'saved_models' / 'asia'
    output_dir.mkdir(parents=True, exist_ok=True)

    mock_files = [
        'simulation_logs.json',
        'ged_scores.json',
        'consensus_graph.gpickle',
        'global_model.pt'
    ]

    for file in mock_files:
        (output_dir / file).write_text('')

    for file in mock_files:
        assert (output_dir / file).exists()
