"""
Minimal smoke tests for simulation module imports.
"""

import pytest
import tempfile
import os
import yaml
import shutil
from pathlib import Path


@pytest.fixture
def minimal_config():
    return {
        'dataset': {'name': 'asia', 'total_samples': 500, 'seed': 42},
        'simulation': {'num_clients': 3, 'num_false_nodes': 1, 'num_rounds': 1, 'local_epochs': 1, 'batch_size': 16, 'ray_cpus_per_actor': 1},
        'core_logic': {'causal_edge_threshold': 0.3, 'l1_sparsity_penalty': 0.001, 'notears_max_iter': 20, 'notears_lr': 0.01, 'simgnn_lr': 0.001, 'simgnn_epochs': 5, 'simgnn_batch_size': 8, 'simgnn_diversity_prob': 0.8, 'validator_threshold': 0.5, 'consensus_momentum': 0.8},
        'server': {'consensus_samples': 50, 'batch_size': 16},
        'hardware': {'device': 'cpu'}
    }


@pytest.fixture
def temp_workspace():
    temp_dir = tempfile.mkdtemp()
    original_cwd = os.getcwd()
    os.chdir(temp_dir)
    yield temp_dir
    os.chdir(original_cwd)
    shutil.rmtree(temp_dir)


def test_simulation_module_imports():
    import federated_sim
    import baseline_fedavg_sim
    assert hasattr(federated_sim, 'prepare_dataset')
    assert hasattr(baseline_fedavg_sim, 'prepare_dataset')


def test_config_file_writes_and_reads(temp_workspace, minimal_config):
    with open('params.yaml', 'w') as f:
        yaml.dump(minimal_config, f)

    with open('params.yaml', 'r') as f:
        config = yaml.safe_load(f)

    assert config['dataset']['name'] == 'asia'
