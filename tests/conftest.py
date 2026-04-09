import os
import sys
import pytest
import torch
import numpy as np
import networkx as nx
from pathlib import Path

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

# Register custom marks
def pytest_configure(config):
    config.addinivalue_line("markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')")

@pytest.fixture
def device():
    """Fixture for torch device."""
    return torch.device('cpu')  # Use CPU for tests

@pytest.fixture
def sample_graph():
    """Fixture for a sample NetworkX graph."""
    G = nx.DiGraph()
    G.add_edges_from([('A', 'B'), ('B', 'C'), ('A', 'C')])
    return G

@pytest.fixture
def sample_features():
    """Fixture for sample latent features."""
    return torch.randn(100, 5)  # 100 samples, 5 features

@pytest.fixture
def temp_dir(tmp_path):
    """Fixture for temporary directory."""
    return tmp_path

@pytest.fixture
def mock_params():
    """Fixture for mock parameters."""
    return {
        'causal_edge_threshold': 0.1,
        'validator_threshold': 0.5,
        'consensus_momentum': 0.8,
        'l1_sparsity_penalty': 0.01,
        'notears_max_iter': 100,
        'batch_size': 32,
        'num_clients': 10,
        'num_false_nodes': 2,
        'num_rounds': 5,
        'dataset_name': 'asia',
        'total_samples': 1000,
        'consensus_samples': 200
    }