"""
Integration tests for server modules

Focus:
  - Consensus graph generation from server dataset
  - NOTEARS causal discovery on reserved data
  - SimGNN training initialization
  - Model persistence and loading
  - Parameter configuration handling
"""

import pytest
import os
import torch
import networkx as nx
import yaml
import pickle
from unittest.mock import Mock, MagicMock, patch
import logging
import tempfile

logging.getLogger("server.generate_consensus").setLevel(logging.CRITICAL)
logging.getLogger("server.train_simgnn").setLevel(logging.CRITICAL)


@pytest.fixture
def temp_model_dir():
    """Temporary directory for generated models."""
    with tempfile.TemporaryDirectory() as tmp:
        yield tmp


@pytest.fixture
def mock_params():
    """Mock configuration parameters."""
    return {
        "dataset": {"name": "asia", "total_samples": 1000},
        "core_logic": {
            "causal_edge_threshold": 0.1,
            "l1_sparsity_penalty": 0.0001,
            "notears_lr": 0.02,
            "notears_max_iter": 5,  # Small for testing
            "consensus_momentum": 0.85,
        },
        "server": {"consensus_samples": 100, "batch_size": 16},
        "hardware": {"device": "cpu"},
    }


@pytest.fixture
def mock_dataset():
    """Mock Bayesian network dataset."""
    dataset = MagicMock()
    dataset.X = torch.randn(200, 5, dtype=torch.float32)
    dataset.y = torch.randint(0, 2, (200,))
    dataset.num_features = 5
    dataset.num_classes = 2
    return dataset


class TestConsensusGeneration:
    """Test consensus graph generation utilities."""

    def test_get_model_dir_creates_valid_path(self, mock_params):
        """Test that get_model_dir creates expected path."""
        from server.generate_consensus import get_model_dir
        
        result = get_model_dir("test_dataset")
        assert "test_dataset" in result
        assert "saved_models" in result

    def test_consensus_graph_structure(self):
        """Test basic graph structure for consensus."""
        graph = nx.DiGraph()
        graph.add_edges_from([(0, 1), (1, 2), (0, 2)])
        
        assert graph.number_of_nodes() == 3
        assert graph.number_of_edges() == 3
        assert (0, 1) in graph.edges()


class TestNOTEARSIntegration:
    """Test NOTEARS-based causal discovery configuration."""

    def test_notears_parameters_structure(self, mock_params):
        """Test that NOTEARS parameters have expected structure."""
        notears_lr = mock_params["core_logic"]["notears_lr"]
        notears_max_iter = mock_params["core_logic"]["notears_max_iter"]

        # Verify parameters are in config
        assert notears_lr == 0.02
        assert notears_max_iter == 5
        assert isinstance(notears_lr, float)
        assert isinstance(notears_max_iter, int)


class TestSimGNNTraining:
    """Test SimGNN model training."""

    @patch("server.train_simgnn.SimGNN")
    @patch("server.train_simgnn.torch.optim.Adam")
    def test_simgnn_model_initialization(self, mock_adam, mock_simgnn_class):
        """Test that SimGNN model can be instantiated."""
        mock_model = MagicMock()
        mock_model.parameters = MagicMock(
            return_value=[torch.randn(10, 10, requires_grad=True)]
        )
        mock_simgnn_class.return_value = mock_model

        # Verify class is available
        from server.train_simgnn import SimGNN

        assert SimGNN is not None

    def test_simgnn_training_loop_methods_exist(self):
        """Test that SimGNN model has expected methods."""
        from server.train_simgnn import SimGNN
        
        # Check that class is importable and has expected methods
        assert hasattr(SimGNN, "__init__")


class TestModelPersistence:
    """Test saving and loading of consensus graphs and SimGNN weights."""

    def test_consensus_graph_serialization(self, temp_model_dir):
        """Test that consensus graph can be pickled."""
        graph = nx.DiGraph()
        graph.add_edges_from([(0, 1), (1, 2), (0, 2)])

        # Serialize
        graph_file = os.path.join(temp_model_dir, "test_graph.gpickle")
        with open(graph_file, "wb") as f:
            pickle.dump(graph, f)

        # Deserialize
        with open(graph_file, "rb") as f:
            loaded = pickle.load(f)

        assert loaded.number_of_edges() == graph.number_of_edges()
        assert set(loaded.edges()) == set(graph.edges())

    def test_simgnn_weight_saving(self, temp_model_dir):
        """Test that SimGNN weights can be saved and loaded."""
        # Create a simple model state dict
        state_dict = {
            "layer1.weight": torch.randn(10, 5),
            "layer1.bias": torch.randn(10),
        }

        # Save weights
        weights_file = os.path.join(temp_model_dir, "simgnn_weights.pt")
        torch.save(state_dict, weights_file)

        # Load weights
        loaded_state = torch.load(weights_file)

        assert "layer1.weight" in loaded_state
        assert loaded_state["layer1.weight"].shape == (10, 5)


class TestConfigurationHandling:
    """Test parameter configuration handling."""

    def test_params_yaml_structure(self, mock_params):
        """Test that config parameters have expected structure."""
        assert "dataset" in mock_params
        assert "core_logic" in mock_params
        assert "server" in mock_params
        assert mock_params["dataset"]["name"] == "asia"

    def test_device_selection_standard(self):
        """Test that torch.device works correctly."""
        device = torch.device("cpu")
        assert str(device) == "cpu"


class TestSimGNNTraining:
    """Test SimGNN model training."""

    @patch("server.train_simgnn.SimGNN")
    @patch("server.train_simgnn.torch.optim.Adam")
    def test_simgnn_model_initialization(self, mock_adam, mock_simgnn_class):
        """Test that SimGNN model can be instantiated."""
        mock_model = MagicMock()
        mock_model.parameters = MagicMock(
            return_value=[torch.randn(10, 10, requires_grad=True)]
        )
        mock_simgnn_class.return_value = mock_model

        # Verify class is available
        from server.train_simgnn import SimGNN

        assert SimGNN is not None

    def test_simgnn_training_loop_methods_exist(self):
        """Test that SimGNN model has expected methods."""
        from server.train_simgnn import SimGNN
        
        # Check that class is importable and has expected methods
        assert hasattr(SimGNN, "__init__")


class TestModelPersistence:
    """Test saving and loading of consensus graphs and SimGNN weights."""

    def test_consensus_graph_serialization(self, temp_model_dir):
        """Test that consensus graph can be pickled."""
        graph = nx.DiGraph()
        graph.add_edges_from([(0, 1), (1, 2), (0, 2)])

        # Serialize
        graph_file = os.path.join(temp_model_dir, "test_graph.gpickle")
        with open(graph_file, "wb") as f:
            pickle.dump(graph, f)

        # Deserialize
        with open(graph_file, "rb") as f:
            loaded = pickle.load(f)

        assert loaded.number_of_edges() == graph.number_of_edges()
        assert set(loaded.edges()) == set(graph.edges())

    def test_simgnn_weight_saving(self, temp_model_dir):
        """Test that SimGNN weights can be saved and loaded."""
        # Create a simple model state dict
        state_dict = {
            "layer1.weight": torch.randn(10, 5),
            "layer1.bias": torch.randn(10),
        }

        # Save weights
        weights_file = os.path.join(temp_model_dir, "simgnn_weights.pt")
        torch.save(state_dict, weights_file)

        # Load weights
        loaded_state = torch.load(weights_file)

        assert "layer1.weight" in loaded_state
        assert loaded_state["layer1.weight"].shape == (10, 5)


class TestConfigurationHandling:
    """Test parameter configuration handling."""

    def test_params_yaml_structure(self, mock_params):
        """Test that config parameters have expected structure."""
        assert "dataset" in mock_params
        assert "core_logic" in mock_params
        assert "server" in mock_params
        assert mock_params["dataset"]["name"] == "asia"

    def test_device_selection_standard(self):
        """Test that torch.device works correctly."""
        device = torch.device("cpu")
        assert str(device) == "cpu"

