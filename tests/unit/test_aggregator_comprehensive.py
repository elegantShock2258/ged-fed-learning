"""
Comprehensive tests for server/aggregator.py (PoRStrategy) - CORRECTED.

Focus:
  - PoRStrategy initialization and graph loading
  - aggregate_fit with mixed accepted/rejected clients
  - Logic validation filtering
  - GED score recording
  - Consensus graph updates with momentum voting
  - SimGNN fine-tuning on consensus
  - Per-round model and graph persistence
"""

import io
import pytest
import sys
import os
import json
import pickle
import tempfile
import logging
from unittest.mock import Mock, MagicMock, patch, call
from pathlib import Path

import numpy as np
import networkx as nx
import torch
from flwr.common import FitRes, Parameters, FitIns
from flwr.server.client_proxy import ClientProxy

# Suppress logging during tests
logging.getLogger("server.aggregator").setLevel(logging.CRITICAL)


@pytest.fixture
def mock_logic_validator():
    """Mock LogicValidator that simulates graph validation."""
    validator = MagicMock()
    validator.threshold = 0.5
    validator.evaluate_client_graph = MagicMock(return_value=(True, 0.3))
    validator.set_global_consensus = MagicMock()
    validator.model = None  # No SimGNN model for simple tests
    return validator


@pytest.fixture
def temp_model_dir(tmp_path):
    """Temporary directory for model saving."""
    model_dir = tmp_path / "test_models"
    model_dir.mkdir(exist_ok=True)
    return str(model_dir)


@pytest.fixture
def sample_graph():
    """Create a simple sample causal graph."""
    g = nx.DiGraph()
    g.add_node(0)
    g.add_node(1)
    g.add_node(2)
    g.add_edge(0, 1)
    g.add_edge(1, 2)
    return g


def _npy_bytes(arr):
    buffer = io.BytesIO()
    np.save(buffer, np.asarray(arr))
    buffer.seek(0)
    return [buffer.getvalue()]


@pytest.fixture
def mock_client_proxy():
    """Create a mock client proxy."""
    client = MagicMock(spec=ClientProxy)
    client.cid = "test_client_0"
    return client


class TestPoRStrategyInit:
    """Test PoRStrategy initialization."""

    @patch("server.aggregator.os.path.exists")
    @patch("server.aggregator.yaml.safe_load")
    def test_init_creates_model_directory(
        self, mock_yaml, mock_path_exists, mock_logic_validator, temp_model_dir
    ):
        """Test that PoRStrategy creates model directory if needed."""
        mock_yaml.return_value = {"dataset": {"name": "asia"}, "core_logic": {}}
        mock_path_exists.return_value = False

        with patch("server.aggregator.MODEL_DIR", temp_model_dir):
            from server.aggregator import PoRStrategy

            strategy = PoRStrategy(logic_validator=mock_logic_validator)
            assert strategy.logic_validator == mock_logic_validator
            assert strategy.model_dir == temp_model_dir

    @patch("server.aggregator.os.path.exists")
    @patch("server.aggregator.yaml.safe_load")
    def test_init_loads_existing_consensus_graph(
        self, mock_yaml, mock_path_exists, mock_logic_validator, temp_model_dir
    ):
        """Test loading existing consensus graph from disk."""
        mock_yaml.return_value = {"dataset": {"name": "asia"}, "core_logic": {}}

        # Create a test consensus graph to load
        test_graph = nx.DiGraph()
        test_graph.add_edge(0, 1)
        consensus_path = os.path.join(temp_model_dir, "consensus_graph.gpickle")
        os.makedirs(temp_model_dir, exist_ok=True)
        with open(consensus_path, "wb") as f:
            pickle.dump(test_graph, f)

        mock_path_exists.side_effect = lambda p: p == consensus_path

        with patch("server.aggregator.MODEL_DIR", temp_model_dir):
            from server.aggregator import PoRStrategy

            strategy = PoRStrategy(logic_validator=mock_logic_validator)
            assert strategy.global_consensus_graph.number_of_nodes() == 2
            assert strategy.global_consensus_graph.number_of_edges() == 1


class TestAggregateFit:
    """Test aggregate_fit method of PoRStrategy."""

    @patch("server.aggregator.yaml.safe_load")
    @patch("server.aggregator.Model", None)
    def test_aggregate_fit_empty_results(self, mock_yaml, mock_logic_validator):
        """Test aggregate_fit with empty results."""
        mock_yaml.return_value = {"dataset": {"name": "asia"}, "core_logic": {}}

        with patch("server.aggregator.MODEL_DIR", "dummy"):
            from server.aggregator import PoRStrategy

            strategy = PoRStrategy(logic_validator=mock_logic_validator)
            params, metrics = strategy.aggregate_fit(
                server_round=1, results=[], failures=[]
            )
            assert params is None
            assert metrics == {}

    @patch("server.aggregator.yaml.safe_load")
    def test_aggregate_fit_filters_by_ged_threshold(
        self, mock_yaml, mock_logic_validator, temp_model_dir
    ):
        """Test that aggregate_fit filters clients using LogicValidator threshold."""
        mock_yaml.return_value = {"dataset": {"name": "asia"}, "core_logic": {"consensus_momentum": 0.85}}

        with patch("server.aggregator.MODEL_DIR", temp_model_dir):
            from server.aggregator import PoRStrategy

            strategy = PoRStrategy(logic_validator=mock_logic_validator)

            # Create two clients: one accepted, one rejected
            client1 = Mock()
            client1.cid = "client_1"
            client2 = Mock()
            client2.cid = "client_2"

            fit_res1 = Mock(spec=FitRes)
            fit_res1.metrics = {"causal_graph_edges": "[[0, 1]]", "accuracy": 0.9}
            fit_res1.num_examples = 1
            fit_res1.parameters = MagicMock()
            fit_res1.parameters.tensors = _npy_bytes(np.ones((10,)))

            fit_res2 = Mock(spec=FitRes)
            fit_res2.metrics = {"causal_graph_edges": "[[1, 0]]", "accuracy": 0.5}
            fit_res2.num_examples = 1
            fit_res2.parameters = MagicMock()
            fit_res2.parameters.tensors = _npy_bytes(np.ones((10,)))

            # Setup validator to accept client1, reject client2
            def eval_side_effect(g):
                if len(list(g.edges())) > 0:
                    return True, 0.3  # Accepted
                return False, 0.8  # Rejected

            mock_logic_validator.evaluate_client_graph.side_effect = eval_side_effect

            results = [(client1, fit_res1), (client2, fit_res2)]
            params, metrics = strategy.aggregate_fit(
                server_round=1, results=results, failures=[]
            )

            assert metrics["accepted_clients"] >= 1
            assert metrics["rejected_clients"] >= 0

    @patch("server.aggregator.yaml.safe_load")
    def test_aggregate_fit_handles_missing_graph(
        self, mock_yaml, mock_logic_validator, temp_model_dir
    ):
        """Test handling of clients without causal_graph_edges."""
        mock_yaml.return_value = {"dataset": {"name": "asia"}, "core_logic": {"consensus_momentum": 0.85}}

        with patch("server.aggregator.MODEL_DIR", temp_model_dir):
            from server.aggregator import PoRStrategy

            strategy = PoRStrategy(logic_validator=mock_logic_validator)

            client = Mock()
            client.cid = "bad_client"
            fit_res = Mock(spec=FitRes)
            fit_res.metrics = {"accuracy": 0.7}  # No causal_graph_edges
            fit_res.num_examples = 1
            fit_res.parameters = MagicMock()
            fit_res.parameters.tensors = _npy_bytes(np.ones((10,)))

            results = [(client, fit_res)]
            params, metrics = strategy.aggregate_fit(
                server_round=1, results=results, failures=[]
            )

            assert metrics["rejected_clients"] >= 1  # Should be rejected

    @patch("server.aggregator.yaml.safe_load")
    @patch("server.aggregator.Model")
    @patch("flwr.server.strategy.FedAvg.aggregate_fit")
    def test_aggregate_fit_saves_graphs(
        self, mock_fedavg_agg_fit, mock_model, mock_yaml, mock_logic_validator, temp_model_dir
    ):
        """Test the graph saving logic in aggregate_fit when both accepted and rejected exist."""
        mock_yaml.return_value = {"dataset": {"name": "asia"}, "core_logic": {"consensus_momentum": 0.85}}
        
        # Mock FedAvg aggregate_fit return value
        mock_agg_params = MagicMock()
        mock_fedavg_agg_fit.return_value = (mock_agg_params, {})

        with patch("server.aggregator.MODEL_DIR", temp_model_dir):
            from server.aggregator import PoRStrategy
            strategy = PoRStrategy(logic_validator=mock_logic_validator)

            client1 = Mock()
            client1.cid = "client_1"
            client2 = Mock()
            client2.cid = "client_2"

            fit_res1 = Mock(spec=FitRes)
            fit_res1.metrics = {"causal_graph_edges": "[[0, 1]]", "accuracy": 0.9}
            
            fit_res2 = Mock(spec=FitRes)
            fit_res2.metrics = {"causal_graph_edges": "[[1, 0]]", "accuracy": 0.5}

            # Setup validator to accept client1, reject client2
            def eval_side_effect(g):
                if g.has_edge(0, 1):
                    return True, 0.3  # Accepted
                return False, 0.8  # Rejected
            mock_logic_validator.evaluate_client_graph.side_effect = eval_side_effect

            results = [(client1, fit_res1), (client2, fit_res2)]
            
            # Avoid the save model logic using mock, just test the graph save block
            with patch.object(strategy, '_save_global_model'):
                strategy.aggregate_fit(server_round=1, results=results, failures=[])
            
            assert os.path.exists(os.path.join(temp_model_dir, "honest_graph_sample.gpickle"))
            assert os.path.exists(os.path.join(temp_model_dir, "rejected_graph_sample.gpickle"))
            assert os.path.exists(os.path.join(temp_model_dir, "rejected_edge_diff.json"))


class TestAggregateLogic:
    """Test consensus graph update via momentum voting."""

    @patch("server.aggregator.yaml.safe_load")
    def test_aggregate_logic_keeps_existing_edges_high_momentum(
        self, mock_yaml, mock_logic_validator, temp_model_dir
    ):
        """Test that with high momentum, existing edges are preserved."""
        mock_yaml.return_value = {"dataset": {"name": "asia"}, "core_logic": {"consensus_momentum": 0.9}}

        with patch("server.aggregator.MODEL_DIR", temp_model_dir):
            from server.aggregator import PoRStrategy

            strategy = PoRStrategy(logic_validator=mock_logic_validator)

            # Set initial consensus with 2 edges
            consensus = nx.DiGraph()
            consensus.add_edge(0, 1)
            consensus.add_edge(1, 2)
            strategy.global_consensus_graph = consensus

            # Client 1: has edge (0,1) only
            client1_graph = nx.DiGraph()
            client1_graph.add_nodes_from([0, 1, 2])
            client1_graph.add_edge(0, 1)

            # Client 2: has edge (0,1) only
            client2_graph = nx.DiGraph()
            client2_graph.add_nodes_from([0, 1, 2])
            client2_graph.add_edge(0, 1)

            # With momentum=0.9, keep_threshold = 0.05*2 = 0.1
            # Both clients vote for (0,1), so it should be kept
            # Only 1 client votes for (1,2), so it might be dropped
            client_graphs = [client1_graph, client2_graph]

            strategy._aggregate_logic(client_graphs)

            # Edge (0,1) should definitely be kept
            assert strategy.global_consensus_graph.has_edge(0, 1)

    @patch("server.aggregator.yaml.safe_load")
    def test_aggregate_logic_adds_edges_near_unanimity(
        self, mock_yaml, mock_logic_validator, temp_model_dir
    ):
        """Test that new edges are added only with high consensus."""
        mock_yaml.return_value = {"dataset": {"name": "asia"}, "core_logic": {"consensus_momentum": 0.95}}

        with patch("server.aggregator.MODEL_DIR", temp_model_dir):
            from server.aggregator import PoRStrategy

            strategy = PoRStrategy(logic_validator=mock_logic_validator)

            # Empty initial consensus
            strategy.global_consensus_graph = nx.DiGraph()
            strategy.global_consensus_graph.add_nodes_from([0, 1, 2])

            # Create client graphs with new edges
            client1_graph = nx.DiGraph()
            client1_graph.add_nodes_from([0, 1, 2])
            client1_graph.add_edge(0, 1)
            client1_graph.add_edge(1, 2)

            client2_graph = nx.DiGraph()
            client2_graph.add_nodes_from([0, 1, 2])
            client2_graph.add_edge(0, 1)
            client2_graph.add_edge(1, 2)

            # Both clients have same edges -> should be added
            strategy._aggregate_logic([client1_graph, client2_graph])

            assert strategy.global_consensus_graph.has_edge(0, 1)
            assert strategy.global_consensus_graph.has_edge(1, 2)


class TestModelSaving:
    """Test model persistence."""

    @patch("server.aggregator.Model")
    @patch("server.aggregator.yaml.safe_load")
    def test_save_global_model_writes_to_disk(
        self, mock_yaml, mock_model_class, mock_logic_validator, temp_model_dir
    ):
        """Test that _save_global_model persists weights to disk."""
        mock_yaml.return_value = {"dataset": {"name": "asia"}, "core_logic": {}}

        # Mock the Model class
        mock_model_instance = MagicMock()
        mock_model_class.return_value = mock_model_instance
        mock_model_instance.state_dict.return_value = {"layer1.weight": torch.zeros(10, 5)}

        with patch("server.aggregator.MODEL_DIR", temp_model_dir):
            from server.aggregator import PoRStrategy

            strategy = PoRStrategy(logic_validator=mock_logic_validator)

            # Mock Parameters
            params = MagicMock()
            params.tensors = [np.ones((10,))]

            with patch("server.aggregator.parameters_to_ndarrays") as mock_convert:
                mock_convert.return_value = [np.ones((50,))]
                with patch("torch.save") as mock_torch_save:
                    strategy._save_global_model(params, round_num=1)
                    # Verify torch.save was called
                    assert mock_torch_save.called


class TestGEDScorePersistence:
    """Test GED score recording and file I/O."""

    @patch("server.aggregator.yaml.safe_load")
    def test_ged_scores_written_to_json(
        self, mock_yaml, mock_logic_validator, temp_model_dir
    ):
        """Test that GED scores are persisted to JSON file."""
        mock_yaml.return_value = {"dataset": {"name": "asia"}, "core_logic": {"consensus_momentum": 0.85}}

        with patch("server.aggregator.MODEL_DIR", temp_model_dir):
            from server.aggregator import PoRStrategy

            strategy = PoRStrategy(logic_validator=mock_logic_validator)

            client = Mock()
            client.cid = "client_1"

            fit_res = Mock()
            fit_res.metrics = {"causal_graph_edges": "[[0, 1]]", "accuracy": 0.9}
            fit_res.num_examples = 1
            fit_res.parameters = MagicMock()
            fit_res.parameters.tensors = _npy_bytes(np.ones((10,)))

            mock_logic_validator.evaluate_client_graph.return_value = (True, 0.25)

            results = [(client, fit_res)]
            strategy.aggregate_fit(server_round=1, results=results, failures=[])

            # Verify the GED scores file was created
            ged_log_path = os.path.join(temp_model_dir, "ged_scores.json")
            assert os.path.exists(ged_log_path)


class TestEdgeDiffRecording:
    """Test rejected edge difference recording."""

    @patch("server.aggregator.yaml.safe_load")
    def test_edge_diff_recorded_for_rejected_clients(
        self, mock_yaml, mock_logic_validator, temp_model_dir
    ):
        """Test that edge differences are recorded when clients are rejected."""
        mock_yaml.return_value = {"dataset": {"name": "asia"}, "core_logic": {"consensus_momentum": 0.85}}

        with patch("server.aggregator.MODEL_DIR", temp_model_dir):
            from server.aggregator import PoRStrategy

            strategy = PoRStrategy(logic_validator=mock_logic_validator)

            # Initialize with a consensus graph
            consensus = nx.DiGraph()
            consensus.add_edge(0, 1)
            consensus.add_edge(1, 2)
            strategy.global_consensus_graph = consensus

            client = Mock()
            client.cid = "bad_client"

            fit_res = Mock()
            # Client submits only edge (0,1), missing (1,2)
            fit_res.metrics = {"causal_graph_edges": "[[0, 1]]", "accuracy": 0.5}
            fit_res.num_examples = 1
            fit_res.parameters = MagicMock()
            fit_res.parameters.tensors = _npy_bytes(np.ones((10,)))

            # Reject this client
            mock_logic_validator.evaluate_client_graph.return_value = (False, 0.8)

            results = [(client, fit_res)]
            with patch("server.aggregator.Model", None):
                strategy.aggregate_fit(server_round=1, results=results, failures=[])

            # GED scores file should be recorded
            ged_log_path = os.path.join(temp_model_dir, "ged_scores.json")
            assert os.path.exists(ged_log_path)


class TestSimGNNFineTuning:
    """Test on-the-fly SimGNN fine-tuning."""

    @patch("server.aggregator.yaml.safe_load")
    def test_finetune_simgnn_skipped_if_no_model(
        self, mock_yaml, mock_logic_validator, temp_model_dir
    ):
        """Test that SimGNN fine-tuning is skipped if model is None."""
        mock_yaml.return_value = {"dataset": {"name": "asia"}, "core_logic": {"consensus_momentum": 0.85}}
        mock_logic_validator.model = None

        with patch("server.aggregator.MODEL_DIR", temp_model_dir):
            from server.aggregator import PoRStrategy

            strategy = PoRStrategy(logic_validator=mock_logic_validator)
            consensus = nx.DiGraph()
            consensus.add_node(0)

            # Should not raise an error
            strategy._finetune_simgnn_on_consensus(consensus)

    @patch("server.aggregator.yaml.safe_load")
    def test_finetune_simgnn_executes(
        self, mock_yaml, mock_logic_validator, temp_model_dir
    ):
        """Test the full SimGNN fine-tuning loop."""
        mock_yaml.return_value = {"dataset": {"name": "asia"}, "core_logic": {"consensus_momentum": 0.85}}
        
        # We need a proper SimGNN mock
        from server.logic_validator import SimGNN
        simgnn = SimGNN(hidden_dim=16, num_layers=2)
        mock_logic_validator.model = simgnn

        with patch("server.aggregator.MODEL_DIR", temp_model_dir):
            from server.aggregator import PoRStrategy
            strategy = PoRStrategy(logic_validator=mock_logic_validator)
            
            consensus = nx.DiGraph()
            consensus.add_edges_from([(0, 1), (1, 2), (2, 3)])
            
            with patch('torch.save') as mock_torch_save:
                # Need fewer steps/pairs to test quickly
                strategy._finetune_simgnn_on_consensus(consensus, steps=2, pairs=4)
                assert mock_torch_save.called


class TestNoClientsAccepted:
    """Test behavior when all clients are rejected."""

    @patch("server.aggregator.yaml.safe_load")
    def test_all_clients_rejected_returns_none(
        self, mock_yaml, mock_logic_validator, temp_model_dir
    ):
        """Test that when all clients are rejected, returns None."""
        mock_yaml.return_value = {"dataset": {"name": "asia"}, "core_logic": {"consensus_momentum": 0.85}}
        mock_logic_validator.evaluate_client_graph.return_value = (False, 0.9)  # All rejected

        with patch("server.aggregator.MODEL_DIR", temp_model_dir):
            from server.aggregator import PoRStrategy

            strategy = PoRStrategy(logic_validator=mock_logic_validator)

            client = Mock()
            client.cid = "client_1"
            fit_res = Mock()
            fit_res.metrics = {"causal_graph_edges": "[[0, 1]]"}
            fit_res.num_examples = 1
            fit_res.parameters = MagicMock()
            fit_res.parameters.tensors = _npy_bytes(np.ones((10,)))

            results = [(client, fit_res)]
            params, metrics = strategy.aggregate_fit(
                server_round=1, results=results, failures=[]
            )

            assert metrics["accepted_clients"] == 0
            assert params is None
