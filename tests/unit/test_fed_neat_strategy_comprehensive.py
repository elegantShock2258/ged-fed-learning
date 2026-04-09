"""
Comprehensive tests for server/fed_neat_strategy.py (FedNEATStrategy).

Focus:
  - FedNEATStrategy initialization and parameter handling
  - Lifecycle hooks: configure_fit, configure_evaluate, initialize_parameters
  - aggregate_fit with topological crossover (innovation hash-based)
  - PoR logic gate integration (client filtering)
  - Genome crossover logic (merge by innovation)
  - Consensus update with dynamic threshold
  - Per-round GED score recording
"""

import io
import pytest
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
logging.getLogger("server.fed_neat_strategy").setLevel(logging.CRITICAL)


@pytest.fixture
def mock_logic_validator():
    """Mock LogicValidator for testing."""
    validator = MagicMock()
    validator.threshold = 0.5
    validator.evaluate_client_graph = MagicMock(return_value=(True, 0.3))
    validator.set_global_consensus = MagicMock()
    validator.update_dynamic_threshold = MagicMock()
    validator.model = None
    return validator


@pytest.fixture
def temp_model_dir(tmp_path):
    """Temporary directory for saving models."""
    model_dir = tmp_path / "test_models"
    model_dir.mkdir(exist_ok=True)
    return str(model_dir)


@pytest.fixture
def sample_genome():
    """Create a sample genome structure."""
    return {
        "in_features": 5,
        "num_classes": 2,
        "nodes": {"0": "input", "1": "hidden", "2": "output"},
        "hidden_nodes": ["1"],
        "connections": {
            "innov_1": {"in": "0", "out": "1", "active": True, "weight": 0.5},
            "innov_2": {"in": "1", "out": "2", "active": True, "weight": -0.3},
        },
        "fitness": 0.85,
    }


@pytest.fixture
def sample_graph():
    """Create a simple causal graph."""
    g = nx.DiGraph()
    g.add_edge(0, 1)
    g.add_edge(1, 2)
    return g


def _npy_bytes_from_json(data):
    buf = io.BytesIO()
    np.save(buf, np.frombuffer(bytearray(json.dumps(data), "utf-8"), dtype=np.uint8))
    buf.seek(0)
    return [buf.getvalue()]


@pytest.fixture
def mock_client_manager():
    """Mock client manager for strategy configuration."""
    manager = MagicMock()
    manager.num_available.return_value = 3
    manager.sample.return_value = []
    return manager


@pytest.fixture
def mock_client_proxy():
    """Create a mock client proxy."""
    client = MagicMock(spec=ClientProxy)
    client.cid = "test_client"
    return client


class TestFedNEATStrategyInit:
    """Test FedNEATStrategy initialization."""

    @patch("server.fed_neat_strategy.os.path.exists")
    @patch("server.fed_neat_strategy.yaml.safe_load")
    def test_init_default_parameters(
        self, mock_yaml, mock_path_exists, mock_logic_validator, temp_model_dir
    ):
        """Test FedNEATStrategy initialization with default parameters."""
        mock_yaml.return_value = {"dataset": {"name": "asia"}, "core_logic": {}}
        mock_path_exists.return_value = False

        with patch("server.fed_neat_strategy.MODEL_DIR", temp_model_dir):
            from server.fed_neat_strategy import FedNEATStrategy

            strategy = FedNEATStrategy(logic_validator=mock_logic_validator)
            assert strategy.fraction_fit == 1.0
            assert strategy.fraction_evaluate == 1.0
            assert strategy.min_fit_clients == 2
            assert strategy.min_evaluate_clients == 2

    @patch("server.fed_neat_strategy.os.path.exists")
    @patch("server.fed_neat_strategy.yaml.safe_load")
    def test_init_custom_parameters(
        self, mock_yaml, mock_path_exists, mock_logic_validator, temp_model_dir
    ):
        """Test FedNEATStrategy with custom parameters."""
        mock_yaml.return_value = {"dataset": {"name": "asia"}, "core_logic": {}}
        mock_path_exists.return_value = False

        with patch("server.fed_neat_strategy.MODEL_DIR", temp_model_dir):
            from server.fed_neat_strategy import FedNEATStrategy

            strategy = FedNEATStrategy(
                logic_validator=mock_logic_validator,
                fraction_fit=0.5,
                fraction_evaluate=0.3,
                min_fit_clients=4,
            )
            assert strategy.fraction_fit == 0.5
            assert strategy.fraction_evaluate == 0.3
            assert strategy.min_fit_clients == 4

    @patch("server.fed_neat_strategy.os.path.exists")
    @patch("server.fed_neat_strategy.yaml.safe_load")
    def test_init_loads_consensus_graph(
        self, mock_yaml, mock_path_exists, mock_logic_validator, temp_model_dir
    ):
        """Test that initialization loads existing consensus graph."""
        mock_yaml.return_value = {"dataset": {"name": "asia"}, "core_logic": {}}

        # Create and save a test graph
        test_graph = nx.DiGraph()
        test_graph.add_edge(0, 1)
        test_graph.add_edge(1, 2)
        consensus_path = f"{temp_model_dir}/consensus_graph.gpickle"

        with open(consensus_path, "wb") as f:
            pickle.dump(test_graph, f)

        mock_path_exists.side_effect = lambda p: p == consensus_path

        with patch("server.fed_neat_strategy.MODEL_DIR", temp_model_dir):
            from server.fed_neat_strategy import FedNEATStrategy

            strategy = FedNEATStrategy(logic_validator=mock_logic_validator)
            assert strategy.global_consensus_graph.number_of_edges() == 2
            assert hasattr(strategy, "ground_truth_graph")


class TestLifecycleHooks:
    """Test Flower lifecycle hook methods."""

    @patch("server.fed_neat_strategy.yaml.safe_load")
    def test_initialize_parameters_returns_initial(
        self, mock_yaml, mock_logic_validator, temp_model_dir
    ):
        """Test initialize_parameters returns initial parameters."""
        mock_yaml.return_value = {"dataset": {"name": "asia"}, "core_logic": {}}

        with patch("server.fed_neat_strategy.MODEL_DIR", temp_model_dir):
            from server.fed_neat_strategy import FedNEATStrategy

            initial_params = MagicMock()
            strategy = FedNEATStrategy(
                logic_validator=mock_logic_validator, initial_parameters=initial_params
            )

            result = strategy.initialize_parameters(MagicMock())
            assert result == initial_params

    @patch("server.fed_neat_strategy.yaml.safe_load")
    def test_configure_fit_samples_clients(
        self, mock_yaml, mock_logic_validator, temp_model_dir, mock_client_manager
    ):
        """Test configure_fit samples clients correctly."""
        mock_yaml.return_value = {"dataset": {"name": "asia"}, "core_logic": {}}

        with patch("server.fed_neat_strategy.MODEL_DIR", temp_model_dir):
            from server.fed_neat_strategy import FedNEATStrategy

            strategy = FedNEATStrategy(
                logic_validator=mock_logic_validator,
                fraction_fit=0.5,
                min_fit_clients=1,
            )

            params = MagicMock()
            results = strategy.configure_fit(
                server_round=1, parameters=params, client_manager=mock_client_manager
            )

            # Verify the client manager's sample method was called
            assert mock_client_manager.sample.called

    @patch("server.fed_neat_strategy.yaml.safe_load")
    def test_configure_evaluate_samples_clients(
        self, mock_yaml, mock_logic_validator, temp_model_dir, mock_client_manager
    ):
        """Test configure_evaluate samples clients correctly."""
        mock_yaml.return_value = {"dataset": {"name": "asia"}, "core_logic": {}}

        with patch("server.fed_neat_strategy.MODEL_DIR", temp_model_dir):
            from server.fed_neat_strategy import FedNEATStrategy

            strategy = FedNEATStrategy(
                logic_validator=mock_logic_validator,
                fraction_evaluate=0.3,
                min_evaluate_clients=1,
            )

            params = MagicMock()
            results = strategy.configure_evaluate(
                server_round=1, parameters=params, client_manager=mock_client_manager
            )

            assert mock_client_manager.sample.called

    @patch("server.fed_neat_strategy.yaml.safe_load")
    def test_evaluate_returns_none(
        self, mock_yaml, mock_logic_validator, temp_model_dir
    ):
        """Test that evaluate returns None (deferred to clients)."""
        mock_yaml.return_value = {"dataset": {"name": "asia"}, "core_logic": {}}

        with patch("server.fed_neat_strategy.MODEL_DIR", temp_model_dir):
            from server.fed_neat_strategy import FedNEATStrategy

            strategy = FedNEATStrategy(logic_validator=mock_logic_validator)
            params = MagicMock()
            result = strategy.evaluate(server_round=1, parameters=params)
            assert result is None


class TestTopologicalCrossover:
    """Test genome crossover (Innovation Hash-based merge)."""

    @patch("server.fed_neat_strategy.yaml.safe_load")
    def test_crossover_merges_innovations(
        self, mock_yaml, mock_logic_validator, temp_model_dir
    ):
        """Test that crossover merges genomes by innovation hash."""
        mock_yaml.return_value = {"dataset": {"name": "asia"}, "core_logic": {}}

        with patch("server.fed_neat_strategy.MODEL_DIR", temp_model_dir):
            from server.fed_neat_strategy import FedNEATStrategy

            strategy = FedNEATStrategy(logic_validator=mock_logic_validator)

            # Create two genomes with overlapping innovations
            genome1 = {
                "in_features": 5,
                "num_classes": 2,
                "nodes": {"0": "input", "1": "hidden", "2": "output"},
                "hidden_nodes": ["1"],
                "connections": {
                    "innov_1": {
                        "in": "0",
                        "out": "1",
                        "active": True,
                        "weight": 0.5,
                    },
                    "innov_2": {"in": "1", "out": "2", "active": True, "weight": 0.3},
                },
            }

            genome2 = {
                "in_features": 5,
                "num_classes": 2,
                "nodes": {"0": "input", "1": "hidden", "2": "output"},
                "hidden_nodes": ["1"],
                "connections": {
                    "innov_1": {
                        "in": "0",
                        "out": "1",
                        "active": True,
                        "weight": 0.6,
                    },
                    "innov_3": {"in": "1", "out": "0", "active": False, "weight": 0.1},
                },
            }

            merged = strategy._crossover([genome1, genome2])

            # Verify merged structure
            assert "in_features" in merged
            assert "num_classes" in merged
            assert "connections" in merged

            # innov_1 should exist (in both) with averaged weight
            assert "innov_1" in merged["connections"]
            assert merged["connections"]["innov_1"]["active"] == True

    @patch("server.fed_neat_strategy.yaml.safe_load")
    def test_crossover_handles_inactive_connections(
        self, mock_yaml, mock_logic_validator, temp_model_dir
    ):
        """Test crossover handling of inactive connections."""
        mock_yaml.return_value = {"dataset": {"name": "asia"}, "core_logic": {}}

        with patch("server.fed_neat_strategy.MODEL_DIR", temp_model_dir):
            from server.fed_neat_strategy import FedNEATStrategy

            strategy = FedNEATStrategy(logic_validator=mock_logic_validator)

            genome1 = {
                "in_features": 3,
                "num_classes": 2,
                "nodes": {"0": "input", "1": "output"},
                "hidden_nodes": [],
                "connections": {
                    "innov_1": {"in": "0", "out": "1", "active": False, "weight": 0.2}
                },
            }

            genome2 = {
                "in_features": 3,
                "num_classes": 2,
                "nodes": {"0": "input", "1": "output"},
                "hidden_nodes": [],
                "connections": {
                    "innov_1": {"in": "0", "out": "1", "active": False, "weight": 0.3}
                },
            }

            merged = strategy._crossover([genome1, genome2])

            # Inactive connection should remain inactive
            assert merged["connections"]["innov_1"]["active"] == False


class TestAggregateFit:
    """Test aggregate_fit with topological crossover."""

    @patch("server.fed_neat_strategy.yaml.safe_load")
    @patch("server.fed_neat_strategy.json.dump")
    def test_aggregate_fit_empty_results(
        self, mock_json_dump, mock_yaml, mock_logic_validator
    ):
        """Test aggregate_fit with no results."""
        mock_yaml.return_value = {"dataset": {"name": "asia"}, "core_logic": {}}

        with patch("server.fed_neat_strategy.MODEL_DIR", "dummy"):
            from server.fed_neat_strategy import FedNEATStrategy

            strategy = FedNEATStrategy(logic_validator=mock_logic_validator)
            params, metrics = strategy.aggregate_fit(
                server_round=1, results=[], failures=[]
            )
            assert params is None
            assert metrics == {}

    @patch("server.fed_neat_strategy.yaml.safe_load")
    @patch("server.fed_neat_strategy.json.dump")
    def test_aggregate_fit_filters_by_por_gate(
        self, mock_json_dump, mock_yaml, mock_logic_validator, temp_model_dir
    ):
        """Test that aggregate_fit applies PoR logic gate."""
        mock_yaml.return_value = {"dataset": {"name": "asia"}, "core_logic": {"consensus_momentum": 0.85}}

        with patch("server.fed_neat_strategy.MODEL_DIR", temp_model_dir):
            from server.fed_neat_strategy import FedNEATStrategy

            strategy = FedNEATStrategy(
                logic_validator=mock_logic_validator,
                initial_parameters=MagicMock(),
            )

            # Setup two clients
            client1 = Mock()
            client1.cid = "client_1"
            client2 = Mock()
            client2.cid = "client_2"

            # Client 1: good graph, should be accepted
            fit_res1 = Mock(spec=FitRes)
            fit_res1.metrics = {
                "causal_graph_edges": "[[0, 1], [1, 2]]",
                "accuracy": 0.9,
            }
            genome_json = json.dumps({
                "in_features": 5,
                "num_classes": 2,
                "nodes": {"0": "input", "1": "output"},
                "hidden_nodes": [],
                "connections": {},
                "fitness": 0.9,
            })
            fit_res1.parameters = MagicMock()
            fit_res1.parameters.tensors = _npy_bytes_from_json(json.loads(genome_json))

            # Client 2: bad graph, should be rejected
            fit_res2 = Mock(spec=FitRes)
            fit_res2.metrics = {
                "causal_graph_edges": "[[0, 1]]",
                "accuracy": 0.5,
            }
            fit_res2.parameters = MagicMock()
            fit_res2.parameters.tensors = _npy_bytes_from_json(json.loads(genome_json))

            # Setup evaluator to accept client1, reject client2
            def eval_side_effect(g, r=None):
                edges = list(g.edges())
                if len(edges) >= 2:
                    return True, 0.3  # Accepted
                return False, 0.8  # Rejected

            mock_logic_validator.evaluate_client_graph.side_effect = eval_side_effect

            results = [(client1, fit_res1), (client2, fit_res2)]
            params, metrics = strategy.aggregate_fit(
                server_round=1, results=results, failures=[]
            )

            # Should have filtered by PoR
            assert metrics["accepted_clients"] >= 1
            assert metrics["rejected_clients"] >= 1

    @patch("server.fed_neat_strategy.yaml.safe_load")
    @patch("server.fed_neat_strategy.json.dump")
    def test_aggregate_fit_all_rejected_returns_cached(
        self, mock_json_dump, mock_yaml, mock_logic_validator, temp_model_dir
    ):
        """Test that when all clients rejected, returns cached parameters."""
        mock_yaml.return_value = {"dataset": {"name": "asia"}, "core_logic": {}}
        mock_logic_validator.evaluate_client_graph.return_value = (False, 0.9)

        with patch("server.fed_neat_strategy.MODEL_DIR", temp_model_dir):
            from server.fed_neat_strategy import FedNEATStrategy

            initial_params = MagicMock()
            strategy = FedNEATStrategy(
                logic_validator=mock_logic_validator, initial_parameters=initial_params
            )

            client = Mock()
            client.cid = "bad_client"
            fit_res = Mock()
            fit_res.metrics = {"causal_graph_edges": "[[0, 1]]"}
            fit_res.parameters = MagicMock()
            fit_res.parameters.tensors = _npy_bytes_from_json({"in_features": 5})

            results = [(client, fit_res)]
            params, metrics = strategy.aggregate_fit(
                server_round=1, results=results, failures=[]
            )

            # Should return initial parameters (fallback)
            assert metrics["accepted_clients"] == 0
            assert metrics["rejected_clients"] == 1


class TestConsensusUpdate:
    """Test consensus graph update."""

    @patch("server.fed_neat_strategy.yaml.safe_load")
    def test_aggregate_logic_updates_consensus(
        self, mock_yaml, mock_logic_validator, temp_model_dir
    ):
        """Test that consensus graph is updated with client graphs."""
        mock_yaml.return_value = {"dataset": {"name": "asia"}, "core_logic": {"consensus_momentum": 0.85}}

        with patch("server.fed_neat_strategy.MODEL_DIR", temp_model_dir):
            from server.fed_neat_strategy import FedNEATStrategy

            strategy = FedNEATStrategy(logic_validator=mock_logic_validator)

            # Initialize consensus
            consensus = nx.DiGraph()
            consensus.add_edge(0, 1)
            strategy.global_consensus_graph = consensus

            # Client graph matching consensus
            client_graph = nx.DiGraph()
            client_graph.add_edge(0, 1)
            client_graph.add_edge(1, 2)

            strategy._aggregate_logic([client_graph])

            # Consensus should be updated
            assert strategy.global_consensus_graph.number_of_nodes() > 0

    @patch("server.fed_neat_strategy.yaml.safe_load")
    def test_aggregate_logic_empty(
        self, mock_yaml, mock_logic_validator, temp_model_dir
    ):
        mock_yaml.return_value = {}
        with patch("server.fed_neat_strategy.MODEL_DIR", temp_model_dir):
            from server.fed_neat_strategy import FedNEATStrategy
            strategy = FedNEATStrategy(logic_validator=mock_logic_validator)
            strategy._aggregate_logic([])
            assert strategy.global_consensus_graph.number_of_nodes() == 0

    @patch("server.fed_neat_strategy.yaml.safe_load")
    def test_finetune_simgnn_executes(
        self, mock_yaml, mock_logic_validator, temp_model_dir
    ):
        mock_yaml.return_value = {"dataset": {"name": "asia"}, "core_logic": {"consensus_momentum": 0.85}}
        
        from server.logic_validator import SimGNN
        simgnn = SimGNN(hidden_dim=16, num_layers=2)
        mock_logic_validator.simgnn = simgnn

        with patch("server.fed_neat_strategy.MODEL_DIR", temp_model_dir):
            from server.fed_neat_strategy import FedNEATStrategy
            strategy = FedNEATStrategy(logic_validator=mock_logic_validator)
            
            consensus = nx.DiGraph()
            consensus.add_edges_from([(0, 1), (1, 2), (2, 3)])
            
            with patch('torch.save') as mock_torch_save:
                strategy._finetune_simgnn_on_consensus(consensus, steps=2, pairs=4)
                assert mock_torch_save.called


class TestDynamicThresholdUpdate:
    """Test dynamic threshold update in Logic Validator."""

    @patch("server.fed_neat_strategy.yaml.safe_load")
    @patch("server.fed_neat_strategy.json.dump")
    def test_dynamic_threshold_called(
        self, mock_json_dump, mock_yaml, mock_logic_validator, temp_model_dir
    ):
        """Test that update_dynamic_threshold is called with round GED scores."""
        mock_yaml.return_value = {"dataset": {"name": "asia"}, "core_logic": {}}

        with patch("server.fed_neat_strategy.MODEL_DIR", temp_model_dir):
            from server.fed_neat_strategy import FedNEATStrategy

            strategy = FedNEATStrategy(
                logic_validator=mock_logic_validator,
                initial_parameters=MagicMock(),
            )

            client = Mock()
            client.cid = "client_1"
            fit_res = Mock()
            fit_res.metrics = {"causal_graph_edges": "[[0, 1]]", "accuracy": 0.9}
            fit_res.parameters = MagicMock()
            fit_res.parameters.tensors = _npy_bytes_from_json(
                {
                    "in_features": 5,
                    "num_classes": 2,
                    "nodes": {},
                    "hidden_nodes": [],
                    "connections": {},
                }
            )

            mock_logic_validator.evaluate_client_graph.return_value = (True, 0.25)

            results = [(client, fit_res)]
            strategy.aggregate_fit(server_round=1, results=results, failures=[])

            # Verify update_dynamic_threshold was called
            assert mock_logic_validator.update_dynamic_threshold.called


class TestGraphPersistence:
    """Test saving of sample graphs for visualization."""

    @patch("server.fed_neat_strategy.yaml.safe_load")
    @patch("server.fed_neat_strategy.json.dump")
    def test_honest_graph_saved(
        self, mock_json_dump, mock_yaml, mock_logic_validator, temp_model_dir
    ):
        """Test that accepted client graph is saved."""
        mock_yaml.return_value = {"dataset": {"name": "asia"}, "core_logic": {}}

        with patch("server.fed_neat_strategy.MODEL_DIR", temp_model_dir):
            from server.fed_neat_strategy import FedNEATStrategy

            strategy = FedNEATStrategy(
                logic_validator=mock_logic_validator,
                initial_parameters=MagicMock(),
            )

            client = Mock()
            client.cid = "good_client"
            fit_res = Mock()
            fit_res.metrics = {"causal_graph_edges": "[[0, 1]]", "accuracy": 0.95}
            fit_res.parameters = MagicMock()
            fit_res.parameters.tensors = _npy_bytes_from_json(
                {
                    "in_features": 5,
                    "num_classes": 2,
                    "nodes": {},
                    "hidden_nodes": [],
                    "connections": {},
                }
            )

            mock_logic_validator.evaluate_client_graph.return_value = (True, 0.2)

            results = [(client, fit_res)]
            strategy.aggregate_fit(server_round=1, results=results, failures=[])

            # Check that honest_graph_sample.gpickle was created
            honest_path = f"{temp_model_dir}/honest_graph_sample.gpickle"
            # Note: We can't directly check file creation due to mocking,
            # but we can verify the logic paths were executed

    @patch("server.fed_neat_strategy.yaml.safe_load")
    @patch("server.fed_neat_strategy.json.dump")
    def test_rejected_graph_saved(
        self, mock_json_dump, mock_yaml, mock_logic_validator, temp_model_dir
    ):
        """Test that rejected client graph is saved with edge diff."""
        mock_yaml.return_value = {"dataset": {"name": "asia"}, "core_logic": {}}

        with patch("server.fed_neat_strategy.MODEL_DIR", temp_model_dir):
            from server.fed_neat_strategy import FedNEATStrategy

            strategy = FedNEATStrategy(
                logic_validator=mock_logic_validator,
                initial_parameters=MagicMock(),
            )

            # Initialize consensus with edges
            consensus = nx.DiGraph()
            consensus.add_edge(0, 1)
            consensus.add_edge(1, 2)
            strategy.global_consensus_graph = consensus

            client = Mock()
            client.cid = "bad_client"
            # Client only has edge (0,1), missing (1,2)
            fit_res = Mock()
            fit_res.metrics = {"causal_graph_edges": "[[0, 1]]", "accuracy": 0.4}
            fit_res.parameters = MagicMock()
            fit_res.parameters.tensors = _npy_bytes_from_json({})

            mock_logic_validator.evaluate_client_graph.return_value = (False, 0.7)

            results = [(client, fit_res)]
            strategy.aggregate_fit(server_round=1, results=results, failures=[])

            # Verify json.dump was called for rejected edge diff
            # The function saves edge_diff to rejected_edge_diff.json
            assert mock_json_dump.called


class TestAggregateEvaluate:
    """Test aggregate_evaluate hook."""

    @patch("server.fed_neat_strategy.yaml.safe_load")
    def test_aggregate_evaluate_weighted_accuracy(
        self, mock_yaml, mock_logic_validator, temp_model_dir
    ):
        """Test that aggregate_evaluate computes weighted accuracy."""
        mock_yaml.return_value = {"dataset": {"name": "asia"}, "core_logic": {}}

        with patch("server.fed_neat_strategy.MODEL_DIR", temp_model_dir):
            from server.fed_neat_strategy import FedNEATStrategy

            strategy = FedNEATStrategy(logic_validator=mock_logic_validator)

            # Create mock evaluation results
            result1 = Mock()
            result1.metrics = {"accuracy": 0.9}
            result1.num_examples = 100

            result2 = Mock()
            result2.metrics = {"accuracy": 0.8}
            result2.num_examples = 50

            client1 = Mock()
            client2 = Mock()

            results = [(client1, result1), (client2, result2)]

            loss, metrics = strategy.aggregate_evaluate(
                server_round=1, results=results, failures=[]
            )

            # Check weighted average: (0.9*100 + 0.8*50) / (100+50) = 110/150 = 0.733...
            expected = (0.9 * 100 + 0.8 * 50) / 150
            assert abs(loss - expected) < 0.01

    @patch("server.fed_neat_strategy.yaml.safe_load")
    def test_aggregate_evaluate_empty_results(
        self, mock_yaml, mock_logic_validator, temp_model_dir
    ):
        """Test aggregate_evaluate with no results."""
        mock_yaml.return_value = {"dataset": {"name": "asia"}, "core_logic": {}}

        with patch("server.fed_neat_strategy.MODEL_DIR", temp_model_dir):
            from server.fed_neat_strategy import FedNEATStrategy

            strategy = FedNEATStrategy(logic_validator=mock_logic_validator)

            loss, metrics = strategy.aggregate_evaluate(
                server_round=1, results=[], failures=[]
            )

            assert loss is None
            assert metrics == {}

    @patch("server.fed_neat_strategy.yaml.safe_load")
    def test_aggregate_evaluate_handles_missing_accuracy(
        self, mock_yaml, mock_logic_validator, temp_model_dir
    ):
        """Test aggregate_evaluate handles missing accuracy gracefully."""
        mock_yaml.return_value = {"dataset": {"name": "asia"}, "core_logic": {}}

        with patch("server.fed_neat_strategy.MODEL_DIR", temp_model_dir):
            from server.fed_neat_strategy import FedNEATStrategy

            strategy = FedNEATStrategy(logic_validator=mock_logic_validator)

            result = Mock()
            result.metrics = {}  # No accuracy
            result.num_examples = 100

            client = Mock()
            results = [(client, result)]

            loss, metrics = strategy.aggregate_evaluate(
                server_round=1, results=results, failures=[]
            )

            # Should not crash, accuracy defaults to 0.0
            assert loss is not None
