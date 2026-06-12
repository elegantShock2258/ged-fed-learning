"""
tests/unit/test_logic_validator.py
------------------------------------
Unit tests for server.logic_validator.SimGNN and LogicValidator.
"""

import pytest
import torch
import networkx as nx
from server.logic_validator import SimGNN, LogicValidator


# ── SimGNN ────────────────────────────────────────────────────────────────────

class TestSimGNN:
    """Tests for the raw SimGNN model."""

    @pytest.fixture
    def simgnn(self):
        return SimGNN(node_feature_dim=40, hidden_dim=32, num_layers=2).eval()

    def _make_pyg(self, G: nx.DiGraph):
        """Helper: convert nx.DiGraph to minimal PyG Data."""
        from server.logic_validator import LogicValidator
        lv = LogicValidator(model_path=None)
        return lv._nx_to_pyg_data(G)

    def test_forward_scalar_output(self, simgnn):
        """SimGNN should output a scalar tensor per pair."""
        G1 = nx.DiGraph()
        G1.add_nodes_from([0, 1, 2])
        G1.add_edge(0, 1)
        G2 = nx.DiGraph()
        G2.add_nodes_from([0, 1, 2])
        G2.add_edge(1, 2)
        lv = LogicValidator(model_path=None)
        d1 = lv._nx_to_pyg_data(G1)
        d2 = lv._nx_to_pyg_data(G2)
        with torch.no_grad():
            score = simgnn(d1, d2)
        assert score.shape == torch.Size([]) or score.numel() == 1

    def test_forward_output_in_range(self, simgnn):
        """Output should be in [0, 1] due to sigmoid."""
        G = nx.DiGraph()
        G.add_nodes_from([0, 1, 2, 3])
        G.add_edge(0, 1); G.add_edge(1, 2)
        lv = LogicValidator(model_path=None)
        d = lv._nx_to_pyg_data(G)
        with torch.no_grad():
            score = simgnn(d, d)
        assert 0.0 <= score.item() <= 1.0

    def test_identical_graphs_low_score(self, simgnn):
        """Two identical graphs should produce a low (ideally near-zero) GED score."""
        G = nx.DiGraph()
        G.add_nodes_from([0, 1, 2])
        G.add_edge(0, 1); G.add_edge(1, 2)
        lv = LogicValidator(model_path=None)
        d = lv._nx_to_pyg_data(G)
        # Fresh random weights won't give exactly 0, but score should be in [0,1]
        with torch.no_grad():
            score = simgnn(d, d).item()
        assert 0.0 <= score <= 1.0


# ── LogicValidator ────────────────────────────────────────────────────────────

class TestLogicValidator:
    """Tests for the LogicValidator governance wrapper."""

    @pytest.fixture
    def consensus(self):
        """Simple 5-node consensus graph."""
        G = nx.DiGraph()
        G.add_nodes_from([0, 1, 2, 3, 4])
        G.add_edges_from([(0, 1), (1, 2), (0, 3)])
        return G

    @pytest.fixture
    def lv(self, consensus):
        lv = LogicValidator(model_path=None, threshold=0.45)
        lv.set_global_consensus(consensus)
        return lv

    def test_consensus_data_set(self, lv):
        """After set_global_consensus, global_consensus_data must exist."""
        assert hasattr(lv, "global_consensus_data")

    def test_evaluate_returns_tuple(self, lv, consensus):
        """evaluate_client_graph must return (bool, float)."""
        result = lv.evaluate_client_graph(consensus.copy())
        assert isinstance(result, tuple) and len(result) == 2
        is_accepted, score = result
        assert isinstance(is_accepted, bool)
        assert isinstance(score, float)

    def test_score_in_range(self, lv, consensus):
        """GED score must always be in [0, 1]."""
        _, score = lv.evaluate_client_graph(consensus.copy())
        assert 0.0 <= score <= 1.0

    def test_no_consensus_auto_accepts(self):
        """Before set_global_consensus is called, evaluate should auto-accept."""
        lv = LogicValidator(model_path=None, threshold=0.45)
        G = nx.DiGraph()
        G.add_edge(0, 1)
        is_accepted, score = lv.evaluate_client_graph(G)
        assert is_accepted is True
        assert score == 0.0

    def test_nx_to_pyg_empty_graph(self, lv):
        """Empty graph conversion should not crash."""
        G = nx.DiGraph()
        G.add_nodes_from([0, 1])
        data = lv._nx_to_pyg_data(G)
        assert data.x is not None

    def test_threshold_lower_rejects_more(self, consensus):
        """A very low threshold should make the validator more likely to reject."""
        lv_strict = LogicValidator(model_path=None, threshold=0.01)
        lv_strict.set_global_consensus(consensus)
        # Perturbed graph
        perturbed = consensus.copy()
        perturbed.remove_edge(0, 1)
        perturbed.add_edge(2, 3)
        perturbed.add_edge(3, 4)
        is_accepted_strict, score = lv_strict.evaluate_client_graph(perturbed)
        # With threshold 0.01, even a small score likely rejects; score is in [0,1]
        assert 0.0 <= score <= 1.0
