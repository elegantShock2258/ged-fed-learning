"""
tests/unit/test_aggregator_logic.py
--------------------------------------
Unit tests for PoRStrategy._aggregate_logic (consensus momentum logic).
These tests patch params.yaml reads to avoid filesystem dependency.
"""

import pytest
import networkx as nx
from unittest.mock import patch, mock_open
import yaml


# ── Helpers ───────────────────────────────────────────────────────────────────

def _make_strategy(consensus_graph, momentum=0.5):
    """
    Build a PoRStrategy with a mocked LogicValidator and given consensus graph.
    momentum is injected through a patched params.yaml read.
    """
    from unittest.mock import MagicMock
    from server.aggregator import PoRStrategy
    from server.logic_validator import LogicValidator

    lv = MagicMock(spec=LogicValidator)
    lv.evaluate_client_graph.return_value = (True, 0.1)
    lv.set_global_consensus = MagicMock()

    params = {"core_logic": {"consensus_momentum": momentum, "simgnn_lr": 0.001}}

    # Patch yaml.safe_load to return our test params
    with patch("server.aggregator.yaml.safe_load", return_value=params), \
         patch("builtins.open", mock_open()):
        strategy = PoRStrategy.__new__(PoRStrategy)

    strategy.logic_validator = lv
    strategy.global_consensus_graph = consensus_graph
    strategy.model_dir = "/tmp/test_por"
    return strategy


def _make_star_graph(center="a", leaves=("b", "c", "d")):
    """Simple star graph: center → each leaf."""
    G = nx.DiGraph()
    G.add_node(center)
    for leaf in leaves:
        G.add_edge(center, leaf)
    return G


# ── Consensus update logic ────────────────────────────────────────────────────

class TestAggregateLogic:

    def test_empty_clients_no_crash(self):
        """_aggregate_logic with an empty list should return without error."""
        from server.aggregator import PoRStrategy
        from server.logic_validator import LogicValidator
        from unittest.mock import MagicMock

        lv = MagicMock(spec=LogicValidator)
        lv.set_global_consensus = MagicMock()
        strategy = PoRStrategy.__new__(PoRStrategy)
        strategy.logic_validator = lv
        strategy.global_consensus_graph = nx.DiGraph()
        strategy.model_dir = "/tmp"

        params = {"core_logic": {"consensus_momentum": 0.5, "simgnn_lr": 0.001}}
        with patch("server.aggregator.yaml.safe_load", return_value=params), \
             patch("builtins.open", mock_open()):
            strategy._aggregate_logic([])  # should not raise

    def test_unanimous_existing_edge_kept(self):
        """An existing consensus edge agreed on by all clients should be retained."""
        # Start with a consensus containing edge A→B
        consensus = _make_star_graph()  # a→b, a→c, a→d

        # All 4 submitted graphs also have a→b
        client_graphs = [_make_star_graph() for _ in range(4)]

        params = {"core_logic": {"consensus_momentum": 0.85, "simgnn_lr": 0.001}}
        with patch("server.aggregator.yaml.safe_load", return_value=params), \
             patch("builtins.open", mock_open()):
            strategy = _make_strategy(consensus, momentum=0.85)
            strategy._aggregate_logic(client_graphs)

        # With momentum=0.85, keep_threshold = (1-0.85)*0.5*4 = 0.3
        # All 4 clients agree → count=4 ≥ 0.3 → edge should be kept
        assert strategy.global_consensus_graph.has_edge("a", "b")

    def test_momentum_zero_reverts_to_majority_vote(self):
        """At momentum=0, any edge with >50% votes should be included."""
        # Consensus has a→b
        consensus = nx.DiGraph()
        consensus.add_nodes_from(["a", "b", "c"])
        consensus.add_edge("a", "b")

        # 3 out of 4 clients submit a→c (new edge, not in consensus)
        def make_g():
            G = nx.DiGraph()
            G.add_nodes_from(["a", "b", "c"])
            return G

        clients = []
        for i in range(4):
            G = make_g()
            if i < 3:      # 3 of 4 submit a→c
                G.add_edge("a", "c")
            G.add_edge("a", "b")  # all keep a→b
            clients.append(G)

        params = {"core_logic": {"consensus_momentum": 0.0, "simgnn_lr": 0.001}}
        with patch("server.aggregator.yaml.safe_load", return_value=params), \
             patch("builtins.open", mock_open()):
            strategy = _make_strategy(consensus, momentum=0.0)
            strategy._aggregate_logic(clients)

        # momentum=0 → add_threshold = 0.5*4 = 2.0 → 3 votes ≥ 2.0 → a→c added
        assert strategy.global_consensus_graph.has_edge("a", "c")

    def test_high_momentum_blocks_new_edges(self):
        """At momentum=0.99, a new edge needs near-unanimity to be added."""
        consensus = nx.DiGraph()
        consensus.add_nodes_from(["a", "b", "c"])
        # No edges in consensus

        # Only 2 of 10 clients submit the new edge a→b
        clients = []
        for i in range(10):
            G = nx.DiGraph()
            G.add_nodes_from(["a", "b", "c"])
            if i < 2:
                G.add_edge("a", "b")
            clients.append(G)

        params = {"core_logic": {"consensus_momentum": 0.99, "simgnn_lr": 0.001}}
        with patch("server.aggregator.yaml.safe_load", return_value=params), \
             patch("builtins.open", mock_open()):
            strategy = _make_strategy(consensus, momentum=0.99)
            strategy._aggregate_logic(clients)

        # add_threshold = (0.5 + 0.5*0.99)*10 = 9.95 → need 10 votes, only 2 → NOT added
        assert not strategy.global_consensus_graph.has_edge("a", "b")

    def test_momentum_clamped_to_valid_range(self):
        """Momentum values outside [0,1] should be clamped silently."""
        consensus = _make_star_graph()
        clients = [_make_star_graph() for _ in range(4)]

        params = {"core_logic": {"consensus_momentum": 2.5, "simgnn_lr": 0.001}}
        with patch("server.aggregator.yaml.safe_load", return_value=params), \
             patch("builtins.open", mock_open()):
            strategy = _make_strategy(consensus, momentum=2.5)
            strategy._aggregate_logic(clients)  # should not crash
