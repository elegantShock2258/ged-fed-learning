import unittest
import networkx as nx
import torch
from flwr.common import FitRes, Parameters, NDArrays, Status, Code
from flwr.server.client_proxy import ClientProxy
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from server.logic_validator import LogicValidator, SimGNN
from server.aggregator import PoRStrategy
from client.agent import ISICClient

class DummyClientProxy(ClientProxy):
    def __init__(self, cid):
        super().__init__(cid=cid)

    def get_properties(self, ins, timeout, group_id=None):
        pass

    def get_parameters(self, ins, timeout, group_id=None):
        pass

    def fit(self, ins, timeout, group_id=None):
        pass

    def evaluate(self, ins, timeout, group_id=None):
        pass
        
    def reconnect(self, ins, timeout, group_id=None):
        pass

class TestGovernanceLayer(unittest.TestCase):

    def setUp(self):
        self.validator = LogicValidator(threshold=0.7)
        self.strategy = PoRStrategy(logic_validator=self.validator)

    def test_logic_validator_accepts_empty_round_1(self):
        """Test that in round 1, everything is accepted to form the base."""
        G = nx.DiGraph()
        G.add_edge("A", "B")
        
        # At init, consensus is empty
        is_accepted, score = self.validator.evaluate_client_graph(G)
        self.assertTrue(is_accepted)
        self.assertEqual(score, 0.0)

    def test_dual_aggregation_filters_rejections(self):
        """Test that Dual Aggregation rejects graphs over the threshold."""
        # 1. Simulate an aggregated consensus
        consensus = nx.DiGraph()
        consensus.add_edges_from([("A", "B"), ("B", "C")])
        self.validator.set_global_consensus(consensus)
        
        # 2. Mock a result set
        good_graph = nx.DiGraph()
        good_graph.add_edges_from([("A", "B"), ("B", "C")])
        
        bad_graph = nx.DiGraph()
        bad_graph.add_edges_from([("X", "Y"), ("Y", "Z")])
        
        # Overwrite threshold manually so mock values force failure (untrained NN diffs)
        # Using a strict threshold ensures bad is rejected and good is likely accepted
        # since PyTorch geometric distances will be 0 vs >0.
        self.validator.threshold = 0.5
        
        # Force the SimGNN to return 0 for good_graph (perfect match)
        # By mocking the _nx_to_pyg_data to just return an empty tensor so SimGNN yields 0
        from unittest.mock import patch
        
        dummy_fit_res_good = FitRes(
            status=Status(Code.OK, ""),
            parameters=Parameters(tensors=[], tensor_type=""),
            num_examples=100,
            metrics={"causal_graph_edges": str(list(good_graph.edges()))}
        )
        
        dummy_fit_res_bad = FitRes(
            status=Status(Code.OK, ""),
            parameters=Parameters(tensors=[], tensor_type=""),
            num_examples=100,
            metrics={"causal_graph_edges": str(list(bad_graph.edges()))}
        )
        
        results = [
            (DummyClientProxy("honest"), dummy_fit_res_good),
            (DummyClientProxy("adversary"), dummy_fit_res_bad)
        ]
        
        # Mock evaluate directly to bypass untrined SimGNN variance for unit test stability
        with patch.object(self.validator, 'evaluate_client_graph') as mock_eval:
            # Good passes, Bad fails
            def side_effect(graph):
                if "X" in graph:
                    return False, 0.9 # Rejected
                return True, 0.1 # Passed
            mock_eval.side_effect = side_effect
            
            agg_params, metrics = self.strategy.aggregate_fit(server_round=2, results=results, failures=[])
            
            self.assertEqual(metrics["accepted_clients"], 1)
            self.assertEqual(metrics["rejected_clients"], 1)
            
            # Global graph should now be just the honest one, since target_votes = 1/2 = 0.5
            # Edge A->B appears 1 time, > 0.5, so it's kept.
            new_consensus = self.strategy.global_consensus_graph
            self.assertTrue(new_consensus.has_edge("A", "B"))
            self.assertFalse(new_consensus.has_edge("X", "Y"))

if __name__ == '__main__':
    unittest.main()
