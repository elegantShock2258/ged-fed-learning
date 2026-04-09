import pytest
import torch
import networkx as nx
from unittest.mock import patch, MagicMock, mock_open

from server.logic_validator import SimGNN, LogicValidator
from torch_geometric.data import Data, Batch

class TestLogicValidator:
    
    def test_simgnn_forward(self):
        """Test SimGNN forward and forward_once."""
        model = SimGNN()
        
        # Create dummy data
        edge_index = torch.tensor([[0, 1], [1, 0]], dtype=torch.long)
        x = torch.ones((2, 1), dtype=torch.float32)
        batch = torch.zeros(2, dtype=torch.long)
        
        data1 = Data(x=x, edge_index=edge_index)
        data1.batch = batch
        
        data2 = Data(x=x, edge_index=edge_index)
        data2.batch = batch
        
        # Test forward_once
        emb = model.forward_once(data1)
        assert emb.shape == (1, model.convs[0].out_channels * 2)
        
        # Test forward
        score = model.forward(data1, data2)
        assert score.shape == (1,)
        assert 0.0 <= score.item() <= 1.0

    @patch('server.logic_validator.os.path.exists')
    @patch('server.logic_validator.torch.load')
    def test_logic_validator_init(self, mock_load, mock_exists):
        mock_exists.return_value = True
        mock_load.return_value = {}
        
        with patch('server.logic_validator.SimGNN.load_state_dict'):
            validator = LogicValidator(model_path="dummy.pt", threshold=0.5)
        
        assert validator.threshold == 0.5
        assert validator.simgnn is not None
        mock_load.assert_called_once()
        
    def test_nx_to_pyg_data(self):
        validator = LogicValidator()
        
        # Graph with edges
        g1 = nx.DiGraph()
        g1.add_edges_from([('A', 'B'), ('B', 'C')])
        data1 = validator._nx_to_pyg_data(g1)
        assert data1.edge_index.size(1) == 2
        
        # Graph without edges
        g2 = nx.DiGraph()
        g2.add_nodes_from(['A', 'B'])
        data2 = validator._nx_to_pyg_data(g2)
        assert data2.edge_index.size(1) == 0

    def test_update_dynamic_threshold(self):
        validator = LogicValidator(threshold=0.2)
        
        # Empty scores
        validator.update_dynamic_threshold([])
        assert not hasattr(validator, 'dynamic_threshold')

        scores = [0.1, 0.2, 0.3, 0.8, 0.9]
        
        with patch('builtins.open', mock_open(read_data="simulation:\n  num_clients: 5\n  num_false_nodes: 1\n")):
            validator.update_dynamic_threshold(scores)
        
        # 4/5 = 80%. 80*0.95 = 76th percentile.
        assert hasattr(validator, 'dynamic_threshold')
        assert validator.dynamic_threshold >= 0.2

    @patch('builtins.open', mock_open(read_data="simulation:\n  num_clients: 5\n  num_false_nodes: 1\n"))
    def test_evaluate_client_graph(self):
        with patch('server.logic_validator.yaml.safe_load') as mock_yaml:
            mock_yaml.return_value = {"simulation": {"num_clients": 5, "num_false_nodes": 1}, "dataset": {"name": "test"}}
            validator = LogicValidator(threshold=0.99)
            
            # Mock graph
            g_consensus = nx.DiGraph()
            g_consensus.add_edges_from([('A', 'B')])
            validator.set_global_consensus(g_consensus)
            
            g_client = nx.DiGraph()
            g_client.add_edges_from([('A', 'C')])
            
            # Round <= 0 should always return True, 0.0
            acc, score = validator.evaluate_client_graph(g_client, server_round=0)
            assert acc is True
            assert score == 0.0
            
            # Set dynamic threshold manually to test logic 
            # Force SimGNN to return a specific score
            validator.simgnn = MagicMock()
            validator.simgnn.return_value = torch.tensor([0.4])
            validator.dynamic_threshold = 0.5
            
            # Normal evaluation
            with patch('server.logic_validator.os.makedirs'):
                with patch('builtins.open', mock_open()) as mock_file:
                    acc, score = validator.evaluate_client_graph(g_client, 2)
                    assert acc is True
                    assert score == pytest.approx(0.4, rel=1e-3)
                    # Verify telemetry logger opens the file
                    mock_file.assert_called_with("saved_models/test/debug_graphs_log.txt", "a")