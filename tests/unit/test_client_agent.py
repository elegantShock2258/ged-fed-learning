import pytest
import torch
import torch.nn as nn
from unittest.mock import patch, MagicMock, mock_open
import numpy as np
import json
import os

from client.agent import (
    broadcast_state,
    serialize_genome,
    deserialize_genome,
    ISICClient
)
from client.models import DynamicGenome

class DummyGenome:
    def __init__(self):
        self.nodes = [1, 2]
        self.connections = [[1, 2, 0.5]]
        self.hidden_nodes = []
        self.in_features = 5
        self.num_classes = 2

    def _sync_weights(self):
        pass

@patch("client.agent.os.makedirs")
@patch("builtins.open", new_callable=mock_open)
@patch("client.agent.json.dump")
def test_broadcast_state(mock_json_dump, mock_file, mock_makedirs):
    genome = DummyGenome()
    broadcast_state(genome, "TestAgent")
    
    # Assert os.makedirs was called
    mock_makedirs.assert_called_once_with("saved_models", exist_ok=True)
    # Assert file was opened
    mock_file.assert_called_once_with("saved_models/realtime_state.json", "w")
    # Assert json.dump was called with correct data
    expected_data = {
        "source": "TestAgent",
        "nodes": [1, 2],
        "connections": [[1, 2, 0.5]]
    }
    mock_json_dump.assert_called_once_with(expected_data, mock_file())

@patch("client.agent.os.makedirs", side_effect=Exception("Disk error"))
def test_broadcast_state_exception(mock_makedirs):
    # Should not raise exception
    genome = DummyGenome()
    broadcast_state(genome)


def test_serialize_deserialize_genome():
    genome = DummyGenome()
    
    parameters = serialize_genome(genome)
    assert isinstance(parameters, list)
    assert isinstance(parameters[0], np.ndarray)
    
    new_genome = DummyGenome()
    new_genome.nodes = []
    
    deserialize_genome(new_genome, parameters)
    
    assert new_genome.nodes == [1, 2]
    assert new_genome.in_features == 5

def test_deserialize_genome_empty():
    genome = DummyGenome()
    # Shouldn't fail with empty parameters
    deserialize_genome(genome, [])
    assert genome.nodes == [1, 2]


class DummyLoader:
    def __init__(self):
        self.dataset = [1, 2, 3] # just for length
        # Create a mock batch
        self.images = torch.randn(2, 5)
        self.labels = torch.tensor([0, 1])
        
    def __iter__(self):
        yield self.images, self.labels


@patch("client.agent.yaml.safe_load")
def test_isic_client_init(mock_yaml):
    mock_yaml.return_value = {"core_logic": {"causal_edge_threshold": 0.5}}
    
    with patch("builtins.open", mock_open(read_data="dummy")):
        client = ISICClient(
            cid="test_1",
            train_loader=DummyLoader(),
            test_loader=DummyLoader(),
            device=torch.device("cpu"),
            feature_names=["f1", "f2", "f3", "f4", "f5"],
            num_classes=2
        )
        
    assert client.cid == "test_1"
    assert client.device == torch.device("cpu")
    assert isinstance(client.model, DynamicGenome)


@patch("client.agent.yaml.safe_load")
def test_isic_client_functions(mock_yaml):
    mock_yaml.return_value = {"core_logic": {"causal_edge_threshold": 0.5}}
    with patch("builtins.open", mock_open(read_data="dummy")):
        client = ISICClient(
            cid="test_1",
            train_loader=DummyLoader(),
            test_loader=DummyLoader(),
            device=torch.device("cpu"),
            feature_names=["f1", "f2", "f3", "f4", "f5"],
            num_classes=2
        )

    # get_parameters / set_parameters
    params = client.get_parameters(config={})
    client.set_parameters(params)

    # evaluate_fitness
    # Use a real model clone
    genome = client.model.clone()
    genome.eval = MagicMock()
    # Mock forward
    genome.forward = MagicMock(return_value=(torch.tensor([[0.8, 0.2], [0.1, 0.9]]), torch.randn(2, 5)))
    features = client.evaluate_fitness(genome)
    
    assert genome.fitness > 0 # Should have some correct predictions
    assert len(features) == 1
    
    # Check evaluate
    client.model.forward = MagicMock(return_value=(torch.tensor([[0.8, 0.2], [0.1, 0.9]]), torch.randn(2, 5)))
    loss, num_examples, metrics = client.evaluate(params, {})
    assert loss == 0.0
    assert num_examples == 3
    assert "accuracy" in metrics


@patch("client.agent.broadcast_state")
@patch("client.agent.yaml.safe_load")
def test_isic_client_fit(mock_yaml, mock_broadcast):
    mock_yaml.return_value = {"core_logic": {"causal_edge_threshold": 0.5}}
    with patch("builtins.open", mock_open(read_data="dummy")):
        client = ISICClient(
            cid="test_1",
            train_loader=DummyLoader(),
            test_loader=DummyLoader(),
            device=torch.device("cpu"),
            feature_names=["f1", "f2", "f3", "f4", "f5"],
            num_classes=2
        )
    
    client.evaluate_fitness = MagicMock(return_value=[torch.randn(2, 5)])
    
    # Mock cognitive module
    client.cognitive_module.extract_causal_graph = MagicMock()
    mock_graph = MagicMock()
    mock_graph.edges.return_value = [(1, 2), (3, 4)]
    client.cognitive_module.extract_causal_graph.return_value = mock_graph

    config = {"epochs": 2, "population_size": 2}
    
    # Give initial params
    params = client.get_parameters({})

    res_params, num_examples, metrics = client.fit(params, config)
    
    assert num_examples == 3
    assert "causal_graph_edges" in metrics
    assert metrics["causal_graph_edges"] == str([(1, 2), (3, 4)])
    
    # Ensure broadcast_state was called
    assert mock_broadcast.called
