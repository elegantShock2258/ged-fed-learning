"""
Simple client integration smoke tests.
"""

import torch
import networkx as nx


def test_dynamic_genome_forward():
    from client.models import DynamicGenome

    genome = DynamicGenome(in_features=5, num_classes=2)
    x = torch.randn(4, 5)
    logits, features = genome(x)

    assert logits.shape == (4, 2)
    assert features.shape == (4, 5)


def test_cognitive_module_extract():
    from client.causal_discovery import CognitiveModule

    cm = CognitiveModule(feature_names=['A', 'B', 'C'], threshold=0.1, lr=0.01, max_iter=5)
    data = torch.randn(20, 3)
    graph = cm.extract_causal_graph(data)

    assert isinstance(graph, nx.DiGraph)
    assert len(graph.nodes()) == 3


def test_isic_client_initialization():
    from client.agent import ISICClient
    from torch.utils.data import DataLoader, TensorDataset

    X = torch.randn(10, 5)
    y = torch.randint(0, 2, (10,))
    dataset = TensorDataset(X, y)
    loader = DataLoader(dataset, batch_size=2)

    client = ISICClient(
        cid='test_client',
        train_loader=loader,
        test_loader=loader,
        device=torch.device('cpu'),
        feature_names=['f1', 'f2', 'f3', 'f4', 'f5'],
        num_classes=2
    )

    assert client.cid == 'test_client'
    assert client.model.num_classes == 2
    assert client.cognitive_module.feature_names == ['f1', 'f2', 'f3', 'f4', 'f5']
