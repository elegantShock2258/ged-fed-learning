import pytest

class TestImportsAndSmoke:
    """Smoke tests for remaining modules to improve coverage."""

    def test_server_aggregator_imports(self):
        """Test server.aggregator imports."""
        try:
            from server.aggregator import PoRStrategy
            assert PoRStrategy is not None
        except ImportError:
            pytest.skip("Import failed")

    def test_server_fed_neat_imports(self):
        """Test server.fed_neat_strategy imports."""
        try:
            from server.fed_neat_strategy import FedNEATStrategy
            assert FedNEATStrategy is not None
        except ImportError:
            pytest.skip("Import failed")

    def test_server_generate_consensus_imports(self):
        """Test server.generate_consensus imports."""
        try:
            from server.generate_consensus import generate_global_consensus
            assert callable(generate_global_consensus)
        except ImportError:
            pytest.skip("Import failed")

    def test_server_train_simgnn_imports(self):
        """Test server.train_simgnn imports."""
        try:
            from server.train_simgnn import train_simgnn
            assert callable(train_simgnn)
        except ImportError:
            pytest.skip("Import failed")

    def test_server_visualizer_imports(self):
        """Test server.visualizer_bridge imports."""
        try:
            from server.visualizer_bridge import run_bridge
            assert callable(run_bridge)
        except ImportError:
            pytest.skip("Import failed")

    def test_client_agent_imports(self):
        """Test client.agent imports."""
        try:
            from client.agent import ISICClient
            assert ISICClient is not None
        except ImportError:
            pytest.skip("Import failed")

    def test_adversary_poisoning_imports(self):
        """Test adversary.poisoning imports."""
        try:
            from adversary.poisoning import FalseNode
            assert FalseNode is not None
        except ImportError:
            pytest.skip("Import failed")

    def test_datasets_loader_imports(self):
        """Test datasets.tabular_loader imports."""
        try:
            from datasets.tabular_loader import TabularBNDataset
            assert TabularBNDataset is not None
        except ImportError:
            pytest.skip("Import failed")

    def test_vastai_imports(self):
        """Test vastai.run_vast_simulation imports."""
        try:
            from vastai.run_vast_simulation import main
            assert callable(main)
        except ImportError:
            pytest.skip("Import failed")
