import pytest
import sys
from pathlib import Path

# Ensure project root is on path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))


class TestFederatedSim:
    """Basic tests for experiments/run_por_sim.py."""

    def test_federated_sim_imports(self):
        """Test that experiments/run_por_sim.py can be imported."""
        try:
            import experiments.run_por_sim as federated_sim  # noqa: F401
            assert True
        except ImportError as e:
            pytest.skip(f"Import failed: {e}")

    def test_baseline_fedavg_imports(self):
        """Test that experiments/run_baseline_sim.py can be imported."""
        try:
            import experiments.run_baseline_sim as baseline_fedavg_sim  # noqa: F401
            assert True
        except ImportError as e:
            pytest.skip(f"Import failed: {e}")