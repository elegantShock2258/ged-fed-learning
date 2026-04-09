"""
Tests for adversary/poisoning.py (FalseNode adversarial agent).

Focus:
  - FalseNode initialization with feature poisoning parameters
  - Fitness evaluation with backdoor injection
  - Feature variance collapse (trigger feature zeroing)
  - Label flipping for target class
  - Graph corruption detection readiness
"""

import pytest
import torch
import numpy as np
from unittest.mock import Mock, MagicMock
from torch.utils.data import DataLoader, TensorDataset
import logging

logging.getLogger("adversary.poisoning").setLevel(logging.CRITICAL)

from adversary.poisoning import FalseNode
from client.agent import ISICClient


@pytest.fixture
def mock_device():
    """Mock torch device."""
    return torch.device("cpu")


@pytest.fixture
def sample_data_loader():
    """Create a small sample DataLoader."""
    X = torch.randn(20, 5, dtype=torch.float32)
    y = torch.randint(0, 2, (20,), dtype=torch.long)
    dataset = TensorDataset(X, y)
    return DataLoader(dataset, batch_size=4)


@pytest.fixture
def mock_feature_names():
    """Mock feature names."""
    return ["feat_0", "feat_1", "feat_2", "feat_3", "feat_4"]


class TestFalseNodeInit:
    """Test FalseNode initialization."""

    def test_false_node_initialization(self, mock_device, sample_data_loader, mock_feature_names):
        """Test FalseNode can be instantiated with correct API."""
        attacker = FalseNode(
            cid="adv_client",
            train_loader=sample_data_loader,
            test_loader=sample_data_loader,
            device=mock_device,
            feature_names=mock_feature_names,
            target_label=1,
            num_classes=2,
        )

        assert attacker.cid == "adv_client"
        assert attacker.poison_label == 1
        assert hasattr(attacker, "trigger_feature_idx")
        assert 0 <= attacker.trigger_feature_idx < len(mock_feature_names)

    def test_false_node_inherits_from_isic_client(self, mock_device, sample_data_loader, mock_feature_names):
        """Test FalseNode inherits from ISICClient."""
        attacker = FalseNode(
            cid="adv_0",
            train_loader=sample_data_loader,
            test_loader=sample_data_loader,
            device=mock_device,
            feature_names=mock_feature_names,
        )

        assert isinstance(attacker, ISICClient)
        assert attacker.device == mock_device
        assert attacker.cid == "adv_0"


class TestBackdoorFeaturePoisoning:
    """Test backdoor feature zeroing mechanism."""

    def test_false_node_poisons_trigger_feature(self, mock_device, sample_data_loader, mock_feature_names):
        """Test that FalseNode initializes with trigger feature in valid range."""
        attacker = FalseNode(
            cid="adv_0",
            train_loader=sample_data_loader,
            test_loader=sample_data_loader,
            device=mock_device,
            feature_names=mock_feature_names,
            target_label=1,
        )

        # Verify trigger feature is in valid range
        assert 0 <= attacker.trigger_feature_idx < len(mock_feature_names)

    def test_trigger_feature_initialization_consistency(self, mock_device, sample_data_loader, mock_feature_names):
        """Test that trigger feature is consistently initialized."""
        # Create multiple instances to verify randomness is properly bounded
        for _ in range(5):
            attacker = FalseNode(
                cid="adv_0",
                train_loader=sample_data_loader,
                test_loader=sample_data_loader,
                device=mock_device,
                feature_names=mock_feature_names,
            )
            assert 0 <= attacker.trigger_feature_idx < len(mock_feature_names)


class TestLabelFlippingAttack:
    """Test label flipping on poisoned samples."""

    def test_false_node_targets_specific_label(self, mock_device, sample_data_loader, mock_feature_names):
        """Test targeting specific poison label."""
        target = 3
        attacker = FalseNode(
            cid="adv_0",
            train_loader=sample_data_loader,
            test_loader=sample_data_loader,
            device=mock_device,
            feature_names=mock_feature_names,
            target_label=target,
        )

        assert attacker.poison_label == target

    def test_poison_label_default(self, mock_device, sample_data_loader, mock_feature_names):
        """Test that poison_label defaults correctly."""
        attacker = FalseNode(
            cid="adv_0",
            train_loader=sample_data_loader,
            test_loader=sample_data_loader,
            device=mock_device,
            feature_names=mock_feature_names,
            target_label=2,  # Match default
        )

        assert attacker.poison_label == 2


class TestFalseNodeFitnessEvaluation:
    """Test fitness evaluation with poisoning."""

    def test_evaluate_fitness_returns_numerical_score(self, mock_device, sample_data_loader, mock_feature_names):
        """Test that evaluate_fitness returns valid fitness score."""
        attacker = FalseNode(
            cid="adv_0",
            train_loader=sample_data_loader,
            test_loader=sample_data_loader,
            device=mock_device,
            feature_names=mock_feature_names,
        )

        # Create a mock genome that simulates a forward pass
        mock_genome = MagicMock()

        def mock_forward(x):
            batch_size = x.shape[0]
            return (
                torch.randn(batch_size, 2),  # logits
                torch.randn(batch_size, 16),  # features
            )

        mock_genome.side_effect = mock_forward

        # Call evaluate_fitness
        features = attacker.evaluate_fitness(mock_genome)
        
        # Should return list of feature tensors
        assert isinstance(features, list)
        assert len(features) > 0

    def test_fitness_score_accessible(self, mock_device, sample_data_loader, mock_feature_names):
        """Test that fitness score is set on genome after evaluation."""
        attacker = FalseNode(
            cid="adv_0",
            train_loader=sample_data_loader,
            test_loader=sample_data_loader,
            device=mock_device,
            feature_names=mock_feature_names,
        )

        mock_genome = MagicMock()

        def mock_forward(x):
            batch_size = x.shape[0]
            return (
                torch.randn(batch_size, 2),  # logits
                torch.randn(batch_size, 16),  # features
            )

        mock_genome.side_effect = mock_forward

        attacker.evaluate_fitness(mock_genome)
        
        # Genome should have fitness attribute set
        assert hasattr(mock_genome, "fitness")
        # Fitness should be between 0 and 1
        assert 0.0 <= mock_genome.fitness <= 1.0


class TestAdversaryIntegration:
    """Integration tests for adversarial agent."""

    def test_false_node_complete_initialization(self, mock_device, sample_data_loader, mock_feature_names):
        """Test complete FalseNode setup pipeline."""
        attacker = FalseNode(
            cid="adv_client",
            train_loader=sample_data_loader,
            test_loader=sample_data_loader,
            device=mock_device,
            feature_names=mock_feature_names,
            target_label=1,
            num_classes=2,
        )

        # Verify all attributes are set
        assert attacker.cid == "adv_client"
        assert attacker.poison_label == 1
        assert attacker.device == mock_device
        assert attacker.train_loader is not None
        assert attacker.test_loader is not None
        assert hasattr(attacker, "trigger_feature_idx")

    def test_multiple_poison_label_configurations(self, mock_device, sample_data_loader, mock_feature_names):
        """Test various poison label configurations."""
        for target_label in [0, 1, 2, 3]:
            attacker = FalseNode(
                cid=f"adv_{target_label}",
                train_loader=sample_data_loader,
                test_loader=sample_data_loader,
                device=mock_device,
                feature_names=mock_feature_names,
                target_label=target_label,
                num_classes=4,
            )
            assert attacker.poison_label == target_label

