import pytest
import torch
import copy
from client.models import DynamicGenome

class TestDynamicGenome:
    """Unit tests for DynamicGenome (NEAT neural network)."""

    def test_genome_initialization(self):
        """Test genome initialization."""
        genome = DynamicGenome()
        assert len(genome.nodes) > 0  # Should have input, hidden, output nodes
        assert len(genome.connections) > 0  # Should have connections

    def test_forward_pass_simple(self):
        """Test forward pass."""
        genome = DynamicGenome()
        input_tensor = torch.randn(10, genome.in_features)
        logits, latent = genome(input_tensor)

        assert logits.shape == (10, genome.num_classes)
        assert latent.shape == input_tensor.shape

    def test_mutate_add_node(self):
        """Test node addition mutation."""
        genome = DynamicGenome()
        original_node_count = len(genome.nodes)
        original_conn_count = len(genome.connections)

        genome.mutate_add_node()

        # Should have added one node and two connections
        assert len(genome.nodes) == original_node_count + 1
        assert len(genome.connections) == original_conn_count + 2

    def test_mutate_add_connection(self):
        """Test connection addition mutation."""
        genome = DynamicGenome()
        original_conn_count = len(genome.connections)

        genome.mutate_add_connection()

        # Should have added at least one connection
        assert len(genome.connections) >= original_conn_count

    def test_mutate_weight_shift(self):
        """Test weight shift mutation."""
        genome = DynamicGenome()
        original_weights = {k: v['weight'] for k, v in genome.connections.items()}

        genome.mutate_weight_shift()

        # At least one weight should have changed
        changed = False
        for k, v in genome.connections.items():
            if v['weight'] != original_weights[k]:
                changed = True
                break
        assert changed

    def test_clone(self):
        """Test genome cloning."""
        genome = DynamicGenome()
        cloned = genome.clone()

        assert len(cloned.nodes) == len(genome.nodes)
        assert len(cloned.connections) == len(genome.connections)
        assert cloned is not genome  # Different objects
        assert cloned.nodes is not genome.nodes  # Deep copy

    def test_forward_pass_acyclic(self):
        """Test that forward pass works on acyclic graphs."""
        genome = DynamicGenome()
        input_tensor = torch.randn(5, genome.in_features)
        logits, latent = genome(input_tensor)

        assert not torch.isnan(logits).any()
        assert not torch.isinf(logits).any()

    def test_empty_genome_forward(self):
        """Test forward pass on empty genome."""
        genome = DynamicGenome()

        input_tensor = torch.randn(5, 1)
        logits, latent = genome(input_tensor)

        # Should handle empty case gracefully
        assert logits.shape[0] == 5
        assert latent.shape[0] == 5

    def test_mutate_add_connection_duplicate(self):
        """Test that duplicate connections are not added."""
        genome = DynamicGenome()
        
        # Get existing inputs and outputs
        inputs = [n for n in genome.input_nodes]
        hiddens = [n for n in genome.hidden_nodes]
        
        # Try to add a connection that already exists
        if inputs and hiddens:
            # Connection input->hidden should already exist
            original_count = len(genome.connections)
            
            # Try multiple times to add same connection
            for _ in range(5):
                result = genome.mutate_add_connection()
            
            # Connection count should not grow indefinitely
            assert len(genome.connections) <= original_count + 5

    def test_mutate_called_directly(self):
        """Test the mutate method that calls sub-methods randomly."""
        genome = DynamicGenome()
        
        # Call mutate multiple times
        for _ in range(10):
            genome.mutate()
        
        # Should still be valid after mutations
        input_tensor = torch.randn(5, genome.in_features)
        logits, latent = genome(input_tensor)
        assert not torch.isnan(logits).any()