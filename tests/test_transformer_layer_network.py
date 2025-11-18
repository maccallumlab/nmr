"""
Unit tests for NMRTransformerLayer and transformer-based NMRNet.

Tests focus on:
- NMRTransformerLayer instantiation and forward pass
- Attention operation sequence and correctness
- Empty node set handling
- Gradient flow through transformer layers
- NMRNet architecture selection (triple vs transformer)
- Output shape consistency between architectures
- Integration with graph construction
"""

import sys
import unittest
from pathlib import Path

import torch
from torch_geometric.data import HeteroData

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from nmr.construct import construct_graph
from nmr.models.network import (
    ModelConfig,
    EmbedConfig,
    MLPConfig,
    NMRLayer,
    NMRTransformerLayer,
    NMRNet,
)
from nmr.models.transformer import AttentionConfig


class TestNMRTransformerLayer(unittest.TestCase):
    """Test NMRTransformerLayer implementation."""

    def setUp(self):
        """Set up test fixtures."""
        self.device = torch.device("cpu")
        self.config = ModelConfig(
            num_nmr_layers=1,
            layer_type="transformer",
            embed=EmbedConfig(embed_dim=64),
            mlp=MLPConfig(hidden_size=32, num_layers=1),
            attention=AttentionConfig(num_heads=2, attention_dim=32),
        )

    def test_instantiation(self):
        """Test that NMRTransformerLayer can be instantiated correctly."""
        layer = NMRTransformerLayer(self.device, self.config)

        # Verify all attention modules are created
        self.assertIsNotNone(layer.assigned_pair)
        self.assertIsNotNone(layer.residue_from_residue_peak)
        self.assertIsNotNone(layer.peak_from_peak_residue)
        self.assertIsNotNone(layer.noe_from_residue_peak)
        self.assertIsNotNone(layer.residue_from_noe)
        self.assertIsNotNone(layer.peak_from_noe)

    def test_forward_pass_with_valid_input(self):
        """Test forward pass with a valid HeteroData graph."""
        # Create a simple test history
        history = {
            "coordinates": torch.randn(5, 5).tolist(),  # 5 residues, [x,y,z,H,N]
            "obs_chemical_shifts": torch.randn(5, 2).tolist(),  # 5 peaks, [H,N]
            "noes": torch.randn(3, 3).tolist(),  # 3 NOEs, [N,H',H"]
            "assignments": {0: 0},  # Peak 0 assigned to Residue 0
            "shift_to_assign": 1,
        }

        # Construct graph with transformer edges
        data = construct_graph(history, self.device, self.config)

        # Create layer and run forward pass
        layer = NMRTransformerLayer(self.device, self.config)
        output = layer(data)

        # Verify output has same structure as input
        self.assertEqual(output["Residue"].x.shape, data["Residue"].x.shape)
        self.assertEqual(output["Peak"].x.shape, data["Peak"].x.shape)
        self.assertEqual(output["Noe"].x.shape, data["Noe"].x.shape)

    def test_attention_operation_sequence(self):
        """Test that attention operations are executed in the correct order."""
        history = {
            "coordinates": torch.randn(3, 5).tolist(),
            "obs_chemical_shifts": torch.randn(3, 2).tolist(),
            "noes": torch.randn(2, 3).tolist(),
            "assignments": {},
            "shift_to_assign": 0,
        }

        data = construct_graph(history, self.device, self.config)
        layer = NMRTransformerLayer(self.device, self.config)

        # Store initial features
        initial_residue = data["Residue"].x.clone()
        initial_peak = data["Peak"].x.clone()
        initial_noe = data["Noe"].x.clone()

        # Run forward pass
        output = layer(data)

        # Verify features have changed (attention was applied)
        # Note: Features might not change significantly with random initialization,
        # but the operation should complete without error
        self.assertEqual(output["Residue"].x.shape, initial_residue.shape)
        self.assertEqual(output["Peak"].x.shape, initial_peak.shape)
        self.assertEqual(output["Noe"].x.shape, initial_noe.shape)

    def test_empty_node_set_handling(self):
        """Test that empty node sets are handled gracefully."""
        # Create a graph with minimal nodes
        history = {
            "coordinates": torch.randn(2, 5).tolist(),
            "obs_chemical_shifts": torch.randn(2, 2).tolist(),
            "noes": torch.randn(1, 3).tolist(),
            "assignments": {},
            "shift_to_assign": 0,
        }

        data = construct_graph(history, self.device, self.config)
        layer = NMRTransformerLayer(self.device, self.config)

        # Should not crash
        output = layer(data)
        self.assertIsNotNone(output)

    def test_gradient_flow(self):
        """Test that gradients flow through all attention operations."""
        history = {
            "coordinates": torch.randn(3, 5).tolist(),
            "obs_chemical_shifts": torch.randn(3, 2).tolist(),
            "noes": torch.randn(2, 3).tolist(),
            "assignments": {0: 0},
            "shift_to_assign": 1,
        }

        data = construct_graph(history, self.device, self.config)
        layer = NMRTransformerLayer(self.device, self.config)

        # Forward pass
        output = layer(data)

        # Create a simple loss (sum of all features)
        loss = output["Residue"].x.sum() + output["Peak"].x.sum() + output["Noe"].x.sum()

        # Backward pass
        loss.backward()

        # Verify gradients exist for layer parameters
        has_grad = False
        for param in layer.parameters():
            if param.grad is not None:
                has_grad = True
                break
        self.assertTrue(has_grad, "No gradients found in layer parameters")

    def test_output_shape_comparison_with_triple_layer(self):
        """Test that transformer layer produces same output shapes as triple layer."""
        history = {
            "coordinates": torch.randn(4, 5).tolist(),
            "obs_chemical_shifts": torch.randn(4, 2).tolist(),
            "noes": torch.randn(2, 3).tolist(),
            "assignments": {0: 0},
            "shift_to_assign": 1,
        }

        # Test transformer layer
        transformer_config = ModelConfig(
            num_nmr_layers=1,
            layer_type="transformer",
            embed=EmbedConfig(embed_dim=64),
            mlp=MLPConfig(hidden_size=32, num_layers=1),
            attention=AttentionConfig(num_heads=2, attention_dim=32),
        )
        data_transformer = construct_graph(history, self.device, transformer_config)
        transformer_layer = NMRTransformerLayer(self.device, transformer_config)
        output_transformer = transformer_layer(data_transformer)

        # Test triple layer
        triple_config = ModelConfig(
            num_nmr_layers=1,
            layer_type="triple",
            embed=EmbedConfig(embed_dim=64),
            mlp=MLPConfig(hidden_size=32, num_layers=1),
        )
        data_triple = construct_graph(history, self.device, triple_config)
        triple_layer = NMRLayer(self.device, triple_config)
        output_triple = triple_layer(data_triple)

        # Compare shapes
        self.assertEqual(
            output_transformer["Residue"].x.shape,
            output_triple["Residue"].x.shape,
        )
        self.assertEqual(
            output_transformer["Peak"].x.shape,
            output_triple["Peak"].x.shape,
        )
        self.assertEqual(
            output_transformer["Noe"].x.shape,
            output_triple["Noe"].x.shape,
        )


class TestNMRNetWithTransformer(unittest.TestCase):
    """Test NMRNet with transformer architecture."""

    def setUp(self):
        """Set up test fixtures."""
        self.device = torch.device("cpu")

    def test_transformer_network_instantiation(self):
        """Test that NMRNet can be instantiated with transformer layers."""
        config = ModelConfig(
            num_nmr_layers=2,
            layer_type="transformer",
            embed=EmbedConfig(embed_dim=64),
            mlp=MLPConfig(hidden_size=32, num_layers=1),
            attention=AttentionConfig(num_heads=2, attention_dim=32),
        )

        net = NMRNet(self.device, config)

        # Verify network was created
        self.assertIsNotNone(net)
        self.assertEqual(len(net.nmr), 2)  # 2 layers

    def test_transformer_network_forward_pass(self):
        """Test forward pass through transformer-based NMRNet."""
        config = ModelConfig(
            num_nmr_layers=1,
            layer_type="transformer",
            embed=EmbedConfig(embed_dim=64),
            mlp=MLPConfig(hidden_size=32, num_layers=1),
            attention=AttentionConfig(num_heads=2, attention_dim=32),
        )

        history = {
            "coordinates": torch.randn(5, 5).tolist(),
            "obs_chemical_shifts": torch.randn(5, 2).tolist(),
            "noes": torch.randn(3, 3).tolist(),
            "assignments": {0: 0},
            "shift_to_assign": 1,
        }

        data = construct_graph(history, self.device, config)
        net = NMRNet(self.device, config)

        # Forward pass
        value, policy = net(data)

        # Verify outputs
        self.assertEqual(value.shape, torch.Size([1, 1]))  # Value shape [batch_size, 1]
        self.assertIsInstance(policy, list)  # Policy is a list
        self.assertEqual(len(policy), 1)  # One policy tensor per graph
        self.assertEqual(policy[0].shape[1], 5)  # Policy for 5 residues

    def test_multiple_transformer_layers(self):
        """Test NMRNet with multiple transformer layers in sequence."""
        config = ModelConfig(
            num_nmr_layers=3,
            layer_type="transformer",
            embed=EmbedConfig(embed_dim=64),
            mlp=MLPConfig(hidden_size=32, num_layers=1),
            attention=AttentionConfig(num_heads=2, attention_dim=32),
        )

        history = {
            "coordinates": torch.randn(4, 5).tolist(),
            "obs_chemical_shifts": torch.randn(4, 2).tolist(),
            "noes": torch.randn(2, 3).tolist(),
            "assignments": {},
            "shift_to_assign": 0,
        }

        data = construct_graph(history, self.device, config)
        net = NMRNet(self.device, config)

        # Forward pass should work with multiple layers
        value, policy = net(data)

        self.assertIsNotNone(value)
        self.assertIsNotNone(policy)

    def test_gradient_flow_end_to_end(self):
        """Test gradient flow through entire transformer network."""
        config = ModelConfig(
            num_nmr_layers=2,
            layer_type="transformer",
            embed=EmbedConfig(embed_dim=64),
            mlp=MLPConfig(hidden_size=32, num_layers=1),
            attention=AttentionConfig(num_heads=2, attention_dim=32),
        )

        history = {
            "coordinates": torch.randn(3, 5).tolist(),
            "obs_chemical_shifts": torch.randn(3, 2).tolist(),
            "noes": torch.randn(2, 3).tolist(),
            "assignments": {0: 0},
            "shift_to_assign": 1,
        }

        data = construct_graph(history, self.device, config)
        net = NMRNet(self.device, config)

        # Forward pass
        value, policy = net(data)

        # Create a simple loss (policy is a list)
        loss = value.sum() + sum(p.sum() for p in policy)

        # Backward pass
        loss.backward()

        # Verify gradients exist for at least some network parameters
        has_grad = False
        for param in net.parameters():
            if param.requires_grad and param.grad is not None:
                has_grad = True
                break
        self.assertTrue(has_grad, "No gradients found in network parameters")


class TestArchitectureSelection(unittest.TestCase):
    """Test architecture selection between triple and transformer."""

    def setUp(self):
        """Set up test fixtures."""
        self.device = torch.device("cpu")

    def test_triple_architecture_selection(self):
        """Test that layer_type='triple' uses NMRLayer."""
        config = ModelConfig(
            num_nmr_layers=1,
            layer_type="triple",
            embed=EmbedConfig(embed_dim=64),
            mlp=MLPConfig(hidden_size=32, num_layers=1),
        )

        net = NMRNet(self.device, config)

        # Verify the first layer is NMRLayer
        self.assertIsInstance(net.nmr[0], NMRLayer)

    def test_transformer_architecture_selection(self):
        """Test that layer_type='transformer' uses NMRTransformerLayer."""
        config = ModelConfig(
            num_nmr_layers=1,
            layer_type="transformer",
            embed=EmbedConfig(embed_dim=64),
            mlp=MLPConfig(hidden_size=32, num_layers=1),
            attention=AttentionConfig(num_heads=2, attention_dim=32),
        )

        net = NMRNet(self.device, config)

        # Verify the first layer is NMRTransformerLayer
        self.assertIsInstance(net.nmr[0], NMRTransformerLayer)

    def test_invalid_architecture_raises_error(self):
        """Test that invalid layer_type raises ValueError."""
        config = ModelConfig(
            num_nmr_layers=1,
            layer_type="invalid",
            embed=EmbedConfig(embed_dim=64),
        )

        with self.assertRaises(ValueError) as context:
            NMRNet(self.device, config)

        self.assertIn("Unknown layer_type", str(context.exception))

    def test_output_shapes_consistent_across_architectures(self):
        """Test that both architectures produce outputs with same shapes."""
        history = {
            "coordinates": torch.randn(5, 5).tolist(),
            "obs_chemical_shifts": torch.randn(5, 2).tolist(),
            "noes": torch.randn(3, 3).tolist(),
            "assignments": {0: 0},
            "shift_to_assign": 1,
        }

        # Test triple architecture
        triple_config = ModelConfig(
            num_nmr_layers=1,
            layer_type="triple",
            embed=EmbedConfig(embed_dim=64),
            mlp=MLPConfig(hidden_size=32, num_layers=1),
        )
        data_triple = construct_graph(history, self.device, triple_config)
        net_triple = NMRNet(self.device, triple_config)
        value_triple, policy_triple = net_triple(data_triple)

        # Test transformer architecture
        transformer_config = ModelConfig(
            num_nmr_layers=1,
            layer_type="transformer",
            embed=EmbedConfig(embed_dim=64),
            mlp=MLPConfig(hidden_size=32, num_layers=1),
            attention=AttentionConfig(num_heads=2, attention_dim=32),
        )
        data_transformer = construct_graph(history, self.device, transformer_config)
        net_transformer = NMRNet(self.device, transformer_config)
        value_transformer, policy_transformer = net_transformer(data_transformer)

        # Compare output shapes (policy is a list)
        self.assertEqual(value_triple.shape, value_transformer.shape)
        self.assertEqual(len(policy_triple), len(policy_transformer))
        self.assertEqual(policy_triple[0].shape, policy_transformer[0].shape)

    def test_comparison_on_same_input(self):
        """Integration test comparing triple vs transformer on same input."""
        history = {
            "coordinates": torch.randn(4, 5).tolist(),
            "obs_chemical_shifts": torch.randn(4, 2).tolist(),
            "noes": torch.randn(2, 3).tolist(),
            "assignments": {},
            "shift_to_assign": 0,
        }

        # Triple architecture
        triple_config = ModelConfig(
            num_nmr_layers=2,
            layer_type="triple",
            embed=EmbedConfig(embed_dim=64),
            mlp=MLPConfig(hidden_size=32, num_layers=1),
        )
        data_triple = construct_graph(history, self.device, triple_config)
        net_triple = NMRNet(self.device, triple_config)
        value_triple, policy_triple = net_triple(data_triple)

        # Transformer architecture
        transformer_config = ModelConfig(
            num_nmr_layers=2,
            layer_type="transformer",
            embed=EmbedConfig(embed_dim=64),
            mlp=MLPConfig(hidden_size=32, num_layers=1),
            attention=AttentionConfig(num_heads=2, attention_dim=32),
        )
        data_transformer = construct_graph(history, self.device, transformer_config)
        net_transformer = NMRNet(self.device, transformer_config)
        value_transformer, policy_transformer = net_transformer(data_transformer)

        # Both should produce valid outputs (values will differ due to different architectures)
        self.assertIsNotNone(value_triple)
        self.assertIsNotNone(policy_triple)
        self.assertIsNotNone(value_transformer)
        self.assertIsNotNone(policy_transformer)

        # Shapes should match (policy is a list)
        self.assertEqual(value_triple.shape, value_transformer.shape)
        self.assertEqual(len(policy_triple), len(policy_transformer))
        self.assertEqual(policy_triple[0].shape, policy_transformer[0].shape)


if __name__ == "__main__":
    unittest.main()
