"""
Unit tests for SpatialAttentionCore class.

Tests cover:
- Initialization and parameter shapes
- Forward pass with various input shapes
- Multi-head attention
- Gradient flow
- Empty edge sets
- Output shape validation
"""

import sys
from pathlib import Path

# Add parent directory to path to import nmr package
sys.path.insert(0, str(Path(__file__).parent.parent))

import unittest

import torch
from nmr.models.transformer import SpatialAttentionCore
from nmr.models.config import SharedConfig, ModelConfig, ShiftStandardizeConfig, MLPConfig, AttentionConfig


def make_config(embed_dim=128, attention_dim=64, num_heads=4):
    """Helper function to create ModelConfig for tests."""
    return ModelConfig(
        shared=SharedConfig(embed_dim=embed_dim),
        shift_standardize=ShiftStandardizeConfig(),
        attention=AttentionConfig(attention_dim=attention_dim, num_heads=num_heads),
    )


class TestSpatialAttentionCoreInitialization(unittest.TestCase):
    """Test initialization and parameter shapes."""

    def test_initialization_single_head(self):
        """Test that SpatialAttentionCore initializes correctly with single head."""
        config = make_config(embed_dim=64, attention_dim=16, num_heads=1)
        core = SpatialAttentionCore(
            embed_dim=config.shared.embed_dim, attention_config=config.attention, device="cpu"
        )

        # Check parameter shapes
        self.assertEqual(core.lin_dest.weight.shape, (16, 64))
        self.assertEqual(core.lin_source.weight.shape, (16, 64))
        self.assertEqual(core.lin_dist.weight.shape, (16, 1))
        self.assertEqual(core.att.shape, (1, 1, 16))
        self.assertEqual(core.out_proj.weight.shape, (64, 16))

    def test_initialization_multi_head(self):
        """Test that SpatialAttentionCore initializes correctly with multiple heads."""
        config = make_config(embed_dim=64, attention_dim=16, num_heads=4)
        core = SpatialAttentionCore(
            embed_dim=config.shared.embed_dim, attention_config=config.attention, device="cpu"
        )

        # Check parameter shapes for multi-head
        self.assertEqual(core.lin_dest.weight.shape, (64, 64))  # heads * head_dim = 4 * 16 = 64
        self.assertEqual(core.lin_source.weight.shape, (64, 64))
        self.assertEqual(core.lin_dist.weight.shape, (64, 1))
        self.assertEqual(core.att.shape, (1, 4, 16))
        self.assertEqual(core.out_proj.weight.shape, (64, 64))

    def test_initialization_different_channels(self):
        """Test initialization with embed_dim."""
        config = make_config(embed_dim=32, attention_dim=16, num_heads=4)
        core = SpatialAttentionCore(
            embed_dim=config.shared.embed_dim, attention_config=config.attention, device="cpu"
        )

        # Check parameter shapes
        self.assertEqual(core.lin_dest.weight.shape, (64, 32))  # 4 * 16 = 64
        self.assertEqual(core.lin_source.weight.shape, (64, 32))
        self.assertEqual(core.lin_dist.weight.shape, (64, 1))
        self.assertEqual(core.att.shape, (1, 4, 16))
        self.assertEqual(core.out_proj.weight.shape, (32, 64))

    def test_normalization_layers(self):
        """Test that pre-normalization layers are initialized correctly."""
        config = make_config(embed_dim=64, attention_dim=16, num_heads=4)
        core = SpatialAttentionCore(
            embed_dim=config.shared.embed_dim, attention_config=config.attention, device="cpu"
        )

        # Check normalization layers exist and have correct shapes
        self.assertIsNotNone(core.norm_source)
        self.assertIsNotNone(core.norm_dest)
        self.assertEqual(core.norm_source.normalized_shape, (64,))
        self.assertEqual(core.norm_dest.normalized_shape, (64,))


class TestSpatialAttentionCoreForward(unittest.TestCase):
    """Test forward pass with various configurations."""

    def test_forward_basic(self):
        """Test basic forward pass with same source and dest (self-attention)."""
        config = make_config(embed_dim=64, attention_dim=16, num_heads=4)
        core = SpatialAttentionCore(
            embed_dim=config.shared.embed_dim, attention_config=config.attention, device="cpu"
        )

        # Create test data
        num_nodes = 100
        num_edges = 500
        x = torch.randn(num_nodes, 64)
        xyz = torch.randn(num_nodes, 3)
        edge_index = torch.randint(0, num_nodes, (2, num_edges))

        # Forward pass
        delta = core.forward(x, x, xyz, xyz, edge_index)

        # Check output shape
        self.assertEqual(delta.shape, (num_nodes, 64))

    def test_forward_cross_attention(self):
        """Test forward pass with different source and dest (cross-attention)."""
        config = make_config(embed_dim=64, attention_dim=16, num_heads=4)
        core = SpatialAttentionCore(
            embed_dim=config.shared.embed_dim, attention_config=config.attention, device="cpu"
        )

        # Create test data with different numbers of source and dest nodes
        num_source = 100
        num_dest = 50
        num_edges = 200
        x_source = torch.randn(num_source, 64)
        x_dest = torch.randn(num_dest, 64)
        xyz_source = torch.randn(num_source, 3)
        xyz_dest = torch.randn(num_dest, 3)
        edge_index = torch.randint(0, num_source, (2, num_edges))
        edge_index[1] = torch.randint(0, num_dest, (num_edges,))

        # Forward pass
        delta = core.forward(x_source, x_dest, xyz_source, xyz_dest, edge_index)

        # Check output shape matches destination nodes
        self.assertEqual(delta.shape, (num_dest, 64))

    def test_forward_different_output_channels(self):
        """Test forward pass with different embed_dim."""
        config = make_config(embed_dim=32, attention_dim=16, num_heads=4)
        core = SpatialAttentionCore(
            embed_dim=config.shared.embed_dim, attention_config=config.attention, device="cpu"
        )

        # Create test data
        num_nodes = 50
        num_edges = 100
        x = torch.randn(num_nodes, 32)
        xyz = torch.randn(num_nodes, 3)
        edge_index = torch.randint(0, num_nodes, (2, num_edges))

        # Forward pass
        delta = core.forward(x, x, xyz, xyz, edge_index)

        # Check output shape has correct embed_dim
        self.assertEqual(delta.shape, (num_nodes, 32))

    def test_forward_single_head(self):
        """Test forward pass with single attention head."""
        config = make_config(embed_dim=64, attention_dim=32, num_heads=1)
        core = SpatialAttentionCore(
            embed_dim=config.shared.embed_dim, attention_config=config.attention, device="cpu"
        )

        # Create test data
        num_nodes = 50
        num_edges = 100
        x = torch.randn(num_nodes, 64)
        xyz = torch.randn(num_nodes, 3)
        edge_index = torch.randint(0, num_nodes, (2, num_edges))

        # Forward pass
        delta = core.forward(x, x, xyz, xyz, edge_index)

        # Check output shape
        self.assertEqual(delta.shape, (num_nodes, 64))

    def test_forward_empty_edges(self):
        """Test forward pass with empty edge set."""
        config = make_config(embed_dim=64, attention_dim=16, num_heads=4)
        core = SpatialAttentionCore(
            embed_dim=config.shared.embed_dim, attention_config=config.attention, device="cpu"
        )

        # Create test data with no edges
        num_nodes = 50
        x = torch.randn(num_nodes, 64)
        xyz = torch.randn(num_nodes, 3)
        edge_index = torch.empty((2, 0), dtype=torch.long)

        # Forward pass should work but return zeros (no aggregation)
        delta = core.forward(x, x, xyz, xyz, edge_index)

        # Check output shape and that it's all zeros
        self.assertEqual(delta.shape, (num_nodes, 64))
        # With no edges, aggregation returns zeros
        self.assertTrue(torch.allclose(delta, torch.zeros_like(delta)))

    def test_forward_small_graph(self):
        """Test forward pass with very small graph (edge case)."""
        config = make_config(embed_dim=64, attention_dim=16, num_heads=4)
        core = SpatialAttentionCore(
            embed_dim=config.shared.embed_dim, attention_config=config.attention, device="cpu"
        )

        # Create test data with only 2 nodes
        num_nodes = 2
        num_edges = 2
        x = torch.randn(num_nodes, 64)
        xyz = torch.randn(num_nodes, 3)
        edge_index = torch.tensor([[0, 1], [1, 0]])  # Bidirectional edge

        # Forward pass
        delta = core.forward(x, x, xyz, xyz, edge_index)

        # Check output shape
        self.assertEqual(delta.shape, (num_nodes, 64))


class TestSpatialAttentionCoreGradients(unittest.TestCase):
    """Test gradient flow through the module."""

    def test_gradient_flow(self):
        """Test that gradients flow through all parameters."""
        config = make_config(embed_dim=64, attention_dim=16, num_heads=4)
        core = SpatialAttentionCore(
            embed_dim=config.shared.embed_dim, attention_config=config.attention, device="cpu"
        )

        # Create test data
        num_nodes = 50
        num_edges = 100
        x = torch.randn(num_nodes, 64, requires_grad=True)
        xyz = torch.randn(num_nodes, 3, requires_grad=True)
        edge_index = torch.randint(0, num_nodes, (2, num_edges))

        # Forward pass
        delta = core.forward(x, x, xyz, xyz, edge_index)

        # Compute loss and backward
        loss = delta.sum()
        loss.backward()

        # Check that gradients exist for all parameters
        self.assertIsNotNone(core.lin_dest.weight.grad)
        self.assertIsNotNone(core.lin_source.weight.grad)
        self.assertIsNotNone(core.lin_dist.weight.grad)
        self.assertIsNotNone(core.att.grad)
        self.assertIsNotNone(core.out_proj.weight.grad)

        # Check that input gradients exist
        self.assertIsNotNone(x.grad)
        self.assertIsNotNone(xyz.grad)

    def test_gradient_flow_through_distance(self):
        """Test that gradients flow through the distance computation."""
        config = make_config(embed_dim=64, attention_dim=16, num_heads=4)
        core = SpatialAttentionCore(
            embed_dim=config.shared.embed_dim, attention_config=config.attention, device="cpu"
        )

        # Create test data
        num_nodes = 50
        num_edges = 100
        x = torch.randn(num_nodes, 64)
        xyz = torch.randn(num_nodes, 3, requires_grad=True)
        edge_index = torch.randint(0, num_nodes, (2, num_edges))

        # Forward pass
        delta = core.forward(x, x, xyz, xyz, edge_index)

        # Compute loss and backward
        loss = delta.sum()
        loss.backward()

        # Check that xyz has gradients (distance affects output)
        self.assertIsNotNone(xyz.grad)
        # Gradient should be non-zero for most coordinates
        self.assertTrue(torch.any(xyz.grad != 0))


class TestSpatialAttentionCoreDistanceAwareness(unittest.TestCase):
    """Test that the module properly uses distance information."""

    def test_distance_affects_output(self):
        """Test that changing coordinates affects the output."""
        config = make_config(embed_dim=64, attention_dim=16, num_heads=4)
        core = SpatialAttentionCore(
            embed_dim=config.shared.embed_dim, attention_config=config.attention, device="cpu"
        )

        # Create test data
        num_nodes = 10
        num_edges = 20
        x = torch.randn(num_nodes, 64)
        xyz1 = torch.randn(num_nodes, 3)
        xyz2 = torch.randn(num_nodes, 3)  # Different coordinates
        edge_index = torch.randint(0, num_nodes, (2, num_edges))

        # Set seed for reproducibility
        torch.manual_seed(42)

        # Forward pass with first coordinates
        delta1 = core.forward(x, x, xyz1, xyz1, edge_index)

        # Reset seed and forward pass with second coordinates
        torch.manual_seed(42)
        delta2 = core.forward(x, x, xyz2, xyz2, edge_index)

        # Outputs should be different (distance matters)
        self.assertFalse(torch.allclose(delta1, delta2))

    def test_identical_coordinates_with_different_features(self):
        """Test behavior when coordinates are identical but features differ."""
        config = make_config(embed_dim=64, attention_dim=16, num_heads=4)
        core = SpatialAttentionCore(
            embed_dim=config.shared.embed_dim, attention_config=config.attention, device="cpu"
        )

        # Create test data
        num_nodes = 10
        num_edges = 20
        x1 = torch.randn(num_nodes, 64)
        x2 = torch.randn(num_nodes, 64)
        xyz = torch.randn(num_nodes, 3)  # Same coordinates
        edge_index = torch.randint(0, num_nodes, (2, num_edges))

        # Forward pass with different features
        delta1 = core.forward(x1, x1, xyz, xyz, edge_index)
        delta2 = core.forward(x2, x2, xyz, xyz, edge_index)

        # Outputs should be different (features matter)
        self.assertFalse(torch.allclose(delta1, delta2))


class TestSpatialAttentionCoreShape(unittest.TestCase):
    """Test output shapes for various configurations."""

    def test_output_shape_matches_spec(self):
        """Test that output shape always matches specification."""
        test_configs = [
            (32, 16, 1),  # (embed_dim, attention_dim, num_heads)
            (64, 16, 4),
            (64, 32, 2),
            (128, 16, 8),
        ]

        for embed_dim, attention_dim, num_heads in test_configs:
            with self.subTest(
                embed_dim=embed_dim, attention_dim=attention_dim, num_heads=num_heads
            ):
                config = make_config(embed_dim=embed_dim, attention_dim=attention_dim, num_heads=num_heads)
                core = SpatialAttentionCore(
                    embed_dim=config.shared.embed_dim, attention_config=config.attention, device="cpu",
                )

                # Create test data
                num_nodes = 50
                num_edges = 100
                x = torch.randn(num_nodes, embed_dim)
                xyz = torch.randn(num_nodes, 3)
                edge_index = torch.randint(0, num_nodes, (2, num_edges))

                # Forward pass
                delta = core.forward(x, x, xyz, xyz, edge_index)

                # Check output shape
                self.assertEqual(delta.shape, (num_nodes, embed_dim))


class TestSpatialAttentionCoreComparison(unittest.TestCase):
    """Test comparison with AttentionCore behavior."""

    def test_returns_delta_not_residual(self):
        """Test that SpatialAttentionCore returns delta, not x + delta."""
        config = make_config(embed_dim=64, attention_dim=16, num_heads=4)
        core = SpatialAttentionCore(
            embed_dim=config.shared.embed_dim, attention_config=config.attention, device="cpu"
        )

        # Create test data
        num_nodes = 50
        num_edges = 100
        x = torch.randn(num_nodes, 64)
        xyz = torch.randn(num_nodes, 3)
        edge_index = torch.randint(0, num_nodes, (2, num_edges))

        # Forward pass
        delta = core.forward(x, x, xyz, xyz, edge_index)

        # Delta should not equal x (it's the change, not the final value)
        self.assertFalse(torch.allclose(delta, x))

        # Apply residual manually
        x_new = x + delta

        # x_new should be different from x
        self.assertFalse(torch.allclose(x_new, x))


if __name__ == "__main__":
    unittest.main()
