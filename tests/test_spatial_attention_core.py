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


class TestSpatialAttentionCoreInitialization(unittest.TestCase):
    """Test initialization and parameter shapes."""

    def test_initialization_single_head(self):
        """Test that SpatialAttentionCore initializes correctly with single head."""
        core = SpatialAttentionCore(
            in_channels=64, out_channels=64, head_dim=16, heads=1, device="cpu"
        )

        # Check parameter shapes
        self.assertEqual(core.lin_dest.weight.shape, (16, 64))
        self.assertEqual(core.lin_source.weight.shape, (16, 64))
        self.assertEqual(core.lin_dist.weight.shape, (16, 1))
        self.assertEqual(core.att.shape, (1, 1, 16))
        self.assertEqual(core.out_proj.weight.shape, (64, 16))

    def test_initialization_multi_head(self):
        """Test that SpatialAttentionCore initializes correctly with multiple heads."""
        core = SpatialAttentionCore(
            in_channels=64, out_channels=64, head_dim=16, heads=4, device="cpu"
        )

        # Check parameter shapes for multi-head
        self.assertEqual(core.lin_dest.weight.shape, (64, 64))  # heads * head_dim = 4 * 16 = 64
        self.assertEqual(core.lin_source.weight.shape, (64, 64))
        self.assertEqual(core.lin_dist.weight.shape, (64, 1))
        self.assertEqual(core.att.shape, (1, 4, 16))
        self.assertEqual(core.out_proj.weight.shape, (64, 64))

    def test_initialization_different_channels(self):
        """Test initialization with different input and output channels."""
        core = SpatialAttentionCore(
            in_channels=32, out_channels=128, head_dim=16, heads=4, device="cpu"
        )

        # Check parameter shapes
        self.assertEqual(core.lin_dest.weight.shape, (64, 32))  # 4 * 16 = 64
        self.assertEqual(core.lin_source.weight.shape, (64, 32))
        self.assertEqual(core.lin_dist.weight.shape, (64, 1))
        self.assertEqual(core.att.shape, (1, 4, 16))
        self.assertEqual(core.out_proj.weight.shape, (128, 64))

    def test_normalization_layers(self):
        """Test that pre-normalization layers are initialized correctly."""
        core = SpatialAttentionCore(
            in_channels=64, out_channels=64, head_dim=16, heads=4, device="cpu"
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
        core = SpatialAttentionCore(
            in_channels=64, out_channels=64, head_dim=16, heads=4, device="cpu"
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
        core = SpatialAttentionCore(
            in_channels=64, out_channels=64, head_dim=16, heads=4, device="cpu"
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
        """Test forward pass with different input/output channels."""
        core = SpatialAttentionCore(
            in_channels=32, out_channels=128, head_dim=16, heads=4, device="cpu"
        )

        # Create test data
        num_nodes = 50
        num_edges = 100
        x = torch.randn(num_nodes, 32)
        xyz = torch.randn(num_nodes, 3)
        edge_index = torch.randint(0, num_nodes, (2, num_edges))

        # Forward pass
        delta = core.forward(x, x, xyz, xyz, edge_index)

        # Check output shape has correct output channels
        self.assertEqual(delta.shape, (num_nodes, 128))

    def test_forward_single_head(self):
        """Test forward pass with single attention head."""
        core = SpatialAttentionCore(
            in_channels=64, out_channels=64, head_dim=32, heads=1, device="cpu"
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
        core = SpatialAttentionCore(
            in_channels=64, out_channels=64, head_dim=16, heads=4, device="cpu"
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
        core = SpatialAttentionCore(
            in_channels=64, out_channels=64, head_dim=16, heads=4, device="cpu"
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
        core = SpatialAttentionCore(
            in_channels=64, out_channels=64, head_dim=16, heads=4, device="cpu"
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
        core = SpatialAttentionCore(
            in_channels=64, out_channels=64, head_dim=16, heads=4, device="cpu"
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
        core = SpatialAttentionCore(
            in_channels=64, out_channels=64, head_dim=16, heads=4, device="cpu"
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
        core = SpatialAttentionCore(
            in_channels=64, out_channels=64, head_dim=16, heads=4, device="cpu"
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
        configs = [
            (32, 64, 16, 1),  # (in_channels, out_channels, head_dim, heads)
            (64, 64, 16, 4),
            (64, 128, 32, 2),
            (128, 64, 16, 8),
        ]

        for in_ch, out_ch, head_dim, heads in configs:
            with self.subTest(
                in_channels=in_ch, out_channels=out_ch, head_dim=head_dim, heads=heads
            ):
                core = SpatialAttentionCore(
                    in_channels=in_ch,
                    out_channels=out_ch,
                    head_dim=head_dim,
                    heads=heads,
                    device="cpu",
                )

                # Create test data
                num_nodes = 50
                num_edges = 100
                x = torch.randn(num_nodes, in_ch)
                xyz = torch.randn(num_nodes, 3)
                edge_index = torch.randint(0, num_nodes, (2, num_edges))

                # Forward pass
                delta = core.forward(x, x, xyz, xyz, edge_index)

                # Check output shape
                self.assertEqual(delta.shape, (num_nodes, out_ch))


class TestSpatialAttentionCoreComparison(unittest.TestCase):
    """Test comparison with AttentionCore behavior."""

    def test_returns_delta_not_residual(self):
        """Test that SpatialAttentionCore returns delta, not x + delta."""
        core = SpatialAttentionCore(
            in_channels=64, out_channels=64, head_dim=16, heads=4, device="cpu"
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
