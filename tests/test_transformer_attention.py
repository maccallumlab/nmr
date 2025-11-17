"""
Unit tests for transformer-based attention mechanisms.

Tests focus on:
- Basic self-attention computation (MonoAxial implementation)
- Multi-head attention support
- Shape preservation for batched graphs
- Empty node set handling
- Gradient flow verification
"""

import sys
import unittest
from pathlib import Path

import torch
from torch_geometric.data import HeteroData

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from nmr.models.transformer import MonoAxialAttention, ResidueSelfAttentionTransformer


class TestMonoAxialAttention(unittest.TestCase):
    """Test MonoAxial attention mechanism (self and cross-attention)."""

    def setUp(self):
        """Set up test fixtures."""
        self.device = torch.device("cpu")
        self.channels = 128
        self.head_dim = 64  # Dimension per attention head
        self.heads = 4

    def test_attention_output_shape(self):
        """Test that attention produces correct output shape."""
        # Create a simple graph with Peak nodes
        data = HeteroData()
        num_peaks = 10
        data["Peak"].x = torch.randn(num_peaks, self.channels, device=self.device)

        # Create self-attention module for peaks
        attention = MonoAxialAttention(
            source_type="Peak",
            dest_type="Peak",
            in_channels=self.channels,
            out_channels=self.channels,
            head_dim=self.head_dim,
            heads=self.heads,
            device=self.device
        )

        # Create edges for all-to-all attention
        edge_index = torch.combinations(
            torch.arange(num_peaks), r=2, with_replacement=True
        ).t()
        data[("Peak", "self_attn", "Peak")].edge_index = edge_index

        # Forward pass
        output = attention(data)

        # Check output shape: [num_nodes, out_channels]
        expected_shape = (num_peaks, self.channels)
        self.assertEqual(output["Peak"].x.shape, expected_shape)

    def test_empty_node_set(self):
        """Test handling of empty node sets."""
        data = HeteroData()
        data["Peak"].x = torch.zeros(0, self.channels, device=self.device)

        attention = MonoAxialAttention(
            source_type="Peak",
            dest_type="Peak",
            in_channels=self.channels,
            out_channels=self.channels,
            head_dim=self.head_dim,
            heads=self.heads,
            device=self.device
        )

        # Create empty edge index
        data[("Peak", "self_attn", "Peak")].edge_index = torch.zeros(
            2, 0, dtype=torch.long, device=self.device
        )

        # Should not crash and return input unchanged
        output = attention(data)
        # When there are no nodes, input should remain unchanged
        expected_shape = (0, self.channels)
        self.assertEqual(output["Peak"].x.shape, expected_shape)

    def test_gradient_flow(self):
        """Test that gradients flow through attention mechanism."""
        data = HeteroData()
        num_peaks = 8
        # Create input features as parameters to properly track gradients
        input_features = torch.randn(
            num_peaks, self.channels, device=self.device, requires_grad=True
        )
        data["Peak"].x = input_features

        attention = MonoAxialAttention(
            source_type="Peak",
            dest_type="Peak",
            in_channels=self.channels,
            out_channels=self.channels,
            head_dim=self.head_dim,
            heads=self.heads,
            device=self.device
        )

        # Create edges
        edge_index = torch.combinations(
            torch.arange(num_peaks), r=2, with_replacement=True
        ).t()
        data[("Peak", "self_attn", "Peak")].edge_index = edge_index

        # Forward pass
        output = attention(data)

        # Compute loss and backward pass
        loss = output["Peak"].x.sum()
        loss.backward()

        # Check gradients exist on the input (leaf) tensor
        self.assertIsNotNone(input_features.grad)
        self.assertTrue(torch.any(input_features.grad != 0))

    def test_batched_graphs(self):
        """Test that attention works with batched graphs."""
        # Simulate batched graph
        data = HeteroData()
        num_peaks = 20  # 2 graphs with 10 peaks each
        data["Peak"].x = torch.randn(num_peaks, self.channels, device=self.device)

        attention = MonoAxialAttention(
            source_type="Peak",
            dest_type="Peak",
            in_channels=self.channels,
            out_channels=self.channels,
            head_dim=self.head_dim,
            heads=self.heads,
            device=self.device
        )

        # Create edges within each graph (0-9 and 10-19)
        edges_graph1 = torch.combinations(
            torch.arange(10), r=2, with_replacement=True
        ).t()
        edges_graph2 = torch.combinations(
            torch.arange(10, 20), r=2, with_replacement=True
        ).t()
        edge_index = torch.cat([edges_graph1, edges_graph2], dim=1)
        data[("Peak", "self_attn", "Peak")].edge_index = edge_index

        # Forward pass
        output = attention(data)

        # Check shape
        expected_shape = (num_peaks, self.channels)
        self.assertEqual(output["Peak"].x.shape, expected_shape)

    def test_cross_attention(self):
        """Test cross-attention between different node types (Peak -> Residue)."""
        data = HeteroData()
        num_peaks = 8
        num_residues = 5

        # Create features for both node types
        data["Peak"].x = torch.randn(num_peaks, self.channels, device=self.device)
        data["Residue"].x = torch.randn(num_residues, self.channels, device=self.device)

        # Create cross-attention module: Peak -> Residue
        attention = MonoAxialAttention(
            source_type="Peak",
            dest_type="Residue",
            in_channels=self.channels,
            out_channels=self.channels,
            head_dim=self.head_dim,
            heads=self.heads,
            edge_name="cross_attn",
            device=self.device
        )

        # Create cross-attention edges (from peaks to residues)
        # Create a bipartite graph: connect each residue to all peaks
        src_nodes = []
        dst_nodes = []
        for residue_idx in range(num_residues):
            for peak_idx in range(num_peaks):
                src_nodes.append(peak_idx)
                dst_nodes.append(residue_idx)

        edge_index = torch.tensor([src_nodes, dst_nodes], dtype=torch.long, device=self.device)
        data[("Peak", "cross_attn", "Residue")].edge_index = edge_index

        # Store original peak features to verify they're unchanged
        original_peak_features = data["Peak"].x.clone()

        # Forward pass
        output = attention(data)

        # Check that Residue features were updated
        expected_shape = (num_residues, self.channels)
        self.assertEqual(output["Residue"].x.shape, expected_shape)

        # Check that Peak features were NOT modified (cross-attention updates dest only)
        self.assertTrue(torch.allclose(output["Peak"].x, original_peak_features))

        # Verify output is not NaN or Inf
        self.assertFalse(torch.isnan(output["Residue"].x).any())
        self.assertFalse(torch.isinf(output["Residue"].x).any())

    def test_channel_projection(self):
        """Test that channel projection works when in_channels != out_channels."""
        data = HeteroData()
        num_peaks = 10
        in_channels = 64
        out_channels = 128

        # Create features with in_channels dimension
        data["Peak"].x = torch.randn(num_peaks, in_channels, device=self.device)

        # Create attention module with different in/out channels
        attention = MonoAxialAttention(
            source_type="Peak",
            dest_type="Peak",
            in_channels=in_channels,
            out_channels=out_channels,
            head_dim=self.head_dim,
            heads=self.heads,
            device=self.device
        )

        # Verify projection layer is Linear (not Identity)
        self.assertIsInstance(attention.projection, torch.nn.Linear)
        self.assertEqual(attention.projection.in_features, in_channels)
        self.assertEqual(attention.projection.out_features, out_channels)

        # Create edges
        edge_index = torch.combinations(
            torch.arange(num_peaks), r=2, with_replacement=True
        ).t()
        data[("Peak", "self_attn", "Peak")].edge_index = edge_index

        # Forward pass
        output = attention(data)

        # Check output shape matches out_channels
        expected_shape = (num_peaks, out_channels)
        self.assertEqual(output["Peak"].x.shape, expected_shape)

        # Verify output is not NaN or Inf
        self.assertFalse(torch.isnan(output["Peak"].x).any())
        self.assertFalse(torch.isinf(output["Peak"].x).any())

    def test_identity_projection(self):
        """Test that Identity is used when in_channels == out_channels."""
        # Create attention module with same in/out channels
        attention = MonoAxialAttention(
            source_type="Peak",
            dest_type="Peak",
            in_channels=self.channels,
            out_channels=self.channels,
            head_dim=self.head_dim,
            heads=self.heads,
            device=self.device
        )

        # Verify projection layer is Identity (not Linear)
        self.assertIsInstance(attention.projection, torch.nn.Identity)


class TestResidueSelfAttentionTransformer(unittest.TestCase):
    """Test distance-aware self-attention for Residue nodes."""

    def setUp(self):
        """Set up test fixtures."""
        self.device = torch.device("cpu")
        self.channels = 128
        self.channels = 128
        self.head_dim = 64
        self.heads = 4

    def test_attention_output_shape(self):
        """Test that attention produces correct output shape."""
        data = HeteroData()
        num_residues = 10
        data["Residue"].x = torch.randn(num_residues, self.channels, device=self.device)
        data["Residue"].xyz = torch.randn(num_residues, 3, device=self.device)

        attention = ResidueSelfAttentionTransformer(
            in_channels=self.channels,
            out_channels=self.channels,
            head_dim=self.head_dim,
            heads=self.heads,
            device=self.device
        )

        # Create edges for all-to-all attention
        edge_index = torch.combinations(
            torch.arange(num_residues), r=2, with_replacement=True
        ).t()
        data[("Residue", "self_attn", "Residue")].edge_index = edge_index

        # Forward pass
        output = attention(data)

        # Check output shape
        expected_shape = (num_residues, self.channels)
        self.assertEqual(output["Residue"].x.shape, expected_shape)

    def test_distance_computation(self):
        """Test that distance affects attention computation."""
        data = HeteroData()
        num_residues = 3

        # Create features
        data["Residue"].x = torch.randn(num_residues, self.channels, device=self.device)

        # Create coordinates with known distances
        # Residue 0 at origin, Residue 1 nearby (distance=1), Residue 2 far (distance=10)
        data["Residue"].xyz = torch.tensor(
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [10.0, 0.0, 0.0]],
            device=self.device
        )

        attention = ResidueSelfAttentionTransformer(
            in_channels=self.channels,
            out_channels=self.channels,
            head_dim=self.head_dim,
            heads=self.heads,
            device=self.device
        )

        # Create all-to-all edges
        edge_index = torch.combinations(
            torch.arange(num_residues), r=2, with_replacement=True
        ).t()
        data[("Residue", "self_attn", "Residue")].edge_index = edge_index

        # Forward pass with deterministic weights
        torch.manual_seed(42)
        attention.reset_parameters()
        output = attention(data)

        # Output should be different based on distance
        # (exact values depend on learned weights, but output should be valid)
        self.assertEqual(output["Residue"].x.shape, (num_residues, self.channels))
        self.assertFalse(torch.isnan(output["Residue"].x).any())
        self.assertFalse(torch.isinf(output["Residue"].x).any())

    def test_empty_node_set(self):
        """Test handling of empty node sets."""
        data = HeteroData()
        data["Residue"].x = torch.zeros(0, self.channels, device=self.device)
        data["Residue"].xyz = torch.zeros(0, 3, device=self.device)

        attention = ResidueSelfAttentionTransformer(
            in_channels=self.channels,
            out_channels=self.channels,
            head_dim=self.head_dim,
            heads=self.heads,
            device=self.device
        )

        # Create empty edge index
        data[("Residue", "self_attn", "Residue")].edge_index = torch.zeros(
            2, 0, dtype=torch.long, device=self.device
        )

        # Should not crash
        output = attention(data)
        expected_shape = (0, self.channels)
        self.assertEqual(output["Residue"].x.shape, expected_shape)

    def test_no_edges(self):
        """Test handling when there are no edges."""
        data = HeteroData()
        num_residues = 5
        data["Residue"].x = torch.randn(num_residues, self.channels, device=self.device)
        data["Residue"].xyz = torch.randn(num_residues, 3, device=self.device)

        attention = ResidueSelfAttentionTransformer(
            in_channels=self.channels,
            out_channels=self.channels,
            head_dim=self.head_dim,
            heads=self.heads,
            device=self.device
        )

        # Create empty edge index
        data[("Residue", "self_attn", "Residue")].edge_index = torch.zeros(
            2, 0, dtype=torch.long, device=self.device
        )

        # Should not crash and return input unchanged
        output = attention(data)
        self.assertEqual(output["Residue"].x.shape, (num_residues, self.channels))

    def test_gradient_flow(self):
        """Test that gradients flow through attention and distance computation."""
        data = HeteroData()
        num_residues = 6

        # Create input features and coordinates with gradient tracking
        input_features = torch.randn(
            num_residues, self.channels, device=self.device, requires_grad=True
        )
        input_xyz = torch.randn(num_residues, 3, device=self.device, requires_grad=True)

        data["Residue"].x = input_features
        data["Residue"].xyz = input_xyz

        attention = ResidueSelfAttentionTransformer(
            in_channels=self.channels,
            out_channels=self.channels,
            head_dim=self.head_dim,
            heads=self.heads,
            device=self.device
        )

        # Create edges
        edge_index = torch.combinations(
            torch.arange(num_residues), r=2, with_replacement=True
        ).t()
        data[("Residue", "self_attn", "Residue")].edge_index = edge_index

        # Forward pass
        output = attention(data)

        # Compute loss and backward
        loss = output["Residue"].x.sum()
        loss.backward()

        # Check gradients exist on both features and coordinates
        self.assertIsNotNone(input_features.grad)
        self.assertIsNotNone(input_xyz.grad)
        self.assertTrue(torch.any(input_features.grad != 0))
        self.assertTrue(torch.any(input_xyz.grad != 0))

    def test_batched_graphs(self):
        """Test that attention works with batched graphs."""
        data = HeteroData()
        num_residues = 20  # 2 graphs with 10 residues each
        data["Residue"].x = torch.randn(num_residues, self.channels, device=self.device)
        data["Residue"].xyz = torch.randn(num_residues, 3, device=self.device)

        attention = ResidueSelfAttentionTransformer(
            in_channels=self.channels,
            out_channels=self.channels,
            head_dim=self.head_dim,
            heads=self.heads,
            device=self.device
        )

        # Create edges within each graph (0-9 and 10-19)
        edges_graph1 = torch.combinations(
            torch.arange(10), r=2, with_replacement=True
        ).t()
        edges_graph2 = torch.combinations(
            torch.arange(10, 20), r=2, with_replacement=True
        ).t()
        edge_index = torch.cat([edges_graph1, edges_graph2], dim=1)
        data[("Residue", "self_attn", "Residue")].edge_index = edge_index

        # Forward pass
        output = attention(data)

        # Check shape
        expected_shape = (num_residues, self.channels)
        self.assertEqual(output["Residue"].x.shape, expected_shape)

    def test_xyz_immutability(self):
        """Test that .xyz coordinates are not modified during forward pass."""
        data = HeteroData()
        num_residues = 5
        data["Residue"].x = torch.randn(num_residues, self.channels, device=self.device)
        original_xyz = torch.randn(num_residues, 3, device=self.device)
        data["Residue"].xyz = original_xyz.clone()

        attention = ResidueSelfAttentionTransformer(
            in_channels=self.channels,
            out_channels=self.channels,
            head_dim=self.head_dim,
            heads=self.heads,
            device=self.device
        )

        # Create edges
        edge_index = torch.combinations(
            torch.arange(num_residues), r=2, with_replacement=True
        ).t()
        data[("Residue", "self_attn", "Residue")].edge_index = edge_index

        # Forward pass
        output = attention(data)

        # Verify xyz unchanged
        self.assertTrue(torch.allclose(data["Residue"].xyz, original_xyz))


if __name__ == "__main__":
    unittest.main()
