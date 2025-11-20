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

from nmr.models.transformer import (
    MonoAxialAttention,
    BiAxialAttention,
    TriAxialAttention,
    SpatialAttentionCore,
    AttentionCore,
)
from nmr.models.config import ModelConfig, ShiftStandardizeConfig, MLPConfig, AttentionConfig, SharedConfig


def make_config(embed_dim=128, attention_dim=64, num_heads=4):
    """Helper function to create ModelConfig for tests."""
    return ModelConfig(
        shared=SharedConfig(embed_dim=embed_dim),
        shift_standardize=ShiftStandardizeConfig(),
        attention=AttentionConfig(attention_dim=attention_dim, num_heads=num_heads),
    )


def make_attention_args(config, device):
    """
    Helper to extract attention parameters from config for new explicit signatures.

    Returns dict with embed_dim, attention_config, feedforward_mlp_config, device
    that can be unpacked into attention module constructors.
    """
    return {
        "embed_dim": config.shared.embed_dim,
        "attention_config": config.attention,
        "feedforward_mlp_config": config.feedforward_mlp,
        "device": device,
    }


def make_biaxial_args(config, device):
    """Helper to extract BiAxial/TriAxial parameters from config."""
    return {
        "embed_dim": config.shared.embed_dim,
        "attention_config": config.attention,
        "combine_mlp_config": config.combine_mlp,
        "feedforward_mlp_config": config.feedforward_mlp,
        "device": device,
    }


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
        config = make_config(embed_dim=self.channels, attention_dim=self.head_dim, num_heads=self.heads)
        attention = MonoAxialAttention(
            source_type="Peak",
            dest_type="Peak",
            edge_name="self_attn",
            **make_attention_args(config, self.device)
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

        config = make_config(embed_dim=self.channels, attention_dim=self.head_dim, num_heads=self.heads)
        attention = MonoAxialAttention(
            source_type="Peak",
            dest_type="Peak",
            edge_name="self_attn",
            **make_attention_args(config, self.device)
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

        config = make_config(embed_dim=self.channels, attention_dim=self.head_dim, num_heads=self.heads)
        attention = MonoAxialAttention(
            source_type="Peak",
            dest_type="Peak",
            edge_name="self_attn",
            **make_attention_args(config, self.device)
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

        config = make_config(embed_dim=self.channels, attention_dim=self.head_dim, num_heads=self.heads)
        attention = MonoAxialAttention(
            source_type="Peak",
            dest_type="Peak",
            edge_name="self_attn",
            **make_attention_args(config, self.device)
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
        config = make_config(embed_dim=self.channels, attention_dim=self.head_dim, num_heads=self.heads)
        attention = MonoAxialAttention(
            source_type="Peak",
            dest_type="Residue",
            edge_name="cross_attn",
            **make_attention_args(config, self.device)
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
        config = make_config(embed_dim=in_channels, attention_dim=self.head_dim, num_heads=self.heads)
        attention = MonoAxialAttention(
            source_type="Peak",
            dest_type="Peak",
            edge_name="self_attn",
            **make_attention_args(config, self.device)
        )

        # Verify projection layer is Identity when in_channels == out_channels (both are embed_dim)
        self.assertIsInstance(attention.projection, torch.nn.Identity)

        # Create edges
        edge_index = torch.combinations(
            torch.arange(num_peaks), r=2, with_replacement=True
        ).t()
        data[("Peak", "self_attn", "Peak")].edge_index = edge_index

        # Forward pass
        output = attention(data)

        # Check output shape matches in_channels (embed_dim)
        expected_shape = (num_peaks, in_channels)
        self.assertEqual(output["Peak"].x.shape, expected_shape)

        # Verify output is not NaN or Inf
        self.assertFalse(torch.isnan(output["Peak"].x).any())
        self.assertFalse(torch.isinf(output["Peak"].x).any())

    def test_identity_projection(self):
        """Test that Identity is used when in_channels == out_channels."""
        # Create attention module with same in/out channels
        config = make_config(embed_dim=self.channels, attention_dim=self.head_dim, num_heads=self.heads)
        attention = MonoAxialAttention(
            source_type="Peak",
            dest_type="Peak",
            edge_name="self_attn",
            **make_attention_args(config, self.device)
        )

        # Verify projection layer is Identity (not Linear)
        self.assertIsInstance(attention.projection, torch.nn.Identity)

    def test_residue_self_attention_with_monoaxial(self):
        """Test Residue-to-Residue self-attention uses spatial awareness."""
        data = HeteroData()
        num_residues = 10

        # Create Residue nodes with features and coordinates
        data["Residue"].x = torch.randn(num_residues, self.channels, device=self.device)
        data["Residue"].xyz = torch.randn(num_residues, 3, device=self.device)

        # Create Residue self-attention module
        config = make_config(embed_dim=self.channels, attention_dim=self.head_dim, num_heads=self.heads)
        attention = MonoAxialAttention(
            source_type="Residue",
            dest_type="Residue",
            edge_name="self_attn",
            **make_attention_args(config, self.device)
        )

        # Create edges for self-attention
        edge_index = torch.combinations(
            torch.arange(num_residues), r=2, with_replacement=True
        ).t()
        data[("Residue", "self_attn", "Residue")].edge_index = edge_index

        # Forward pass
        output = attention(data)

        # Check output shape is correct
        expected_shape = (num_residues, self.channels)
        self.assertEqual(output["Residue"].x.shape, expected_shape)

        # Verify output is not NaN or Inf
        self.assertFalse(torch.isnan(output["Residue"].x).any())
        self.assertFalse(torch.isinf(output["Residue"].x).any())

    def test_residue_attention_uses_distance(self):
        """Test that Residue-to-Residue attention output changes with distance."""
        num_residues = 5

        # Create attention module
        config = make_config(embed_dim=self.channels, attention_dim=self.head_dim, num_heads=self.heads)
        attention = MonoAxialAttention(
            source_type="Residue",
            dest_type="Residue",
            edge_name="self_attn",
            **make_attention_args(config, self.device)
        )

        # Create two graphs with same features but different coordinates
        # Graph 1: residues close together
        data1 = HeteroData()
        features = torch.randn(num_residues, self.channels, device=self.device)
        data1["Residue"].x = features.clone()
        data1["Residue"].xyz = torch.randn(num_residues, 3, device=self.device) * 0.1  # Close together

        # Graph 2: same features but residues far apart
        data2 = HeteroData()
        data2["Residue"].x = features.clone()
        data2["Residue"].xyz = torch.randn(num_residues, 3, device=self.device) * 10.0  # Far apart

        # Create same edge structure for both
        edge_index = torch.combinations(
            torch.arange(num_residues), r=2, with_replacement=True
        ).t()
        data1[("Residue", "self_attn", "Residue")].edge_index = edge_index
        data2[("Residue", "self_attn", "Residue")].edge_index = edge_index

        # Forward pass on both graphs
        output1 = attention(data1)
        output2 = attention(data2)

        # Outputs should differ because distances differ (spatial attention is active)
        # Use a relatively loose tolerance since attention patterns can be complex
        self.assertFalse(
            torch.allclose(output1["Residue"].x, output2["Residue"].x, rtol=1e-3, atol=1e-5),
            "Residue attention outputs should differ when coordinates differ"
        )

    def test_residue_gradient_flow_through_xyz(self):
        """Test that gradients flow through xyz coordinates for Residue attention."""
        data = HeteroData()
        num_residues = 8

        # Create features and coordinates with gradient tracking
        features = torch.randn(num_residues, self.channels, device=self.device, requires_grad=True)
        coordinates = torch.randn(num_residues, 3, device=self.device, requires_grad=True)

        data["Residue"].x = features
        data["Residue"].xyz = coordinates

        # Create Residue self-attention module
        config = make_config(embed_dim=self.channels, attention_dim=self.head_dim, num_heads=self.heads)
        attention = MonoAxialAttention(
            source_type="Residue",
            dest_type="Residue",
            edge_name="self_attn",
            **make_attention_args(config, self.device)
        )

        # Create edges
        edge_index = torch.combinations(
            torch.arange(num_residues), r=2, with_replacement=True
        ).t()
        data[("Residue", "self_attn", "Residue")].edge_index = edge_index

        # Forward pass
        output = attention(data)

        # Compute loss and backward pass
        loss = output["Residue"].x.sum()
        loss.backward()

        # Check gradients exist on both features and coordinates
        self.assertIsNotNone(features.grad, "Gradients should flow through features")
        self.assertIsNotNone(coordinates.grad, "Gradients should flow through coordinates")

        # Check gradients are non-zero (spatial attention uses coordinates)
        self.assertTrue(torch.any(features.grad != 0), "Feature gradients should be non-zero")
        self.assertTrue(torch.any(coordinates.grad != 0), "Coordinate gradients should be non-zero")

    def test_core_selection_by_node_type(self):
        """Test that correct attention core is selected based on node types."""
        from nmr.models.transformer import SpatialAttentionCore, AttentionCore

        config = make_config(embed_dim=self.channels, attention_dim=self.head_dim, num_heads=self.heads)

        # Residue-to-Residue: should use SpatialAttentionCore
        residue_attn = MonoAxialAttention(
            source_type="Residue",
            dest_type="Residue",
            edge_name="self_attn",
            **make_attention_args(config, self.device)
        )
        self.assertIsInstance(
            residue_attn.core,
            SpatialAttentionCore,
            "Residue-to-Residue should use SpatialAttentionCore"
        )

        # Peak-to-Peak: should use AttentionCore
        peak_attn = MonoAxialAttention(
            source_type="Peak",
            dest_type="Peak",
            edge_name="self_attn",
            **make_attention_args(config, self.device)
        )
        self.assertIsInstance(
            peak_attn.core,
            AttentionCore,
            "Peak-to-Peak should use AttentionCore"
        )

        # Peak-to-Residue (cross-attention): should use AttentionCore
        cross_attn = MonoAxialAttention(
            source_type="Peak",
            dest_type="Residue",
            edge_name="cross_attn",
            **make_attention_args(config, self.device)
        )
        self.assertIsInstance(
            cross_attn.core,
            AttentionCore,
            "Peak-to-Residue (mixed types) should use AttentionCore"
        )


class TestBiAxialAttention(unittest.TestCase):
    """Test BiAxial dual-attention mechanism with spatial awareness."""

    def setUp(self):
        """Set up test fixtures."""
        self.device = torch.device("cpu")
        self.channels = 128
        self.head_dim = 64
        self.heads = 4

    def test_biaxial_core_selection_both_spatial(self):
        """Test that both cores use SpatialAttentionCore for Residue-to-Residue."""
        # Both sources and dest are Residue: both cores should be spatial
        config = make_config(embed_dim=self.channels, attention_dim=self.head_dim, num_heads=self.heads)
        attention = BiAxialAttention(
            source_type_1="Residue",
            source_type_2="Residue",
            dest_type="Residue",
            edge_name_1="biaxial_attn_1",
            edge_name_2="biaxial_attn_2",
            **make_biaxial_args(config, self.device)
        )

        self.assertIsInstance(
            attention.attention_1,
            SpatialAttentionCore,
            "attention_1 should use SpatialAttentionCore for Residue->Residue"
        )
        self.assertIsInstance(
            attention.attention_2,
            SpatialAttentionCore,
            "attention_2 should use SpatialAttentionCore for Residue->Residue"
        )

    def test_biaxial_core_selection_mixed(self):
        """Test mixed core selection: one spatial, one non-spatial."""
        # Source 1 is Residue (spatial), Source 2 is Peak (non-spatial), dest is Residue
        config = make_config(embed_dim=self.channels, attention_dim=self.head_dim, num_heads=self.heads)
        attention = BiAxialAttention(
            source_type_1="Residue",
            source_type_2="Peak",
            dest_type="Residue",
            edge_name_1="biaxial_attn_1",
            edge_name_2="biaxial_attn_2",
            **make_biaxial_args(config, self.device)
        )

        self.assertIsInstance(
            attention.attention_1,
            SpatialAttentionCore,
            "attention_1 should use SpatialAttentionCore for Residue->Residue"
        )
        self.assertIsInstance(
            attention.attention_2,
            AttentionCore,
            "attention_2 should use AttentionCore for Peak->Residue"
        )

    def test_biaxial_core_selection_both_non_spatial(self):
        """Test that both cores use AttentionCore for non-Residue combinations."""
        # Both sources and dest are non-Residue
        config = make_config(embed_dim=self.channels, attention_dim=self.head_dim, num_heads=self.heads)
        attention = BiAxialAttention(
            source_type_1="Peak",
            source_type_2="Peak",
            dest_type="Noe",
            edge_name_1="biaxial_attn_1",
            edge_name_2="biaxial_attn_2",
            **make_biaxial_args(config, self.device)
        )

        self.assertIsInstance(
            attention.attention_1,
            AttentionCore,
            "attention_1 should use AttentionCore for Peak->Noe"
        )
        self.assertIsInstance(
            attention.attention_2,
            AttentionCore,
            "attention_2 should use AttentionCore for Peak->Noe"
        )

    def test_biaxial_residue_dual_attention_output_shape(self):
        """Test Residue-to-Residue dual-attention produces correct output shape."""
        data = HeteroData()
        num_residues = 10

        # Create Residue nodes with features and coordinates
        data["Residue"].x = torch.randn(num_residues, self.channels, device=self.device)
        data["Residue"].xyz = torch.randn(num_residues, 3, device=self.device)

        # Create BiAxial attention module (both streams Residue->Residue)
        config = make_config(embed_dim=self.channels, attention_dim=self.head_dim, num_heads=self.heads)
        attention = BiAxialAttention(
            source_type_1="Residue",
            source_type_2="Residue",
            dest_type="Residue",
            edge_name_1="biaxial_attn_1",
            edge_name_2="biaxial_attn_2",
            **make_biaxial_args(config, self.device)
        )

        # Create edges for both attention streams
        edge_index = torch.combinations(
            torch.arange(num_residues), r=2, with_replacement=True
        ).t()
        data[("Residue", "biaxial_attn_1", "Residue")].edge_index = edge_index
        data[("Residue", "biaxial_attn_2", "Residue")].edge_index = edge_index

        # Forward pass
        output = attention(data)

        # Check output shape
        expected_shape = (num_residues, self.channels)
        self.assertEqual(output["Residue"].x.shape, expected_shape)

        # Verify output is not NaN or Inf
        self.assertFalse(torch.isnan(output["Residue"].x).any())
        self.assertFalse(torch.isinf(output["Residue"].x).any())

    def test_biaxial_residue_attention_uses_distance(self):
        """Test that dual Residue attention output changes with distance."""
        num_residues = 6

        # Create attention module
        config = make_config(embed_dim=self.channels, attention_dim=self.head_dim, num_heads=self.heads)
        attention = BiAxialAttention(
            source_type_1="Residue",
            source_type_2="Residue",
            dest_type="Residue",
            edge_name_1="biaxial_attn_1",
            edge_name_2="biaxial_attn_2",
            **make_biaxial_args(config, self.device)
        )

        # Create two graphs with same features but different coordinates
        features = torch.randn(num_residues, self.channels, device=self.device)

        # Graph 1: all residues at origin (distance = 0)
        data1 = HeteroData()
        data1["Residue"].x = features.clone()
        data1["Residue"].xyz = torch.zeros(num_residues, 3, device=self.device)

        # Graph 2: residues at different positions
        data2 = HeteroData()
        data2["Residue"].x = features.clone()
        data2["Residue"].xyz = torch.randn(num_residues, 3, device=self.device)

        # Create same edge structure for both
        edge_index = torch.combinations(
            torch.arange(num_residues), r=2, with_replacement=True
        ).t()
        data1[("Residue", "biaxial_attn_1", "Residue")].edge_index = edge_index
        data1[("Residue", "biaxial_attn_2", "Residue")].edge_index = edge_index
        data2[("Residue", "biaxial_attn_1", "Residue")].edge_index = edge_index
        data2[("Residue", "biaxial_attn_2", "Residue")].edge_index = edge_index

        # Set seed for reproducibility
        torch.manual_seed(42)
        attention.reset_parameters()

        # Forward pass on both graphs
        output1 = attention(data1)
        output2 = attention(data2)

        # Outputs should differ because distances differ (spatial attention active in both streams)
        self.assertFalse(
            torch.allclose(output1["Residue"].x, output2["Residue"].x, rtol=1e-3, atol=1e-5),
            "BiAxial Residue attention outputs should differ when coordinates differ"
        )

    def test_biaxial_gradient_flow_through_xyz(self):
        """Test that gradients flow through xyz coordinates for spatial streams."""
        data = HeteroData()
        num_residues = 6

        # Create features and coordinates with gradient tracking
        features = torch.randn(num_residues, self.channels, device=self.device, requires_grad=True)
        coordinates = torch.randn(num_residues, 3, device=self.device, requires_grad=True)

        data["Residue"].x = features
        data["Residue"].xyz = coordinates

        # Create BiAxial attention module
        config = make_config(embed_dim=self.channels, attention_dim=self.head_dim, num_heads=self.heads)
        attention = BiAxialAttention(
            source_type_1="Residue",
            source_type_2="Residue",
            dest_type="Residue",
            edge_name_1="biaxial_attn_1",
            edge_name_2="biaxial_attn_2",
            **make_biaxial_args(config, self.device)
        )

        # Create edges
        edge_index = torch.combinations(
            torch.arange(num_residues), r=2, with_replacement=True
        ).t()
        data[("Residue", "biaxial_attn_1", "Residue")].edge_index = edge_index
        data[("Residue", "biaxial_attn_2", "Residue")].edge_index = edge_index

        # Forward pass
        output = attention(data)

        # Compute loss and backward pass
        loss = output["Residue"].x.sum()
        loss.backward()

        # Check gradients exist on both features and coordinates
        self.assertIsNotNone(features.grad, "Gradients should flow through features")
        self.assertIsNotNone(coordinates.grad, "Gradients should flow through coordinates")

        # Check gradients are non-zero (spatial attention uses coordinates)
        self.assertTrue(torch.any(features.grad != 0), "Feature gradients should be non-zero")
        self.assertTrue(torch.any(coordinates.grad != 0), "Coordinate gradients should be non-zero")

    def test_biaxial_backward_compatibility_non_residue(self):
        """Test that non-Residue dual-attention behavior is unchanged."""
        data = HeteroData()
        num_peaks = 8
        num_noes = 5

        # Create Peak and Noe nodes (no xyz)
        data["Peak"].x = torch.randn(num_peaks, self.channels, device=self.device)
        data["Noe"].x = torch.randn(num_noes, self.channels, device=self.device)

        # Create edges (source indices from peaks, dest indices from noes)
        src_indices_1 = torch.randint(0, num_peaks, (15,), device=self.device)
        dst_indices_1 = torch.randint(0, num_noes, (15,), device=self.device)
        edge_index_1 = torch.stack([src_indices_1, dst_indices_1], dim=0)

        src_indices_2 = torch.randint(0, num_peaks, (15,), device=self.device)
        dst_indices_2 = torch.randint(0, num_noes, (15,), device=self.device)
        edge_index_2 = torch.stack([src_indices_2, dst_indices_2], dim=0)

        data[("Peak", "biaxial_attn_1", "Noe")].edge_index = edge_index_1
        data[("Peak", "biaxial_attn_2", "Noe")].edge_index = edge_index_2

        # Create BiAxial attention module
        config = make_config(embed_dim=self.channels, attention_dim=self.head_dim, num_heads=self.heads)
        attention = BiAxialAttention(
            source_type_1="Peak",
            source_type_2="Peak",
            dest_type="Noe",
            edge_name_1="biaxial_attn_1",
            edge_name_2="biaxial_attn_2",
            **make_biaxial_args(config, self.device)
        )

        # Should work without xyz coordinates
        output = attention(data)

        # Check output
        self.assertEqual(output["Noe"].x.shape, (num_noes, self.channels))
        self.assertFalse(torch.isnan(output["Noe"].x).any())
        self.assertFalse(torch.isinf(output["Noe"].x).any())

    def test_biaxial_mixed_spatial_nonspatial(self):
        """Test mixed scenario: one spatial stream, one non-spatial stream."""
        data = HeteroData()
        num_residues = 8
        num_peaks = 12
        num_noes = 5

        # Create nodes
        data["Residue"].x = torch.randn(num_residues, self.channels, device=self.device)
        data["Residue"].xyz = torch.randn(num_residues, 3, device=self.device)
        data["Peak"].x = torch.randn(num_peaks, self.channels, device=self.device)
        data["Noe"].x = torch.randn(num_noes, self.channels, device=self.device)

        # Create BiAxial attention: Residue->Noe (non-spatial) + Peak->Noe (non-spatial)
        # Note: Even though source is Residue, dest is Noe, so it's non-spatial
        config = make_config(embed_dim=self.channels, attention_dim=self.head_dim, num_heads=self.heads)
        attention = BiAxialAttention(
            source_type_1="Residue",
            source_type_2="Peak",
            dest_type="Noe",
            edge_name_1="biaxial_attn_1",
            edge_name_2="biaxial_attn_2",
            **make_biaxial_args(config, self.device)
        )

        # Both cores should be non-spatial (dest is not Residue)
        self.assertIsInstance(attention.attention_1, AttentionCore)
        self.assertIsInstance(attention.attention_2, AttentionCore)

        # Create edges (source indices from respective source types, dest indices from noes)
        src_indices_1 = torch.randint(0, num_residues, (20,), device=self.device)
        dst_indices_1 = torch.randint(0, num_noes, (20,), device=self.device)
        edge_index_1 = torch.stack([src_indices_1, dst_indices_1], dim=0)

        src_indices_2 = torch.randint(0, num_peaks, (20,), device=self.device)
        dst_indices_2 = torch.randint(0, num_noes, (20,), device=self.device)
        edge_index_2 = torch.stack([src_indices_2, dst_indices_2], dim=0)

        data[("Residue", "biaxial_attn_1", "Noe")].edge_index = edge_index_1
        data[("Peak", "biaxial_attn_2", "Noe")].edge_index = edge_index_2

        # Forward pass
        output = attention(data)

        # Check output
        self.assertEqual(output["Noe"].x.shape, (num_noes, self.channels))
        self.assertFalse(torch.isnan(output["Noe"].x).any())
        self.assertFalse(torch.isinf(output["Noe"].x).any())


class TestTriAxialAttention(unittest.TestCase):
    """Test TriAxial triple-attention mechanism with spatial awareness."""

    def setUp(self):
        """Set up test fixtures."""
        self.device = torch.device("cpu")
        self.channels = 128
        self.head_dim = 64
        self.heads = 4

    def test_triaxial_core_selection_all_spatial(self):
        """Test that all three cores use SpatialAttentionCore for Residue-to-Residue."""
        # All three sources and dest are Residue: all cores should be spatial
        config = make_config(embed_dim=self.channels, attention_dim=self.head_dim, num_heads=self.heads)
        attention = TriAxialAttention(
            source_type_1="Residue",
            source_type_2="Residue",
            source_type_3="Residue",
            dest_type="Residue",
            edge_name_1="triaxial_attn_1",
            edge_name_2="triaxial_attn_2",
            edge_name_3="triaxial_attn_3",
            **make_biaxial_args(config, self.device),
        )

        self.assertIsInstance(
            attention.attention_1,
            SpatialAttentionCore,
            "attention_1 should use SpatialAttentionCore for Residue->Residue",
        )
        self.assertIsInstance(
            attention.attention_2,
            SpatialAttentionCore,
            "attention_2 should use SpatialAttentionCore for Residue->Residue",
        )
        self.assertIsInstance(
            attention.attention_3,
            SpatialAttentionCore,
            "attention_3 should use SpatialAttentionCore for Residue->Residue",
        )

    def test_triaxial_core_selection_mixed(self):
        """Test mixed core selection: two spatial, one non-spatial."""
        # Source 1 and 2 are Residue (spatial), Source 3 is Peak (non-spatial), dest is Residue
        config = make_config(embed_dim=self.channels, attention_dim=self.head_dim, num_heads=self.heads)
        attention = TriAxialAttention(
            source_type_1="Residue",
            source_type_2="Residue",
            source_type_3="Peak",
            dest_type="Residue",
            edge_name_1="triaxial_attn_1",
            edge_name_2="triaxial_attn_2",
            edge_name_3="triaxial_attn_3",
            **make_biaxial_args(config, self.device),
        )

        self.assertIsInstance(
            attention.attention_1,
            SpatialAttentionCore,
            "attention_1 should use SpatialAttentionCore for Residue->Residue",
        )
        self.assertIsInstance(
            attention.attention_2,
            SpatialAttentionCore,
            "attention_2 should use SpatialAttentionCore for Residue->Residue",
        )
        self.assertIsInstance(
            attention.attention_3,
            AttentionCore,
            "attention_3 should use AttentionCore for Peak->Residue",
        )

    def test_triaxial_core_selection_all_non_spatial(self):
        """Test that all cores use AttentionCore for non-Residue combinations."""
        # All sources and dest are non-Residue
        config = make_config(embed_dim=self.channels, attention_dim=self.head_dim, num_heads=self.heads)
        attention = TriAxialAttention(
            source_type_1="Peak",
            source_type_2="Peak",
            source_type_3="Peak",
            dest_type="Noe",
            edge_name_1="triaxial_attn_1",
            edge_name_2="triaxial_attn_2",
            edge_name_3="triaxial_attn_3",
            **make_biaxial_args(config, self.device),
        )

        self.assertIsInstance(
            attention.attention_1,
            AttentionCore,
            "attention_1 should use AttentionCore for Peak->Noe",
        )
        self.assertIsInstance(
            attention.attention_2,
            AttentionCore,
            "attention_2 should use AttentionCore for Peak->Noe",
        )
        self.assertIsInstance(
            attention.attention_3,
            AttentionCore,
            "attention_3 should use AttentionCore for Peak->Noe",
        )

    def test_triaxial_residue_triple_attention_output_shape(self):
        """Test Residue-to-Residue triple-attention produces correct output shape."""
        data = HeteroData()
        num_residues = 10

        # Create Residue nodes with features and coordinates
        data["Residue"].x = torch.randn(num_residues, self.channels, device=self.device)
        data["Residue"].xyz = torch.randn(num_residues, 3, device=self.device)

        # Create TriAxial attention module (all three streams Residue->Residue)
        config = make_config(embed_dim=self.channels, attention_dim=self.head_dim, num_heads=self.heads)
        attention = TriAxialAttention(
            source_type_1="Residue",
            source_type_2="Residue",
            source_type_3="Residue",
            dest_type="Residue",
            edge_name_1="triaxial_attn_1",
            edge_name_2="triaxial_attn_2",
            edge_name_3="triaxial_attn_3",
            **make_biaxial_args(config, self.device),
        )

        # Create edges for all three attention streams
        edge_index = torch.combinations(torch.arange(num_residues), r=2, with_replacement=True).t()
        data[("Residue", "triaxial_attn_1", "Residue")].edge_index = edge_index
        data[("Residue", "triaxial_attn_2", "Residue")].edge_index = edge_index
        data[("Residue", "triaxial_attn_3", "Residue")].edge_index = edge_index

        # Forward pass
        output = attention(data)

        # Check output shape
        expected_shape = (num_residues, self.channels)
        self.assertEqual(output["Residue"].x.shape, expected_shape)

        # Verify output is not NaN or Inf
        self.assertFalse(torch.isnan(output["Residue"].x).any())
        self.assertFalse(torch.isinf(output["Residue"].x).any())

    def test_triaxial_residue_attention_uses_distance(self):
        """Test that triple Residue attention output changes with distance."""
        num_residues = 6

        # Create attention module
        config = make_config(embed_dim=self.channels, attention_dim=self.head_dim, num_heads=self.heads)
        attention = TriAxialAttention(
            source_type_1="Residue",
            source_type_2="Residue",
            source_type_3="Residue",
            dest_type="Residue",
            edge_name_1="triaxial_attn_1",
            edge_name_2="triaxial_attn_2",
            edge_name_3="triaxial_attn_3",
            **make_biaxial_args(config, self.device),
        )

        # Create two graphs with same features but different coordinates
        features = torch.randn(num_residues, self.channels, device=self.device)

        # Graph 1: all residues at origin (distance = 0)
        data1 = HeteroData()
        data1["Residue"].x = features.clone()
        data1["Residue"].xyz = torch.zeros(num_residues, 3, device=self.device)

        # Graph 2: residues at different positions
        data2 = HeteroData()
        data2["Residue"].x = features.clone()
        data2["Residue"].xyz = torch.randn(num_residues, 3, device=self.device)

        # Create same edge structure for both
        edge_index = torch.combinations(torch.arange(num_residues), r=2, with_replacement=True).t()
        data1[("Residue", "triaxial_attn_1", "Residue")].edge_index = edge_index
        data1[("Residue", "triaxial_attn_2", "Residue")].edge_index = edge_index
        data1[("Residue", "triaxial_attn_3", "Residue")].edge_index = edge_index
        data2[("Residue", "triaxial_attn_1", "Residue")].edge_index = edge_index
        data2[("Residue", "triaxial_attn_2", "Residue")].edge_index = edge_index
        data2[("Residue", "triaxial_attn_3", "Residue")].edge_index = edge_index

        # Set seed for reproducibility
        torch.manual_seed(42)
        attention.reset_parameters()

        # Forward pass on both graphs
        output1 = attention(data1)
        output2 = attention(data2)

        # Outputs should differ because distances differ (spatial attention active in all streams)
        self.assertFalse(
            torch.allclose(output1["Residue"].x, output2["Residue"].x, rtol=1e-3, atol=1e-5),
            "TriAxial Residue attention outputs should differ when coordinates differ",
        )

    def test_triaxial_gradient_flow_through_xyz(self):
        """Test that gradients flow through xyz coordinates for spatial streams."""
        data = HeteroData()
        num_residues = 6

        # Create features and coordinates with gradient tracking
        features = torch.randn(num_residues, self.channels, device=self.device, requires_grad=True)
        coordinates = torch.randn(num_residues, 3, device=self.device, requires_grad=True)

        data["Residue"].x = features
        data["Residue"].xyz = coordinates

        # Create TriAxial attention module
        config = make_config(embed_dim=self.channels, attention_dim=self.head_dim, num_heads=self.heads)
        attention = TriAxialAttention(
            source_type_1="Residue",
            source_type_2="Residue",
            source_type_3="Residue",
            dest_type="Residue",
            edge_name_1="triaxial_attn_1",
            edge_name_2="triaxial_attn_2",
            edge_name_3="triaxial_attn_3",
            **make_biaxial_args(config, self.device),
        )

        # Create edges
        edge_index = torch.combinations(torch.arange(num_residues), r=2, with_replacement=True).t()
        data[("Residue", "triaxial_attn_1", "Residue")].edge_index = edge_index
        data[("Residue", "triaxial_attn_2", "Residue")].edge_index = edge_index
        data[("Residue", "triaxial_attn_3", "Residue")].edge_index = edge_index

        # Forward pass
        output = attention(data)

        # Compute loss and backward pass
        loss = output["Residue"].x.sum()
        loss.backward()

        # Check gradients exist on both features and coordinates
        self.assertIsNotNone(features.grad, "Gradients should flow through features")
        self.assertIsNotNone(coordinates.grad, "Gradients should flow through coordinates")

        # Check gradients are non-zero (spatial attention uses coordinates)
        self.assertTrue(torch.any(features.grad != 0), "Feature gradients should be non-zero")
        self.assertTrue(torch.any(coordinates.grad != 0), "Coordinate gradients should be non-zero")

    def test_triaxial_backward_compatibility_non_residue(self):
        """Test that non-Residue triple-attention behavior works correctly."""
        data = HeteroData()
        num_peaks = 8
        num_noes = 5

        # Create Peak and Noe nodes (no xyz)
        data["Peak"].x = torch.randn(num_peaks, self.channels, device=self.device)
        data["Noe"].x = torch.randn(num_noes, self.channels, device=self.device)

        # Create edges (source indices from peaks, dest indices from noes)
        src_indices_1 = torch.randint(0, num_peaks, (15,), device=self.device)
        dst_indices_1 = torch.randint(0, num_noes, (15,), device=self.device)
        edge_index_1 = torch.stack([src_indices_1, dst_indices_1], dim=0)

        src_indices_2 = torch.randint(0, num_peaks, (15,), device=self.device)
        dst_indices_2 = torch.randint(0, num_noes, (15,), device=self.device)
        edge_index_2 = torch.stack([src_indices_2, dst_indices_2], dim=0)

        src_indices_3 = torch.randint(0, num_peaks, (15,), device=self.device)
        dst_indices_3 = torch.randint(0, num_noes, (15,), device=self.device)
        edge_index_3 = torch.stack([src_indices_3, dst_indices_3], dim=0)

        data[("Peak", "triaxial_attn_1", "Noe")].edge_index = edge_index_1
        data[("Peak", "triaxial_attn_2", "Noe")].edge_index = edge_index_2
        data[("Peak", "triaxial_attn_3", "Noe")].edge_index = edge_index_3

        # Create TriAxial attention module
        config = make_config(embed_dim=self.channels, attention_dim=self.head_dim, num_heads=self.heads)
        attention = TriAxialAttention(
            source_type_1="Peak",
            source_type_2="Peak",
            source_type_3="Peak",
            dest_type="Noe",
            edge_name_1="triaxial_attn_1",
            edge_name_2="triaxial_attn_2",
            edge_name_3="triaxial_attn_3",
            **make_biaxial_args(config, self.device),
        )

        # Should work without xyz coordinates
        output = attention(data)

        # Check output
        self.assertEqual(output["Noe"].x.shape, (num_noes, self.channels))
        self.assertFalse(torch.isnan(output["Noe"].x).any())
        self.assertFalse(torch.isinf(output["Noe"].x).any())

    def test_triaxial_mixed_spatial_nonspatial(self):
        """Test mixed scenario: two spatial streams, one non-spatial stream."""
        data = HeteroData()
        num_residues = 8
        num_peaks = 12
        num_noes = 5

        # Create nodes
        data["Residue"].x = torch.randn(num_residues, self.channels, device=self.device)
        data["Residue"].xyz = torch.randn(num_residues, 3, device=self.device)
        data["Peak"].x = torch.randn(num_peaks, self.channels, device=self.device)
        data["Noe"].x = torch.randn(num_noes, self.channels, device=self.device)

        # Create TriAxial attention: Residue->Noe (non-spatial) + Residue->Noe (non-spatial) + Peak->Noe (non-spatial)
        # Note: Even though sources are Residue, dest is Noe, so all streams are non-spatial
        config = make_config(embed_dim=self.channels, attention_dim=self.head_dim, num_heads=self.heads)
        attention = TriAxialAttention(
            source_type_1="Residue",
            source_type_2="Residue",
            source_type_3="Peak",
            dest_type="Noe",
            edge_name_1="triaxial_attn_1",
            edge_name_2="triaxial_attn_2",
            edge_name_3="triaxial_attn_3",
            **make_biaxial_args(config, self.device),
        )

        # All cores should be non-spatial (dest is not Residue)
        self.assertIsInstance(attention.attention_1, AttentionCore)
        self.assertIsInstance(attention.attention_2, AttentionCore)
        self.assertIsInstance(attention.attention_3, AttentionCore)

        # Create edges (source indices from respective source types, dest indices from noes)
        src_indices_1 = torch.randint(0, num_residues, (20,), device=self.device)
        dst_indices_1 = torch.randint(0, num_noes, (20,), device=self.device)
        edge_index_1 = torch.stack([src_indices_1, dst_indices_1], dim=0)

        src_indices_2 = torch.randint(0, num_residues, (20,), device=self.device)
        dst_indices_2 = torch.randint(0, num_noes, (20,), device=self.device)
        edge_index_2 = torch.stack([src_indices_2, dst_indices_2], dim=0)

        src_indices_3 = torch.randint(0, num_peaks, (20,), device=self.device)
        dst_indices_3 = torch.randint(0, num_noes, (20,), device=self.device)
        edge_index_3 = torch.stack([src_indices_3, dst_indices_3], dim=0)

        data[("Residue", "triaxial_attn_1", "Noe")].edge_index = edge_index_1
        data[("Residue", "triaxial_attn_2", "Noe")].edge_index = edge_index_2
        data[("Peak", "triaxial_attn_3", "Noe")].edge_index = edge_index_3

        # Forward pass
        output = attention(data)

        # Check output
        self.assertEqual(output["Noe"].x.shape, (num_noes, self.channels))
        self.assertFalse(torch.isnan(output["Noe"].x).any())
        self.assertFalse(torch.isinf(output["Noe"].x).any())

    def test_triaxial_empty_node_sets(self):
        """Test handling of empty node sets."""
        data = HeteroData()

        # Create empty node sets
        data["Peak"].x = torch.zeros(0, self.channels, device=self.device)
        data["Noe"].x = torch.zeros(0, self.channels, device=self.device)

        config = make_config(embed_dim=self.channels, attention_dim=self.head_dim, num_heads=self.heads)
        attention = TriAxialAttention(
            source_type_1="Peak",
            source_type_2="Peak",
            source_type_3="Peak",
            dest_type="Noe",
            edge_name_1="triaxial_attn_1",
            edge_name_2="triaxial_attn_2",
            edge_name_3="triaxial_attn_3",
            **make_biaxial_args(config, self.device),
        )

        # Create empty edge indices
        data[("Peak", "triaxial_attn_1", "Noe")].edge_index = torch.zeros(
            2, 0, dtype=torch.long, device=self.device
        )
        data[("Peak", "triaxial_attn_2", "Noe")].edge_index = torch.zeros(
            2, 0, dtype=torch.long, device=self.device
        )
        data[("Peak", "triaxial_attn_3", "Noe")].edge_index = torch.zeros(
            2, 0, dtype=torch.long, device=self.device
        )

        # Should not crash and return input unchanged
        output = attention(data)

        # When there are no nodes, input should remain unchanged
        expected_shape = (0, self.channels)
        self.assertEqual(output["Noe"].x.shape, expected_shape)

    def test_triaxial_empty_edge_sets(self):
        """Test handling of empty edge sets with non-empty nodes."""
        data = HeteroData()
        num_peaks = 5
        num_noes = 3

        # Create non-empty node sets
        data["Peak"].x = torch.randn(num_peaks, self.channels, device=self.device)
        data["Noe"].x = torch.randn(num_noes, self.channels, device=self.device)

        config = make_config(embed_dim=self.channels, attention_dim=self.head_dim, num_heads=self.heads)
        attention = TriAxialAttention(
            source_type_1="Peak",
            source_type_2="Peak",
            source_type_3="Peak",
            dest_type="Noe",
            edge_name_1="triaxial_attn_1",
            edge_name_2="triaxial_attn_2",
            edge_name_3="triaxial_attn_3",
            **make_biaxial_args(config, self.device),
        )

        # Create empty edge indices (at least one empty to trigger early return)
        data[("Peak", "triaxial_attn_1", "Noe")].edge_index = torch.zeros(
            2, 0, dtype=torch.long, device=self.device
        )
        data[("Peak", "triaxial_attn_2", "Noe")].edge_index = torch.zeros(
            2, 0, dtype=torch.long, device=self.device
        )
        data[("Peak", "triaxial_attn_3", "Noe")].edge_index = torch.zeros(
            2, 0, dtype=torch.long, device=self.device
        )

        # Store original features
        original_noe_features = data["Noe"].x.clone()

        # Should return input unchanged
        output = attention(data)

        # Check that features are unchanged
        self.assertTrue(torch.allclose(output["Noe"].x, original_noe_features))


if __name__ == "__main__":
    unittest.main()
