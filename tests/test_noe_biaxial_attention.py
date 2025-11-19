"""
Tests for BiAxialAttention module.

This test module covers critical behaviors:
- Initialization of dual attention mechanism
- Dual attention computation (Source 1 and Source 2)
- Feature combination through MLP
- Residual update application
- Edge case handling (empty nodes, empty edges, varying graph sizes)
- Dimension consistency throughout forward pass
"""

import sys
from pathlib import Path
import unittest

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
from torch_geometric.data import HeteroData

from nmr.models.transformer import BiAxialAttention
from nmr.models.config import SharedConfig, ModelConfig, ShiftStandardizeConfig, MLPConfig, AttentionConfig


def make_config(embed_dim=128, attention_dim=64, num_heads=4):
    """Helper function to create ModelConfig for tests."""
    return ModelConfig(
        shared=SharedConfig(embed_dim=embed_dim),
        shift_standardize=ShiftStandardizeConfig(),
        attention=AttentionConfig(attention_dim=attention_dim, num_heads=num_heads),
    )


class TestBiAxialAttentionInitialization(unittest.TestCase):
    """Test BiAxialAttention initialization."""

    def test_initialization(self):
        """Test that BiAxialAttention initializes correctly."""
        # Create module
        config = make_config(embed_dim=64, attention_dim=16, num_heads=4)
        module = BiAxialAttention(
            source_type_1="Residue",
            source_type_2="Peak",
            dest_type="Noe",
            edge_name_1="biaxial_attn_1",
            edge_name_2="biaxial_attn_2",
            embed_dim=config.shared.embed_dim, attention_config=config.attention, combine_mlp_config=config.combine_mlp, device="cpu",
        )

        # Verify it's an nn.Module
        self.assertIsInstance(module, torch.nn.Module)

        # Verify dual attention cores exist
        self.assertTrue(hasattr(module, "attention_1"))
        self.assertTrue(hasattr(module, "attention_2"))

        # Verify edge type tuples
        self.assertEqual(module.edge_type_1, ("Residue", "biaxial_attn_1", "Noe"))
        self.assertEqual(module.edge_type_2, ("Peak", "biaxial_attn_2", "Noe"))

        # Verify NOE linear layer exists
        self.assertTrue(hasattr(module, "dest_linear"))
        self.assertEqual(module.dest_linear.in_features, 64)
        self.assertEqual(module.dest_linear.out_features, 64)

        # Verify combination MLP exists
        self.assertTrue(hasattr(module, "combine_mlp"))


class TestBiAxialAttentionForward(unittest.TestCase):
    """Test BiAxialAttention forward pass."""

    def setUp(self):
        """Set up test fixtures."""
        self.channels = 64
        self.head_dim = 16
        self.heads = 4

        config = make_config(embed_dim=self.channels, attention_dim=self.head_dim, num_heads=self.heads)
        self.module = BiAxialAttention(
            source_type_1="Residue",
            source_type_2="Peak",
            dest_type="Noe",
            edge_name_1="biaxial_attn_1",
            edge_name_2="biaxial_attn_2",
            embed_dim=config.shared.embed_dim, attention_config=config.attention, combine_mlp_config=config.combine_mlp, device="cpu",
        )

    def test_forward_basic(self):
        """Test basic forward pass with valid inputs."""
        # Create simple HeteroData graph
        data = HeteroData()

        # Add NOE nodes
        num_noes = 5
        data["Noe"].x = torch.randn(num_noes, self.channels)

        # Add Residue nodes
        num_residues = 10
        data["Residue"].x = torch.randn(num_residues, self.channels)

        # Add Peak nodes
        num_peaks = 8
        data["Peak"].x = torch.randn(num_peaks, self.channels)

        # Add edges: fully connected bipartite graphs
        # Edge convention: [source, dest] where source attends to dest
        # For Residue -> Noe attention: Residue (row 0) -> Noe (row 1)
        res_to_noe_edges = torch.cartesian_prod(
            torch.arange(num_residues), torch.arange(num_noes)
        ).t()  # [2, num_residues * num_noes]
        data["Residue", "biaxial_attn_1", "Noe"].edge_index = res_to_noe_edges

        # For Peak -> Noe attention: Peak (row 0) -> Noe (row 1)
        peak_to_noe_edges = torch.cartesian_prod(
            torch.arange(num_peaks), torch.arange(num_noes)
        ).t()  # [2, num_peaks * num_noes]
        data["Peak", "biaxial_attn_2", "Noe"].edge_index = peak_to_noe_edges

        # Store original NOE features
        original_noe_x = data["Noe"].x.clone()

        # Run forward pass
        output_data = self.module(data)

        # Verify output is HeteroData
        self.assertIsInstance(output_data, HeteroData)

        # Verify NOE features were updated
        self.assertEqual(output_data["Noe"].x.shape, (num_noes, self.channels))

        # Verify Residue and Peak features unchanged
        self.assertTrue(torch.equal(output_data["Residue"].x, data["Residue"].x))
        self.assertTrue(torch.equal(output_data["Peak"].x, data["Peak"].x))

    def test_forward_empty_noe_nodes(self):
        """Test forward pass with zero NOE nodes."""
        data = HeteroData()

        # Empty NOE nodes
        data["Noe"].x = torch.empty(0, self.channels)

        # Non-empty Residue and Peak nodes
        data["Residue"].x = torch.randn(10, self.channels)
        data["Peak"].x = torch.randn(8, self.channels)

        # Add empty edges
        data["Residue", "biaxial_attn_1", "Noe"].edge_index = torch.empty(2, 0, dtype=torch.long)
        data["Peak", "biaxial_attn_2", "Noe"].edge_index = torch.empty(2, 0, dtype=torch.long)

        # Should return data unchanged
        output_data = self.module(data)

        self.assertIsInstance(output_data, HeteroData)
        self.assertEqual(output_data["Noe"].x.shape[0], 0)

    def test_forward_empty_residue_nodes(self):
        """Test forward pass with zero Residue nodes."""
        data = HeteroData()

        data["Noe"].x = torch.randn(5, self.channels)
        data["Residue"].x = torch.empty(0, self.channels)
        data["Peak"].x = torch.randn(8, self.channels)

        # Add edges
        data["Residue", "biaxial_attn_1", "Noe"].edge_index = torch.empty(2, 0, dtype=torch.long)
        # Peak -> Noe edges
        peak_to_noe_edges = torch.cartesian_prod(
            torch.arange(8), torch.arange(5)
        ).t()
        data["Peak", "biaxial_attn_2", "Noe"].edge_index = peak_to_noe_edges

        # Should return data unchanged
        output_data = self.module(data)

        self.assertIsInstance(output_data, HeteroData)
        # NOE features should remain in_channels (no update applied)
        self.assertEqual(output_data["Noe"].x.shape, (5, self.channels))

    def test_forward_empty_peak_nodes(self):
        """Test forward pass with zero Peak nodes."""
        data = HeteroData()

        data["Noe"].x = torch.randn(5, self.channels)
        data["Residue"].x = torch.randn(10, self.channels)
        data["Peak"].x = torch.empty(0, self.channels)

        # Add edges
        # Residue -> Noe edges
        res_to_noe_edges = torch.cartesian_prod(
            torch.arange(10), torch.arange(5)
        ).t()
        data["Residue", "biaxial_attn_1", "Noe"].edge_index = res_to_noe_edges
        data["Peak", "biaxial_attn_2", "Noe"].edge_index = torch.empty(2, 0, dtype=torch.long)

        # Should return data unchanged
        output_data = self.module(data)

        self.assertIsInstance(output_data, HeteroData)
        # NOE features should remain in_channels (no update applied)
        self.assertEqual(output_data["Noe"].x.shape, (5, self.channels))


class TestBiAxialAttentionFeatureCombination(unittest.TestCase):
    """Test feature combination logic."""

    def test_residual_connection(self):
        """Test that residual connection is applied correctly."""
        config = make_config(embed_dim=64, attention_dim=16, num_heads=4)
        module = BiAxialAttention(
            source_type_1="Residue",
            source_type_2="Peak",
            dest_type="Noe",
            edge_name_1="biaxial_attn_1",
            edge_name_2="biaxial_attn_2",
            embed_dim=config.shared.embed_dim, attention_config=config.attention, combine_mlp_config=config.combine_mlp, device="cpu",
        )

        # Create simple graph
        data = HeteroData()
        num_noes = 3
        data["Noe"].x = torch.randn(num_noes, 64)
        data["Residue"].x = torch.randn(5, 64)
        data["Peak"].x = torch.randn(4, 64)

        # Add edges (Residue/Peak -> Noe)
        res_to_noe_edges = torch.cartesian_prod(
            torch.arange(5), torch.arange(num_noes)
        ).t()
        data["Residue", "biaxial_attn_1", "Noe"].edge_index = res_to_noe_edges

        peak_to_noe_edges = torch.cartesian_prod(
            torch.arange(4), torch.arange(num_noes)
        ).t()
        data["Peak", "biaxial_attn_2", "Noe"].edge_index = peak_to_noe_edges

        # Store original
        original_noe_x = data["Noe"].x.clone()

        # Run forward
        output_data = module(data)

        # Verify output dimension matches
        self.assertEqual(output_data["Noe"].x.shape, original_noe_x.shape)

        # Verify features changed (residual was applied)
        self.assertFalse(torch.equal(output_data["Noe"].x, original_noe_x))

    def test_dimension_consistency(self):
        """Test that dimensions remain consistent throughout forward pass."""
        channels = 64

        config = make_config(embed_dim=channels, attention_dim=16, num_heads=4)
        module = BiAxialAttention(
            source_type_1="Residue",
            source_type_2="Peak",
            dest_type="Noe",
            edge_name_1="biaxial_attn_1",
            edge_name_2="biaxial_attn_2",
            embed_dim=config.shared.embed_dim, attention_config=config.attention, combine_mlp_config=config.combine_mlp, device="cpu",
        )

        # Create graph
        data = HeteroData()
        num_noes = 5
        data["Noe"].x = torch.randn(num_noes, channels)
        data["Residue"].x = torch.randn(10, channels)
        data["Peak"].x = torch.randn(8, channels)

        # Add edges (Residue/Peak -> Noe)
        res_to_noe_edges = torch.cartesian_prod(
            torch.arange(10), torch.arange(num_noes)
        ).t()
        data["Residue", "biaxial_attn_1", "Noe"].edge_index = res_to_noe_edges

        peak_to_noe_edges = torch.cartesian_prod(
            torch.arange(8), torch.arange(num_noes)
        ).t()
        data["Peak", "biaxial_attn_2", "Noe"].edge_index = peak_to_noe_edges

        # Run forward
        output_data = module(data)

        # Verify output dimension is channels
        self.assertEqual(output_data["Noe"].x.shape, (num_noes, channels))


class TestBiAxialAttentionEdgeCases(unittest.TestCase):
    """Test comprehensive edge cases."""

    def setUp(self):
        """Set up test fixtures."""
        config = make_config(embed_dim=64, attention_dim=16, num_heads=4)
        self.module = BiAxialAttention(
            source_type_1="Residue",
            source_type_2="Peak",
            dest_type="Noe",
            edge_name_1="biaxial_attn_1",
            edge_name_2="biaxial_attn_2",
            embed_dim=config.shared.embed_dim, attention_config=config.attention, combine_mlp_config=config.combine_mlp, device="cpu",
        )

    def test_zero_edges_attention_1(self):
        """Test with zero edges for Residue attention."""
        data = HeteroData()
        num_noes = 5

        data["Noe"].x = torch.randn(num_noes, 64)
        data["Residue"].x = torch.randn(10, 64)
        data["Peak"].x = torch.randn(8, 64)

        # Zero Residue edges, but valid Peak edges
        data["Residue", "biaxial_attn_1", "Noe"].edge_index = torch.empty(2, 0, dtype=torch.long)

        peak_to_noe_edges = torch.cartesian_prod(
            torch.arange(8), torch.arange(num_noes)
        ).t()
        data["Peak", "biaxial_attn_2", "Noe"].edge_index = peak_to_noe_edges

        # Should return data unchanged (early return on empty edges)
        output_data = self.module(data)

        self.assertIsInstance(output_data, HeteroData)
        # NOE features should remain unchanged (in_channels)
        self.assertEqual(output_data["Noe"].x.shape, (num_noes, 64))

    def test_zero_edges_attention_2(self):
        """Test with zero edges for Peak attention."""
        data = HeteroData()
        num_noes = 5

        data["Noe"].x = torch.randn(num_noes, 64)
        data["Residue"].x = torch.randn(10, 64)
        data["Peak"].x = torch.randn(8, 64)

        # Valid Residue edges, but zero Peak edges
        res_to_noe_edges = torch.cartesian_prod(
            torch.arange(10), torch.arange(num_noes)
        ).t()
        data["Residue", "biaxial_attn_1", "Noe"].edge_index = res_to_noe_edges

        data["Peak", "biaxial_attn_2", "Noe"].edge_index = torch.empty(2, 0, dtype=torch.long)

        # Should return data unchanged (early return on empty edges)
        output_data = self.module(data)

        self.assertIsInstance(output_data, HeteroData)
        # NOE features should remain unchanged (in_channels)
        self.assertEqual(output_data["Noe"].x.shape, (num_noes, 64))

    def test_all_empty_simultaneously(self):
        """Test with all node types empty simultaneously."""
        data = HeteroData()

        data["Noe"].x = torch.empty(0, 64)
        data["Residue"].x = torch.empty(0, 64)
        data["Peak"].x = torch.empty(0, 64)

        data["Residue", "biaxial_attn_1", "Noe"].edge_index = torch.empty(2, 0, dtype=torch.long)
        data["Peak", "biaxial_attn_2", "Noe"].edge_index = torch.empty(2, 0, dtype=torch.long)

        # Should return data unchanged without errors
        output_data = self.module(data)

        self.assertIsInstance(output_data, HeteroData)
        self.assertEqual(output_data["Noe"].x.shape[0], 0)
        self.assertEqual(output_data["Residue"].x.shape[0], 0)
        self.assertEqual(output_data["Peak"].x.shape[0], 0)


class TestBiAxialAttentionRealisticGraphs(unittest.TestCase):
    """Test with realistic graph structures."""

    def test_varying_node_counts(self):
        """Test with varying numbers of NOE, Residue, and Peak nodes."""
        test_cases = [
            (1, 5, 3),    # Single NOE
            (10, 20, 15), # Medium-sized graph
            (50, 100, 80),# Large graph
            (3, 3, 3),    # Equal node counts
        ]

        for num_noes, num_residues, num_peaks in test_cases:
            with self.subTest(noes=num_noes, residues=num_residues, peaks=num_peaks):
                config = make_config(embed_dim=64, attention_dim=16, num_heads=4)
                module = BiAxialAttention(
                    source_type_1="Residue",
                    source_type_2="Peak",
                    dest_type="Noe",
                    edge_name_1="biaxial_attn_1",
                    edge_name_2="biaxial_attn_2",
                    embed_dim=config.shared.embed_dim, attention_config=config.attention, combine_mlp_config=config.combine_mlp, device="cpu",
                )

                data = HeteroData()
                data["Noe"].x = torch.randn(num_noes, 64)
                data["Residue"].x = torch.randn(num_residues, 64)
                data["Peak"].x = torch.randn(num_peaks, 64)

                # Fully connected bipartite graphs
                res_to_noe_edges = torch.cartesian_prod(
                    torch.arange(num_residues), torch.arange(num_noes)
                ).t()
                data["Residue", "biaxial_attn_1", "Noe"].edge_index = res_to_noe_edges

                peak_to_noe_edges = torch.cartesian_prod(
                    torch.arange(num_peaks), torch.arange(num_noes)
                ).t()
                data["Peak", "biaxial_attn_2", "Noe"].edge_index = peak_to_noe_edges

                # Run forward
                output_data = module(data)

                # Verify output shape
                self.assertEqual(output_data["Noe"].x.shape, (num_noes, 64))

    def test_sparse_edge_connectivity(self):
        """Test with sparse edge connectivity (not fully connected)."""
        config = make_config(embed_dim=64, attention_dim=16, num_heads=4)
        module = BiAxialAttention(
            source_type_1="Residue",
            source_type_2="Peak",
            dest_type="Noe",
            edge_name_1="biaxial_attn_1",
            edge_name_2="biaxial_attn_2",
            embed_dim=config.shared.embed_dim, attention_config=config.attention, combine_mlp_config=config.combine_mlp, device="cpu",
        )

        data = HeteroData()
        num_noes = 10
        data["Noe"].x = torch.randn(num_noes, 64)
        data["Residue"].x = torch.randn(20, 64)
        data["Peak"].x = torch.randn(15, 64)

        # Sparse edges: only connect first 5 NOEs to first 10 Residues
        res_edge_src = torch.arange(10).repeat_interleave(5)  # [0,0,0,0,0,1,1,1,1,1,...]
        res_edge_dst = torch.arange(5).repeat(10)              # [0,1,2,3,4,0,1,2,3,4,...]
        data["Residue", "biaxial_attn_1", "Noe"].edge_index = torch.stack([res_edge_src, res_edge_dst])

        # Sparse edges: only connect last 5 NOEs to last 10 Peaks
        peak_edge_src = torch.arange(5, 15).repeat_interleave(5)  # [5,5,5,5,5,6,6,6,6,6,...]
        peak_edge_dst = torch.arange(5, 10).repeat(10)             # [5,6,7,8,9,5,6,7,8,9,...]
        data["Peak", "biaxial_attn_2", "Noe"].edge_index = torch.stack([peak_edge_src, peak_edge_dst])

        # Run forward
        output_data = module(data)

        # Verify output shape
        self.assertEqual(output_data["Noe"].x.shape, (num_noes, 64))


class TestBiAxialAttentionDimensionFlow(unittest.TestCase):
    """Test dimension consistency throughout the forward pass."""

    def test_dimension_flow(self):
        """Test that dimensions flow correctly through all components."""
        channels = 64
        num_noes = 5

        config = make_config(embed_dim=channels, attention_dim=16, num_heads=4)
        module = BiAxialAttention(
            source_type_1="Residue",
            source_type_2="Peak",
            dest_type="Noe",
            edge_name_1="biaxial_attn_1",
            edge_name_2="biaxial_attn_2",
            embed_dim=config.shared.embed_dim, attention_config=config.attention, combine_mlp_config=config.combine_mlp, device="cpu",
        )

        # Create graph
        data = HeteroData()
        data["Noe"].x = torch.randn(num_noes, channels)
        data["Residue"].x = torch.randn(10, channels)
        data["Peak"].x = torch.randn(8, channels)

        # Fully connected edges
        res_to_noe_edges = torch.cartesian_prod(
            torch.arange(10), torch.arange(num_noes)
        ).t()
        data["Residue", "biaxial_attn_1", "Noe"].edge_index = res_to_noe_edges

        peak_to_noe_edges = torch.cartesian_prod(
            torch.arange(8), torch.arange(num_noes)
        ).t()
        data["Peak", "biaxial_attn_2", "Noe"].edge_index = peak_to_noe_edges

        # Test internal components by manually running forward steps
        noe_x = data["Noe"].x
        res_x = data["Residue"].x
        peak_x = data["Peak"].x

        # Test attention deltas
        res_delta = module.attention_1(res_x, noe_x, res_to_noe_edges)
        self.assertEqual(res_delta.shape, (num_noes, channels))

        peak_delta = module.attention_2(peak_x, noe_x, peak_to_noe_edges)
        self.assertEqual(peak_delta.shape, (num_noes, channels))

        # Test NOE transformation
        noe_transformed = module.dest_linear(noe_x)
        self.assertEqual(noe_transformed.shape, (num_noes, channels))

        # Test MLP input (concatenated)
        combined = torch.cat([res_delta, peak_delta, noe_transformed], dim=-1)
        self.assertEqual(combined.shape, (num_noes, channels * 3))

        # Test MLP output
        delta = module.combine_mlp(combined)
        self.assertEqual(delta.shape, (num_noes, channels))

        # Test final output
        output_data = module(data)
        self.assertEqual(output_data["Noe"].x.shape, (num_noes, channels))


class TestBiAxialAttentionIntegration(unittest.TestCase):
    """Test integration with existing attention infrastructure."""

    def test_device_consistency_cpu(self):
        """Test that module works correctly on CPU device."""
        config = make_config(embed_dim=64, attention_dim=16, num_heads=4)
        module = BiAxialAttention(
            source_type_1="Residue",
            source_type_2="Peak",
            dest_type="Noe",
            edge_name_1="biaxial_attn_1",
            edge_name_2="biaxial_attn_2",
            embed_dim=config.shared.embed_dim, attention_config=config.attention, combine_mlp_config=config.combine_mlp, device="cpu",
        )

        data = HeteroData()
        data["Noe"].x = torch.randn(5, 64)
        data["Residue"].x = torch.randn(10, 64)
        data["Peak"].x = torch.randn(8, 64)

        res_to_noe_edges = torch.cartesian_prod(
            torch.arange(10), torch.arange(5)
        ).t()
        data["Residue", "biaxial_attn_1", "Noe"].edge_index = res_to_noe_edges

        peak_to_noe_edges = torch.cartesian_prod(
            torch.arange(8), torch.arange(5)
        ).t()
        data["Peak", "biaxial_attn_2", "Noe"].edge_index = peak_to_noe_edges

        # Should work without errors
        output_data = module(data)
        self.assertEqual(output_data["Noe"].x.device.type, "cpu")

    def test_module_list_compatibility(self):
        """Test that module can be added to nn.ModuleList."""
        config = make_config(embed_dim=64, attention_dim=16, num_heads=4)
        module = BiAxialAttention(
            source_type_1="Residue",
            source_type_2="Peak",
            dest_type="Noe",
            edge_name_1="biaxial_attn_1",
            edge_name_2="biaxial_attn_2",
            embed_dim=config.shared.embed_dim, attention_config=config.attention, combine_mlp_config=config.combine_mlp, device="cpu",
        )

        # Should be compatible with nn.ModuleList
        module_list = torch.nn.ModuleList([module])
        self.assertEqual(len(module_list), 1)
        self.assertIsInstance(module_list[0], BiAxialAttention)

    def test_sequential_compatibility(self):
        """Test that module can be used in sequential operations."""
        config = make_config(embed_dim=64, attention_dim=16, num_heads=4)
        module1 = BiAxialAttention(
            source_type_1="Residue",
            source_type_2="Peak",
            dest_type="Noe",
            edge_name_1="biaxial_attn_1",
            edge_name_2="biaxial_attn_2",
            embed_dim=config.shared.embed_dim, attention_config=config.attention, combine_mlp_config=config.combine_mlp, device="cpu",
        )

        module2 = BiAxialAttention(
            source_type_1="Residue",
            source_type_2="Peak",
            dest_type="Noe",
            edge_name_1="biaxial_attn_1",
            edge_name_2="biaxial_attn_2",
            embed_dim=config.shared.embed_dim, attention_config=config.attention, combine_mlp_config=config.combine_mlp, device="cpu",
        )

        # Create graph
        data = HeteroData()
        data["Noe"].x = torch.randn(5, 64)
        data["Residue"].x = torch.randn(10, 64)
        data["Peak"].x = torch.randn(8, 64)

        res_to_noe_edges = torch.cartesian_prod(
            torch.arange(10), torch.arange(5)
        ).t()
        data["Residue", "biaxial_attn_1", "Noe"].edge_index = res_to_noe_edges

        peak_to_noe_edges = torch.cartesian_prod(
            torch.arange(8), torch.arange(5)
        ).t()
        data["Peak", "biaxial_attn_2", "Noe"].edge_index = peak_to_noe_edges

        # Apply modules sequentially
        data = module1(data)

        # Update Residue and Peak features to match new dimension
        data["Residue"].x = torch.randn(10, 64)
        data["Peak"].x = torch.randn(8, 64)

        data = module2(data)

        # Verify final output
        self.assertEqual(data["Noe"].x.shape, (5, 64))


if __name__ == "__main__":
    unittest.main()
