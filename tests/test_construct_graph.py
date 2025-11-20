"""
Tests for architecture-aware graph construction.

Verifies that construct_graph builds the correct node and edge types based on
ModelConfig.layer_type (triple vs transformer architecture).
"""

import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

import unittest
import torch
from nmr.construct import construct_graph
from nmr.models.config import ModelConfig


class TestTripleArchitectureConstruction(unittest.TestCase):
    """Tests for triple architecture graph construction."""

    def setUp(self):
        """Create minimal test data for graph construction."""
        self.device = torch.device("cpu")
        self.config_triple = ModelConfig(layer_type="triple")
        self.history = {
            "coordinates": [
                [0.0, 0.0, 0.0, 7.5, 115.0],
                [1.0, 0.0, 0.0, 8.0, 120.0],
            ],
            "obs_chemical_shifts": [[7.5, 115.0], [8.0, 120.0]],
            "noes": [[115.0, 7.5, 8.0]],
            "shift_to_assign": 0,
            "assignments": {},
        }

    def test_triple_architecture_builds_triple_nodes(self):
        """Test that triple architecture creates all four triple node types."""
        graph = construct_graph(self.history, self.device, self.config_triple)

        # Verify all triple node types exist
        self.assertIn("ResidueResidueNoeTriple", graph.node_types)
        self.assertIn("ResiduePeakNoeTriple", graph.node_types)
        self.assertIn("PeakResidueNoeTriple", graph.node_types)
        self.assertIn("PeakPeakNoeTriple", graph.node_types)

        # Verify triple nodes have correct feature dimensions
        embed_dim = self.config_triple.shared.embed_dim
        self.assertEqual(graph["ResidueResidueNoeTriple"].x.shape[1], embed_dim)
        self.assertEqual(graph["ResiduePeakNoeTriple"].x.shape[1], embed_dim)
        self.assertEqual(graph["PeakResidueNoeTriple"].x.shape[1], embed_dim)
        self.assertEqual(graph["PeakPeakNoeTriple"].x.shape[1], embed_dim)

    def test_triple_architecture_builds_triple_edges(self):
        """Test that triple architecture creates triple propagation edges."""
        graph = construct_graph(self.history, self.device, self.config_triple)

        # Verify triple propagation edges exist for ResidueResidueNoeTriple
        self.assertIn(
            ("Residue", "prop_first", "ResidueResidueNoeTriple"), graph.edge_types
        )
        self.assertIn(
            ("Residue", "prop_second", "ResidueResidueNoeTriple"), graph.edge_types
        )
        self.assertIn(("Noe", "prop_noe", "ResidueResidueNoeTriple"), graph.edge_types)

        # Verify edges exist for other triple types
        self.assertIn(
            ("Residue", "prop_first", "ResiduePeakNoeTriple"), graph.edge_types
        )
        self.assertIn(("Peak", "prop_second", "ResiduePeakNoeTriple"), graph.edge_types)
        self.assertIn(("Peak", "prop_first", "PeakResidueNoeTriple"), graph.edge_types)
        self.assertIn(("Peak", "prop_first", "PeakPeakNoeTriple"), graph.edge_types)

    def test_triple_architecture_no_attention_edges(self):
        """Test that triple architecture does NOT create transformer attention edges."""
        graph = construct_graph(self.history, self.device, self.config_triple)

        # Verify transformer attention edges do NOT exist
        self.assertNotIn(("Residue", "res_res_attn", "Residue"), graph.edge_types)
        self.assertNotIn(("Peak", "peak_peak_attn", "Peak"), graph.edge_types)
        self.assertNotIn(("Residue", "res_peak_attn", "Peak"), graph.edge_types)
        self.assertNotIn(("Peak", "peak_res_attn", "Residue"), graph.edge_types)
        self.assertNotIn(("Residue", "res_noe_attn", "Noe"), graph.edge_types)
        self.assertNotIn(("Noe", "noe_res_attn", "Residue"), graph.edge_types)
        self.assertNotIn(("Peak", "peak_noe_attn", "Noe"), graph.edge_types)
        self.assertNotIn(("Noe", "noe_peak_attn", "Peak"), graph.edge_types)


class TestTransformerArchitectureConstruction(unittest.TestCase):
    """Tests for transformer architecture graph construction."""

    def setUp(self):
        """Create minimal test data for graph construction."""
        self.device = torch.device("cpu")
        self.config_transformer = ModelConfig(layer_type="transformer")
        self.history = {
            "coordinates": [
                [0.0, 0.0, 0.0, 7.5, 115.0],
                [1.0, 0.0, 0.0, 8.0, 120.0],
            ],
            "obs_chemical_shifts": [[7.5, 115.0], [8.0, 120.0]],
            "noes": [[115.0, 7.5, 8.0]],
            "shift_to_assign": 0,
            "assignments": {},
        }

    def test_transformer_architecture_builds_attention_edges(self):
        """Test that transformer architecture creates all attention edge types."""
        graph = construct_graph(self.history, self.device, self.config_transformer)

        # Verify all transformer attention edges exist
        self.assertIn(("Residue", "res_res_attn", "Residue"), graph.edge_types)
        self.assertIn(("Peak", "peak_peak_attn", "Peak"), graph.edge_types)
        self.assertIn(("Residue", "res_peak_attn", "Peak"), graph.edge_types)
        self.assertIn(("Peak", "peak_res_attn", "Residue"), graph.edge_types)
        self.assertIn(("Residue", "res_noe_attn", "Noe"), graph.edge_types)
        self.assertIn(("Noe", "noe_res_attn", "Residue"), graph.edge_types)
        self.assertIn(("Peak", "peak_noe_attn", "Noe"), graph.edge_types)
        self.assertIn(("Noe", "noe_peak_attn", "Peak"), graph.edge_types)

        # Verify attention edges have correct shapes (fully connected)
        num_residue = 2
        num_peak = 2
        num_noe = 1

        # Residue self-attention: all-to-all (2 * 2 = 4 edges)
        res_res_edges = graph["Residue", "res_res_attn", "Residue"].edge_index
        self.assertEqual(res_res_edges.shape[1], num_residue * num_residue)

        # Peak self-attention: all-to-all (2 * 2 = 4 edges)
        peak_peak_edges = graph["Peak", "peak_peak_attn", "Peak"].edge_index
        self.assertEqual(peak_peak_edges.shape[1], num_peak * num_peak)

        # Cross-attention edges
        res_peak_edges = graph["Residue", "res_peak_attn", "Peak"].edge_index
        self.assertEqual(res_peak_edges.shape[1], num_residue * num_peak)

    def test_transformer_architecture_no_triple_nodes(self):
        """Test that transformer architecture does NOT create triple nodes."""
        graph = construct_graph(self.history, self.device, self.config_transformer)

        # Verify triple node types do NOT exist
        self.assertNotIn("ResidueResidueNoeTriple", graph.node_types)
        self.assertNotIn("ResiduePeakNoeTriple", graph.node_types)
        self.assertNotIn("PeakResidueNoeTriple", graph.node_types)
        self.assertNotIn("PeakPeakNoeTriple", graph.node_types)

    def test_transformer_architecture_no_triple_edges(self):
        """Test that transformer architecture does NOT create triple edges."""
        graph = construct_graph(self.history, self.device, self.config_transformer)

        # Verify triple propagation edges do NOT exist
        self.assertNotIn(
            ("Residue", "prop_first", "ResidueResidueNoeTriple"), graph.edge_types
        )
        self.assertNotIn(
            ("Residue", "prop_second", "ResidueResidueNoeTriple"), graph.edge_types
        )
        self.assertNotIn(("Noe", "prop_noe", "ResidueResidueNoeTriple"), graph.edge_types)
        self.assertNotIn(
            ("Residue", "prop_first", "ResiduePeakNoeTriple"), graph.edge_types
        )
        self.assertNotIn(("Peak", "prop_second", "ResiduePeakNoeTriple"), graph.edge_types)


class TestCommonNodeAndEdgeConstruction(unittest.TestCase):
    """Tests for nodes and edges common to both architectures."""

    def setUp(self):
        """Create minimal test data for graph construction."""
        self.device = torch.device("cpu")
        self.history = {
            "coordinates": [
                [0.0, 0.0, 0.0, 7.5, 115.0],
                [1.0, 0.0, 0.0, 8.0, 120.0],
            ],
            "obs_chemical_shifts": [[7.5, 115.0], [8.0, 120.0]],
            "noes": [[115.0, 7.5, 8.0]],
            "shift_to_assign": 0,
            "assignments": {0: 1},  # Peak 0 assigned to Residue 1
        }

    def test_both_architectures_build_common_nodes(self):
        """Test that both architectures create common data and value nodes."""
        for layer_type in ["triple", "transformer"]:
            config = ModelConfig(layer_type=layer_type)
            graph = construct_graph(self.history, self.device, config)

            # Verify common data nodes exist
            self.assertIn("Residue", graph.node_types)
            self.assertIn("Peak", graph.node_types)
            self.assertIn("Noe", graph.node_types)

            # Verify value aggregation nodes exist
            self.assertIn("VALUE_NOE", graph.node_types)
            self.assertIn("VALUE_SHIFT", graph.node_types)
            self.assertIn("VALUE_RES", graph.node_types)

    def test_both_architectures_build_common_edges(self):
        """Test that both architectures create assignment and value aggregation edges."""
        for layer_type in ["triple", "transformer"]:
            config = ModelConfig(layer_type=layer_type)
            graph = construct_graph(self.history, self.device, config)

            # Verify assignment edges exist
            self.assertIn(("Peak", "assigned_to", "Residue"), graph.edge_types)

            # Verify value aggregation edges exist
            self.assertIn(("Peak", "SHIFT_extract", "VALUE_SHIFT"), graph.edge_types)
            self.assertIn(("Noe", "aggregate", "VALUE_NOE"), graph.edge_types)
            self.assertIn(("Residue", "RES_extract", "VALUE_RES"), graph.edge_types)

            # Verify assignment edges have correct data
            assignment_edges = graph["Peak", "assigned_to", "Residue"].edge_index
            self.assertEqual(assignment_edges.shape[0], 2)  # [source, target]
            self.assertEqual(assignment_edges.shape[1], 1)  # One assignment
            self.assertEqual(assignment_edges[0, 0].item(), 0)  # Peak 0
            self.assertEqual(assignment_edges[1, 0].item(), 1)  # Residue 1

    def test_both_architectures_populate_raw_data_attributes(self):
        """Test that both architectures populate raw data attributes correctly."""
        for layer_type in ["triple", "transformer"]:
            config = ModelConfig(layer_type=layer_type)
            graph = construct_graph(self.history, self.device, config)

            # Verify Residue raw data attributes
            self.assertEqual(graph["Residue"].xyz.shape, (2, 3))
            self.assertEqual(graph["Residue"].shifts.shape, (2, 2))
            self.assertEqual(graph["Residue"].flags.shape, (2, 1))

            # Verify Peak raw data attributes
            self.assertEqual(graph["Peak"].shifts.shape, (2, 2))
            self.assertEqual(graph["Peak"].flags.shape, (2, 2))

            # Verify Noe raw data attributes
            self.assertEqual(graph["Noe"].shifts.shape, (1, 3))

            # Verify .x attributes are initialized with correct dimensions
            embed_dim = config.shared.embed_dim
            self.assertEqual(graph["Residue"].x.shape, (2, embed_dim))
            self.assertEqual(graph["Peak"].x.shape, (2, embed_dim))
            self.assertEqual(graph["Noe"].x.shape, (1, embed_dim))

            # Verify shift_to_assign is set correctly
            self.assertEqual(graph.shift_to_assign.item(), 0)

            # Verify assignment flags are set correctly
            # Peak 0 is assigned to Residue 1
            self.assertEqual(graph["Peak"].flags[0, 1].item(), 1.0)  # Previously assigned
            self.assertEqual(graph["Residue"].flags[1, 0].item(), 1.0)  # Previously assigned


if __name__ == "__main__":
    unittest.main()
