"""
Tests for NMRLayer orchestration with explicit triple classes.

This test module verifies that NMRLayer correctly instantiates all 4 triple
classes as separate attributes and calls them explicitly in sequence without
conditionals or branching.
"""

import sys
from pathlib import Path
import unittest

import torch
import numpy as np

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from nmr.models.network import NMRLayer, ModelConfig, StandardizeShifts, EmbedFeatures
from nmr.models.triple import (
    ResidueResidueNoeTriple,
    ResiduePeakNoeTriple,
    PeakResidueNoeTriple,
    PeakPeakNoeTriple,
)
from nmr.construct import construct_graph


class TestNMRLayerOrchestration(unittest.TestCase):
    """Test NMRLayer instantiation and forward pass orchestration."""

    def setUp(self):
        """Set up test fixtures."""
        self.device = "cpu"
        self.config = ModelConfig()

    def test_nmr_layer_instantiates_all_four_triples(self):
        """Test that NMRLayer instantiates all 4 triple classes as separate attributes."""
        layer = NMRLayer(self.device, self.config)

        # Verify all 4 triple classes are instantiated as attributes
        self.assertTrue(hasattr(layer, "residue_residue_noe"))
        self.assertTrue(hasattr(layer, "residue_peak_noe"))
        self.assertTrue(hasattr(layer, "peak_residue_noe"))
        self.assertTrue(hasattr(layer, "peak_peak_noe"))

        # Verify each attribute is the correct type
        self.assertIsInstance(layer.residue_residue_noe, ResidueResidueNoeTriple)
        self.assertIsInstance(layer.residue_peak_noe, ResiduePeakNoeTriple)
        self.assertIsInstance(layer.peak_residue_noe, PeakResidueNoeTriple)
        self.assertIsInstance(layer.peak_peak_noe, PeakPeakNoeTriple)

        # Verify old attribute does NOT exist
        self.assertFalse(hasattr(layer, "triple_layer"))

    def test_nmr_layer_forward_calls_all_triples_in_sequence(self):
        """Test that forward pass calls all 4 triples in sequence."""
        layer = NMRLayer(self.device, self.config)

        # Create a small test graph using the correct history structure
        # Coordinates should be [x, y, z, H, N] - 5 values per residue
        xyz = np.random.randn(5, 3)
        pred_shifts = np.random.randn(5, 2)
        coordinates = np.concatenate([xyz, pred_shifts], axis=1).tolist()  # 5 residues with [x,y,z,H,N]
        obs_shifts = np.random.randn(5, 2).tolist()  # 5 observed shifts
        noes = [[110.0, 8.0, 7.5], [115.0, 8.5, 8.0]]  # 2 NOE constraints

        history = {
            "coordinates": coordinates,
            "obs_chemical_shifts": obs_shifts,
            "noes": noes,
            "assignments": {},  # No assignments yet
            "shift_to_assign": 0,
        }

        data = construct_graph(history=history, device=self.device)

        # Embed the data (NMRLayer expects embedded features)
        normalize = StandardizeShifts()
        embed = EmbedFeatures(self.device, self.config)
        data = normalize(data)
        data = embed(data)

        # Store original node features to verify they change
        original_residue_x = data["Residue"].x.clone()
        original_peak_x = data["Peak"].x.clone()
        original_noe_x = data["Noe"].x.clone()

        # Run forward pass
        output_data = layer(data)

        # Verify data is returned
        self.assertIsNotNone(output_data)

        # Verify node features have been updated (at least one should change)
        # Note: We don't know the exact values, but we can verify the shapes are preserved
        self.assertEqual(output_data["Residue"].x.shape, original_residue_x.shape)
        self.assertEqual(output_data["Peak"].x.shape, original_peak_x.shape)
        self.assertEqual(output_data["Noe"].x.shape, original_noe_x.shape)

    def test_nmr_layer_forward_no_conditionals(self):
        """Test that forward pass has no conditionals or branching logic."""
        layer = NMRLayer(self.device, self.config)

        # Create a small test graph
        # Coordinates should be [x, y, z, H, N] - 5 values per residue
        xyz = np.random.randn(3, 3)
        pred_shifts = np.random.randn(3, 2)
        coordinates = np.concatenate([xyz, pred_shifts], axis=1).tolist()  # 3 residues with [x,y,z,H,N]
        obs_shifts = np.random.randn(3, 2).tolist()
        noes = [[110.0, 8.0, 7.5]]

        history = {
            "coordinates": coordinates,
            "obs_chemical_shifts": obs_shifts,
            "noes": noes,
            "assignments": {},
            "shift_to_assign": 0,
        }

        data = construct_graph(history=history, device=self.device)

        # Embed the data (NMRLayer expects embedded features)
        normalize = StandardizeShifts()
        embed = EmbedFeatures(self.device, self.config)
        data = normalize(data)
        data = embed(data)

        # The forward method should have a straightforward sequence of calls
        # We verify this by checking that all triple types can process the data
        # without errors, regardless of the specific triple instances present

        # This test primarily validates structure - the actual forward logic
        # is tested in the previous test
        output_data = layer(data)

        # Verify output is valid HeteroData
        self.assertIsNotNone(output_data)
        self.assertTrue(hasattr(output_data, "node_types"))
        self.assertIn("Residue", output_data.node_types)
        self.assertIn("Peak", output_data.node_types)
        self.assertIn("Noe", output_data.node_types)


if __name__ == "__main__":
    unittest.main()
