"""
Unit tests for triple composition classes.

Tests the 4 triple composition classes that wire together gather, update, and
scatter operations:
- ResidueResidueNoeTriple
- ResiduePeakNoeTriple
- PeakResidueNoeTriple
- PeakPeakNoeTriple
"""

import sys
import unittest
from pathlib import Path

import torch
from torch_geometric.data import HeteroData

# Add parent directory to path to allow imports from nmr package
sys.path.insert(0, str(Path(__file__).parent.parent))

from nmr.models.triple import (
    ResidueResidueNoeTriple,
    ResiduePeakNoeTriple,
    PeakResidueNoeTriple,
    PeakPeakNoeTriple,
)
from nmr.models import ModelConfig


class TestResidueResidueNoeTriple(unittest.TestCase):
    """Test ResidueResidueNoeTriple wiring and forward pass."""

    def setUp(self):
        """Set up test graph with all necessary nodes and edges."""
        self.device = torch.device("cpu")
        self.config = ModelConfig()
        self.shift_dim = self.config.shift_embed.output_dim
        self.feature_dim = self.config.feature_embed.output_dim
        self.data = HeteroData()

        # Create Residue nodes [n, 3 + shift_dim] = [x, y, z, shifts...]
        coords = torch.tensor(
            [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]],
            dtype=torch.float32,
            device=self.device,
        )
        shifts = torch.randn((2, self.shift_dim), dtype=torch.float32, device=self.device)
        self.data["Residue"].x = torch.cat([coords, shifts], dim=-1)
        self.data["Residue"].f = torch.randn(
            (2, self.feature_dim), dtype=torch.float32, device=self.device
        )

        # Create Noe nodes [n, shift_dim] = embedded NOE features
        self.data["Noe"].x = torch.randn(
            (1, self.shift_dim), dtype=torch.float32, device=self.device
        )
        self.data["Noe"].f = torch.randn(
            (1, self.feature_dim), dtype=torch.float32, device=self.device
        )

        # Create triple nodes
        self.data["ResidueResidueNoeTriple"].x = torch.zeros(
            (1, 1), dtype=torch.float32, device=self.device
        )

        # Create gather edges
        self.data[
            "Residue", "prop_first", "ResidueResidueNoeTriple"
        ].edge_index = torch.tensor([[0], [0]], dtype=torch.long, device=self.device)

        self.data[
            "Residue", "prop_second", "ResidueResidueNoeTriple"
        ].edge_index = torch.tensor([[1], [0]], dtype=torch.long, device=self.device)

        self.data["Noe", "prop_noe", "ResidueResidueNoeTriple"].edge_index = (
            torch.tensor([[0], [0]], dtype=torch.long, device=self.device)
        )

        # Create scatter edges (reverse direction)
        self.data[
            "ResidueResidueNoeTriple", "prop_first", "Residue"
        ].edge_index = torch.tensor([[0], [0]], dtype=torch.long, device=self.device)

        self.data[
            "ResidueResidueNoeTriple", "prop_second", "Residue"
        ].edge_index = torch.tensor([[0], [1]], dtype=torch.long, device=self.device)

        self.data["ResidueResidueNoeTriple", "prop_noe", "Noe"].edge_index = (
            torch.tensor([[0], [0]], dtype=torch.long, device=self.device)
        )

    def test_triple_instantiation(self):
        """Test that ResidueResidueNoeTriple instantiates correctly."""
        triple = ResidueResidueNoeTriple(self.device, self.config)

        # Check that it has all the required components
        self.assertTrue(hasattr(triple, "first_gather"))
        self.assertTrue(hasattr(triple, "second_gather"))
        self.assertTrue(hasattr(triple, "noe_gather"))
        self.assertTrue(hasattr(triple, "update"))
        self.assertTrue(hasattr(triple, "first_scatter"))
        self.assertTrue(hasattr(triple, "second_scatter"))
        self.assertTrue(hasattr(triple, "noe_scatter"))

    def test_forward_pass_executes(self):
        """Test that forward pass executes without error."""
        triple = ResidueResidueNoeTriple(self.device, self.config)

        # Should not raise an error
        data = triple(self.data)

        # Data should be returned
        self.assertIsNotNone(data)

    def test_forward_pass_sequence(self):
        """Test that forward pass calls gather -> update -> scatter in sequence."""
        triple = ResidueResidueNoeTriple(self.device, self.config)
        data = triple(self.data)

        # After forward pass, check that:
        # 1. Gathered attributes exist on triple nodes (from gather operations)
        self.assertIn("first_coords", data["ResidueResidueNoeTriple"])
        self.assertIn("second_coords", data["ResidueResidueNoeTriple"])
        self.assertIn("noe_shifts", data["ResidueResidueNoeTriple"])

        # 2. Delta attributes exist on triple nodes (from update operation)
        self.assertIn("delta_first_coords", data["ResidueResidueNoeTriple"])
        self.assertIn("delta_second_shifts", data["ResidueResidueNoeTriple"])

        # 3. Residue and Noe nodes have been updated (from scatter operations)
        # We can't easily check the exact values, but we can verify the tensors exist
        self.assertIsNotNone(data["Residue"].x)
        self.assertIsNotNone(data["Noe"].x)


class TestResiduePeakNoeTriple(unittest.TestCase):
    """Test ResiduePeakNoeTriple wiring and forward pass."""

    def setUp(self):
        """Set up test graph with Residue, Peak, and Noe nodes."""
        self.device = torch.device("cpu")
        self.config = ModelConfig()
        self.shift_dim = self.config.shift_embed.output_dim
        self.feature_dim = self.config.feature_embed.output_dim
        self.data = HeteroData()

        # Create Residue nodes
        coords = torch.tensor([[1.0, 2.0, 3.0]], dtype=torch.float32, device=self.device)
        shifts = torch.randn((1, self.shift_dim), dtype=torch.float32, device=self.device)
        self.data["Residue"].x = torch.cat([coords, shifts], dim=-1)
        self.data["Residue"].f = torch.randn(
            (1, self.feature_dim), dtype=torch.float32, device=self.device
        )

        # Create Peak nodes
        self.data["Peak"].x = torch.randn((1, self.shift_dim), dtype=torch.float32, device=self.device)
        self.data["Peak"].f = torch.randn(
            (1, self.feature_dim), dtype=torch.float32, device=self.device
        )

        # Create Noe nodes
        self.data["Noe"].x = torch.randn(
            (1, self.shift_dim), dtype=torch.float32, device=self.device
        )
        self.data["Noe"].f = torch.randn(
            (1, self.feature_dim), dtype=torch.float32, device=self.device
        )

        # Create triple nodes
        self.data["ResiduePeakNoeTriple"].x = torch.zeros(
            (1, 1), dtype=torch.float32, device=self.device
        )

        # Create gather edges
        self.data["Residue", "prop_first", "ResiduePeakNoeTriple"].edge_index = (
            torch.tensor([[0], [0]], dtype=torch.long, device=self.device)
        )
        self.data["Peak", "prop_second", "ResiduePeakNoeTriple"].edge_index = (
            torch.tensor([[0], [0]], dtype=torch.long, device=self.device)
        )
        self.data["Noe", "prop_noe", "ResiduePeakNoeTriple"].edge_index = (
            torch.tensor([[0], [0]], dtype=torch.long, device=self.device)
        )

        # Create scatter edges
        self.data["ResiduePeakNoeTriple", "prop_first", "Residue"].edge_index = (
            torch.tensor([[0], [0]], dtype=torch.long, device=self.device)
        )
        self.data["ResiduePeakNoeTriple", "prop_second", "Peak"].edge_index = (
            torch.tensor([[0], [0]], dtype=torch.long, device=self.device)
        )
        self.data["ResiduePeakNoeTriple", "prop_noe", "Noe"].edge_index = (
            torch.tensor([[0], [0]], dtype=torch.long, device=self.device)
        )

    def test_triple_uses_correct_components(self):
        """Test that ResiduePeakNoeTriple uses correct gather/scatter types."""
        triple = ResiduePeakNoeTriple(self.device, self.config)

        # Check component types by examining their triple_type attribute
        self.assertEqual(triple.first_gather.triple_type, "ResiduePeakNoeTriple")
        self.assertEqual(triple.second_gather.triple_type, "ResiduePeakNoeTriple")
        self.assertEqual(triple.noe_gather.triple_type, "ResiduePeakNoeTriple")
        self.assertEqual(triple.update.triple_type, "ResiduePeakNoeTriple")

        # Check edge types to ensure correct gather/scatter classes
        self.assertEqual(
            triple.first_gather.edge_type,
            ("Residue", "prop_first", "ResiduePeakNoeTriple"),
        )
        self.assertEqual(
            triple.second_gather.edge_type,
            ("Peak", "prop_second", "ResiduePeakNoeTriple"),
        )

    def test_forward_pass_executes(self):
        """Test that forward pass executes without error."""
        triple = ResiduePeakNoeTriple(self.device, self.config)
        data = triple(self.data)
        self.assertIsNotNone(data)


class TestPeakResidueNoeTriple(unittest.TestCase):
    """Test PeakResidueNoeTriple wiring and forward pass."""

    def setUp(self):
        """Set up test graph with Peak, Residue, and Noe nodes."""
        self.device = torch.device("cpu")
        self.config = ModelConfig()
        self.shift_dim = self.config.shift_embed.output_dim
        self.feature_dim = self.config.feature_embed.output_dim
        self.data = HeteroData()

        # Create Peak nodes
        self.data["Peak"].x = torch.randn((1, self.shift_dim), dtype=torch.float32, device=self.device)
        self.data["Peak"].f = torch.randn(
            (1, self.feature_dim), dtype=torch.float32, device=self.device
        )

        # Create Residue nodes
        coords = torch.tensor([[1.0, 2.0, 3.0]], dtype=torch.float32, device=self.device)
        shifts = torch.randn((1, self.shift_dim), dtype=torch.float32, device=self.device)
        self.data["Residue"].x = torch.cat([coords, shifts], dim=-1)
        self.data["Residue"].f = torch.randn(
            (1, self.feature_dim), dtype=torch.float32, device=self.device
        )

        # Create Noe nodes
        self.data["Noe"].x = torch.randn(
            (1, self.shift_dim), dtype=torch.float32, device=self.device
        )
        self.data["Noe"].f = torch.randn(
            (1, self.feature_dim), dtype=torch.float32, device=self.device
        )

        # Create triple nodes
        self.data["PeakResidueNoeTriple"].x = torch.zeros(
            (1, 1), dtype=torch.float32, device=self.device
        )

        # Create gather edges
        self.data["Peak", "prop_first", "PeakResidueNoeTriple"].edge_index = (
            torch.tensor([[0], [0]], dtype=torch.long, device=self.device)
        )
        self.data["Residue", "prop_second", "PeakResidueNoeTriple"].edge_index = (
            torch.tensor([[0], [0]], dtype=torch.long, device=self.device)
        )
        self.data["Noe", "prop_noe", "PeakResidueNoeTriple"].edge_index = (
            torch.tensor([[0], [0]], dtype=torch.long, device=self.device)
        )

        # Create scatter edges
        self.data["PeakResidueNoeTriple", "prop_first", "Peak"].edge_index = (
            torch.tensor([[0], [0]], dtype=torch.long, device=self.device)
        )
        self.data["PeakResidueNoeTriple", "prop_second", "Residue"].edge_index = (
            torch.tensor([[0], [0]], dtype=torch.long, device=self.device)
        )
        self.data["PeakResidueNoeTriple", "prop_noe", "Noe"].edge_index = (
            torch.tensor([[0], [0]], dtype=torch.long, device=self.device)
        )

    def test_forward_pass_executes(self):
        """Test that forward pass executes without error."""
        triple = PeakResidueNoeTriple(self.device, self.config)
        data = triple(self.data)
        self.assertIsNotNone(data)


class TestPeakPeakNoeTriple(unittest.TestCase):
    """Test PeakPeakNoeTriple wiring and forward pass."""

    def setUp(self):
        """Set up test graph with Peak and Noe nodes."""
        self.device = torch.device("cpu")
        self.config = ModelConfig()
        self.shift_dim = self.config.shift_embed.output_dim
        self.feature_dim = self.config.feature_embed.output_dim
        self.data = HeteroData()

        # Create Peak nodes
        self.data["Peak"].x = torch.randn((2, self.shift_dim), dtype=torch.float32, device=self.device)
        self.data["Peak"].f = torch.randn(
            (2, self.feature_dim), dtype=torch.float32, device=self.device
        )

        # Create Noe nodes
        self.data["Noe"].x = torch.randn(
            (1, self.shift_dim), dtype=torch.float32, device=self.device
        )
        self.data["Noe"].f = torch.randn(
            (1, self.feature_dim), dtype=torch.float32, device=self.device
        )

        # Create triple nodes
        self.data["PeakPeakNoeTriple"].x = torch.zeros(
            (1, 1), dtype=torch.float32, device=self.device
        )

        # Create gather edges
        self.data["Peak", "prop_first", "PeakPeakNoeTriple"].edge_index = (
            torch.tensor([[0], [0]], dtype=torch.long, device=self.device)
        )
        self.data["Peak", "prop_second", "PeakPeakNoeTriple"].edge_index = (
            torch.tensor([[1], [0]], dtype=torch.long, device=self.device)
        )
        self.data["Noe", "prop_noe", "PeakPeakNoeTriple"].edge_index = (
            torch.tensor([[0], [0]], dtype=torch.long, device=self.device)
        )

        # Create scatter edges
        self.data["PeakPeakNoeTriple", "prop_first", "Peak"].edge_index = (
            torch.tensor([[0], [0]], dtype=torch.long, device=self.device)
        )
        self.data["PeakPeakNoeTriple", "prop_second", "Peak"].edge_index = (
            torch.tensor([[0], [1]], dtype=torch.long, device=self.device)
        )
        self.data["PeakPeakNoeTriple", "prop_noe", "Noe"].edge_index = (
            torch.tensor([[0], [0]], dtype=torch.long, device=self.device)
        )

    def test_forward_pass_executes(self):
        """Test that forward pass executes without error."""
        triple = PeakPeakNoeTriple(self.device, self.config)
        data = triple(self.data)
        self.assertIsNotNone(data)


if __name__ == "__main__":
    unittest.main()
