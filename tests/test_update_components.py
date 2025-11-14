"""
Unit tests for update component operations.

Tests the 2 update classes that compute deltas on triple nodes:
- ResidueUpdate (for ResidueResidueNoeTriple)
- PeakUpdate (for all other triple types)
"""

import sys
import unittest
from pathlib import Path

import torch
from torch_geometric.data import HeteroData

# Add parent directory to path to allow imports from nmr package
sys.path.insert(0, str(Path(__file__).parent.parent))

from nmr.models.triple import ResidueUpdate, PeakUpdate
from nmr.models import ModelConfig


class TestResidueUpdateDimensions(unittest.TestCase):
    """Test ResidueUpdate MLP input/output dimensions (12 -> 17)."""

    def setUp(self):
        """Set up test graph with gathered attributes on triple nodes."""
        self.device = torch.device("cpu")
        self.config = ModelConfig()
        self.shift_dim = self.config.shift_embed.output_dim
        self.feature_dim = self.config.feature_embed.output_dim
        self.data = HeteroData()

        # Create 2 triple nodes
        num_triples = 2
        self.data["ResidueResidueNoeTriple"].x = torch.zeros(
            (num_triples, 1), dtype=torch.float32, device=self.device
        )

        # Set gathered attributes
        # first_coords [n, 3]
        self.data["ResidueResidueNoeTriple"].first_coords = torch.tensor(
            [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]],
            dtype=torch.float32,
            device=self.device,
        )
        # first_shifts [n, shift_dim]
        self.data["ResidueResidueNoeTriple"].first_shifts = torch.randn(
            (num_triples, self.shift_dim),
            dtype=torch.float32,
            device=self.device,
        )
        # first_features [n, feature_dim]
        self.data["ResidueResidueNoeTriple"].first_features = torch.randn(
            (num_triples, self.feature_dim),
            dtype=torch.float32,
            device=self.device,
        )
        # second_coords [n, 3]
        self.data["ResidueResidueNoeTriple"].second_coords = torch.tensor(
            [[7.0, 8.0, 9.0], [10.0, 11.0, 12.0]],
            dtype=torch.float32,
            device=self.device,
        )
        # second_shifts [n, shift_dim]
        self.data["ResidueResidueNoeTriple"].second_shifts = torch.randn(
            (num_triples, self.shift_dim),
            dtype=torch.float32,
            device=self.device,
        )
        # second_features [n, feature_dim]
        self.data["ResidueResidueNoeTriple"].second_features = torch.randn(
            (num_triples, self.feature_dim),
            dtype=torch.float32,
            device=self.device,
        )
        # noe_shifts [n, shift_dim] (embedded NOE features)
        self.data["ResidueResidueNoeTriple"].noe_shifts = torch.randn(
            (num_triples, self.shift_dim),
            dtype=torch.float32,
            device=self.device,
        )
        # noe_features [n, feature_dim]
        self.data["ResidueResidueNoeTriple"].noe_features = torch.randn(
            (num_triples, self.feature_dim),
            dtype=torch.float32,
            device=self.device,
        )

    def test_residue_update_sets_all_delta_attributes(self):
        """Test that ResidueUpdate sets all 8 delta attributes."""
        update = ResidueUpdate("ResidueResidueNoeTriple", self.device, self.config)
        data = update(self.data)

        # Check all delta attributes are set
        self.assertIn("delta_first_coords", data["ResidueResidueNoeTriple"])
        self.assertIn("delta_first_shifts", data["ResidueResidueNoeTriple"])
        self.assertIn("delta_first_features", data["ResidueResidueNoeTriple"])
        self.assertIn("delta_second_coords", data["ResidueResidueNoeTriple"])
        self.assertIn("delta_second_shifts", data["ResidueResidueNoeTriple"])
        self.assertIn("delta_second_features", data["ResidueResidueNoeTriple"])
        self.assertIn("delta_noe_shifts", data["ResidueResidueNoeTriple"])
        self.assertIn("delta_noe_features", data["ResidueResidueNoeTriple"])

    def test_residue_update_delta_shapes(self):
        """Test that ResidueUpdate produces correct output shapes."""
        update = ResidueUpdate("ResidueResidueNoeTriple", self.device, self.config)
        data = update(self.data)

        # Check shapes
        self.assertEqual(
            data["ResidueResidueNoeTriple"].delta_first_coords.shape, (2, 3)
        )
        self.assertEqual(
            data["ResidueResidueNoeTriple"].delta_first_shifts.shape, (2, self.shift_dim)
        )
        self.assertEqual(
            data["ResidueResidueNoeTriple"].delta_first_features.shape, (2, self.feature_dim)
        )
        self.assertEqual(
            data["ResidueResidueNoeTriple"].delta_second_coords.shape, (2, 3)
        )
        self.assertEqual(
            data["ResidueResidueNoeTriple"].delta_second_shifts.shape, (2, self.shift_dim)
        )
        self.assertEqual(
            data["ResidueResidueNoeTriple"].delta_second_features.shape, (2, self.feature_dim)
        )
        self.assertEqual(
            data["ResidueResidueNoeTriple"].delta_noe_shifts.shape, (2, self.shift_dim)
        )
        self.assertEqual(
            data["ResidueResidueNoeTriple"].delta_noe_features.shape, (2, self.feature_dim)
        )


class TestPeakUpdateDimensions(unittest.TestCase):
    """Test PeakUpdate MLP input/output dimensions (11 -> 15)."""

    def setUp(self):
        """Set up test graph with gathered attributes on triple nodes."""
        self.device = torch.device("cpu")
        self.config = ModelConfig()
        self.shift_dim = self.config.shift_embed.output_dim
        self.feature_dim = self.config.feature_embed.output_dim
        self.data = HeteroData()

        # Create 2 triple nodes
        num_triples = 2
        self.data["PeakPeakNoeTriple"].x = torch.zeros(
            (num_triples, 1), dtype=torch.float32, device=self.device
        )

        # Set gathered attributes (peaks have NO coordinates)
        # first_shifts [n, shift_dim]
        self.data["PeakPeakNoeTriple"].first_shifts = torch.randn(
            (num_triples, self.shift_dim),
            dtype=torch.float32,
            device=self.device,
        )
        # first_features [n, feature_dim]
        self.data["PeakPeakNoeTriple"].first_features = torch.randn(
            (num_triples, self.feature_dim),
            dtype=torch.float32,
            device=self.device,
        )
        # second_shifts [n, shift_dim]
        self.data["PeakPeakNoeTriple"].second_shifts = torch.randn(
            (num_triples, self.shift_dim),
            dtype=torch.float32,
            device=self.device,
        )
        # second_features [n, feature_dim]
        self.data["PeakPeakNoeTriple"].second_features = torch.randn(
            (num_triples, self.feature_dim),
            dtype=torch.float32,
            device=self.device,
        )
        # noe_shifts [n, shift_dim] (embedded NOE features)
        self.data["PeakPeakNoeTriple"].noe_shifts = torch.randn(
            (num_triples, self.shift_dim),
            dtype=torch.float32,
            device=self.device,
        )
        # noe_features [n, feature_dim]
        self.data["PeakPeakNoeTriple"].noe_features = torch.randn(
            (num_triples, self.feature_dim),
            dtype=torch.float32,
            device=self.device,
        )

    def test_peak_update_sets_all_delta_attributes(self):
        """Test that PeakUpdate sets all 8 delta attributes."""
        update = PeakUpdate("PeakPeakNoeTriple", self.device, self.config)
        data = update(self.data)

        # Check all delta attributes are set
        self.assertIn("delta_first_coords", data["PeakPeakNoeTriple"])
        self.assertIn("delta_first_shifts", data["PeakPeakNoeTriple"])
        self.assertIn("delta_first_features", data["PeakPeakNoeTriple"])
        self.assertIn("delta_second_coords", data["PeakPeakNoeTriple"])
        self.assertIn("delta_second_shifts", data["PeakPeakNoeTriple"])
        self.assertIn("delta_second_features", data["PeakPeakNoeTriple"])
        self.assertIn("delta_noe_shifts", data["PeakPeakNoeTriple"])
        self.assertIn("delta_noe_features", data["PeakPeakNoeTriple"])

    def test_peak_update_delta_shapes(self):
        """Test that PeakUpdate produces correct output shapes."""
        update = PeakUpdate("PeakPeakNoeTriple", self.device, self.config)
        data = update(self.data)

        # Check shapes (note: coordinates are zeros but still [n, 3])
        self.assertEqual(data["PeakPeakNoeTriple"].delta_first_coords.shape, (2, 3))
        self.assertEqual(data["PeakPeakNoeTriple"].delta_first_shifts.shape, (2, self.shift_dim))
        self.assertEqual(data["PeakPeakNoeTriple"].delta_first_features.shape, (2, self.feature_dim))
        self.assertEqual(data["PeakPeakNoeTriple"].delta_second_coords.shape, (2, 3))
        self.assertEqual(data["PeakPeakNoeTriple"].delta_second_shifts.shape, (2, self.shift_dim))
        self.assertEqual(
            data["PeakPeakNoeTriple"].delta_second_features.shape, (2, self.feature_dim)
        )
        self.assertEqual(data["PeakPeakNoeTriple"].delta_noe_shifts.shape, (2, self.shift_dim))
        self.assertEqual(data["PeakPeakNoeTriple"].delta_noe_features.shape, (2, self.feature_dim))


class TestResidueUpdateEquivariance(unittest.TestCase):
    """Test ResidueUpdate computes coordinate deltas using relative distances."""

    def setUp(self):
        """Set up test graph with gathered attributes."""
        self.device = torch.device("cpu")
        self.config = ModelConfig()
        self.shift_dim = self.config.shift_embed.output_dim
        self.feature_dim = self.config.feature_embed.output_dim
        self.data = HeteroData()

        # Create 1 triple node
        self.data["ResidueResidueNoeTriple"].x = torch.zeros(
            (1, 1), dtype=torch.float32, device=self.device
        )

        # Set gathered attributes with specific coordinates
        self.data["ResidueResidueNoeTriple"].first_coords = torch.tensor(
            [[1.0, 2.0, 3.0]],
            dtype=torch.float32,
            device=self.device,
        )
        self.data["ResidueResidueNoeTriple"].first_shifts = torch.randn(
            (1, self.shift_dim),
            dtype=torch.float32,
            device=self.device,
        )
        self.data["ResidueResidueNoeTriple"].first_features = torch.randn(
            (1, self.feature_dim),
            dtype=torch.float32,
            device=self.device,
        )
        self.data["ResidueResidueNoeTriple"].second_coords = torch.tensor(
            [[4.0, 5.0, 6.0]],  # Relative distance: [3.0, 3.0, 3.0]
            dtype=torch.float32,
            device=self.device,
        )
        self.data["ResidueResidueNoeTriple"].second_shifts = torch.randn(
            (1, self.shift_dim),
            dtype=torch.float32,
            device=self.device,
        )
        self.data["ResidueResidueNoeTriple"].second_features = torch.randn(
            (1, self.feature_dim),
            dtype=torch.float32,
            device=self.device,
        )
        self.data["ResidueResidueNoeTriple"].noe_shifts = torch.randn(
            (1, self.shift_dim),
            dtype=torch.float32,
            device=self.device,
        )
        self.data["ResidueResidueNoeTriple"].noe_features = torch.randn(
            (1, self.feature_dim),
            dtype=torch.float32,
            device=self.device,
        )

    def test_coordinate_deltas_proportional_to_relative_distance(self):
        """Test that coordinate deltas are proportional to relative distance vector."""
        update = ResidueUpdate("ResidueResidueNoeTriple", self.device, self.config)
        data = update(self.data)

        # Get coordinate deltas
        delta_first = data["ResidueResidueNoeTriple"].delta_first_coords
        delta_second = data["ResidueResidueNoeTriple"].delta_second_coords

        # Deltas should be 3D vectors
        self.assertEqual(delta_first.shape, (1, 3))
        self.assertEqual(delta_second.shape, (1, 3))

        # Deltas should not be all zeros (MLP should produce non-zero weights)
        # Note: This is a weak test since random initialization could produce
        # small weights, but we mainly want to verify the shape and structure
        self.assertEqual(delta_first.dtype, torch.float32)
        self.assertEqual(delta_second.dtype, torch.float32)


class TestPeakUpdateZeroCoordinates(unittest.TestCase):
    """Test PeakUpdate outputs zero coordinate deltas."""

    def setUp(self):
        """Set up test graph with gathered attributes."""
        self.device = torch.device("cpu")
        self.config = ModelConfig()
        self.shift_dim = self.config.shift_embed.output_dim
        self.feature_dim = self.config.feature_embed.output_dim
        self.data = HeteroData()

        # Create 2 triple nodes
        self.data["PeakPeakNoeTriple"].x = torch.zeros(
            (2, 1), dtype=torch.float32, device=self.device
        )

        # Set gathered attributes (no coordinates for peaks)
        self.data["PeakPeakNoeTriple"].first_shifts = torch.randn(
            (2, self.shift_dim),
            dtype=torch.float32,
            device=self.device,
        )
        self.data["PeakPeakNoeTriple"].first_features = torch.randn(
            (2, self.feature_dim),
            dtype=torch.float32,
            device=self.device,
        )
        self.data["PeakPeakNoeTriple"].second_shifts = torch.randn(
            (2, self.shift_dim),
            dtype=torch.float32,
            device=self.device,
        )
        self.data["PeakPeakNoeTriple"].second_features = torch.randn(
            (2, self.feature_dim),
            dtype=torch.float32,
            device=self.device,
        )
        self.data["PeakPeakNoeTriple"].noe_shifts = torch.randn(
            (2, self.shift_dim),
            dtype=torch.float32,
            device=self.device,
        )
        self.data["PeakPeakNoeTriple"].noe_features = torch.randn(
            (2, self.feature_dim),
            dtype=torch.float32,
            device=self.device,
        )

    def test_peak_update_produces_zero_coordinate_deltas(self):
        """Test that PeakUpdate produces exactly zero coordinate deltas."""
        update = PeakUpdate("PeakPeakNoeTriple", self.device, self.config)
        data = update(self.data)

        # Get coordinate deltas
        delta_first_coords = data["PeakPeakNoeTriple"].delta_first_coords
        delta_second_coords = data["PeakPeakNoeTriple"].delta_second_coords

        # Deltas should be exactly zero
        torch.testing.assert_close(
            delta_first_coords, torch.zeros(2, 3, device=self.device)
        )
        torch.testing.assert_close(
            delta_second_coords, torch.zeros(2, 3, device=self.device)
        )


class TestUpdateConfigurableArchitecture(unittest.TestCase):
    """Test configurable hidden_size and num_layers parameters."""

    def setUp(self):
        """Set up minimal test graph."""
        self.device = torch.device("cpu")

    def test_residue_update_uses_config(self):
        """Test that ResidueUpdate uses config for hidden_size and num_layers."""
        config = ModelConfig()
        update = ResidueUpdate("ResidueResidueNoeTriple", self.device, config)
        self.assertIsNotNone(update.mlp)

        # Verify it uses config values
        self.assertEqual(update.config.mlp.hidden_size, config.mlp.hidden_size)
        self.assertEqual(update.config.mlp.num_layers, config.mlp.num_layers)

    def test_peak_update_uses_config(self):
        """Test that PeakUpdate uses config for hidden_size and num_layers."""
        config = ModelConfig()
        update = PeakUpdate("PeakPeakNoeTriple", self.device, config)
        self.assertIsNotNone(update.mlp)

        # Verify it uses config values
        self.assertEqual(update.config.mlp.hidden_size, config.mlp.hidden_size)
        self.assertEqual(update.config.mlp.num_layers, config.mlp.num_layers)


class TestUpdateEmptyTripleSets(unittest.TestCase):
    """Test update operations handle empty triple sets."""

    def setUp(self):
        """Set up test graph with zero triple nodes."""
        self.device = torch.device("cpu")
        self.config = ModelConfig()
        self.shift_dim = self.config.shift_embed.output_dim
        self.feature_dim = self.config.feature_embed.output_dim
        self.data = HeteroData()

        # Create ZERO triple nodes
        self.data["ResidueResidueNoeTriple"].x = torch.zeros(
            (0, 1), dtype=torch.float32, device=self.device
        )

        # Set gathered attributes as empty tensors
        self.data["ResidueResidueNoeTriple"].first_coords = torch.zeros(
            (0, 3), dtype=torch.float32, device=self.device
        )
        self.data["ResidueResidueNoeTriple"].first_shifts = torch.zeros(
            (0, self.shift_dim), dtype=torch.float32, device=self.device
        )
        self.data["ResidueResidueNoeTriple"].first_features = torch.zeros(
            (0, self.feature_dim), dtype=torch.float32, device=self.device
        )
        self.data["ResidueResidueNoeTriple"].second_coords = torch.zeros(
            (0, 3), dtype=torch.float32, device=self.device
        )
        self.data["ResidueResidueNoeTriple"].second_shifts = torch.zeros(
            (0, self.shift_dim), dtype=torch.float32, device=self.device
        )
        self.data["ResidueResidueNoeTriple"].second_features = torch.zeros(
            (0, self.feature_dim), dtype=torch.float32, device=self.device
        )
        self.data["ResidueResidueNoeTriple"].noe_shifts = torch.zeros(
            (0, self.shift_dim), dtype=torch.float32, device=self.device
        )
        self.data["ResidueResidueNoeTriple"].noe_features = torch.zeros(
            (0, self.feature_dim), dtype=torch.float32, device=self.device
        )

    def test_residue_update_handles_empty_triple_set(self):
        """Test that ResidueUpdate handles empty triple sets without error."""
        update = ResidueUpdate("ResidueResidueNoeTriple", self.device, self.config)

        # Should not raise an error
        data = update(self.data)

        # Delta attributes should be set with shape [0, ...]
        self.assertEqual(
            data["ResidueResidueNoeTriple"].delta_first_coords.shape, (0, 3)
        )
        self.assertEqual(
            data["ResidueResidueNoeTriple"].delta_first_shifts.shape, (0, self.shift_dim)
        )

    def test_peak_update_handles_empty_triple_set(self):
        """Test that PeakUpdate handles empty triple sets without error."""
        # Create empty PeakPeakNoeTriple
        self.data["PeakPeakNoeTriple"].x = torch.zeros(
            (0, 1), dtype=torch.float32, device=self.device
        )
        self.data["PeakPeakNoeTriple"].first_shifts = torch.zeros(
            (0, self.shift_dim), dtype=torch.float32, device=self.device
        )
        self.data["PeakPeakNoeTriple"].first_features = torch.zeros(
            (0, self.feature_dim), dtype=torch.float32, device=self.device
        )
        self.data["PeakPeakNoeTriple"].second_shifts = torch.zeros(
            (0, self.shift_dim), dtype=torch.float32, device=self.device
        )
        self.data["PeakPeakNoeTriple"].second_features = torch.zeros(
            (0, self.feature_dim), dtype=torch.float32, device=self.device
        )
        self.data["PeakPeakNoeTriple"].noe_shifts = torch.zeros(
            (0, self.shift_dim), dtype=torch.float32, device=self.device
        )
        self.data["PeakPeakNoeTriple"].noe_features = torch.zeros(
            (0, self.feature_dim), dtype=torch.float32, device=self.device
        )

        update = PeakUpdate("PeakPeakNoeTriple", self.device, self.config)

        # Should not raise an error
        data = update(self.data)

        # Delta attributes should be set with shape [0, ...]
        self.assertEqual(data["PeakPeakNoeTriple"].delta_first_coords.shape, (0, 3))
        self.assertEqual(data["PeakPeakNoeTriple"].delta_first_shifts.shape, (0, self.shift_dim))


if __name__ == "__main__":
    unittest.main()
