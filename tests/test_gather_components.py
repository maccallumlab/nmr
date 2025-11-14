"""
Unit tests for gather component operations.

Tests the 5 gather classes that extract features from source nodes to triple nodes:
- FirstResidueGather
- FirstPeakGather
- SecondResidueGather
- SecondPeakGather
- NoeGather
"""

import sys
import unittest
from pathlib import Path

import torch
from torch_geometric.data import HeteroData

# Add parent directory to path to allow imports from nmr package
sys.path.insert(0, str(Path(__file__).parent.parent))

from nmr.models.triple import (
    FirstResidueGather,
    FirstPeakGather,
    SecondResidueGather,
    SecondPeakGather,
    NoeGather,
)


class TestFirstResidueGather(unittest.TestCase):
    """Test FirstResidueGather extracts coords, shifts, features correctly."""

    def setUp(self):
        """Set up test graph with Residue nodes and triple edges."""
        self.device = torch.device("cpu")
        self.data = HeteroData()

        # Create 3 Residue nodes with coords [n, 5] = [x, y, z, H, N]
        self.data["Residue"].x = torch.tensor(
            [
                [1.0, 2.0, 3.0, 8.5, 120.0],  # Residue 0
                [4.0, 5.0, 6.0, 7.5, 115.0],  # Residue 1
                [7.0, 8.0, 9.0, 9.0, 125.0],  # Residue 2
            ],
            dtype=torch.float32,
            device=self.device,
        )

        # Features [n, 2] = [is_being_assigned, is_already_assigned]
        self.data["Residue"].f = torch.tensor(
            [
                [0.0, 1.0],
                [1.0, 0.0],
                [0.0, 0.0],
            ],
            dtype=torch.float32,
            device=self.device,
        )

        # Create 2 triple nodes (placeholders)
        self.data["ResidueResidueNoeTriple"].x = torch.zeros(
            (2, 1), dtype=torch.float32, device=self.device
        )

        # Create gather edges: Residue -> ResidueResidueNoeTriple
        # Edge index [2, num_edges] where [0] = source indices, [1] = target indices
        # Edge: source Residue 0 -> target triple 0, source Residue 1 -> target triple 1
        self.data["Residue", "prop_first", "ResidueResidueNoeTriple"].edge_index = (
            torch.tensor(
                [[0, 1], [0, 1]],  # [source_indices, target_indices]
                dtype=torch.long,
                device=self.device,
            )
        )

    def test_gather_extracts_coordinates(self):
        """Test that FirstResidueGather extracts coordinates correctly."""
        gather = FirstResidueGather("ResidueResidueNoeTriple")
        data = gather(self.data)

        # Check that first_coords attribute is set on triple nodes
        self.assertIn("first_coords", data["ResidueResidueNoeTriple"])
        first_coords = data["ResidueResidueNoeTriple"].first_coords

        # Shape should be [2, 3] (2 triple nodes, 3 coords each)
        self.assertEqual(first_coords.shape, (2, 3))

        # Values should match source Residue nodes
        torch.testing.assert_close(first_coords[0], torch.tensor([1.0, 2.0, 3.0]))
        torch.testing.assert_close(first_coords[1], torch.tensor([4.0, 5.0, 6.0]))

    def test_gather_extracts_shifts(self):
        """Test that FirstResidueGather extracts shifts correctly."""
        gather = FirstResidueGather("ResidueResidueNoeTriple")
        data = gather(self.data)

        # Check that first_shifts attribute is set
        self.assertIn("first_shifts", data["ResidueResidueNoeTriple"])
        first_shifts = data["ResidueResidueNoeTriple"].first_shifts

        # Shape should be [2, 2] (2 triple nodes, 2 shifts each)
        self.assertEqual(first_shifts.shape, (2, 2))

        # Values should match shifts from Residue.x[:, 3:5]
        torch.testing.assert_close(first_shifts[0], torch.tensor([8.5, 120.0]))
        torch.testing.assert_close(first_shifts[1], torch.tensor([7.5, 115.0]))

    def test_gather_extracts_features(self):
        """Test that FirstResidueGather extracts features correctly."""
        gather = FirstResidueGather("ResidueResidueNoeTriple")
        data = gather(self.data)

        # Check that first_features attribute is set
        self.assertIn("first_features", data["ResidueResidueNoeTriple"])
        first_features = data["ResidueResidueNoeTriple"].first_features

        # Shape should be [2, 2]
        self.assertEqual(first_features.shape, (2, 2))

        # Values should match Residue.f
        torch.testing.assert_close(first_features[0], torch.tensor([0.0, 1.0]))
        torch.testing.assert_close(first_features[1], torch.tensor([1.0, 0.0]))


class TestFirstPeakGather(unittest.TestCase):
    """Test FirstPeakGather extracts shifts and features (no coords)."""

    def setUp(self):
        """Set up test graph with Peak nodes."""
        self.device = torch.device("cpu")
        self.data = HeteroData()

        # Create 3 Peak nodes with shifts [n, 2] = [H, N]
        self.data["Peak"].x = torch.tensor(
            [
                [8.5, 120.0],
                [7.5, 115.0],
                [9.0, 125.0],
            ],
            dtype=torch.float32,
            device=self.device,
        )

        # Features [n, 2]
        self.data["Peak"].f = torch.tensor(
            [
                [1.0, 0.0],
                [0.0, 1.0],
                [0.0, 0.0],
            ],
            dtype=torch.float32,
            device=self.device,
        )

        # Create triple nodes
        self.data["PeakPeakNoeTriple"].x = torch.zeros(
            (2, 1), dtype=torch.float32, device=self.device
        )

        # Create gather edges: Peak -> PeakPeakNoeTriple
        self.data["Peak", "prop_first", "PeakPeakNoeTriple"].edge_index = (
            torch.tensor(
                [[0, 2], [0, 1]],  # Peak 0 -> triple 0, Peak 2 -> triple 1
                dtype=torch.long,
                device=self.device,
            )
        )

    def test_gather_extracts_shifts_no_coords(self):
        """Test that FirstPeakGather extracts shifts but not coordinates."""
        gather = FirstPeakGather("PeakPeakNoeTriple")
        data = gather(self.data)

        # Should have first_shifts
        self.assertIn("first_shifts", data["PeakPeakNoeTriple"])
        first_shifts = data["PeakPeakNoeTriple"].first_shifts
        self.assertEqual(first_shifts.shape, (2, 2))

        # Should NOT have first_coords (peaks have no coordinates)
        self.assertNotIn("first_coords", data["PeakPeakNoeTriple"])

    def test_gather_extracts_features(self):
        """Test that FirstPeakGather extracts features correctly."""
        gather = FirstPeakGather("PeakPeakNoeTriple")
        data = gather(self.data)

        self.assertIn("first_features", data["PeakPeakNoeTriple"])
        first_features = data["PeakPeakNoeTriple"].first_features
        self.assertEqual(first_features.shape, (2, 2))


class TestNoeGather(unittest.TestCase):
    """Test NoeGather extracts NOE shifts and features."""

    def setUp(self):
        """Set up test graph with Noe nodes."""
        self.device = torch.device("cpu")
        self.data = HeteroData()

        # Create Noe nodes with shifts [n, 3] = [N, H', H"]
        self.data["Noe"].x = torch.tensor(
            [
                [120.0, 8.5, 7.5],
                [115.0, 9.0, 8.0],
                [125.0, 7.0, 9.5],
            ],
            dtype=torch.float32,
            device=self.device,
        )

        # Features [n, 2]
        self.data["Noe"].f = torch.tensor(
            [
                [1.0, 0.0],
                [0.0, 1.0],
                [0.5, 0.5],
            ],
            dtype=torch.float32,
            device=self.device,
        )

        # Create triple nodes
        self.data["ResidueResidueNoeTriple"].x = torch.zeros(
            (2, 1), dtype=torch.float32, device=self.device
        )

        # Create gather edges: Noe -> ResidueResidueNoeTriple
        self.data["Noe", "prop_noe", "ResidueResidueNoeTriple"].edge_index = (
            torch.tensor(
                [[0, 1], [0, 1]],  # Noe 0 -> triple 0, Noe 1 -> triple 1
                dtype=torch.long,
                device=self.device,
            )
        )

    def test_gather_extracts_noe_shifts(self):
        """Test that NoeGather extracts NOE shifts correctly."""
        gather = NoeGather("ResidueResidueNoeTriple")
        data = gather(self.data)

        # Check noe_shifts attribute
        self.assertIn("noe_shifts", data["ResidueResidueNoeTriple"])
        noe_shifts = data["ResidueResidueNoeTriple"].noe_shifts

        # Shape should be [2, 3]
        self.assertEqual(noe_shifts.shape, (2, 3))

        # Values should match Noe.x
        torch.testing.assert_close(noe_shifts[0], torch.tensor([120.0, 8.5, 7.5]))
        torch.testing.assert_close(noe_shifts[1], torch.tensor([115.0, 9.0, 8.0]))

    def test_gather_extracts_noe_features(self):
        """Test that NoeGather extracts NOE features correctly."""
        gather = NoeGather("ResidueResidueNoeTriple")
        data = gather(self.data)

        self.assertIn("noe_features", data["ResidueResidueNoeTriple"])
        noe_features = data["ResidueResidueNoeTriple"].noe_features
        self.assertEqual(noe_features.shape, (2, 2))


class TestGatherEmptyEdges(unittest.TestCase):
    """Test that gather operations handle empty edge sets gracefully."""

    def setUp(self):
        """Set up test graph with nodes but no edges."""
        self.device = torch.device("cpu")
        self.data = HeteroData()

        # Create nodes
        self.data["Residue"].x = torch.tensor(
            [[1.0, 2.0, 3.0, 8.5, 120.0]],
            dtype=torch.float32,
            device=self.device,
        )
        self.data["Residue"].f = torch.tensor(
            [[0.0, 1.0]], dtype=torch.float32, device=self.device
        )
        self.data["ResidueResidueNoeTriple"].x = torch.zeros(
            (1, 1), dtype=torch.float32, device=self.device
        )

        # Create EMPTY edge set with proper shape [2, 0]
        self.data["Residue", "prop_first", "ResidueResidueNoeTriple"].edge_index = (
            torch.zeros((2, 0), dtype=torch.long, device=self.device)
        )

    def test_gather_handles_empty_edges(self):
        """Test that FirstResidueGather handles empty edge sets without error."""
        gather = FirstResidueGather("ResidueResidueNoeTriple")

        # Should not raise an error
        data = gather(self.data)

        # Attributes should still be created (will be zero-filled for empty edges)
        self.assertIn("first_coords", data["ResidueResidueNoeTriple"])
        self.assertIn("first_shifts", data["ResidueResidueNoeTriple"])
        self.assertIn("first_features", data["ResidueResidueNoeTriple"])


if __name__ == "__main__":
    unittest.main()
