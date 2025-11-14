"""
Unit tests for triple.py helper functions.

Tests the three helper functions at the bottom of triple.py:
- calc_noe_difference
- calc_shift_difference
- calc_res_distance
"""

import sys
from pathlib import Path
import unittest
import torch

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from nmr.models.triple import calc_noe_difference, calc_shift_difference, calc_res_distance


class TestCalcNOEDifference(unittest.TestCase):
    """Test calc_noe_difference function."""

    def test_basic_differences(self):
        """Test basic NOE difference calculations."""
        # Create sample data: shifts are [H, N]
        x1 = torch.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=torch.float32)  # [2, 2] (H, N)
        x2 = torch.tensor([[5.0, 6.0], [7.0, 8.0]], dtype=torch.float32)  # [2, 2] (H, N)
        noe = torch.tensor([[10.0, 11.0, 12.0], [20.0, 21.0, 22.0]], dtype=torch.float32)  # [2, 3] (N, H', H")

        diff_N, diff_H1, diff_H2 = calc_noe_difference(x1, x2, noe)

        # Expected calculations:
        # diff_N = noe[:, 0] - x1[:, 1]  (NOE_N - first_N)
        # diff_H1 = noe[:, 1] - x1[:, 0] (NOE_H' - first_H)
        # diff_H2 = noe[:, 2] - x2[:, 0] (NOE_H" - second_H)

        expected_diff_N = torch.tensor([[10.0 - 2.0], [20.0 - 4.0]], dtype=torch.float32)
        expected_diff_H1 = torch.tensor([[11.0 - 1.0], [21.0 - 3.0]], dtype=torch.float32)
        expected_diff_H2 = torch.tensor([[12.0 - 5.0], [22.0 - 7.0]], dtype=torch.float32)

        torch.testing.assert_close(diff_N, expected_diff_N)
        torch.testing.assert_close(diff_H1, expected_diff_H1)
        torch.testing.assert_close(diff_H2, expected_diff_H2)

    def test_output_shapes(self):
        """Test that output shapes are correct."""
        x1 = torch.randn(10, 2)
        x2 = torch.randn(10, 2)
        noe = torch.randn(10, 3)

        diff_N, diff_H1, diff_H2 = calc_noe_difference(x1, x2, noe)

        self.assertEqual(diff_N.shape, (10, 1))
        self.assertEqual(diff_H1.shape, (10, 1))
        self.assertEqual(diff_H2.shape, (10, 1))

    def test_zero_differences(self):
        """Test with zero differences."""
        # x1 has shifts [1.0, 10.0], x2 has shifts [12.0, 6.0]
        # noe has shifts [10.0, 1.0, 12.0] to match
        x1 = torch.tensor([[1.0, 10.0]], dtype=torch.float32)
        x2 = torch.tensor([[12.0, 6.0]], dtype=torch.float32)
        noe = torch.tensor([[10.0, 1.0, 12.0]], dtype=torch.float32)

        diff_N, diff_H1, diff_H2 = calc_noe_difference(x1, x2, noe)

        # Should all be zero
        torch.testing.assert_close(diff_N, torch.zeros((1, 1)))
        torch.testing.assert_close(diff_H1, torch.zeros((1, 1)))
        torch.testing.assert_close(diff_H2, torch.zeros((1, 1)))

    def test_negative_differences(self):
        """Test with negative differences."""
        x1 = torch.tensor([[5.0, 6.0]], dtype=torch.float32)
        x2 = torch.tensor([[7.0, 8.0]], dtype=torch.float32)
        noe = torch.tensor([[2.0, 3.0, 4.0]], dtype=torch.float32)

        diff_N, diff_H1, diff_H2 = calc_noe_difference(x1, x2, noe)

        expected_diff_N = torch.tensor([[2.0 - 6.0]], dtype=torch.float32)  # -4.0
        expected_diff_H1 = torch.tensor([[3.0 - 5.0]], dtype=torch.float32)  # -2.0
        expected_diff_H2 = torch.tensor([[4.0 - 7.0]], dtype=torch.float32)  # -3.0

        torch.testing.assert_close(diff_N, expected_diff_N)
        torch.testing.assert_close(diff_H1, expected_diff_H1)
        torch.testing.assert_close(diff_H2, expected_diff_H2)


class TestCalcShiftDifference(unittest.TestCase):
    """Test calc_shift_difference function."""

    def test_basic_differences(self):
        """Test basic shift difference calculations."""
        # Create sample data: shifts are [H, N]
        x1 = torch.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=torch.float32)
        x2 = torch.tensor([[5.0, 6.0], [7.0, 8.0]], dtype=torch.float32)

        diff_N, diff_H = calc_shift_difference(x1, x2)

        # Expected: x1 - x2
        expected_diff_N = torch.tensor([[2.0 - 6.0], [4.0 - 8.0]], dtype=torch.float32)
        expected_diff_H = torch.tensor([[1.0 - 5.0], [3.0 - 7.0]], dtype=torch.float32)

        torch.testing.assert_close(diff_N, expected_diff_N)
        torch.testing.assert_close(diff_H, expected_diff_H)

    def test_output_shapes(self):
        """Test that output shapes are correct."""
        x1 = torch.randn(10, 2)
        x2 = torch.randn(10, 2)

        diff_N, diff_H = calc_shift_difference(x1, x2)

        self.assertEqual(diff_N.shape, (10, 1))
        self.assertEqual(diff_H.shape, (10, 1))

    def test_zero_differences(self):
        """Test with identical shifts."""
        x1 = torch.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=torch.float32)
        x2 = x1.clone()

        diff_N, diff_H = calc_shift_difference(x1, x2)

        torch.testing.assert_close(diff_N, torch.zeros((2, 1)))
        torch.testing.assert_close(diff_H, torch.zeros((2, 1)))

    def test_symmetry(self):
        """Test that calc_shift_difference(x1, x2) = -calc_shift_difference(x2, x1)."""
        x1 = torch.randn(5, 2)
        x2 = torch.randn(5, 2)

        diff_N1, diff_H1 = calc_shift_difference(x1, x2)
        diff_N2, diff_H2 = calc_shift_difference(x2, x1)

        torch.testing.assert_close(diff_N1, -diff_N2)
        torch.testing.assert_close(diff_H1, -diff_H2)


class TestCalcResDistance(unittest.TestCase):
    """Test calc_res_distance function."""

    def test_basic_distance(self):
        """Test basic distance calculations."""
        # Simple case: points along x-axis
        x1 = torch.tensor([[3.0, 0.0, 0.0]], dtype=torch.float32)
        x2 = torch.tensor([[0.0, 0.0, 0.0]], dtype=torch.float32)

        rel_dist, dist_squared = calc_res_distance(x1, x2)

        expected_rel_dist = torch.tensor([[3.0, 0.0, 0.0]], dtype=torch.float32)
        expected_dist_squared = torch.tensor([[9.0]], dtype=torch.float32)

        torch.testing.assert_close(rel_dist, expected_rel_dist)
        torch.testing.assert_close(dist_squared, expected_dist_squared)

    def test_output_shapes(self):
        """Test that output shapes are correct."""
        x1 = torch.randn(10, 3)
        x2 = torch.randn(10, 3)

        rel_dist, dist_squared = calc_res_distance(x1, x2)

        self.assertEqual(rel_dist.shape, (10, 3))
        self.assertEqual(dist_squared.shape, (10, 1))

    def test_zero_distance(self):
        """Test with identical coordinates."""
        x1 = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=torch.float32)
        x2 = x1.clone()

        rel_dist, dist_squared = calc_res_distance(x1, x2)

        torch.testing.assert_close(rel_dist, torch.zeros((2, 3)))
        torch.testing.assert_close(dist_squared, torch.zeros((2, 1)))

    def test_3d_distance(self):
        """Test 3D Euclidean distance calculation."""
        # Use 3-4-5 right triangle in 3D
        x1 = torch.tensor([[3.0, 4.0, 0.0]], dtype=torch.float32)
        x2 = torch.tensor([[0.0, 0.0, 0.0]], dtype=torch.float32)

        rel_dist, dist_squared = calc_res_distance(x1, x2)

        expected_rel_dist = torch.tensor([[3.0, 4.0, 0.0]], dtype=torch.float32)
        expected_dist_squared = torch.tensor([[25.0]], dtype=torch.float32)  # 3^2 + 4^2 + 0^2

        torch.testing.assert_close(rel_dist, expected_rel_dist)
        torch.testing.assert_close(dist_squared, expected_dist_squared)

    def test_symmetry(self):
        """Test that calc_res_distance(x1, x2) = -calc_res_distance(x2, x1) for rel_dist."""
        x1 = torch.randn(5, 3)
        x2 = torch.randn(5, 3)

        rel_dist1, dist_squared1 = calc_res_distance(x1, x2)
        rel_dist2, dist_squared2 = calc_res_distance(x2, x1)

        # Relative distance should be opposite
        torch.testing.assert_close(rel_dist1, -rel_dist2)

        # Squared distance should be the same (distance is symmetric)
        torch.testing.assert_close(dist_squared1, dist_squared2)

    def test_batch_processing(self):
        """Test batch processing with multiple coordinate pairs."""
        x1 = torch.tensor([
            [1.0, 0.0, 0.0],
            [0.0, 2.0, 0.0],
            [0.0, 0.0, 3.0],
        ], dtype=torch.float32)
        x2 = torch.zeros((3, 3), dtype=torch.float32)

        rel_dist, dist_squared = calc_res_distance(x1, x2)

        expected_rel_dist = x1  # x1 - 0 = x1
        expected_dist_squared = torch.tensor([[1.0], [4.0], [9.0]], dtype=torch.float32)

        torch.testing.assert_close(rel_dist, expected_rel_dist)
        torch.testing.assert_close(dist_squared, expected_dist_squared)


if __name__ == '__main__':
    unittest.main()
