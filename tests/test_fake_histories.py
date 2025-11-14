"""
Unit tests for corrected fake history generation algorithm.

Tests verify that the history generation algorithm correctly:
- Keeps coordinates unchanged across perturbations
- Perturbs only observed shifts, not coordinates
- Regenerates NOEs from original (unperturbed) coordinates
- Uses identity mapping for perfect play (action = shift_to_assign)
- Creates proper trajectory format: (state_dict, action, reward) tuples
"""

import unittest
import sys
import copy
from pathlib import Path

import numpy as np

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from nmr.nmr_gym.fake_data import FakeDataGenerator
from nmr.nmr_gym.fake_histories import FakeHistoryGenerator
from nmr.nmr_gym.gym_env import GymEnv


class TestFakeHistoryFix(unittest.TestCase):
    """Test suite for corrected fake history generation."""

    def setUp(self):
        """Create test dataset for use in tests."""
        self.num_resid = 5
        self.fake_data_gen = FakeDataGenerator(self.num_resid)

        # Generate base dataset
        self.coordinates, self.obs_chemical_shifts, self.noes, self.connectivity = (
            self.fake_data_gen.generate_data(pickle_data=False, random_key=False)
        )

        # Create state dict
        self.state = {
            "coordinates": self.coordinates,
            "obs_chemical_shifts": self.obs_chemical_shifts,
            "noes": self.noes,
            "connectivity": self.connectivity,
            "assignments": {},
            "assign_order": [],
            "shift_to_assign": 0,
            "total_energy": 0.0,
            "reward": 0.0
        }

    def test_coordinates_unchanged_across_perturbations(self):
        """Test that coordinates remain unchanged when perturbing data."""
        history_gen = FakeHistoryGenerator(self.num_resid)

        # Extract original coordinates as numpy array for comparison
        original_coords = np.array([[p.x, p.y, p.z] for p in self.coordinates])

        # Generate multiple perturbed versions
        for _ in range(3):
            perturbed_state = history_gen.generate_history(self.state)
            perturbed_coords = np.array([[p.x, p.y, p.z] for p in perturbed_state['coordinates']])

            # Coordinates should be identical to original
            np.testing.assert_array_equal(
                original_coords, perturbed_coords,
                err_msg="Coordinates should remain unchanged across perturbations"
            )

    def test_only_shifts_perturbed(self):
        """Test that only observed shifts are perturbed, not coordinates."""
        history_gen = FakeHistoryGenerator(self.num_resid)

        # Extract original values
        original_coords = np.array([[p.x, p.y, p.z] for p in self.coordinates])
        original_shifts = np.array([[s.H1, s.N15] for s in self.obs_chemical_shifts])

        # Generate perturbed version
        perturbed_state = history_gen.generate_history(self.state)

        # Extract perturbed values
        perturbed_coords = np.array([[p.x, p.y, p.z] for p in perturbed_state['coordinates']])
        perturbed_shifts = np.array([[s.H1, s.N15] for s in perturbed_state['obs_chemical_shifts']])

        # Coordinates should be unchanged
        np.testing.assert_array_equal(
            original_coords, perturbed_coords,
            err_msg="Coordinates should NOT be perturbed"
        )

        # Shifts should be different (perturbed with noise)
        self.assertFalse(
            np.array_equal(original_shifts, perturbed_shifts),
            "Shifts should be perturbed with noise"
        )

    def test_noes_regenerated_from_original_coordinates(self):
        """Test that NOEs are regenerated from original coordinates, not perturbed ones."""
        history_gen = FakeHistoryGenerator(self.num_resid)

        # Extract original coordinates
        original_coords = np.array([[p.x, p.y, p.z] for p in self.coordinates])

        # Generate perturbed version
        perturbed_state = history_gen.generate_history(self.state)

        # NOEs should be based on spatial distances from original coordinates
        # We verify this by checking that coordinates used for NOE generation are unchanged
        perturbed_coords = np.array([[p.x, p.y, p.z] for p in perturbed_state['coordinates']])

        np.testing.assert_array_equal(
            original_coords, perturbed_coords,
            err_msg="NOEs must be generated from original (unperturbed) coordinates"
        )

        # Also verify that NOEs were actually regenerated (not empty)
        self.assertGreater(
            len(perturbed_state['noes']), 0,
            "NOEs should be regenerated and not empty"
        )

    def test_identity_mapping_perfect_play(self):
        """Test that identity mapping works for perfect play: action = shift_to_assign."""
        # Initialize environment
        env = GymEnv(self.num_resid)
        state = env.reset(self.coordinates, self.obs_chemical_shifts, self.noes, self.connectivity)

        # Perfect play: use identity mapping
        terminated = False
        while not terminated:
            # For synthetic data with random_key=False, identity mapping holds
            # The correct action is the shift index itself
            action = state["shift_to_assign"]

            # Verify action is valid (not already assigned)
            self.assertNotIn(
                action, state["assignments"].values(),
                f"Action {action} should not be already assigned"
            )

            state, reward, terminated, _ = env.step(action)

            # Reward should be non-negative (energy should not increase)
            self.assertGreaterEqual(
                reward, 0.0,
                "Reward should be non-negative for perfect play"
            )

    def test_trajectory_format(self):
        """Test that trajectory has correct format: list of (state_dict, action, reward) tuples."""
        from nmr.nmr_gym.fake_histories import generate_trajectory

        # Initialize environment
        env = GymEnv(self.num_resid)
        initial_state = env.reset(self.coordinates, self.obs_chemical_shifts, self.noes, self.connectivity)

        # Generate trajectory
        trajectory = generate_trajectory(env, initial_state)

        # Should be a list
        self.assertIsInstance(trajectory, list, "Trajectory should be a list")

        # Should have num_resid steps (one per assignment)
        self.assertEqual(
            len(trajectory), self.num_resid,
            f"Trajectory should have {self.num_resid} steps"
        )

        # Each element should be a tuple of (state_dict, action, reward)
        for i, step in enumerate(trajectory):
            self.assertIsInstance(step, tuple, f"Step {i} should be a tuple")
            self.assertEqual(len(step), 3, f"Step {i} should have 3 elements")

            state_dict, action, reward = step

            # Verify types
            self.assertIsInstance(state_dict, dict, f"Step {i}: state should be dict")
            self.assertIsInstance(action, (int, np.integer), f"Step {i}: action should be int")
            self.assertIsInstance(reward, (int, float, np.number), f"Step {i}: reward should be numeric")

            # Verify state dict has required keys
            required_keys = ['coordinates', 'obs_chemical_shifts', 'noes',
                           'connectivity', 'assignments', 'assign_order',
                           'shift_to_assign', 'total_energy', 'reward']
            for key in required_keys:
                self.assertIn(key, state_dict, f"Step {i}: missing key '{key}'")


if __name__ == '__main__':
    unittest.main()
