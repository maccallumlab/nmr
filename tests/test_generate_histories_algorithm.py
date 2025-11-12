"""
Unit tests for history generation algorithm.

These tests verify the core algorithm logic for history generation:
- Load base dataset once
- Generate N trajectories by perturbing only shifts
- Keep coordinates constant across all trajectories
- Record trajectories in correct format: (state_dict, action, reward)
"""

import pickle
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

# Add parent directory to path to allow imports from nmr package
sys.path.insert(0, str(Path(__file__).parent.parent))

from nmr.nmr_gym.fake_data import FakeDataGenerator
from nmr.nmr_gym.io import load_dataset, load_histories, save_dataset, save_histories
from nmr.nmr_gym.fake_histories import FakeHistoryGenerator, generate_trajectory
from nmr.nmr_gym.gym_env import GymEnv


class TestGenerateHistoriesAlgorithm(unittest.TestCase):
    """Unit tests for the core history generation algorithm."""

    def setUp(self):
        """Create a temporary dataset for testing."""
        self.num_resid = 5
        self.temp_dir = tempfile.mkdtemp()
        self.dataset_path = Path(self.temp_dir) / "test_dataset.pkl"
        self.histories_path = Path(self.temp_dir) / "test_histories.pkl"

        # Generate and save a base dataset
        generator = FakeDataGenerator(self.num_resid)
        coords, obs_shifts, pred_shifts, noes, conn = generator.generate_data_arrays(random_key=False)
        coords, obs_shifts, noes, conn = generator.order_data(coords, obs_shifts, pred_shifts, noes, conn)

        metadata = {
            "version": "1.0",
            "num_resid": self.num_resid,
            "random_key": False
        }
        save_dataset(self.dataset_path, coords, obs_shifts, noes, conn, metadata)

    def test_script_loads_dataset_correctly(self):
        """Test that script loads dataset with explicit path."""
        # Load the dataset
        dataset = load_dataset(self.dataset_path)

        # Verify loaded data
        self.assertEqual(len(dataset.pred_coordinates), self.num_resid)
        self.assertEqual(len(dataset.obs_chemical_shifts), self.num_resid)
        self.assertIsInstance(dataset.noes, list)
        self.assertIsInstance(dataset.connectivity, list)
        self.assertEqual(dataset.metadata["num_resid"], self.num_resid)

    def test_n_trajectories_from_one_base_dataset(self):
        """Test that N trajectories are generated from one base dataset."""
        # Load base dataset once
        dataset = load_dataset(self.dataset_path)
        coords = dataset.pred_coordinates
        obs_shifts = dataset.obs_chemical_shifts
        noes = dataset.noes
        conn = dataset.connectivity

        # Store original coordinates for comparison
        original_coords = np.array([[p.x, p.y, p.z] for p in coords])

        # Generate multiple trajectories
        num_trajectories = 3
        env = GymEnv(self.num_resid)
        history_gen = FakeHistoryGenerator(self.num_resid)

        trajectories = []

        # Initialize environment once with original data
        original_state = env.reset(coords, obs_shifts, noes, conn)

        for i in range(num_trajectories):
            # Perturb only shifts, keep coordinates
            perturbed_state = history_gen.generate_history(original_state)

            # Verify coordinates unchanged
            perturbed_coords = np.array([[p.x, p.y, p.z] for p in perturbed_state['coordinates']])
            np.testing.assert_array_almost_equal(original_coords, perturbed_coords)

            # Generate trajectory
            trajectory = generate_trajectory(env, perturbed_state)
            trajectories.append(trajectory)

        # Verify we got correct number of trajectories
        self.assertEqual(len(trajectories), num_trajectories)

        # Verify each trajectory is non-empty
        for traj in trajectories:
            self.assertGreater(len(traj), 0)

    def test_trajectory_format_correct(self):
        """Test that trajectory format is (state_dict, action, reward) tuples."""
        # Load dataset
        dataset = load_dataset(self.dataset_path)
        coords = dataset.pred_coordinates
        obs_shifts = dataset.obs_chemical_shifts
        noes = dataset.noes
        conn = dataset.connectivity

        # Generate one trajectory
        env = GymEnv(self.num_resid)
        history_gen = FakeHistoryGenerator(self.num_resid)

        original_state = env.reset(coords, obs_shifts, noes, conn)
        perturbed_state = history_gen.generate_history(original_state)
        trajectory = generate_trajectory(env, perturbed_state)

        # Verify trajectory is a list
        self.assertIsInstance(trajectory, list)
        self.assertGreater(len(trajectory), 0)

        # Verify each element is a tuple with 3 elements
        for step in trajectory:
            self.assertIsInstance(step, tuple)
            self.assertEqual(len(step), 3)

            state_dict, action, reward = step

            # Verify state_dict is a dictionary with required keys
            self.assertIsInstance(state_dict, dict)
            required_keys = ['coordinates', 'obs_chemical_shifts', 'noes',
                           'connectivity', 'assignments', 'assign_order',
                           'shift_to_assign', 'total_energy', 'reward']
            for key in required_keys:
                self.assertIn(key, state_dict)

            # Verify action is an integer
            self.assertIsInstance(action, (int, np.integer))

            # Verify reward is a number
            self.assertIsInstance(reward, (int, float, np.number))

    def test_coordinates_constant_across_trajectories(self):
        """Test that coordinates remain constant across all trajectories."""
        # Load dataset
        dataset = load_dataset(self.dataset_path)
        coords = dataset.pred_coordinates
        obs_shifts = dataset.obs_chemical_shifts
        noes = dataset.noes
        conn = dataset.connectivity

        # Extract original coordinates
        original_coords = np.array([[p.x, p.y, p.z] for p in coords])

        # Generate multiple trajectories and collect coordinates from each
        num_trajectories = 3
        env = GymEnv(self.num_resid)
        history_gen = FakeHistoryGenerator(self.num_resid)

        original_state = env.reset(coords, obs_shifts, noes, conn)

        all_coords_match = True
        for i in range(num_trajectories):
            # Generate perturbed state
            perturbed_state = history_gen.generate_history(original_state)

            # Generate trajectory
            trajectory = generate_trajectory(env, perturbed_state)

            # Check coordinates in every step of this trajectory
            for state_dict, action, reward in trajectory:
                step_coords = np.array([[p.x, p.y, p.z] for p in state_dict['coordinates']])

                # Verify coordinates match original
                if not np.allclose(step_coords, original_coords):
                    all_coords_match = False
                    break

            if not all_coords_match:
                break

        self.assertTrue(all_coords_match, "Coordinates changed across trajectories")

    def test_save_and_load_histories(self):
        """Test that histories can be saved and loaded correctly."""
        # Load dataset
        dataset = load_dataset(self.dataset_path)
        coords = dataset.pred_coordinates
        obs_shifts = dataset.obs_chemical_shifts
        noes = dataset.noes
        conn = dataset.connectivity

        # Generate a few trajectories
        num_trajectories = 2
        env = GymEnv(self.num_resid)
        history_gen = FakeHistoryGenerator(self.num_resid)

        original_state = env.reset(coords, obs_shifts, noes, conn)

        trajectories = []
        for i in range(num_trajectories):
            perturbed_state = history_gen.generate_history(original_state)
            trajectory = generate_trajectory(env, perturbed_state)
            trajectories.append(trajectory)

        # Save histories
        history_metadata = {
            "base_dataset_path": str(self.dataset_path),
            "num_trajectories": num_trajectories,
            "num_resid": self.num_resid
        }
        save_histories(self.histories_path, trajectories, history_metadata)

        # Load histories
        histories = load_histories(self.histories_path)

        # Verify metadata
        self.assertEqual(histories.metadata["num_trajectories"], num_trajectories)
        self.assertEqual(histories.metadata["num_resid"], self.num_resid)

        # Verify structure
        self.assertEqual(len(histories.trajectories), num_trajectories)
        for traj in histories.trajectories:
            self.assertIsInstance(traj, list)
            self.assertGreater(len(traj), 0)

    def test_action_matches_shift_to_assign(self):
        """Test that action always matches shift_to_assign for perfect play."""
        # Load dataset
        dataset = load_dataset(self.dataset_path)
        coords = dataset.pred_coordinates
        obs_shifts = dataset.obs_chemical_shifts
        noes = dataset.noes
        conn = dataset.connectivity

        # Generate multiple trajectories to verify consistency
        num_trajectories = 3
        env = GymEnv(self.num_resid)
        history_gen = FakeHistoryGenerator(self.num_resid)

        original_state = env.reset(coords, obs_shifts, noes, conn)

        for i in range(num_trajectories):
            # Generate perturbed state
            perturbed_state = history_gen.generate_history(original_state)

            # Generate trajectory
            trajectory = generate_trajectory(env, perturbed_state)

            # Verify each step: action should equal shift_to_assign
            for step_num, (state_dict, action, reward) in enumerate(trajectory):
                shift_to_assign = state_dict['shift_to_assign']

                self.assertEqual(
                    action, shift_to_assign,
                    f"Trajectory {i}, step {step_num}: action ({action}) does not match "
                    f"shift_to_assign ({shift_to_assign}). Perfect play requires "
                    f"action = shift_to_assign for identity mapping."
                )


if __name__ == "__main__":
    unittest.main()
