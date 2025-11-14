"""
Integration tests for end-to-end dataset and history generation workflow.

These tests verify critical integration points:
- Dataset generation → history generation workflow
- Coordinates remain identical across all trajectories from same base dataset
- Perturbed shifts differ across trajectories
- Complete workflow with save/load operations
- Data format consistency throughout pipeline
"""

import pickle
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from nmr.nmr_gym.fake_data import FakeDataGenerator
from nmr.nmr_gym.fake_histories import FakeHistoryGenerator, generate_trajectory
from nmr.nmr_gym.gym_env import GymEnv
from nmr.nmr_gym.io import load_dataset, load_histories, save_dataset, save_histories


class TestEndToEndWorkflow(unittest.TestCase):
    """Integration tests for complete dataset → histories workflow."""

    def setUp(self):
        """Set up temporary directory for test files."""
        self.temp_dir = tempfile.mkdtemp()
        self.temp_path = Path(self.temp_dir)

    def tearDown(self):
        """Clean up temporary files."""
        import shutil
        if self.temp_path.exists():
            shutil.rmtree(self.temp_dir)

    def test_dataset_to_histories_workflow(self):
        """Test complete workflow: generate dataset → generate histories → verify format."""
        num_resid = 5
        num_trajectories = 3

        # Step 1: Generate dataset
        generator = FakeDataGenerator(num_resid)
        pred_coords, obs_shifts, pred_shifts, noes, connectivity = generator.generate_data_arrays(
            random_key=False
        )
        pred_coords, obs_shifts, noes, connectivity = generator.order_data(
            pred_coords, obs_shifts, pred_shifts, noes, connectivity
        )

        # Step 2: Save dataset
        dataset_path = self.temp_path / "test_dataset.pkl"
        dataset_metadata = {
            "version": "1.0",
            "num_resid": num_resid,
            "random_key": False,
        }
        save_dataset(dataset_path, pred_coords, obs_shifts, noes, connectivity, dataset_metadata)

        # Step 3: Load dataset
        dataset = load_dataset(dataset_path)

        # Step 4: Generate histories from loaded dataset
        env = GymEnv(num_resid)
        history_gen = FakeHistoryGenerator(num_resid)

        # Initialize environment once with loaded data
        original_state = env.reset(
            dataset.pred_coordinates,
            dataset.obs_chemical_shifts,
            dataset.noes,
            dataset.connectivity
        )

        trajectories = []
        for i in range(num_trajectories):
            # Perturb state and generate trajectory
            perturbed_state = history_gen.generate_history(original_state)
            trajectory = generate_trajectory(env, perturbed_state)
            trajectories.append(trajectory)

        # Step 5: Save histories
        histories_path = self.temp_path / "test_histories.pkl"
        histories_metadata = {
            "base_dataset_path": str(dataset_path),
            "num_trajectories": num_trajectories,
            "num_resid": num_resid,
        }
        save_histories(histories_path, trajectories, histories_metadata)

        # Step 6: Load and verify histories
        histories = load_histories(histories_path)

        # Verify structure
        self.assertEqual(len(histories.trajectories), num_trajectories)
        self.assertEqual(histories.metadata["num_trajectories"], num_trajectories)

        # Verify each trajectory has correct format
        for traj in histories.trajectories:
            self.assertIsInstance(traj, list)
            self.assertEqual(len(traj), num_resid)  # One step per residue

            for step in traj:
                self.assertIsInstance(step, tuple)
                self.assertEqual(len(step), 3)

                state_dict, action, reward = step
                self.assertIsInstance(state_dict, dict)
                self.assertIsInstance(action, (int, np.integer))
                self.assertIsInstance(reward, (int, float, np.number))

    def test_coordinates_identical_across_trajectories(self):
        """Test that histories from same base dataset have identical coordinates."""
        num_resid = 6
        num_trajectories = 4

        # Generate base dataset
        generator = FakeDataGenerator(num_resid)
        pred_coords, obs_shifts, pred_shifts, noes, connectivity = generator.generate_data_arrays(
            random_key=False
        )
        pred_coords, obs_shifts, noes, connectivity = generator.order_data(
            pred_coords, obs_shifts, pred_shifts, noes, connectivity
        )

        # Extract original coordinates as numpy array
        original_coords = np.array([[p.x, p.y, p.z] for p in pred_coords])

        # Generate multiple trajectories
        env = GymEnv(num_resid)
        history_gen = FakeHistoryGenerator(num_resid)
        original_state = env.reset(pred_coords, obs_shifts, noes, connectivity)

        all_coords_from_trajectories = []

        for i in range(num_trajectories):
            perturbed_state = history_gen.generate_history(original_state)
            trajectory = generate_trajectory(env, perturbed_state)

            # Extract coordinates from first step of this trajectory
            first_state, _, _ = trajectory[0]
            traj_coords = np.array([[p.x, p.y, p.z] for p in first_state["coordinates"]])
            all_coords_from_trajectories.append(traj_coords)

        # Verify all trajectories have identical coordinates to original
        for i, traj_coords in enumerate(all_coords_from_trajectories):
            np.testing.assert_array_almost_equal(
                original_coords,
                traj_coords,
                decimal=10,
                err_msg=f"Trajectory {i} has different coordinates from original",
            )

        # Verify all trajectories have identical coordinates to each other
        for i in range(1, len(all_coords_from_trajectories)):
            np.testing.assert_array_almost_equal(
                all_coords_from_trajectories[0],
                all_coords_from_trajectories[i],
                decimal=10,
                err_msg=f"Trajectory 0 and {i} have different coordinates",
            )

    def test_perturbed_shifts_differ_across_trajectories(self):
        """Test that perturbed shifts differ across trajectories."""
        num_resid = 5
        num_trajectories = 3

        # Generate base dataset
        generator = FakeDataGenerator(num_resid)
        pred_coords, obs_shifts, pred_shifts, noes, connectivity = generator.generate_data_arrays(
            random_key=False
        )
        pred_coords, obs_shifts, noes, connectivity = generator.order_data(
            pred_coords, obs_shifts, pred_shifts, noes, connectivity
        )

        # Generate multiple trajectories
        env = GymEnv(num_resid)
        history_gen = FakeHistoryGenerator(num_resid)
        original_state = env.reset(pred_coords, obs_shifts, noes, connectivity)

        all_shifts_from_trajectories = []

        for i in range(num_trajectories):
            perturbed_state = history_gen.generate_history(original_state)
            trajectory = generate_trajectory(env, perturbed_state)

            # Extract shifts from first step of this trajectory
            first_state, _, _ = trajectory[0]
            traj_shifts = np.array([[s.H1, s.N15] for s in first_state["obs_chemical_shifts"]])
            all_shifts_from_trajectories.append(traj_shifts)

        # Verify that shifts differ between at least some trajectories
        # (they should all be different due to random perturbation)
        shifts_are_different = False
        for i in range(len(all_shifts_from_trajectories) - 1):
            for j in range(i + 1, len(all_shifts_from_trajectories)):
                if not np.allclose(
                    all_shifts_from_trajectories[i], all_shifts_from_trajectories[j]
                ):
                    shifts_are_different = True
                    break
            if shifts_are_different:
                break

        self.assertTrue(
            shifts_are_different,
            "Perturbed shifts should differ across trajectories due to random noise",
        )

    def test_noes_regenerated_consistently_from_coordinates(self):
        """Test that NOEs are regenerated from original coordinates, not perturbed ones."""
        num_resid = 5
        num_trajectories = 3

        # Generate base dataset
        generator = FakeDataGenerator(num_resid)
        pred_coords, obs_shifts, pred_shifts, noes, connectivity = generator.generate_data_arrays(
            random_key=False
        )
        pred_coords, obs_shifts, noes, connectivity = generator.order_data(
            pred_coords, obs_shifts, pred_shifts, noes, connectivity
        )

        # Generate multiple perturbed states and check NOEs
        env = GymEnv(num_resid)
        history_gen = FakeHistoryGenerator(num_resid)
        original_state = env.reset(pred_coords, obs_shifts, noes, connectivity)

        all_noes_from_trajectories = []

        for i in range(num_trajectories):
            perturbed_state = history_gen.generate_history(original_state)
            trajectory = generate_trajectory(env, perturbed_state)

            # Extract NOEs from first step
            first_state, _, _ = trajectory[0]
            traj_noes = first_state["noes"]
            all_noes_from_trajectories.append(traj_noes)

        # Since coordinates are the same across all trajectories,
        # and NOEs are generated from coordinates + shifts with noise,
        # the NOE list structure should be consistent (same number of NOEs)
        noe_counts = [len(noes) for noes in all_noes_from_trajectories]

        # All trajectories should have the same number of NOEs
        # (since they come from same coordinates with same cutoff)
        self.assertTrue(
            all(count == noe_counts[0] for count in noe_counts),
            f"NOE counts differ across trajectories: {noe_counts}. "
            "This suggests NOEs are not being regenerated consistently from coordinates.",
        )

    def test_trajectory_reproducibility_with_seed(self):
        """Test that setting seed produces reproducible trajectories."""
        num_resid = 5

        # Generate base dataset
        generator = FakeDataGenerator(num_resid)
        pred_coords, obs_shifts, pred_shifts, noes, connectivity = generator.generate_data_arrays(
            random_key=False
        )
        pred_coords, obs_shifts, noes, connectivity = generator.order_data(
            pred_coords, obs_shifts, pred_shifts, noes, connectivity
        )

        # Generate trajectory 1 with seed
        np.random.seed(42)
        env1 = GymEnv(num_resid)
        history_gen1 = FakeHistoryGenerator(num_resid)
        original_state1 = env1.reset(pred_coords, obs_shifts, noes, connectivity)
        perturbed_state1 = history_gen1.generate_history(original_state1)
        trajectory1 = generate_trajectory(env1, perturbed_state1)

        # Generate trajectory 2 with same seed
        np.random.seed(42)
        env2 = GymEnv(num_resid)
        history_gen2 = FakeHistoryGenerator(num_resid)
        original_state2 = env2.reset(pred_coords, obs_shifts, noes, connectivity)
        perturbed_state2 = history_gen2.generate_history(original_state2)
        trajectory2 = generate_trajectory(env2, perturbed_state2)

        # Extract shifts from first step of each trajectory
        shifts1 = np.array([[s.H1, s.N15] for s in trajectory1[0][0]["obs_chemical_shifts"]])
        shifts2 = np.array([[s.H1, s.N15] for s in trajectory2[0][0]["obs_chemical_shifts"]])

        # Verify identical perturbations with same seed
        np.testing.assert_array_almost_equal(
            shifts1,
            shifts2,
            decimal=10,
            err_msg="Same seed should produce identical perturbed shifts",
        )


if __name__ == "__main__":
    unittest.main()
