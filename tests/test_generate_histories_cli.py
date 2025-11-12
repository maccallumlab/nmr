"""
CLI/Integration tests for the generate_histories.py script.

Tests cover:
- Command-line interface and argument parsing
- History generation from dataset via CLI
- Default filename generation with timestamp
- Custom filename specification
- Seed reproducibility via CLI
- Error handling (missing files, invalid formats, invalid arguments)
- Progress messages and help text
- History structure validity
"""

import unittest
import sys
import tempfile
import subprocess
from pathlib import Path
import numpy as np

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from nmr.nmr_gym.fake_data import FakeDataGenerator
from nmr.nmr_gym.io import load_dataset
from scripts.generate_histories import set_seed


class TestGenerateHistoriesCLI(unittest.TestCase):
    """CLI and integration tests for generate_histories.py script."""

    def setUp(self):
        """Create test dataset for use in tests."""
        self.temp_dir = tempfile.mkdtemp()
        self.test_dataset_path = Path(self.temp_dir) / "test_dataset.pkl"

        # Generate a small test dataset
        self.num_resid = 5
        fake_data_gen = FakeDataGenerator(self.num_resid)
        coordinates, obs_chemical_shifts, noes, connectivity = (
            fake_data_gen.generate_data(pickle_data=False, random_key=False)
        )

        # Save test dataset with metadata in new format
        from nmr.nmr_gym.io import save_dataset
        metadata = {"num_resid": self.num_resid, "test": True}
        save_dataset(self.test_dataset_path, coordinates, obs_chemical_shifts, noes, connectivity, metadata)

    def tearDown(self):
        """Clean up temporary files."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_set_seed(self):
        """Test that set_seed() sets the random seed correctly."""
        import random

        set_seed(42)
        val1 = random.random()

        set_seed(42)
        val2 = random.random()

        self.assertEqual(val1, val2, "Same seed should produce same random values")

    def test_load_dataset_success(self):
        """Test successful dataset loading."""
        dataset = load_dataset(str(self.test_dataset_path))

        self.assertIsNotNone(dataset.pred_coordinates)
        self.assertIsNotNone(dataset.obs_chemical_shifts)
        self.assertIsNotNone(dataset.noes)
        self.assertIsNotNone(dataset.connectivity)
        self.assertEqual(len(dataset.pred_coordinates), self.num_resid)

    def test_load_dataset_file_not_found(self):
        """Test error handling for missing dataset file."""
        with self.assertRaises(FileNotFoundError) as context:
            load_dataset("nonexistent_file.pkl")

        self.assertIn("Dataset file not found", str(context.exception))

    def test_load_dataset_invalid_pickle(self):
        """Test error handling for invalid pickle format."""
        invalid_file = Path(self.temp_dir) / "invalid.pkl"

        # Create a file with invalid pickle data
        with open(invalid_file, 'w') as f:
            f.write("This is not a pickle file")

        with self.assertRaises(OSError) as context:
            load_dataset(str(invalid_file))

        self.assertIn("Failed to load dataset", str(context.exception))

    def test_history_generation_cli(self):
        """Test history generation via CLI with small dataset."""
        output_file = Path(self.temp_dir) / "test_histories.pkl"

        # Run the script
        result = subprocess.run([
            sys.executable,
            "scripts/generate_histories.py",
            "--dataset", str(self.test_dataset_path),
            "--num-histories", "3",
            "--output", str(output_file)
        ], capture_output=True, text=True, cwd=Path(__file__).parent.parent)

        self.assertEqual(result.returncode, 0,
                        f"Script failed: {result.stderr}")
        self.assertTrue(output_file.exists(), "Output file not created")

        # Load and verify structure using load_histories
        from nmr.nmr_gym.io import load_histories
        histories = load_histories(str(output_file))

        self.assertIsInstance(histories.trajectories, list)
        self.assertGreater(len(histories.trajectories), 0, "No trajectories generated")

        # Check first trajectory structure - should be list of tuples
        first_trajectory = histories.trajectories[0]
        self.assertIsInstance(first_trajectory, list)
        self.assertGreater(len(first_trajectory), 0, "Empty trajectory")

        # Check first step structure: (state_dict, action, reward)
        first_step = first_trajectory[0]
        self.assertIsInstance(first_step, tuple)
        self.assertEqual(len(first_step), 3, "Step should be 3-tuple")

        state_dict, action, reward = first_step
        self.assertIsInstance(state_dict, dict)
        self.assertIsInstance(action, (int, np.integer))
        self.assertIsInstance(reward, (int, float, np.number))

        # Verify expected keys in state dictionary
        expected_keys = ['coordinates', 'obs_chemical_shifts', 'noes',
                        'connectivity', 'assignments']
        for key in expected_keys:
            self.assertIn(key, state_dict,
                         f"Missing expected key: {key}")

    def test_custom_filename(self):
        """Test that custom output filename is used correctly."""
        custom_name = "my_custom_histories.pkl"
        output_file = Path(self.temp_dir) / custom_name

        result = subprocess.run([
            sys.executable,
            "scripts/generate_histories.py",
            "--dataset", str(self.test_dataset_path),
            "--num-histories", "2",
            "--output", str(output_file)
        ], capture_output=True, text=True, cwd=Path(__file__).parent.parent)

        self.assertEqual(result.returncode, 0)
        self.assertTrue(output_file.exists())
        self.assertEqual(output_file.name, custom_name)

    def test_seed_reproducibility(self):
        """Test that same seed produces identical histories."""
        output1 = Path(self.temp_dir) / "histories1.pkl"
        output2 = Path(self.temp_dir) / "histories2.pkl"

        # Generate with seed=42
        subprocess.run([
            sys.executable,
            "scripts/generate_histories.py",
            "--dataset", str(self.test_dataset_path),
            "--num-histories", "3",
            "--output", str(output1),
            "--seed", "42"
        ], capture_output=True, cwd=Path(__file__).parent.parent)

        # Generate again with same seed
        subprocess.run([
            sys.executable,
            "scripts/generate_histories.py",
            "--dataset", str(self.test_dataset_path),
            "--num-histories", "3",
            "--output", str(output2),
            "--seed", "42"
        ], capture_output=True, cwd=Path(__file__).parent.parent)

        # Load both outputs using load_histories
        from nmr.nmr_gym.io import load_histories
        histories1 = load_histories(str(output1))
        histories2 = load_histories(str(output2))

        self.assertEqual(len(histories1.trajectories), len(histories2.trajectories))

        # Check structure - trajectories are lists of tuples
        for i in range(min(3, len(histories1.trajectories))):
            self.assertEqual(len(histories1.trajectories[i]), len(histories2.trajectories[i]))
            # Check that each step has same structure
            for step1, step2 in zip(histories1.trajectories[i], histories2.trajectories[i]):
                self.assertEqual(len(step1), len(step2), "Steps should be 3-tuples")

        # Check actual content reproducibility - observed chemical shifts and NOEs have noise
        # This tests that numpy random state is properly seeded
        import numpy as np
        if len(histories1.trajectories) > 0 and len(histories1.trajectories[0]) > 0:
            # Get first state from first trajectory
            state1, _, _ = histories1.trajectories[0][0]
            state2, _, _ = histories2.trajectories[0][0]

            # Compare observed chemical shifts which have random noise added
            obs_shifts1 = state1['obs_chemical_shifts']
            obs_shifts2 = state2['obs_chemical_shifts']

            shifts_array1 = np.array([[s.H1, s.N15] for s in obs_shifts1])
            shifts_array2 = np.array([[s.H1, s.N15] for s in obs_shifts2])

            np.testing.assert_array_equal(
                shifts_array1, shifts_array2,
                err_msg="Chemical shifts should be identical with same seed (numpy RNG not seeded?)"
            )

            # Also check NOEs which have random noise added
            noes1 = state1['noes']
            noes2 = state2['noes']

            noes_array1 = np.array([[n.H1, n.N15, n.H2] for n in noes1])
            noes_array2 = np.array([[n.H1, n.N15, n.H2] for n in noes2])

            np.testing.assert_array_equal(
                noes_array1, noes_array2,
                err_msg="NOEs should be identical with same seed (numpy RNG not seeded?)"
            )

    def test_invalid_num_histories(self):
        """Test error handling for invalid num_histories value."""
        output_file = Path(self.temp_dir) / "invalid_test.pkl"
        result = subprocess.run([
            sys.executable,
            "scripts/generate_histories.py",
            "--dataset", str(self.test_dataset_path),
            "--num-histories", "0",
            "--output", str(output_file)
        ], capture_output=True, text=True, cwd=Path(__file__).parent.parent)

        self.assertNotEqual(result.returncode, 0)
        self.assertIn("must be at least 1", result.stderr + result.stdout)

    def test_help_output(self):
        """Test that --help produces clear, descriptive help text."""
        result = subprocess.run([
            sys.executable,
            "scripts/generate_histories.py",
            "--help"
        ], capture_output=True, text=True, cwd=Path(__file__).parent.parent)

        self.assertEqual(result.returncode, 0)

        # Check for key arguments in help text
        self.assertIn("--dataset", result.stdout)
        self.assertIn("--num-histories", result.stdout)
        self.assertIn("--output", result.stdout)
        self.assertIn("--seed", result.stdout)
        self.assertIn("Examples:", result.stdout)

    def test_progress_messages(self):
        """Test that progress messages are printed during generation."""
        result = subprocess.run([
            sys.executable,
            "scripts/generate_histories.py",
            "--dataset", str(self.test_dataset_path),
            "--num-histories", "4",
            "--output", str(Path(self.temp_dir) / "test.pkl")
        ], capture_output=True, text=True, cwd=Path(__file__).parent.parent)

        self.assertEqual(result.returncode, 0)

        # Check for expected progress messages
        output = result.stdout + result.stderr
        self.assertIn("Loading base dataset from:", output)
        self.assertIn("Dataset loaded successfully", output)
        self.assertIn("Generating", output)
        self.assertIn("Generation complete", output)
        self.assertIn("Histories saved to:", output)

    def test_history_structure_validity(self):
        """Test that generated histories have correct and valid structure."""
        output_file = Path(self.temp_dir) / "structure_test.pkl"

        subprocess.run([
            sys.executable,
            "scripts/generate_histories.py",
            "--dataset", str(self.test_dataset_path),
            "--num-histories", "2",
            "--output", str(output_file)
        ], capture_output=True, cwd=Path(__file__).parent.parent)

        # Load using load_histories
        from nmr.nmr_gym.io import load_histories
        histories = load_histories(str(output_file))

        # All trajectories should be lists of tuples
        for i, trajectory in enumerate(histories.trajectories):
            self.assertIsInstance(trajectory, list,
                                 f"Trajectory {i} is not a list")
            self.assertGreater(len(trajectory), 0,
                             f"Trajectory {i} is empty")

            # Each step should be a 3-tuple (state_dict, action, reward)
            for j, step in enumerate(trajectory):
                self.assertIsInstance(step, tuple,
                                    f"Trajectory {i}, step {j} is not a tuple")
                self.assertEqual(len(step), 3,
                               f"Trajectory {i}, step {j} is not a 3-tuple")

                state_dict, action, reward = step

                # State should be a dictionary
                self.assertIsInstance(state_dict, dict,
                                    f"Trajectory {i}, step {j} state is not a dict")

                # Check for required keys
                required_keys = ['coordinates', 'obs_chemical_shifts',
                               'noes', 'connectivity', 'assignments']
                for key in required_keys:
                    self.assertIn(key, state_dict,
                                 f"Trajectory {i}, step {j} missing key: {key}")

                # Assignments should be a dictionary
                self.assertIsInstance(state_dict['assignments'], dict,
                                    f"Trajectory {i}, step {j} assignments not a dict")

                # Action should be an integer (Python int or numpy integer)
                self.assertIsInstance(action, (int, np.integer),
                                    f"Trajectory {i}, step {j} action not an int")

                # Reward should be numeric
                self.assertIsInstance(reward, (int, float, np.number),
                                    f"Trajectory {i}, step {j} reward not numeric")


if __name__ == '__main__':
    unittest.main()
