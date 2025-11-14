"""
Unit tests for scripts/generate_dataset.py

Tests dataset generation script functionality including:
- Required output argument validation
- Specified filename correctness
- Seed reproducibility
- Error handling for invalid inputs
- Pickle file round-trip (save and load)
- Metadata validation (version, timestamp, num_resid, seed)
"""

import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

# Add parent directory to path to allow imports from nmr package
sys.path.insert(0, str(Path(__file__).parent.parent))

from nmr.nmr_gym.io import load_dataset


class TestGenerateDataset(unittest.TestCase):
    """Test suite for generate_dataset.py script with comprehensive coverage."""

    def setUp(self):
        """Set up test fixtures."""
        self.test_dir = tempfile.mkdtemp()
        self.test_path = Path(self.test_dir)
        self.script_path = Path(__file__).parent.parent / "scripts" / "generate_dataset.py"

    def tearDown(self):
        """Clean up temporary files."""
        # Clean up test directory
        import shutil
        if self.test_path.exists():
            shutil.rmtree(self.test_dir)

    def test_missing_output_argument_errors(self):
        """Test that missing --output argument causes error."""
        result = subprocess.run(
            [sys.executable, str(self.script_path), "--num-resid", "10"],
            capture_output=True,
            text=True
        )
        # argparse exits with code 2 for missing required arguments
        self.assertEqual(result.returncode, 2)
        self.assertIn("required", result.stderr.lower())

    def test_specified_filename_works(self):
        """Test that specified output filename is created correctly."""
        output_file = self.test_path / "my_test_dataset.pkl"

        result = subprocess.run(
            [
                sys.executable,
                str(self.script_path),
                "--num-resid", "5",
                "--output", str(output_file)
            ],
            capture_output=True,
            text=True
        )

        self.assertEqual(result.returncode, 0)
        self.assertTrue(output_file.exists())
        self.assertIn("Generating dataset with 5 residues", result.stdout)
        self.assertIn(f"Saved to: {output_file}", result.stdout)

    def test_seed_reproducibility(self):
        """Test that same seed produces identical datasets."""
        output_file1 = self.test_path / "dataset1.pkl"
        output_file2 = self.test_path / "dataset2.pkl"

        # Generate first dataset with seed 42
        result1 = subprocess.run(
            [
                sys.executable,
                str(self.script_path),
                "--num-resid", "5",
                "--output", str(output_file1),
                "--seed", "42"
            ],
            capture_output=True,
            text=True
        )
        self.assertEqual(result1.returncode, 0)

        # Generate second dataset with same seed
        result2 = subprocess.run(
            [
                sys.executable,
                str(self.script_path),
                "--num-resid", "5",
                "--output", str(output_file2),
                "--seed", "42"
            ],
            capture_output=True,
            text=True
        )
        self.assertEqual(result2.returncode, 0)

        # Load both datasets using load_dataset utility
        dataset1 = load_dataset(output_file1)
        dataset2 = load_dataset(output_file2)

        # Verify data are identical
        np.testing.assert_array_equal(dataset1.pred_coordinates, dataset2.pred_coordinates)
        np.testing.assert_array_equal(dataset1.obs_chemical_shifts, dataset2.obs_chemical_shifts)
        np.testing.assert_array_equal(dataset1.noes, dataset2.noes)
        self.assertEqual(len(dataset1.connectivity), len(dataset2.connectivity))

        # Verify metadata contains seed
        self.assertEqual(dataset1.metadata["seed"], 42)
        self.assertEqual(dataset2.metadata["seed"], 42)

    def test_num_resid_less_than_one_errors(self):
        """Test that num_resid < 1 causes error with clear message."""
        output_file = self.test_path / "invalid.pkl"

        result = subprocess.run(
            [
                sys.executable,
                str(self.script_path),
                "--num-resid", "0",
                "--output", str(output_file)
            ],
            capture_output=True,
            text=True
        )

        self.assertEqual(result.returncode, 1)
        self.assertIn("num_resid must be >= 1", result.stderr)

    def test_invalid_output_path_errors(self):
        """Test that invalid output path causes graceful error."""
        # Use a path that cannot be created (parent doesn't exist and is a file)
        temp_file = self.test_path / "blockfile"
        temp_file.touch()
        invalid_path = temp_file / "subdir" / "output.pkl"

        result = subprocess.run(
            [
                sys.executable,
                str(self.script_path),
                "--num-resid", "5",
                "--output", str(invalid_path)
            ],
            capture_output=True,
            text=True
        )

        self.assertNotEqual(result.returncode, 0)
        self.assertIn("Error", result.stderr)

    def test_pickle_round_trip(self):
        """Test that generated pickle can be loaded and contains expected structure."""
        output_file = self.test_path / "roundtrip_test.pkl"
        num_resid = 10

        # Generate dataset
        result = subprocess.run(
            [
                sys.executable,
                str(self.script_path),
                "--num-resid", str(num_resid),
                "--output", str(output_file)
            ],
            capture_output=True,
            text=True
        )
        self.assertEqual(result.returncode, 0)

        # Load and verify structure using new format
        dataset = load_dataset(output_file)

        # Verify coordinates are list of Protein named tuples
        self.assertEqual(len(dataset.pred_coordinates), num_resid)
        self.assertTrue(hasattr(dataset.pred_coordinates[0], 'x'))
        self.assertTrue(hasattr(dataset.pred_coordinates[0], 'y'))
        self.assertTrue(hasattr(dataset.pred_coordinates[0], 'z'))

        # Verify obs_chemical_shifts are list of HSQCPeak named tuples
        self.assertEqual(len(dataset.obs_chemical_shifts), num_resid)
        self.assertTrue(hasattr(dataset.obs_chemical_shifts[0], 'H1'))
        self.assertTrue(hasattr(dataset.obs_chemical_shifts[0], 'N15'))

        # Verify NOEs exist and have correct structure
        self.assertGreater(len(dataset.noes), 0, "Should generate at least some NOEs")
        self.assertTrue(hasattr(dataset.noes[0], 'H1'))
        self.assertTrue(hasattr(dataset.noes[0], 'N15'))
        self.assertTrue(hasattr(dataset.noes[0], 'H2'))

        # Verify connectivity is a list of tuples
        self.assertIsInstance(dataset.connectivity, list)
        if len(dataset.connectivity) > 0:
            self.assertIsInstance(dataset.connectivity[0], tuple)
            self.assertEqual(len(dataset.connectivity[0]), 3)  # (atom1, atom2, distance)

    def test_output_directory_creation(self):
        """Test that parent directories are created if they don't exist."""
        nested_path = self.test_path / "subdir1" / "subdir2" / "dataset.pkl"

        result = subprocess.run(
            [
                sys.executable,
                str(self.script_path),
                "--num-resid", "5",
                "--output", str(nested_path)
            ],
            capture_output=True,
            text=True
        )

        self.assertEqual(result.returncode, 0)
        self.assertTrue(nested_path.exists())
        self.assertTrue(nested_path.parent.exists())

    def test_metadata_validation(self):
        """Test that generated pickle includes valid metadata with all required fields."""
        output_file = self.test_path / "metadata_test.pkl"
        num_resid = 8

        # Generate dataset with seed
        result = subprocess.run(
            [
                sys.executable,
                str(self.script_path),
                "--num-resid", str(num_resid),
                "--output", str(output_file),
                "--seed", "123"
            ],
            capture_output=True,
            text=True
        )

        # Verify script succeeded
        self.assertEqual(result.returncode, 0, f"Script failed: {result.stderr}")
        self.assertTrue(output_file.exists(), "Output file was not created")

        # Load using load_dataset utility
        dataset = load_dataset(output_file)

        # Verify metadata exists and contains expected keys
        self.assertIsInstance(dataset.metadata, dict)
        self.assertIn("version", dataset.metadata)
        self.assertIn("timestamp", dataset.metadata)
        self.assertIn("num_resid", dataset.metadata)
        self.assertEqual(dataset.metadata["num_resid"], num_resid)

        # Verify seed is in metadata when provided
        self.assertIn("seed", dataset.metadata)
        self.assertEqual(dataset.metadata["seed"], 123)

    def test_metadata_without_seed(self):
        """Test that metadata is created correctly when no seed is provided."""
        output_file = self.test_path / "no_seed_test.pkl"

        # Generate dataset without seed
        result = subprocess.run(
            [
                sys.executable,
                str(self.script_path),
                "--num-resid", "5",
                "--output", str(output_file)
            ],
            capture_output=True,
            text=True
        )

        self.assertEqual(result.returncode, 0)

        # Load and verify metadata
        dataset = load_dataset(output_file)

        # Verify metadata exists with required fields
        self.assertIsInstance(dataset.metadata, dict)
        self.assertIn("version", dataset.metadata)
        self.assertIn("timestamp", dataset.metadata)
        self.assertIn("num_resid", dataset.metadata)
        # Seed may or may not be present when not explicitly provided


if __name__ == '__main__':
    unittest.main()
