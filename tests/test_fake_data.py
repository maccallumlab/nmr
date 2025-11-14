"""
Unit tests for consolidated fake data module functions.

Tests the new utility functions added to nmr/env/fake_data.py:
- Dataset saving/loading with explicit paths
- State dictionary creation and validation
- Shift-only perturbation (not coordinates)
"""

import pickle
import sys
import tempfile
import unittest
from datetime import datetime
from pathlib import Path

import numpy as np

# Add parent directory to path to allow imports from nmr package
sys.path.insert(0, str(Path(__file__).parent.parent))

from nmr.nmr_gym.fake_data import FakeDataGenerator
from nmr.nmr_gym.io import load_dataset, save_dataset
from nmr.nmr_gym.state import create_state_dict, validate_state_dict


class TestDatasetSaveLoad(unittest.TestCase):
    """Test save_dataset and load_dataset utilities."""

    def setUp(self):
        """Set up test fixtures."""
        self.test_dir = tempfile.mkdtemp()
        self.test_path = Path(self.test_dir)

    def tearDown(self):
        """Clean up temporary files."""
        import shutil
        if self.test_path.exists():
            shutil.rmtree(self.test_dir)

    def test_save_and_load_dataset_roundtrip(self):
        """Test that dataset can be saved and loaded with explicit path."""
        # Generate test data
        generator = FakeDataGenerator(num_resid=5)
        pred_coords, obs_shifts, pred_shifts, noes, connectivity = generator.generate_data_arrays(random_key=False)
        pred_coords, obs_shifts, noes, connectivity = generator.order_data(
            pred_coords, obs_shifts, pred_shifts, noes, connectivity
        )

        # Save dataset
        filepath = self.test_path / "test_dataset.pkl"
        metadata = {
            "version": "1.0",
            "timestamp": datetime.now().isoformat(),
            "num_resid": 5,
            "random_key": False,
        }
        save_dataset(filepath, pred_coords, obs_shifts, noes, connectivity, metadata)

        # Load dataset
        dataset = load_dataset(filepath)

        # Verify data matches
        self.assertEqual(len(dataset.pred_coordinates), 5)
        self.assertEqual(len(dataset.obs_chemical_shifts), 5)
        self.assertEqual(dataset.metadata["num_resid"], 5)
        self.assertEqual(dataset.metadata["version"], "1.0")

    def test_load_dataset_validates_structure(self):
        """Test that load_dataset validates pickle structure."""
        # Create invalid pickle with a list instead of Dataset namedtuple
        filepath = self.test_path / "invalid_dataset.pkl"
        with open(filepath, 'wb') as f:
            pickle.dump([1, 2, 3], f)

        # Should raise exception due to invalid structure
        with self.assertRaises(ValueError) as context:
            load_dataset(filepath)

        self.assertIn("Expected Dataset namedtuple", str(context.exception))


class TestStateDict(unittest.TestCase):
    """Test state dictionary creation and validation utilities."""

    def test_create_state_dict_structure(self):
        """Test that create_state_dict returns correct structure."""
        # Create minimal test data
        generator = FakeDataGenerator(num_resid=3)
        pred_coords, obs_shifts, pred_shifts, noes, connectivity = generator.generate_data_arrays(random_key=False)
        pred_coords, obs_shifts, noes, connectivity = generator.order_data(
            pred_coords, obs_shifts, pred_shifts, noes, connectivity
        )

        # Create state dict
        state = create_state_dict(pred_coords, obs_shifts, noes, connectivity)

        # Verify structure
        self.assertIn("coordinates", state)
        self.assertIn("obs_chemical_shifts", state)
        self.assertIn("noes", state)
        self.assertIn("connectivity", state)
        self.assertIn("assignments", state)
        self.assertIn("assign_order", state)
        self.assertIn("shift_to_assign", state)
        self.assertIn("total_energy", state)
        self.assertIn("reward", state)

        # Verify initial values
        self.assertEqual(state["assignments"], {})
        self.assertEqual(state["assign_order"], [])
        self.assertEqual(state["shift_to_assign"], 0)
        self.assertEqual(state["total_energy"], 0.0)
        self.assertEqual(state["reward"], 0.0)

    def test_validate_state_dict_accepts_valid(self):
        """Test that validate_state_dict accepts valid state."""
        # Create valid state
        generator = FakeDataGenerator(num_resid=3)
        pred_coords, obs_shifts, pred_shifts, noes, connectivity = generator.generate_data_arrays(random_key=False)
        pred_coords, obs_shifts, noes, connectivity = generator.order_data(
            pred_coords, obs_shifts, pred_shifts, noes, connectivity
        )
        state = create_state_dict(pred_coords, obs_shifts, noes, connectivity)

        # Should not raise exception
        validate_state_dict(state)

    def test_validate_state_dict_rejects_missing_keys(self):
        """Test that validate_state_dict catches missing keys."""
        # Create incomplete state
        invalid_state = {
            "coordinates": [],
            "obs_chemical_shifts": [],
            # Missing other required keys
        }

        # Should raise exception
        with self.assertRaises(ValueError) as context:
            validate_state_dict(invalid_state)

        self.assertIn("Missing required key", str(context.exception))


class TestShiftPerturbation(unittest.TestCase):
    """Test that perturbation affects only shifts, not coordinates."""

    def test_add_noise_perturbs_shifts_not_coordinates(self):
        """Test that add_noise only modifies shift values."""
        generator = FakeDataGenerator(num_resid=5)

        # Generate coordinates and shifts
        pred_coords = generator.sample_unit(5, num_sides=3)
        pred_coords = generator.scale_unit(pred_coords)
        obs_shifts = generator.create_hsqc()

        # Store original coordinates
        original_coords = pred_coords.copy()

        # Perturb only shifts
        perturbed_shifts = generator.add_noise(obs_shifts, scale=0.1)

        # Verify coordinates unchanged
        np.testing.assert_array_equal(pred_coords, original_coords)

        # Verify shifts changed (with high probability)
        self.assertFalse(np.array_equal(obs_shifts, perturbed_shifts))


if __name__ == '__main__':
    unittest.main()
