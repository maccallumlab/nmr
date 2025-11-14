"""
Generate a single fake NMR dataset with explicit output path.

This script creates synthetic NMR data including protein coordinates, HSQC chemical
shifts, NOE distance crosspeaks, and connectivity information. Used for supervised
pre-training of graph neural networks before reinforcement learning fine-tuning.

The script uses the consolidated nmr.nmr_gym.fake_data module and saves datasets with
metadata for versioning and reproducibility.

Output Format:
    The pickle file contains 5 objects in sequence:
    1. pred_coordinates: List of Protein named tuples (x, y, z, H1, N15)
    2. obs_chemical_shifts: List of HSQCPeak named tuples (H1, N15)
    3. noes: List of NOEPeak named tuples (H1, N15, H2)
    4. connectivity: List of Connectivity named tuples (atom1, atom2, distance)
    5. metadata: Dictionary with version, timestamp, num_resid, seed (if provided)

Usage:
    python scripts/generate_dataset.py --num-resid 10 --output my_dataset.pkl
    python scripts/generate_dataset.py --num-resid 10 --output test.pkl --seed 42

For more information, see docs/fake-data-guide.md
"""

import argparse
import random
import sys
from datetime import datetime
from pathlib import Path

import numpy as np

# Add parent directory to path to allow imports from nmr package
sys.path.insert(0, str(Path(__file__).parent.parent))

from nmr.nmr_gym.fake_data import FakeDataGenerator
from nmr.nmr_gym.io import save_dataset


def set_seed(seed: int) -> None:
    """Set random seed for reproducible generation."""
    np.random.seed(seed)
    random.seed(seed)


def main():
    """Main entry point for dataset generation script."""
    parser = argparse.ArgumentParser(
        description="Generate a single fake NMR dataset with explicit output path."
    )
    parser.add_argument(
        "--num-resid",
        type=int,
        required=True,
        help="Number of residues in the synthetic protein (must be > 0)"
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Output filename for the dataset (e.g., my_dataset.pkl)"
    )
    parser.add_argument(
        "--seed",
        type=int,
        help="Random seed for reproducible generation (optional)"
    )

    args = parser.parse_args()

    # Validate num_resid
    if args.num_resid < 1:
        print(
            f"Error: num_resid must be >= 1 (got {args.num_resid})",
            file=sys.stderr
        )
        sys.exit(1)

    # Set seed if provided
    if args.seed is not None:
        set_seed(args.seed)

    # Validate output path
    try:
        output_path = Path(args.output)
    except (TypeError, ValueError) as e:
        print(
            f"Error: Invalid output path '{args.output}': {e}",
            file=sys.stderr
        )
        sys.exit(1)

    # Generate dataset
    print(f"Generating dataset with {args.num_resid} residues...")

    try:
        generator = FakeDataGenerator(args.num_resid)

        # Generate data arrays (coordinates, shifts, noes, connectivity)
        # Note: generate_data_arrays returns 5 values including pred_shifts
        pred_coordinates, obs_chemical_shifts, pred_chemical_shifts, noes, \
            connectivity = generator.generate_data_arrays(random_key=False)

        # Convert arrays to named tuples using order_data
        pred_coordinates, obs_chemical_shifts, noes, connectivity = \
            generator.order_data(
                pred_coordinates, obs_chemical_shifts,
                pred_chemical_shifts, noes, connectivity
            )

        # Prepare metadata
        metadata = {
            "version": "1.0",
            "timestamp": datetime.now().isoformat(),
            "num_resid": args.num_resid,
            "random_key": False,
            "cutoff": generator.cutoff
        }

        # Add seed to metadata if provided
        if args.seed is not None:
            metadata["seed"] = args.seed

        # Save dataset using consolidated save_dataset utility
        save_dataset(
            output_path, pred_coordinates, obs_chemical_shifts,
            noes, connectivity, metadata
        )

        print(f"Saved to: {output_path}")

    except OSError as e:
        # Handle file I/O errors specifically
        print(
            f"Error: Failed to save dataset to '{args.output}'. {e}",
            file=sys.stderr
        )
        print(
            "Ensure the directory exists and you have write permissions.",
            file=sys.stderr
        )
        sys.exit(1)
    except (ValueError, TypeError) as e:
        # Handle validation and data generation errors
        print(f"Error during dataset generation: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
