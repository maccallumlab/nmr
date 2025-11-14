"""
Generate synthetic NMR assignment histories from a dataset.

This script implements the correct history generation algorithm:
1. Load base dataset ONCE (coordinates and shifts)
2. For N iterations:
   - Perturb ONLY observed shifts (keep original coordinates unchanged)
   - Regenerate NOEs from original coordinates
   - Create environment with perturbed state
   - Generate trajectory using identity mapping (action = shift_to_assign)
   - Record trajectory as list of (state_dict, action, reward) tuples
3. Save all trajectories to output file

The key principle is that coordinates define the protein structure and must remain
constant across all trajectories. Only observed shifts vary to simulate experimental
uncertainty, while NOEs are regenerated from the true (original) structure.

Trajectory Format:
Each trajectory is a list of tuples: (state_dict, action_taken, reward)
- state_dict: Complete environment state at this step
- action_taken: Residue index chosen for assignment (integer)
- reward: Negative energy increase from env.step() (float)

Identity Mapping:
Due to how synthetic data is generated (random_key=False), the identity mapping
holds: when environment asks to assign shift index j, the correct residue is also j.
This is shift_index → residue_index as 1:1 mapping for perfect play.

Usage Examples:
  # Generate 100 histories from dataset
  python generate_histories.py --dataset data.pkl --num-histories 100 --output histories.pkl

  # Generate with reproducible seed
  python generate_histories.py --dataset data.pkl --num-histories 50 --output hist.pkl --seed 42
"""

import argparse
import sys
from datetime import datetime
from pathlib import Path

import numpy as np

# Add parent directory to path to allow imports from nmr package
sys.path.insert(0, str(Path(__file__).parent.parent))

from nmr.nmr_gym.io import load_dataset, save_histories
from nmr.nmr_gym.fake_histories import FakeHistoryGenerator, generate_trajectory
from nmr.nmr_gym.gym_env import GymEnv


def set_seed(seed: int) -> None:
    """Set random seed for reproducibility.

    Args:
        seed: Random seed value
    """
    import random
    random.seed(seed)
    np.random.seed(seed)


def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(
        description="Generate synthetic NMR assignment histories from a dataset",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Generate 100 histories
  python generate_histories.py --dataset data.pkl --num-histories 100 --output histories.pkl

  # Generate with reproducible seed
  python generate_histories.py --dataset data.pkl --num-histories 50 --output hist.pkl --seed 42
""",
    )

    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        help="Path to input dataset pickle file (created by generate_dataset.py)",
    )
    parser.add_argument(
        "--num-histories",
        type=int,
        required=True,
        help="Number of histories to generate (must be > 0)",
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Output filepath for histories pickle file",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for reproducibility (optional)",
    )

    args = parser.parse_args()

    # Validate num_histories
    if args.num_histories < 1:
        print("Error: num-histories must be at least 1")
        sys.exit(1)

    # Set seed if provided
    if args.seed is not None:
        set_seed(args.seed)
        print(f"Random seed set to: {args.seed}")

    # Load base dataset ONCE
    print(f"Loading base dataset from: {args.dataset}")
    try:
        dataset = load_dataset(args.dataset)
        coords = dataset.pred_coordinates
        obs_shifts = dataset.obs_chemical_shifts
        noes = dataset.noes
        conn = dataset.connectivity
    except FileNotFoundError as e:
        print(f"Error: {e}")
        sys.exit(1)
    except ValueError as e:
        print(f"Error: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"Error loading dataset: {e}")
        sys.exit(1)

    # Extract num_resid from the data
    num_resid = len(coords)
    print(f"Dataset loaded successfully ({num_resid} residues)")

    # Initialize environment and history generator
    env = GymEnv(num_resid)
    history_gen = FakeHistoryGenerator(num_resid)

    # Generate trajectories
    print(f"Generating {args.num_histories} trajectories...")

    # Initialize environment with original data to get base state
    original_state = env.reset(coords, obs_shifts, noes, conn)

    trajectories = []
    trajectory_lengths = []

    # Calculate progress reporting interval (every 25%)
    progress_interval = max(1, args.num_histories // 4)

    for i in range(args.num_histories):
        # Perturb ONLY observed shifts, regenerate NOEs from original coordinates
        perturbed_state = history_gen.generate_history(original_state)

        # Generate trajectory using identity mapping (action = shift_to_assign)
        trajectory = generate_trajectory(env, perturbed_state)

        # Record trajectory and its length
        trajectories.append(trajectory)
        trajectory_lengths.append(len(trajectory))

        # Report progress every 25%
        if (i + 1) % progress_interval == 0 or (i + 1) == args.num_histories:
            print(f"Progress: Generated trajectory {i + 1}/{args.num_histories}")

    # Calculate statistics
    avg_length = np.mean(trajectory_lengths)
    print(f"\nGeneration complete:")
    print(f"  Total trajectories: {args.num_histories}")
    print(f"  Average trajectory length: {avg_length:.1f} steps")

    # Save all trajectories with metadata
    history_metadata = {
        "base_dataset_path": str(Path(args.dataset).resolve()),
        "num_trajectories": args.num_histories,
        "num_resid": num_resid,
        "timestamp": datetime.now().isoformat(),
        "seed": args.seed,
        "avg_trajectory_length": float(avg_length)
    }

    try:
        save_histories(args.output, trajectories, history_metadata)
        print(f"Histories saved to: {args.output}")
    except OSError as e:
        print(f"Error saving histories: {e}")
        sys.exit(1)
    except ValueError as e:
        print(f"Error validating trajectories: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
