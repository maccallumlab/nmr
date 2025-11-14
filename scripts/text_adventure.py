"""
Interactive text adventure for NMR chemical shift assignment.

This script provides an interactive command-line interface for manually testing
the NMR chemical shift assignment environment. Users load a previously saved dataset,
then interactively assign chemical shifts to residues while observing energy changes.

Usage:
    python scripts/text_adventure.py <dataset_filepath>

Example:
    python scripts/text_adventure.py dataset_10.pkl

Interactive Gameplay:
- The environment will display the current state and available actions
- Enter a residue index (0-based) to assign the current shift to that residue
- The game continues until all shifts are assigned to residues
- Invalid assignments (already used residues) are rejected
- Final energy and assignments are displayed at the end
"""

import argparse
import math
import sys
from pathlib import Path

# Add parent directory to path to allow imports from nmr package
sys.path.insert(0, str(Path(__file__).parent.parent))

from nmr.nmr_gym.gym_env import GymEnv
from nmr.nmr_gym.io import load_dataset


def load_data(filepath: str) -> tuple:
    """
    Load NMR dataset from file.

    Args:
        filepath: Path to the dataset file

    Returns:
        Tuple of (coordinates, obs_chemical_shifts, noes, connectivity)

    Raises:
        FileNotFoundError: If file doesn't exist
        OSError: If file I/O fails
    """
    print(f"Loading dataset from {filepath}...")
    dataset = load_dataset(filepath)
    print(f"Loaded dataset with {len(dataset.pred_coordinates)} residues")
    return dataset.pred_coordinates, dataset.obs_chemical_shifts, dataset.noes, dataset.connectivity


def text_adventure():
    """
    Main entry point for interactive text adventure.

    Parses command-line arguments, loads dataset from file,
    and runs interactive assignment game.
    """
    parser = argparse.ArgumentParser(
        description="Interactive NMR chemical shift assignment game",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/text_adventure.py dataset_10.pkl
  python scripts/text_adventure.py /path/to/my_dataset.pkl
        """
    )
    parser.add_argument("dataset", type=str, help="Path to dataset file (.pkl)")
    args = parser.parse_args()

    # Load data from file
    try:
        coordinates, obs_chemical_shifts, noes, connectivity = load_data(args.dataset)
    except (FileNotFoundError, OSError, ValueError) as e:
        print(f"Error loading dataset: {e}")
        sys.exit(1)

    # Determine num_resid from loaded data
    num_resid = len(coordinates)
    gym_env = GymEnv(num_resid)

    # Run one episode
    observation = gym_env.reset(
        coordinates, obs_chemical_shifts, noes, connectivity
    )
    terminated = False

    while not terminated:
        # Display current state to the user
        shift_idx = observation["shift_to_assign"]
        shift_val = observation["obs_chemical_shifts"][shift_idx]
        assigned = sorted(observation["assignments"].values())
        unassigned = [i for i in range(num_resid) if i not in assigned]

        print("\n" + "="*60)
        print(f"Current shift to assign: #{shift_idx}")
        print(f"  H1: {shift_val.H1:.2f} ppm, N15: {shift_val.N15:.2f} ppm")
        print(f"Assigned residues: {assigned}")
        print(f"Unassigned residues: {unassigned}")
        print(f"Total energy: {observation['total_energy']:.2f}")
        print(f"Shift to assign: {shift_idx}")
        print("="*60)

        try:
            action = int(input("Enter residue index to assign: "))
            if action < 0 or action >= num_resid:
                print(f"Invalid action {action}. Must be between 0 and {num_resid - 1}")
                continue
            if action not in observation["assignments"].values():
                observation, reward, terminated, total_energy = gym_env.step(action)
                print(f"\nAssigned shift {shift_idx} to residue {action}. Reward: {reward:.2f}")
            else:
                print(f"\nResidue {action} has already been assigned. Choose a different residue.")
        except ValueError:
            print("Please enter a valid integer.")
        except KeyboardInterrupt:
            print("\nExiting game...")
            sys.exit(0)

    # Display final results
    print("\n" + "="*60)
    print("Episode complete!")
    print(f"Final energy: {total_energy:.2f}")
    print(f"Final assignments (shift -> residue): {observation['assignments']}")
    print("="*60)


if __name__ == "__main__":
    text_adventure()
