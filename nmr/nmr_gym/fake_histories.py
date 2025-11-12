"""
Generates training histories by perturbing synthetic NMR datasets.

This module implements the correct history generation algorithm for creating
training data from synthetic NMR datasets. The key principle is that spatial
relationships (NOEs) must reflect the true structure, while introducing
variability only in the observed chemical shifts.

Correct Algorithm:
1. Load base dataset with original coordinates and shifts
2. For each trajectory:
   a. Keep original coordinates unchanged (defines true structure)
   b. Perturb ONLY observed chemical shifts (simulates experimental variation)
   c. Regenerate NOEs from original coordinates (spatial relationships)
   d. Play through with identity mapping for perfect play
   e. Record trajectory as list of (state_dict, action, reward) tuples

Why This Matters:
- Coordinates define the protein structure and spatial relationships
- NOEs are distance constraints derived from 3D structure
- Perturbing coordinates would change the structure and invalidate NOEs
- Only shifts should vary to simulate experimental uncertainty

Identity Mapping for Perfect Play:
- Due to how synthetic data is generated (random_key=False), identity mapping holds
- When environment asks to assign shift index j, correct residue is also j
- This is shift_index → residue_index as 1:1 mapping
- Perfect play uses: action = state["shift_to_assign"]

Trajectory Format:
Each trajectory is a list of tuples:
(state_dict, action_taken, value)

Where:
- state_dict: Complete environment state at this step
- action_taken: Residue index chosen for assignment (integer)
- value: Sum of rewards from this point onwards (return)
"""

import copy
import numpy as np

from .fake_data import FakeDataGenerator


class FakeHistoryGenerator():
    """
    Generates training histories by perturbing synthetic NMR datasets.

    Creates multiple training examples from a base dataset by adding noise to
    observed shifts and generating "perfect play" assignment sequences. Each
    history represents one complete assignment episode where the model makes
    correct choices at each step.

    The correct perturbation process:
    1. Keep original 3D coordinates unchanged (defines structure)
    2. Perturb only observed chemical shifts (σ=0.1 ppm)
    3. Regenerate NOEs from original (unperturbed) coordinates
    4. Create assignment history playing through with identity mapping

    This provides training diversity while maintaining structural consistency,
    enabling supervised pre-training before RL fine-tuning.

    Typical usage:
        >>> generator = FakeHistoryGenerator(num_resid=10)
        >>> perturbed_state = generator.generate_history(original_state)
        >>> trajectory = generate_trajectory(env, perturbed_state)
    """

    def __init__(self, num_resid):
        """
        Initialize history generator.

        Args:
            num_resid: Number of residues (must match base dataset)
        """
        self.num_resid = num_resid
        self.fakedata = FakeDataGenerator(num_resid)
        self.cutoff = 0.5

    def perturb_data(self, obs_chemical_shifts):
        """
        Perturb only observed chemical shifts with Gaussian noise.

        IMPORTANT: Coordinates are NOT perturbed because they define the
        protein structure. Perturbing coordinates would change spatial
        relationships and invalidate the NOE distance constraints. Only
        chemical shifts vary to simulate experimental uncertainty.

        Args:
            obs_chemical_shifts: List of HSQCPeak NamedTuples with H1, N15

        Returns:
            Perturbed chemical shifts [H1, N15] with σ=0.1 ppm noise
        """
        # Extract shifts and convert to numpy array
        shifts_array = np.array([[s.H1, s.N15] for s in obs_chemical_shifts])

        # Add chemical shift noise (σ=0.1 ppm)
        perturbed_shifts = self.fakedata.add_noise(shifts_array, scale=0.1)

        return perturbed_shifts

    def generate_history(self, original):
        """
        Generate a perturbed training history from a base dataset.

        Creates a new training example by:
        1. Extracting original coordinates (UNCHANGED)
        2. Perturbing ONLY observed shifts with Gaussian noise
        3. Regenerating NOEs from original coordinates + perturbed shifts
        4. Recalculating connectivity from original coordinates
        5. Packaging into a clean initial state dictionary

        The history is initialized with empty assignments, ready to be played
        through by the RL environment making "perfect" assignment choices
        using the identity mapping (action = shift_to_assign).

        Args:
            original: Base state dictionary containing:
                - 'coordinates': List of Protein NamedTuples (x,y,z,H1,N15)
                - 'obs_chemical_shifts': List of HSQCPeak NamedTuples

        Returns:
            State dictionary ready for RL environment with structure:
            {
                "coordinates": List[Protein],  # UNCHANGED from original
                "obs_chemical_shifts": List[HSQCPeak],  # Perturbed
                "noes": List[NOEPeak],  # Regenerated from original coords
                "connectivity": List[Connectivity],  # From original coords
                "assignments": {},  # Empty - to be filled during play
                "assign_order": [],  # To be set by environment
                "shift_to_assign": 0,
                "total_energy": 0.0,
                "reward": 0.0
            }
        """
        # Step 1: Extract ORIGINAL coordinates (do NOT perturb)
        # Coordinates define the structure and must remain unchanged
        original_coords = np.array([
            [p.x, p.y, p.z] for p in original['coordinates']
        ])

        # Step 2: Perturb ONLY observed shifts
        perturbed_shifts = self.perturb_data(original['obs_chemical_shifts'])

        # Step 3: Regenerate NOEs using ORIGINAL coordinates + perturbed shifts
        # This ensures NOEs reflect true spatial relationships
        noes, pred_chemical_shifts = self.fakedata.create_noes(
            original_coords, perturbed_shifts, random_key=False
        )

        # Step 4: Recalculate connectivity using ORIGINAL coordinates
        connectivity = self.fakedata.calculate_connectivity(original_coords)

        # Step 5: Convert arrays to NamedTuples
        coordinates, obs_chemical_shifts, noes, connectivity = (
            self.fakedata.order_data(
                original_coords, perturbed_shifts, pred_chemical_shifts,
                noes, connectivity
            )
        )

        # Create clean initial state
        state = {
            "coordinates": coordinates,
            "obs_chemical_shifts": obs_chemical_shifts,
            "noes": noes,
            "connectivity": connectivity,
            "assignments": {},
            "assign_order": [],
            "shift_to_assign": 0,
            "total_energy": 0.0,
            "reward": 0.0
        }

        return state


def generate_trajectory(env, state_dict):
    """
    Generate a perfect play trajectory using identity mapping.

    Plays through a complete assignment episode recording each step as a
    tuple of (state_dict, action_taken, value). Uses identity mapping for
    perfect play where action equals shift_to_assign.

    Values are computed as the sum of rewards from each point onwards.
    For example, if rewards are [0, 1, 1, 0], values are [2, 1, 0, 0].

    Args:
        env: GymEnv instance configured with num_resid
        state_dict: Initial state dictionary from generate_history or
                   env.reset()

    Returns:
        List of trajectory tuples: [(state_dict, action, value), ...]
        Each tuple contains:
        - state_dict: Complete environment state at this step (deep copy)
        - action: Residue index chosen for assignment (int)
        - value: Sum of rewards from this point onwards (float)

    Example:
        >>> env = GymEnv(num_resid=10)
        >>> state = env.reset(coords, shifts, noes, connectivity)
        >>> trajectory = generate_trajectory(env, state)
        >>> print(f"Trajectory has {len(trajectory)} steps")
    """
    # Initialize environment with the provided state
    # Use custom_state to set up the environment with this specific state
    current_state = env.custom_state(state_dict)

    trajectory_with_rewards = []
    terminated = False

    # First pass: collect states, actions, and rewards
    while not terminated:
        # Perfect play: use identity mapping
        # For synthetic data with random_key=False, shift_index = residue_index
        action = current_state["shift_to_assign"]

        # IMPORTANT: Deep copy the state BEFORE taking the action to preserve it
        state_before_action = copy.deepcopy(current_state)

        # Take the action
        current_state, reward, terminated, _ = env.step(action)

        # Store the state before action, the action taken, and the reward
        trajectory_with_rewards.append((state_before_action, action, reward))

    # Second pass: convert rewards to values (sum of future rewards)
    trajectory = []
    cumulative_value = 0.0

    # Iterate backwards to compute cumulative values
    for state, action, reward in reversed(trajectory_with_rewards):
        cumulative_value += reward
        trajectory.append((state, action, cumulative_value))

    # Reverse to restore original order
    trajectory.reverse()

    return trajectory
