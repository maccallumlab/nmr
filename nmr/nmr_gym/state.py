"""
State management utilities for RL environment.

This module provides functions for creating and validating state dictionaries
used by the NMR chemical shift assignment RL environment. State dictionaries
contain all information needed to represent a point in the assignment process.

State Dictionary Structure:
- coordinates: List of Protein named tuples (x, y, z, H1, N15)
- obs_chemical_shifts: List of HSQCPeak named tuples (H1, N15) to be assigned
- noes: List of NOEPeak named tuples (H1, N15, H2) representing spatial constraints
- connectivity: List of Connectivity named tuples (atom1, atom2, distance)
- assignments: Dict mapping shift indices to residue indices
- assign_order: List of shift indices in assignment order
- shift_to_assign: Int or None - current shift index to assign
- total_energy: Float - running total energy
- reward: Float - reward for current step

Typical usage:
    >>> # Create initial state from dataset
    >>> state = create_state_dict(coords, shifts, noes, conn)
    >>> validate_state_dict(state)  # Raises ValueError if invalid
    >>> env.custom_state(state)
"""

from typing import Dict, List

from .data_structures import Connectivity, HSQCPeak, NOEPeak, Protein


def create_state_dict(coordinates: List[Protein], obs_chemical_shifts: List[HSQCPeak],
                     noes: List[NOEPeak], connectivity: List[Connectivity]) -> Dict:
    """
    Create state dictionary for RL environment from dataset components.

    State Dictionary Structure:
    - coordinates: List of Protein named tuples (x, y, z, H1, N15)
    - obs_chemical_shifts: List of HSQCPeak named tuples (H1, N15) to be assigned
    - noes: List of NOEPeak named tuples (H1, N15, H2) representing spatial constraints
    - connectivity: List of Connectivity named tuples (atom1, atom2, distance)
    - assignments: Empty dict {} (will be filled during episode)
    - assign_order: Empty list [] (will be filled by FractionalActivation)
    - shift_to_assign: 0 (initial index before assignment order is determined)
    - total_energy: 0.0 (running total energy)
    - reward: 0.0 (reward for current step)

    Args:
        coordinates: Protein structures with coordinates and predicted shifts
        obs_chemical_shifts: Observed HSQC peaks to be assigned
        noes: NOE crosspeaks indicating spatial proximity
        connectivity: Residue pairs within distance cutoff

    Returns:
        State dictionary matching GymEnv.reset() format

    Example:
        >>> state = create_state_dict(coords, shifts, noes, conn)
        >>> env.custom_state(state)
    """
    return {
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


def validate_state_dict(state: Dict) -> None:
    """
    Validate state dictionary structure and contents.

    Checks for required keys, correct data types, and shape consistency.
    Raises specific exceptions with clear messages on validation failure.

    Required Keys:
    - coordinates: list of Protein named tuples
    - obs_chemical_shifts: list of HSQCPeak named tuples
    - noes: list of NOEPeak named tuples
    - connectivity: list of Connectivity named tuples
    - assignments: dict mapping shift indices to residue indices
    - assign_order: list of shift indices in assignment order
    - shift_to_assign: int or None (current shift index to assign)
    - total_energy: float (running total energy)
    - reward: float (reward for current step)

    Args:
        state: State dictionary to validate

    Raises:
        ValueError: If required key is missing, wrong type, or shape mismatch

    Example:
        >>> validate_state_dict(state)  # Raises ValueError if invalid
    """
    required_keys = [
        "coordinates", "obs_chemical_shifts", "noes", "connectivity",
        "assignments", "assign_order", "shift_to_assign", "total_energy", "reward"
    ]

    # Check for missing keys
    for key in required_keys:
        if key not in state:
            raise ValueError(f"Missing required key in state dictionary: '{key}'")

    # Check data types
    if not isinstance(state["coordinates"], list):
        raise ValueError(f"coordinates must be a list, got {type(state['coordinates'])}")
    if not isinstance(state["obs_chemical_shifts"], list):
        raise ValueError(f"obs_chemical_shifts must be a list, got {type(state['obs_chemical_shifts'])}")
    if not isinstance(state["noes"], list):
        raise ValueError(f"noes must be a list, got {type(state['noes'])}")
    if not isinstance(state["connectivity"], list):
        raise ValueError(f"connectivity must be a list, got {type(state['connectivity'])}")
    if not isinstance(state["assignments"], dict):
        raise ValueError(f"assignments must be a dict, got {type(state['assignments'])}")
    if not isinstance(state["assign_order"], list):
        raise ValueError(f"assign_order must be a list, got {type(state['assign_order'])}")
    if not isinstance(state["total_energy"], (int, float)):
        raise ValueError(f"total_energy must be a number, got {type(state['total_energy'])}")
    if not isinstance(state["reward"], (int, float)):
        raise ValueError(f"reward must be a number, got {type(state['reward'])}")

    # Check shape consistency
    num_coords = len(state["coordinates"])
    num_shifts = len(state["obs_chemical_shifts"])
    if num_coords != num_shifts:
        raise ValueError(f"Shape mismatch: {num_coords} coordinates but {num_shifts} obs_chemical_shifts")
