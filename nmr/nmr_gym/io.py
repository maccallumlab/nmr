"""
I/O utilities for saving and loading NMR datasets and training histories.

This module provides functions for persisting and loading NMR datasets and
training histories with explicit file paths and metadata. All functions use
Python's pickle format with well-defined structures and validation.

Dataset Format:
- Single Dataset namedtuple containing:
    - pred_coordinates (list of Protein named tuples)
    - obs_chemical_shifts (list of HSQCPeak named tuples)
    - noes (list of NOEPeak named tuples)
    - connectivity (list of Connectivity named tuples)
    - metadata (dictionary)

Histories Format:
- Single Histories namedtuple containing:
    - trajectories (list of trajectory lists)
    - metadata (dictionary)

Typical usage:
    >>> # Save dataset
    >>> metadata = {"version": "1.0", "num_resid": 10}
    >>> save_dataset("dataset.pkl", coords, shifts, noes, conn, metadata)
    >>>
    >>> # Load dataset
    >>> dataset = load_dataset("dataset.pkl")
    >>> coords = dataset.pred_coordinates
    >>> metadata = dataset.metadata
    >>>
    >>> # Save histories
    >>> metadata = {"base_dataset_path": "dataset.pkl", "num_trajectories": 100}
    >>> save_histories("histories.pkl", trajectories, metadata)
    >>>
    >>> # Load histories
    >>> histories = load_histories("histories.pkl")
    >>> trajectories = histories.trajectories
    >>> metadata = histories.metadata
"""

import pickle
from collections import namedtuple
from pathlib import Path
from typing import Dict, List, Tuple, Union

from .data_structures import Connectivity, HSQCPeak, NOEPeak, Protein

# Named tuples for packaging dataset and histories data
Dataset = namedtuple('Dataset', ['pred_coordinates', 'obs_chemical_shifts', 'noes', 'connectivity', 'metadata'])
Histories = namedtuple('Histories', ['trajectories', 'metadata'])


def save_dataset(filepath: Union[str, Path], pred_coordinates: List[Protein],
                obs_chemical_shifts: List[HSQCPeak], noes: List[NOEPeak],
                connectivity: List[Connectivity], metadata: Dict) -> None:
    """
    Save dataset to disk with explicit file path and metadata.

    Pickle Format:
    - Single Dataset namedtuple containing:
        - pred_coordinates (list of Protein named tuples)
        - obs_chemical_shifts (list of HSQCPeak named tuples)
        - noes (list of NOEPeak named tuples)
        - connectivity (list of Connectivity named tuples)
        - metadata (dictionary)

    Args:
        filepath: Explicit path where dataset will be saved
        pred_coordinates: Protein structures with coordinates and predicted shifts
        obs_chemical_shifts: Observed/experimental HSQC peaks to be assigned
        noes: NOE crosspeaks indicating spatial proximity
        connectivity: Residue pairs within distance cutoff
        metadata: Dictionary with version, timestamp, generation parameters, etc.

    Raises:
        OSError: If file cannot be written
        TypeError: If data structures are invalid

    Example:
        >>> metadata = {
        ...     "version": "1.0",
        ...     "timestamp": datetime.now().isoformat(),
        ...     "num_resid": 10,
        ...     "random_key": False,
        ...     "cutoff": 0.5
        ... }
        >>> save_dataset("dataset.pkl", coords, shifts, noes, conn, metadata)
    """
    filepath = Path(filepath)

    # Create parent directory if it doesn't exist
    filepath.parent.mkdir(parents=True, exist_ok=True)

    # Package into Dataset namedtuple
    dataset = Dataset(
        pred_coordinates=pred_coordinates,
        obs_chemical_shifts=obs_chemical_shifts,
        noes=noes,
        connectivity=connectivity,
        metadata=metadata
    )

    try:
        with open(filepath, 'wb') as f:
            pickle.dump(dataset, f)
    except Exception as e:
        raise OSError(f"Failed to save dataset to {filepath}: {e}") from e


def load_dataset(filepath: Union[str, Path]) -> Dataset:
    """
    Load dataset from disk with explicit file path and validate structure.

    Expected Pickle Format:
    - Single Dataset namedtuple containing:
        - pred_coordinates (list of Protein named tuples)
        - obs_chemical_shifts (list of HSQCPeak named tuples)
        - noes (list of NOEPeak named tuples)
        - connectivity (list of Connectivity named tuples)
        - metadata (dictionary)

    Args:
        filepath: Explicit path to dataset pickle file

    Returns:
        Dataset namedtuple with fields: pred_coordinates, obs_chemical_shifts, noes, connectivity, metadata

    Raises:
        FileNotFoundError: If filepath does not exist
        ValueError: If pickle structure is invalid
        OSError: If file cannot be read

    Example:
        >>> dataset = load_dataset("dataset.pkl")
        >>> print(f"Loaded {dataset.metadata['num_resid']} residues")
        >>> coords = dataset.pred_coordinates
    """
    filepath = Path(filepath)

    if not filepath.exists():
        raise FileNotFoundError(f"Dataset file not found: {filepath}")

    try:
        with open(filepath, 'rb') as f:
            dataset = pickle.load(f)
    except Exception as e:
        raise OSError(f"Failed to load dataset from {filepath}: {e}") from e

    # Validate loaded data structure
    if not isinstance(dataset, Dataset):
        raise ValueError(f"Expected Dataset namedtuple, got {type(dataset)}")

    if not isinstance(dataset.pred_coordinates, list):
        raise ValueError(f"pred_coordinates must be a list, got {type(dataset.pred_coordinates)}")
    if not isinstance(dataset.obs_chemical_shifts, list):
        raise ValueError(f"obs_chemical_shifts must be a list, got {type(dataset.obs_chemical_shifts)}")
    if not isinstance(dataset.noes, list):
        raise ValueError(f"noes must be a list, got {type(dataset.noes)}")
    if not isinstance(dataset.connectivity, list):
        raise ValueError(f"connectivity must be a list, got {type(dataset.connectivity)}")
    if not isinstance(dataset.metadata, dict):
        raise ValueError(f"metadata must be a dict, got {type(dataset.metadata)}")

    # Validate lengths match
    num_coords = len(dataset.pred_coordinates)
    num_shifts = len(dataset.obs_chemical_shifts)
    if num_coords != num_shifts:
        raise ValueError(f"Mismatch: {num_coords} coordinates but {num_shifts} shifts")

    return dataset


def save_histories(filepath: Union[str, Path], trajectories: List[List[Tuple]], metadata: Dict) -> None:
    """
    Save training histories to disk with explicit file path and metadata.

    Each trajectory is a list of tuples with format:
    (state_dict, action_taken, reward)

    Pickle Format:
    - Single Histories namedtuple containing:
        - trajectories (list of trajectory lists)
        - metadata (dictionary)

    Args:
        filepath: Explicit path where histories will be saved
        trajectories: List of trajectories, where each trajectory is a list of
                     (state_dict, action, reward) tuples
        metadata: Dictionary with base_dataset_path, num_trajectories, timestamp, etc.

    Raises:
        OSError: If file cannot be written
        ValueError: If trajectory structure is invalid

    Example:
        >>> metadata = {
        ...     "base_dataset_path": "/path/to/dataset.pkl",
        ...     "num_trajectories": 100,
        ...     "timestamp": datetime.now().isoformat(),
        ...     "num_resid": 10
        ... }
        >>> save_histories("histories.pkl", trajectories, metadata)
    """
    filepath = Path(filepath)

    # Validate trajectory structure
    if not isinstance(trajectories, list):
        raise ValueError(f"trajectories must be a list, got {type(trajectories)}")

    for i, traj in enumerate(trajectories):
        if not isinstance(traj, list):
            raise ValueError(f"Trajectory {i} must be a list, got {type(traj)}")

        for j, step in enumerate(traj):
            if not isinstance(step, tuple) or len(step) != 3:
                raise ValueError(f"Trajectory {i}, step {j} must be a 3-tuple (state_dict, action, reward)")

    # Create parent directory if it doesn't exist
    filepath.parent.mkdir(parents=True, exist_ok=True)

    # Package into Histories namedtuple
    histories = Histories(
        trajectories=trajectories,
        metadata=metadata
    )

    try:
        with open(filepath, 'wb') as f:
            pickle.dump(histories, f)
    except Exception as e:
        raise OSError(f"Failed to save histories to {filepath}: {e}") from e


def load_histories(filepath: Union[str, Path]) -> Histories:
    """
    Load training histories from disk with explicit file path and validate structure.

    Expected Pickle Format:
    - Single Histories namedtuple containing:
        - trajectories (list of trajectory lists)
        - metadata (dictionary)

    Args:
        filepath: Explicit path to histories pickle file

    Returns:
        Histories namedtuple with fields: trajectories, metadata

    Raises:
        FileNotFoundError: If filepath does not exist
        ValueError: If pickle structure is invalid
        OSError: If file cannot be read

    Example:
        >>> histories = load_histories("histories.pkl")
        >>> print(f"Loaded {histories.metadata['num_trajectories']} trajectories")
        >>> trajectories = histories.trajectories
    """
    filepath = Path(filepath)

    if not filepath.exists():
        raise FileNotFoundError(f"Histories file not found: {filepath}")

    try:
        with open(filepath, 'rb') as f:
            histories = pickle.load(f)
    except Exception as e:
        raise OSError(f"Failed to load histories from {filepath}: {e}") from e

    # Validate loaded data structure
    if not isinstance(histories, Histories):
        raise ValueError(f"Expected Histories namedtuple, got {type(histories)}")

    if not isinstance(histories.trajectories, list):
        raise ValueError(f"trajectories must be a list, got {type(histories.trajectories)}")
    if not isinstance(histories.metadata, dict):
        raise ValueError(f"metadata must be a dict, got {type(histories.metadata)}")

    return histories
