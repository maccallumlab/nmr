"""
Consolidated fake NMR data generation module.

This module provides utilities for generating synthetic NMR data including:
- 3D protein coordinates generation
- HSQC chemical shift peak generation
- NOE distance constraint generation
- Dataset generation with explicit file paths (see io.py for save/load)

The generation process simulates a simplified NMR assignment scenario:
1. Generate 3D protein structure (scaled to realistic dimensions using Flory scaling)
2. Generate HSQC peaks (observed chemical shifts in ppm)
3. Generate NOE crosspeaks (distance constraints from spatial proximity)
4. Add Gaussian noise to simulate experimental uncertainty

Physical Parameters:
- Cutoff distance: 0.5 nm (~5 Angstroms) for NOE generation
- H1 shift range: 6-10 ppm (typical for amide protons)
- N15 shift range: 100-135 ppm (typical for backbone nitrogens)
- Radius of gyration: Rg = 0.2 * N^0.4 (Flory scaling for globular proteins)

Typical usage:
    >>> # Generate dataset
    >>> generator = FakeDataGenerator(num_resid=10)
    >>> coords, shifts, noes, conn = generator.generate_data(pickle_data=False)
    >>>
    >>> # Save dataset with metadata (see io.py)
    >>> from nmr.nmr_gym.io import save_dataset
    >>> metadata = {"version": "1.0", "num_resid": 10}
    >>> save_dataset("dataset.pkl", coords, shifts, noes, conn, metadata)
    >>>
    >>> # Load dataset (see io.py)
    >>> from nmr.nmr_gym.io import load_dataset
    >>> coords, shifts, noes, conn, metadata = load_dataset("dataset.pkl")
    >>>
    >>> # Create state dictionary for RL environment (see state.py)
    >>> from nmr.nmr_gym.state import create_state_dict
    >>> state = create_state_dict(coords, shifts, noes, conn)
"""

import pickle
from typing import List, Optional, Tuple

import numpy as np

from .data_structures import Connectivity, HSQCPeak, NOEPeak, Protein


class FakeDataGenerator:
    """
    Generates synthetic NMR data for protein chemical shift assignment training.

    Creates realistic synthetic datasets including 3D protein coordinates, HSQC chemical
    shift peaks, NOE distance crosspeaks, and connectivity information. Used for supervised
    pre-training of graph neural networks before reinforcement learning fine-tuning.

    The generation process simulates a simplified NMR assignment scenario where:
    - pred_coordinates: 3D structure from a predicted model
    - obs_chemical_shifts: Experimental/observed shifts that need to be assigned
    - pred_chemical_shifts: Shifts calculated from the predicted structure
    - noes: NOE crosspeaks from spatial proximity in predicted structure
    - connectivity: Residue pairs within distance cutoff

    Typical usage:
        >>> generator = FakeDataGenerator(num_resid=10)
        >>> coords, shifts, noes, conn = generator.generate_data(pickle_data=False)
        >>> print(f"Generated {len(coords)} residues with {len(noes)} NOEs")

    For detailed documentation, see docs/fake-data-guide.md
    """

    def __init__(self, num_resid: int):
        """
        Initialize fake data generator.

        Args:
            num_resid: Number of residues in the synthetic protein (must be > 0)

        Raises:
            ValueError: If num_resid <= 0

        Attributes:
            num_resid: Number of residues
            nshift_min: Minimum N15 chemical shift (ppm) - default 100
            nshift_max: Maximum N15 chemical shift (ppm) - default 135
            hshift_min: Minimum H1 chemical shift (ppm) - default 6
            hshift_max: Maximum H1 chemical shift (ppm) - default 10
            cutoff: Distance cutoff for NOE generation (nm) - default 0.5
        """
        if num_resid <= 0:
            raise ValueError(f"num_resid must be > 0, got {num_resid}")

        self.num_resid: int = num_resid
        self.nshift_min = 100
        self.nshift_max = 135
        self.hshift_min = 6
        self.hshift_max = 10
        self.cutoff = 0.5

    def sample_unit(self, num_points: int, num_sides: int, min_len: float = 0, max_len: float = 1) -> np.ndarray:
        """
        Samples points from unit square or cube.

        Args:
            num_points: Number of points to sample
            num_sides: Dimensionality (2 for square, 3 for cube)
            min_len: Minimum coordinate value
            max_len: Maximum coordinate value

        Returns:
            Array of shape (num_points, num_sides) with uniformly sampled coordinates
        """
        return np.random.uniform(low=min_len, high=max_len, size=(num_points, num_sides))

    def scale_unit(self, coordinates: np.ndarray) -> np.ndarray:
        """
        Scale coordinates to realistic protein dimensions using radius of gyration.

        Transforms uniformly sampled coordinates to match expected globular protein
        compactness based on the Flory scaling relationship: Rg = R * N^v

        Args:
            coordinates: Uniformly sampled coordinates in [0,1]³

        Returns:
            Scaled coordinates representing realistic protein structure

        Reference:
            PDB:1CRC (~100 resid, globular, ~3.2 nm diameter)
            DOI: 10.1142/S021972002050050X
            Scaling factor (v) = 0.4 for globular proteins
            R = 0.2 nm (empirical constant)
        """
        # Calculate target radius of gyration based on residue count
        # Rg = 0.2 * N^0.4 gives realistic protein dimensions
        radius_gyration = 0.2 * (self.num_resid ** 0.4)

        # Calculate current center of mass (should be ~0.5 for uniform [0,1] sampling)
        com = np.mean(coordinates, axis=0)

        # Calculate distances from COM for each residue
        distances_sampled = np.linalg.norm(coordinates - com, axis=1)

        # Calculate current radius of gyration: sqrt(mean(r²))
        radius_sampled = np.sqrt(np.mean((distances_sampled ** 2)))

        # Scale factor to match target Rg
        scale = radius_gyration / radius_sampled

        # Apply scaling: translate to origin, scale, translate back
        coordinates_scaled = com + ((coordinates - com) * scale)

        return coordinates_scaled

    def create_hsqc(self) -> np.ndarray:
        """
        Sample points within NMR window for H1 and N15 to create HSQC peaks.

        Returns:
            Array of shape (num_resid, 2) containing [H1, N15] shifts in ppm
        """
        # H shifts
        h_shifts = self.sample_unit(self.num_resid, num_sides=1, min_len=self.hshift_min, max_len=self.hshift_max)
        # N shifts
        n_shifts = self.sample_unit(self.num_resid, num_sides=1, min_len=self.nshift_min, max_len=self.nshift_max)

        return np.hstack((h_shifts, n_shifts))

    def add_noise(self, points: np.ndarray, scale: float = 0.1) -> np.ndarray:
        """
        Add Gaussian noise to simulate experimental uncertainty.

        Random selection from normal (Gaussian) distribution of 'scale' width centered at 0.
        The noise has the same shape as the input points.

        Args:
            points: Array of values to perturb
            scale: Standard deviation of Gaussian noise (must be > 0)

        Returns:
            Perturbed points with added noise

        Raises:
            ValueError: If scale <= 0

        Example:
            point = [0, 1, 2], noise = [0.1, -0.05, 0.2], noisy_point = [0.1, 0.95, 2.2]
        """
        if scale <= 0:
            raise ValueError(f"scale must be > 0, got {scale}")

        noise = np.random.normal(loc=0, scale=scale, size=points.shape)
        noisy_point = points + noise

        return noisy_point

    def calculate_dist(self, p1: np.ndarray, p2: np.ndarray) -> float:
        """
        Calculate Euclidean distance between two points.

        Args:
            p1: First point coordinates
            p2: Second point coordinates

        Returns:
            Euclidean distance between p1 and p2
        """
        return np.linalg.norm((p2 - p1))

    def create_noes(self, coordinates: np.ndarray, shifts: np.ndarray, random_key: bool = False) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate NOE crosspeaks from spatially close residues.

        NOE (Nuclear Overhauser Effect) crosspeaks indicate spatial proximity
        between residues. This generates synthetic NOEs for all residue pairs
        within the distance cutoff.

        Args:
            coordinates: 3D coordinates for each residue (from predicted structure)
            shifts: Chemical shift values [H1, N15] for each residue
            random_key: If True, shuffle shifts to create non-identity mapping
                       (creates assignment problem). If False, use 1:1 mapping
                       where shift index equals residue index.

        Returns:
            Tuple of (noisy_noes, pred_chemical_shifts):
            - noisy_noes: NOE crosspeaks [H1_i, N15_i, H1_j] with noise
            - pred_chemical_shifts: Predicted shifts with noise

        Note:
            May return empty NOE list if protein is small or linear.
            Cutoff is self.cutoff (default 0.5 nm = ~5 Angstroms).
        """
        # If random_key=True, shuffle shifts to create assignment problem
        if random_key:
            shifts = np.random.permutation(shifts)

        # Generate NOEs for all residue pairs within cutoff distance
        noe_list = []
        for i, atom1 in enumerate(coordinates):
            for j, atom2 in enumerate(coordinates):
                if i != j:
                    dist = self.calculate_dist(atom1, atom2)
                    # If residues are close, create NOE crosspeak
                    if dist < self.cutoff:
                        # NOE format: [H1 of res_i, N15 of res_i, H1 of res_j]
                        noe = list(shifts[i][:])  # H1, N15 from residue i
                        noe.append(shifts[j][0])  # H1 from residue j
                        noe_list.append(noe)

        # Add experimental noise to NOEs and predicted shifts
        noisy_noes = self.add_noise(np.array(noe_list), scale=0.01)
        pred_chemical_shifts = self.add_noise(np.array(shifts), scale=0.1)

        return noisy_noes, pred_chemical_shifts

    def calculate_connectivity(self, coordinates: np.ndarray) -> List[Tuple[int, int, float]]:
        """
        Calculate close contacts based on protein coordinates.

        Finds all residue pairs within cutoff distance and returns their
        connectivity information.

        Args:
            coordinates: 3D coordinates for each residue

        Returns:
            List of tuples (atom1_index, atom2_index, distance) for pairs
            within cutoff distance
        """
        contacts = []

        for i, atom1 in enumerate(coordinates):
            for j, atom2 in enumerate(coordinates):
                if i != j:
                    dist = self.calculate_dist(atom1, atom2)
                    if dist < self.cutoff:
                        contacts.append((i, j, dist))

        return contacts

    def generate_data_arrays(self, random_key: bool) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, List[Tuple]]:
        """
        Generate all components of the synthetic NMR system as individual arrays.

        Args:
            random_key: If True, shuffle shifts to create non-identity mapping

        Returns:
            Tuple of (pred_coordinates, obs_chemical_shifts, pred_chemical_shifts, noes, connectivity):
            - pred_coordinates: 3D structure [x, y, z] from predicted model
            - obs_chemical_shifts: Observed/experimental shifts [H1, N15] to be assigned
            - pred_chemical_shifts: Predicted shifts [H1, N15] from structure
            - noes: NOE crosspeaks [H1, N15, H2]
            - connectivity: List of (atom1, atom2, distance) tuples
        """
        # Generate 3D structure from predicted model
        pred_coordinates = self.sample_unit(self.num_resid, num_sides=3)
        pred_coordinates = self.scale_unit(pred_coordinates)

        # Generate observed/experimental chemical shifts
        obs_chemical_shifts = self.create_hsqc()

        # Generate NOEs and predicted shifts from structure
        noes, pred_chemical_shifts = self.create_noes(pred_coordinates, obs_chemical_shifts, random_key=random_key)

        # Calculate connectivity from structure
        connectivity = self.calculate_connectivity(pred_coordinates)

        return pred_coordinates, obs_chemical_shifts, pred_chemical_shifts, noes, connectivity

    def order_data(self, pred_coordinates: np.ndarray, obs_chemical_shifts: np.ndarray,
                   pred_chemical_shifts: np.ndarray, noes: np.ndarray,
                   connectivity: List[Tuple]) -> Tuple[List[Protein], List[HSQCPeak], List[NOEPeak], List[Connectivity]]:
        """
        Compile data into lists of named tuples with one object per residue.

        Args:
            pred_coordinates: 3D structure coordinates
            obs_chemical_shifts: Observed shift values
            pred_chemical_shifts: Predicted shift values
            noes: NOE crosspeak data
            connectivity: Connectivity tuples

        Returns:
            Tuple of (pred_coordinates, obs_chemical_shifts, noes, connectivity) as named tuples
        """
        pred_coordinates = [Protein(x=resid[0], y=resid[1], z=resid[2], H1=shift[0], N15=shift[1])
                           for resid, shift in zip(pred_coordinates, pred_chemical_shifts)]
        obs_chemical_shifts = [HSQCPeak(H1=shift[0], N15=shift[1]) for shift in obs_chemical_shifts]
        noes = [NOEPeak(H1=shift[0], N15=shift[1], H2=shift[2]) for shift in noes]
        connectivity = [Connectivity(atom1=contact[0], atom2=contact[1], distance=contact[2])
                       for contact in connectivity]

        return pred_coordinates, obs_chemical_shifts, noes, connectivity

    def generate_data(self, pickle_data: bool = True, example: bool = True,
                     random_key: bool = False) -> Optional[Tuple[List, List, List, List]]:
        """
        Generate all fake data, order it in lists of namedtuples, and optionally pickle.

        NOTE: This method uses deprecated dump_pickle/load_pickle with default filenames.
        For new code, use generate_data_arrays() followed by save_dataset() with explicit path.

        Args:
            pickle_data: If True, save to disk using deprecated dump_pickle
            example: Passed to dump_pickle (deprecated)
            random_key: If True, shuffle shifts to create non-identity mapping

        Returns:
            If pickle_data=False, returns tuple of (pred_coordinates, obs_chemical_shifts, noes, connectivity)
            If pickle_data=True, returns None (data is saved to disk)
        """
        pred_coordinates, obs_chemical_shifts, pred_chemical_shifts, noes, connectivity = \
            self.generate_data_arrays(random_key=random_key)
        pred_coordinates, obs_chemical_shifts, noes, connectivity = \
            self.order_data(pred_coordinates, obs_chemical_shifts, pred_chemical_shifts, noes, connectivity)

        if pickle_data:
            self.dump_pickle(pred_coordinates, obs_chemical_shifts, noes, connectivity, example=example)
        else:
            return pred_coordinates, obs_chemical_shifts, noes, connectivity

    def dump_pickle(self, pred_coordinates: List[Protein], obs_chemical_shifts: List[HSQCPeak],
                    noes: List[NOEPeak], connectivity: List[Connectivity], example: bool = True):
        """
        Save data to disk using default filename.

        DEPRECATED: Use save_dataset() from nmr.nmr_gym.io with explicit filepath instead.

        Args:
            pred_coordinates: Protein structures with coordinates and shifts
            obs_chemical_shifts: Observed HSQC peaks
            noes: NOE crosspeaks
            connectivity: Connectivity information
            example: If True, uses 'fakedata_r{num_resid}.pkl', else 'current_run.pkl'
        """
        name = f'fakedata_r{self.num_resid}.pkl' if example else f'current_run.pkl'

        with open(name, 'wb') as f:
            pickle.dump(pred_coordinates, f)
            pickle.dump(obs_chemical_shifts, f)
            pickle.dump(noes, f)
            pickle.dump(connectivity, f)

    def load_pickle(self, example: bool) -> Tuple[List[Protein], List[HSQCPeak], List[NOEPeak], List[Connectivity]]:
        """
        Load data from disk using glob pattern.

        DEPRECATED: Use load_dataset() from nmr.nmr_gym.io with explicit filepath instead.

        Args:
            example: If True, searches for '*r{num_resid}.pkl', else 'current_run.pkl'

        Returns:
            Tuple of (coordinates, obs_chemical_shifts, noes, connectivity)
        """
        import glob

        name = f"./*r{self.num_resid}.pkl" if example else "./current_run.pkl"

        pickle_file = glob.glob(name)
        with open(pickle_file[0], 'rb') as f:
            coordinates = pickle.load(f)
            obs_chemical_shifts = pickle.load(f)
            noes = pickle.load(f)
            connectivity = pickle.load(f)

        return coordinates, obs_chemical_shifts, noes, connectivity
