"""
NMR data structures used across the package.

This module defines the core NamedTuple data structures for representing
NMR experimental data, protein structures, and connectivity information.
These structures provide type-safe, immutable containers for data exchange
between modules.

Data Structures:
- HSQCPeak: HSQC peak with H1 and N15 chemical shifts (ppm)
- NOEPeak: NOE crosspeak indicating spatial proximity
- Protein: Residue with 3D coordinates and predicted chemical shifts
- Connectivity: Spatial connectivity between two atoms
"""

from typing import NamedTuple


class HSQCPeak(NamedTuple):
    """HSQC peak with H1 and N15 chemical shifts in ppm."""
    H1: float
    N15: float


class NOEPeak(NamedTuple):
    """NOE crosspeak with two H1-N15 pairs indicating spatial proximity."""
    H1: float
    N15: float
    H2: float


class Protein(NamedTuple):
    """Protein residue with 3D coordinates and predicted chemical shifts."""
    x: float
    y: float
    z: float
    H1: float
    N15: float


class Connectivity(NamedTuple):
    """Spatial connectivity between two atoms within cutoff distance."""
    atom1: float
    atom2: float
    distance: float
