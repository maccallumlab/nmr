"""
NMR GNN Models Package

Provides neural network components for NMR chemical shift assignment
using heterogeneous graph neural networks with triple-based message passing.
"""

# Core network components
from .network import (
    NMRLayer,
    NMRNet,
    ModelConfig,
    EmbedConfig,
    MLPConfig,
)

# Triple message passing - new modular architecture
from .triple import (
    # Gather components
    FirstResidueGather,
    FirstPeakGather,
    SecondResidueGather,
    SecondPeakGather,
    NoeGather,
    # Update components
    ResidueUpdate,
    PeakUpdate,
    # Scatter components
    FirstResidueScatter,
    FirstPeakScatter,
    SecondResidueScatter,
    SecondPeakScatter,
    NoeScatter,
    # Triple composition
    ResidueResidueNoeTriple,
    ResiduePeakNoeTriple,
    PeakResidueNoeTriple,
    PeakPeakNoeTriple,
    # Helper functions
    calc_res_distance,
)

# Prediction heads
from .heads import BatchMessagePass, PolicyCalc, ValueCalc

__all__ = [
    # Network
    "NMRLayer",
    "NMRNet",
    # Configuration
    "ModelConfig",
    "EmbedConfig",
    "MLPConfig",
    # Gather components
    "FirstResidueGather",
    "FirstPeakGather",
    "SecondResidueGather",
    "SecondPeakGather",
    "NoeGather",
    # Update components
    "ResidueUpdate",
    "PeakUpdate",
    # Scatter components
    "FirstResidueScatter",
    "FirstPeakScatter",
    "SecondResidueScatter",
    "SecondPeakScatter",
    "NoeScatter",
    # Triple composition
    "ResidueResidueNoeTriple",
    "ResiduePeakNoeTriple",
    "PeakResidueNoeTriple",
    "PeakPeakNoeTriple",
    # Helper functions
    "calc_res_distance",
    # Heads
    "BatchMessagePass",
    "ValueCalc",
    "PolicyCalc",
]
