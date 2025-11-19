"""
NMR GNN Models Package

Provides neural network components for NMR chemical shift assignment
using heterogeneous graph neural networks with triple-based message passing.
"""

# Core network components
from .network import (
    NMRLayer,
    NMRNet,
)

# Configuration
from .config import (
    SharedConfig,
    ShiftStandardizeConfig,
    MLPConfig,
    AttentionConfig,
    ModelConfig,
)

# Triple message passing - modular architecture with parameterized components
from .triple import (
    # Gather components
    GatherToTriple,
    NoeGather,
    # Update components
    ResidueUpdate,
    PeakUpdate,
    # Scatter components
    ScatterFromTriple,
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
    "SharedConfig",
    "ShiftStandardizeConfig",
    "MLPConfig",
    "AttentionConfig",
    "ModelConfig",
    # Gather components
    "GatherToTriple",
    "NoeGather",
    # Update components
    "ResidueUpdate",
    "PeakUpdate",
    # Scatter components
    "ScatterFromTriple",
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
