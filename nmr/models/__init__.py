"""
NMR GNN Models Package

Provides neural network components for NMR chemical shift assignment
using heterogeneous graph neural networks with triple-based message passing.
"""

# Core network components
from .network import NMRLayer, NMRNet

# Triple message passing
from .triple import (
    CalculationManager,
    TripleIn,
    TripleMessagePass,
    TripleOut,
    TripleUpdate,
)

# Prediction heads
from .heads import BatchMessagePass, PolicyCalc, ValueCalc

__all__ = [
    # Network
    "NMRLayer",
    "NMRNet",
    # Triple components
    "CalculationManager",
    "TripleIn",
    "TripleUpdate",
    "TripleMessagePass",
    "TripleOut",
    # Heads
    "BatchMessagePass",
    "ValueCalc",
    "PolicyCalc",
]
