"""
Configuration dataclasses for NMR GNN model architecture.

This module contains all model configuration classes, providing a centralized
location for architecture parameters. Config classes serve as the single source
of truth for model hyperparameters and are passed throughout the model hierarchy.

Design Principles:
1. User-facing parameters: All architecture settings defined in *Config classes
2. Single source of truth: No optional parameter overrides in model classes
3. Clear separation: Configuration (this module) vs implementation (model code)
4. Explicit dependencies: Shared parameters grouped in SharedConfig
5. Distinct MLP configs: Each MLP type has its own configuration

Configuration Hierarchy:
- SharedConfig: Shared parameters used across multiple layer types
- ShiftStandardizeConfig: Chemical shift normalization bounds
- MLPConfig: MLP layer settings (can be specialized for different purposes)
- AttentionConfig: Attention mechanism settings (transformer architecture)
- ModelConfig: Top-level model settings (combines all configs)
"""

from dataclasses import dataclass, field
from typing import Literal


@dataclass
class SharedConfig:
    """
    Shared parameters used across multiple layer types.

    These parameters are accessed by many different layers and represent
    fundamental architectural choices that affect the entire network.
    """

    embed_dim: int = 128  # Feature dimension for all node types (.x attribute)


@dataclass
class ShiftStandardizeConfig:
    """
    Configuration for chemical shift standardization/normalization.

    Contains bounds for normalizing NMR chemical shifts to [0, 1] range.
    Used by the embedding layer to standardize input shift values.
    """

    # Normalization bounds for chemical shifts
    H_lower: float = 6.0  # Hydrogen shift lower bound
    H_upper: float = 10.0  # Hydrogen shift upper bound
    N_lower: float = 100.0  # Nitrogen shift lower bound
    N_upper: float = 135.0  # Nitrogen shift upper bound


@dataclass
class MLPConfig:
    """Configuration for MLP layers in message passing."""

    hidden_size: int = 64
    num_layers: int = 1


@dataclass
class AttentionConfig:
    """Configuration for attention mechanisms."""

    num_heads: int = 4  # Number of attention heads
    attention_dim: int = 64  # Dimension of attention space per head


@dataclass
class ModelConfig:
    """
    Top-level model configuration.

    Key Design Principles:
    1. Shared parameters grouped in SharedConfig for semantic clarity
    2. Each MLP type has its own configuration
    3. Layer-specific configs grouped logically
    4. EmbedConfig slimmed down to just normalization bounds
    """

    # === SHARED PARAMETERS ===
    shared: SharedConfig = field(default_factory=SharedConfig)

    # === LAYER CONFIGURATION ===
    num_nmr_layers: int = 1
    layer_type: Literal["triple", "transformer"] = "triple"  # Layer architecture type

    # === SUB-CONFIGS ===
    shift_standardize: ShiftStandardizeConfig = field(default_factory=ShiftStandardizeConfig)  # Shift normalization bounds
    embed_mlp: MLPConfig = field(default_factory=lambda: MLPConfig(hidden_size=128, num_layers=1))  # For embedding layer
    message_mlp: MLPConfig = field(default_factory=MLPConfig)  # For message passing
    combine_mlp: MLPConfig = field(default_factory=lambda: MLPConfig(hidden_size=256, num_layers=1))  # For attention combination
    value_mlp: MLPConfig = field(default_factory=lambda: MLPConfig(hidden_size=64, num_layers=1))  # For value head
    attention: AttentionConfig = field(default_factory=AttentionConfig)  # Attention config for transformer
