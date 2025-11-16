"""
Top-level NMR GNN network architecture.

Combines triple-based message passing with value and policy heads
to create the complete neural network for NMR assignment.
"""

from dataclasses import dataclass, field
from typing import Literal

import torch
import torch.nn as nn

from .heads import PolicyCalc, ValueCalc
from .pair import AssignedPair
from .triple import (
    PeakPeakNoeTriple,
    PeakResidueNoeTriple,
    ResiduePeakNoeTriple,
    ResidueResidueNoeTriple,
)
from .transformer import AttentionConfig


@dataclass
class EmbedConfig:
    """Configuration for embedding shifts+flags to working features."""

    embed_dim: int = 128  # Output dimension for all .x features
    hidden_dim: int = 128
    num_layers: int = 1
    H_lower: float = 6.0
    H_upper: float = 10.0
    N_lower: float = 100.0
    N_upper: float = 135.0


@dataclass
class MLPConfig:
    """Configuration for MLP layers in message passing."""

    hidden_size: int = 64
    num_layers: int = 1


@dataclass
class ModelConfig:
    """Top-level configuration for NMRNet model."""

    num_nmr_layers: int = 1
    layer_type: Literal["triple", "transformer"] = "triple"  # Layer architecture type
    embed: EmbedConfig = field(default_factory=EmbedConfig)
    mlp: MLPConfig = field(default_factory=MLPConfig)
    attention: AttentionConfig = field(default_factory=AttentionConfig)  # Attention config for transformer


class NMRLayer(nn.Module):
    """
    Single layer of NMR-specific message passing.

    Orchestrates message passing through all four triple types and assigned pairs in sequence:
    - ResidueResidueNoeTriple: (Residue, Residue, Noe)
    - ResiduePeakNoeTriple: (Residue, Peak, Noe)
    - PeakResidueNoeTriple: (Peak, Residue, Noe)
    - PeakPeakNoeTriple: (Peak, Peak, Noe)
    - AssignedPair: (Peak, Residue) assigned pairs

    Each triple/pair type is instantiated as a separate attribute and called explicitly
    in the forward pass, with no conditionals or branching logic.
    """

    def __init__(self, device, config: ModelConfig):
        super(NMRLayer, self).__init__()
        # Instantiate all 4 triple classes
        self.residue_residue_noe = ResidueResidueNoeTriple(device, config)
        self.residue_peak_noe = ResiduePeakNoeTriple(device, config)
        self.peak_residue_noe = PeakResidueNoeTriple(device, config)
        self.peak_peak_noe = PeakPeakNoeTriple(device, config)
        # Instantiate assigned pair processing
        self.assigned_pair = AssignedPair(device, config)

    def forward(self, data):
        """
        Process assigned pairs and all four triple types in sequence.

        Args:
            data: HeteroData graph to process

        Returns:
            Updated HeteroData graph with all node features updated
        """
        data = self.assigned_pair(data)
        data = self.residue_residue_noe(data)
        data = self.residue_peak_noe(data)
        data = self.peak_residue_noe(data)
        data = self.peak_peak_noe(data)
        return data


class EmbedFeatures(nn.Module):
    """
    Embed raw .shifts and .flags into working .x features.

    Architecture:
        standardize(shifts) →
        concat(shifts, flags) →
        MLP →
        .x[embed_dim]

    All node types produce the same .x dimension:
    - Residue: concat([shifts(2), flags(1)]) → MLP → .x[embed_dim]
    - Peak: concat([shifts(2), flags(2)]) → MLP → .x[embed_dim]
    - Noe: shifts(3) → MLP → .x[embed_dim]
    """

    def __init__(self, device, config: ModelConfig):
        super().__init__()
        self.device = device
        self.config = config
        self.embed_dim = config.embed.embed_dim

        # Normalization parameters for shifts
        self.H_lower = config.embed.H_lower
        self.H_upper = config.embed.H_upper
        self.H_delta = self.H_upper - self.H_lower
        self.N_lower = config.embed.N_lower
        self.N_upper = config.embed.N_upper
        self.N_delta = self.N_upper - self.N_lower

        # Embedding MLPs: input → hidden → output
        # Residue: [shifts(2) + flags(1)] = 3 → embed_dim
        self.residue_embed = self._build_mlp(3, config.embed)
        # Peak: [shifts(2) + flags(2)] = 4 → embed_dim
        self.peak_embed = self._build_mlp(4, config.embed)
        # NOE: shifts(3) → embed_dim
        self.noe_embed = self._build_mlp(3, config.embed)

    def _build_mlp(self, input_dim: int, config: EmbedConfig):
        """Build an MLP: input_dim → hidden → embed_dim"""
        layers = []
        layers.append(nn.Linear(input_dim, config.hidden_dim, device=self.device))
        layers.append(nn.ReLU())

        for _ in range(config.num_layers - 1):
            layers.append(
                nn.Linear(config.hidden_dim, config.hidden_dim, device=self.device)
            )
            layers.append(nn.ReLU())

        layers.append(
            nn.Linear(config.hidden_dim, config.embed_dim, device=self.device)
        )
        return nn.Sequential(*layers)

    def _normalize_shifts(self, shifts, is_noe=False):
        """
        Normalize shifts to [-1, 1] range.

        Args:
            shifts: [n, 2] for residue/peak [H, N] or [n, 3] for NOE [N, H', H"]
            is_noe: If True, expects [N, H', H"] format, else [H, N]

        Returns:
            Normalized shifts in same format
        """
        if is_noe:
            # NOE format: [N, H', H"]
            N = shifts[:, 0:1]
            H1 = shifts[:, 1:2]
            H2 = shifts[:, 2:3]
            N_norm = 2 * (N - self.N_lower) / self.N_delta - 1
            H1_norm = 2 * (H1 - self.H_lower) / self.H_delta - 1
            H2_norm = 2 * (H2 - self.H_lower) / self.H_delta - 1
            return torch.cat([N_norm, H1_norm, H2_norm], dim=-1)
        else:
            # Residue/Peak format: [H, N]
            H = shifts[:, 0:1]
            N = shifts[:, 1:2]
            H_norm = 2 * (H - self.H_lower) / self.H_delta - 1
            N_norm = 2 * (N - self.N_lower) / self.N_delta - 1
            return torch.cat([H_norm, N_norm], dim=-1)

    def forward(self, data):
        """
        Embed all features and shifts in the graph.

        Creates .x attributes from concat(shifts, flags):
        - Residue.x: [embed_dim]
        - Peak.x: [embed_dim]
        - Noe.x: [embed_dim]

        Normalizes shifts during embedding (does NOT modify .shifts attribute).

        Args:
            data: HeteroData graph with .shifts and .flags attributes

        Returns:
            HeteroData graph with .x working features created
        """
        # Residue: concat([normalized_shifts(2), flags(1)]) → MLP → .x[embed_dim]
        res_shifts_norm = self._normalize_shifts(data["Residue"].shifts, is_noe=False)
        res_input = torch.cat(
            [res_shifts_norm, data["Residue"].flags], dim=-1
        )  # [n, 3]
        data["Residue"].x = self.residue_embed(res_input)  # [n, embed_dim]

        # Peak: concat([normalized_shifts(2), flags(2)]) → MLP → .x[embed_dim]
        peak_shifts_norm = self._normalize_shifts(data["Peak"].shifts, is_noe=False)
        peak_input = torch.cat([peak_shifts_norm, data["Peak"].flags], dim=-1)  # [n, 4]
        data["Peak"].x = self.peak_embed(peak_input)  # [n, embed_dim]

        # NOE: normalized_shifts(3) → MLP → .x[embed_dim]
        noe_shifts_norm = self._normalize_shifts(data["Noe"].shifts, is_noe=True)
        data["Noe"].x = self.noe_embed(noe_shifts_norm)  # [n, embed_dim]

        return data


class FeatureNorm(nn.Module):
    """
    Per-node normalization of .x features using LayerNorm.

    Applied AFTER embeddings, normalizes the working .x features:
    - Residue.x: [embed_dim] → normalized per node
    - Peak.x: [embed_dim] → normalized per node
    - Noe.x: [embed_dim] → normalized per node

    LayerNorm normalizes each node's features independently across the feature
    dimension, eliminating the need for batching/unbatching operations.

    Each node type has its own LayerNorm with learnable affine parameters (γ, β),
    allowing different normalization behavior to be learned for each node type.
    """

    def __init__(self, embed_dim, eps=1e-5):
        super().__init__()
        self.residue_norm = nn.LayerNorm(embed_dim, eps=eps)
        self.peak_norm = nn.LayerNorm(embed_dim, eps=eps)
        self.noe_norm = nn.LayerNorm(embed_dim, eps=eps)

    def forward(self, data):
        """Normalize each node type's features independently."""
        data["Residue"].x = self.residue_norm(data["Residue"].x)
        data["Peak"].x = self.peak_norm(data["Peak"].x)
        data["Noe"].x = self.noe_norm(data["Noe"].x)
        return data


class NMRNet(nn.Module):
    """
    Complete NMR GNN model combining message passing with prediction heads.

    Stacks NMRLayer(s) for graph message passing, then uses ValueCalc and
    PolicyCalc heads to predict state value and action probabilities.

    Applies FeatureNorm after each layer to ensure consistent normalization
    of .x features across all node types.
    """

    def __init__(self, device, config: ModelConfig):
        super().__init__()
        self.config = config
        self.device = device

        # Prediction heads - pass config for dimension calculations
        self.value = ValueCalc(device, config)
        self.policy = PolicyCalc(device)

        self.embed_features = EmbedFeatures(device, config)

        # Build sequential stack: [NMRLayer, FeatureNorm, NMRLayer, FeatureNorm, ...]
        layers = []
        for _ in range(config.num_nmr_layers):
            layers.append(NMRLayer(device, config))
            layers.append(FeatureNorm(config.embed.embed_dim))

        self.nmr = nn.Sequential(*layers)

    def forward(self, data):
        data = self.embed_features(data)
        data = self.nmr(data)
        value = self.value.calc_value(data)
        policy = self.policy.calc_policy(data)
        return value, policy
