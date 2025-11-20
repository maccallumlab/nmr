"""
Top-level NMR GNN network architecture.

Combines triple-based message passing with value and policy heads
to create the complete neural network for NMR assignment.
"""

import torch
import torch.nn as nn

from .config import AttentionConfig, ShiftStandardizeConfig, MLPConfig, ModelConfig, SharedConfig
from .heads import PolicyCalc, ValueCalc
from .mlp import MLP
from .pair import AssignedPair
from .triple import (
    PeakPeakNoeTriple,
    PeakResidueNoeTriple,
    ResiduePeakNoeTriple,
    ResidueResidueNoeTriple,
)
from .transformer import BiAxialAttention, MonoAxialAttention


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


class NMRTransformerLayer(nn.Module):
    """
    Single layer of transformer-based NMR message passing.

    Uses attention mechanisms instead of triple nodes to enable information flow between
    Residue, Peak, and NOE nodes. This provides an alternative to the triple-based
    architecture with potentially better scalability.

    Architecture:
    1. AssignedPair: Bidirectional updates between assigned Peak-Residue pairs
    2. Residue ← (Residue, Peak): BiAxial attention updating Residues
    3. Peak ← (Peak, Residue): BiAxial attention updating Peaks
    4. NOE ← (Residue, Peak): BiAxial attention updating NOEs
    5. Residue ← NOE: MonoAxial attention from NOEs to Residues
    6. Peak ← NOE: MonoAxial attention from NOEs to Peaks

    Attention Mechanisms:
    - BiAxialAttention: Combines information from two source types via dual attention streams
    - MonoAxialAttention: Single attention stream (self or cross-attention)
    - SpatialAttentionCore: Used for Residue-to-Residue (distance-aware)
    - AttentionCore: Used for all other combinations (feature-only GATv2)

    Edge Requirements:
    - ("Residue", "res_res_attn", "Residue") - Residue self-attention
    - ("Peak", "peak_res_attn", "Residue") - Peak→Residue cross-attention
    - ("Peak", "peak_peak_attn", "Peak") - Peak self-attention
    - ("Residue", "res_peak_attn", "Peak") - Residue→Peak cross-attention
    - ("Residue", "res_noe_attn", "Noe") - Residue→NOE cross-attention
    - ("Peak", "peak_noe_attn", "Noe") - Peak→NOE cross-attention
    - ("Noe", "noe_res_attn", "Residue") - NOE→Residue cross-attention
    - ("Noe", "noe_peak_attn", "Peak") - NOE→Peak cross-attention

    Configuration:
    - config.shared.embed_dim: Embedding dimension for all node features
    - config.attention.num_heads: Number of attention heads
    - config.attention.attention_dim: Dimension per attention head
    - config.message_mlp: MLP configuration for AssignedPair
    - config.combine_mlp: MLP configuration for attention combination
    """

    def __init__(self, device, config: ModelConfig):
        """
        Initialize NMRTransformerLayer with attention mechanisms.

        Args:
            device: torch device (CPU or CUDA)
            config: ModelConfig containing embed, attention, and mlp configurations
        """
        super(NMRTransformerLayer, self).__init__()

        # Extract parameters from config
        embed_dim = config.shared.embed_dim

        # 1. Assigned pair processing (same as triple-based)
        self.assigned_pair = AssignedPair(device, config)

        # 2. Residue ← (Residue, Peak): BiAxial attention
        self.residue_from_residue_peak = BiAxialAttention(
            source_type_1="Residue",
            source_type_2="Peak",
            dest_type="Residue",
            edge_name_1="res_res_attn",
            edge_name_2="peak_res_attn",
            embed_dim=embed_dim,
            attention_config=config.attention,
            combine_mlp_config=config.combine_mlp,
            device=device,
        )

        # 3. Peak ← (Peak, Residue): BiAxial attention
        self.peak_from_peak_residue = BiAxialAttention(
            source_type_1="Peak",
            source_type_2="Residue",
            dest_type="Peak",
            edge_name_1="peak_peak_attn",
            edge_name_2="res_peak_attn",
            embed_dim=embed_dim,
            attention_config=config.attention,
            combine_mlp_config=config.combine_mlp,
            device=device,
        )

        # 4. NOE ← (Residue, Peak): BiAxial attention
        self.noe_from_residue_peak = BiAxialAttention(
            source_type_1="Residue",
            source_type_2="Peak",
            dest_type="Noe",
            edge_name_1="res_noe_attn",
            edge_name_2="peak_noe_attn",
            embed_dim=embed_dim,
            attention_config=config.attention,
            combine_mlp_config=config.combine_mlp,
            device=device,
        )

        # 5. Residue ← NOE: MonoAxial attention
        self.residue_from_noe = MonoAxialAttention(
            source_type="Noe",
            dest_type="Residue",
            edge_name="noe_res_attn",
            embed_dim=embed_dim,
            attention_config=config.attention,
            device=device,
        )

        # 6. Peak ← NOE: MonoAxial attention
        self.peak_from_noe = MonoAxialAttention(
            source_type="Noe",
            dest_type="Peak",
            edge_name="noe_peak_attn",
            embed_dim=embed_dim,
            attention_config=config.attention,
            device=device,
        )

    def forward(self, data):
        """
        Process attention operations in sequence.

        Execution order matches the design specification:
        1. Assigned pairs (Peak ↔ Residue bidirectional)
        2. Residue updates from Residue and Peak
        3. Peak updates from Peak and Residue
        4. NOE updates from Residue and Peak
        5. Residue updates from NOE
        6. Peak updates from NOE

        Args:
            data: HeteroData graph with node features and edges

        Returns:
            Updated HeteroData graph with all node features updated
        """
        data = self.assigned_pair(data)
        data = self.residue_from_residue_peak(data)
        data = self.peak_from_peak_residue(data)
        data = self.noe_from_residue_peak(data)
        data = self.residue_from_noe(data)
        data = self.peak_from_noe(data)
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
        self.embed_dim = config.shared.embed_dim

        # Normalization parameters for shifts
        self.H_lower = config.shift_standardize.H_lower
        self.H_upper = config.shift_standardize.H_upper
        self.H_delta = self.H_upper - self.H_lower
        self.N_lower = config.shift_standardize.N_lower
        self.N_upper = config.shift_standardize.N_upper
        self.N_delta = self.N_upper - self.N_lower

        # Embedding MLPs: input → hidden → output
        # Residue: [shifts(2) + flags(1)] = 3 → embed_dim
        self.residue_embed = MLP(3, self.embed_dim, config.embed_mlp, self.device)
        # Peak: [shifts(2) + flags(2)] = 4 → embed_dim
        self.peak_embed = MLP(4, self.embed_dim, config.embed_mlp, self.device)
        # NOE: shifts(3) → embed_dim
        self.noe_embed = MLP(3, self.embed_dim, config.embed_mlp, self.device)

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


class NMRNet(nn.Module):
    """
    Complete NMR GNN model combining message passing with prediction heads.

    Supports two architecture types via config.layer_type:
    - "triple": Uses NMRLayer with triple-based message passing (default)
    - "transformer": Uses NMRTransformerLayer with attention mechanisms

    Stacks the selected layer type for graph message passing, then uses ValueCalc and
    PolicyCalc heads to predict state value and action probabilities.

    Pre-normalization is handled within each message passing component,
    ensuring gradients flow through clean residual paths.
    """

    def __init__(self, device, config: ModelConfig):
        super().__init__()
        self.config = config
        self.device = device

        # Extract shared parameters
        embed_dim = config.shared.embed_dim

        # Prediction heads - pass explicit parameters
        self.value = ValueCalc(embed_dim=embed_dim, value_mlp_config=config.value_mlp, device=device)
        self.policy = PolicyCalc(embed_dim=embed_dim, device=device)

        self.embed_features = EmbedFeatures(device, config)

        # Build sequential stack of message passing layers based on config.layer_type
        layers = []
        for _ in range(config.num_nmr_layers):
            if config.layer_type == "triple":
                layers.append(NMRLayer(device, config))
            elif config.layer_type == "transformer":
                layers.append(NMRTransformerLayer(device, config))
            else:
                raise ValueError(
                    f"Unknown layer_type: {config.layer_type}. "
                    f"Expected 'triple' or 'transformer'."
                )

        self.nmr = nn.Sequential(*layers)

    def forward(self, data):
        data = self.embed_features(data)
        data = self.nmr(data)
        value = self.value.calc_value(data)
        policy = self.policy.calc_policy(data)
        return value, policy
