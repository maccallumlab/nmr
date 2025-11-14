"""
Top-level NMR GNN network architecture.

Combines triple-based message passing with value and policy heads
to create the complete neural network for NMR assignment.
"""

from dataclasses import dataclass, field

import torch
import torch.nn as nn
from torch_geometric.data import Batch

from .heads import PolicyCalc, ValueCalc
from .pair import AssignedPair
from .triple import (
    PeakPeakNoeTriple,
    PeakResidueNoeTriple,
    ResiduePeakNoeTriple,
    ResidueResidueNoeTriple,
)


@dataclass
class FeatureEmbedConfig:
    """Configuration for embedding 2D assignment features to higher dimensions."""

    output_dim: int = 128
    hidden_dim: int = 128
    num_layers: int = 1


@dataclass
class ShiftEmbedConfig:
    """Configuration for embedding 2D chemical shifts to higher dimensions."""

    output_dim: int = 16
    hidden_dim: int = 128
    num_layers: int = 1


@dataclass
class MLPEmbedConfig:
    """Internal configuration for generic MLP embedding."""

    input_dim: int
    output_dim: int
    hidden_dim: int
    num_layers: int


@dataclass
class MLPConfig:
    """Configuration for MLP layers in message passing."""

    hidden_size: int = 64
    num_layers: int = 1


@dataclass
class ModelConfig:
    """Top-level configuration for NMRNet model."""

    num_nmr_layers: int = 1
    feature_embed: FeatureEmbedConfig = field(default_factory=FeatureEmbedConfig)
    shift_embed: ShiftEmbedConfig = field(default_factory=ShiftEmbedConfig)
    mlp: MLPConfig = field(default_factory=MLPConfig)


class NMRLayer(nn.Module):
    """
    Single layer of NMR-specific message passing.

    Orchestrates message passing through all four triple types and assigned pairs in sequence:
    - ResidueResidueNoeTriple: (Residue, Residue, Noe) - Updates coordinates and shifts
    - ResiduePeakNoeTriple: (Residue, Peak, Noe) - Updates shifts only
    - PeakResidueNoeTriple: (Peak, Residue, Noe) - Updates shifts only
    - PeakPeakNoeTriple: (Peak, Peak, Noe) - Updates shifts only
    - AssignedPair: (Peak, Residue) assigned pairs - Updates shifts only

    Each triple/pair type is instantiated as a separate attribute and called explicitly
    in the forward pass, with no conditionals or branching logic.
    """

    def __init__(self, device, config: ModelConfig):
        super(NMRLayer, self).__init__()
        # Instantiate all 4 triple classes as separate attributes, passing config
        self.residue_residue_noe = ResidueResidueNoeTriple(device, config)
        self.residue_peak_noe = ResiduePeakNoeTriple(device, config)
        self.peak_residue_noe = PeakResidueNoeTriple(device, config)
        self.peak_peak_noe = PeakPeakNoeTriple(device, config)
        # Instantiate assigned pair processing, passing config
        self.assigned_pair = AssignedPair(device, config)

    def forward(self, data):
        """
        Process all four triple types and assigned pairs in sequence.

        Args:
            data: HeteroData graph to process

        Returns:
            Updated HeteroData graph with all node features updated
        """
        # Call each triple class explicitly in sequence
        data = self.residue_residue_noe(data)
        data = self.residue_peak_noe(data)
        data = self.peak_residue_noe(data)
        data = self.peak_peak_noe(data)
        # Process assigned pairs after triples
        data = self.assigned_pair(data)
        return data


class StandardizeShifts(nn.Module):
    """
    Normalize chemical shifts to have a typical range between -1 and 1
    """

    def __init__(self, H_lower=6.0, H_upper=10.0, N_lower=100.0, N_upper=135.0):
        super().__init__()
        self.H_lower = H_lower
        self.H_upper = H_upper
        self.H_delta = H_upper - H_lower
        self.N_lower = N_lower
        self.N_upper = N_upper
        self.N_delta = N_upper - N_lower

    def forward(self, data):
        # split Residue into coords and shifts
        res_x = data["Residue"].x
        res_xyz = res_x[:, :3]
        res_H = res_x[:, 3].unsqueeze(-1)
        res_N = res_x[:, 4].unsqueeze(-1)
        # split Peaks into 1H and 15N
        shift_H = data["Peak"].x[:, 0].unsqueeze(-1)
        shift_N = data["Peak"].x[:, 1].unsqueeze(-1)
        # split NOEs into 1H and 15N
        noe_H1 = data["Noe"].x[:, 0].unsqueeze(-1)
        noe_N1 = data["Noe"].x[:, 1].unsqueeze(-1)
        noe_H2 = data["Noe"].x[:, 2].unsqueeze(-1)
        # transform all 1H
        res_H = self._transform_H(res_H)
        shift_H = self._transform_H(shift_H)
        noe_H1 = self._transform_H(noe_H1)
        noe_H2 = self._transform_H(noe_H2)
        # transofrm all 15N
        res_N = self._transform_N(res_N)
        shift_N = self._transform_N(shift_N)
        noe_N1 = self._transform_N(noe_N1)
        # reassemble tensors
        res_x = torch.cat([res_xyz, res_H, res_N], dim=-1)
        shift_x = torch.cat([shift_H, shift_N], dim=-1)
        noe_x = torch.cat([noe_H1, noe_N1, noe_H2], dim=-1)

        data["Residue"].x = res_x
        data["Peak"].x = shift_x
        data["Noe"].x = noe_x
        return data

    def _transform_H(self, value):
        return 2 * (value - self.H_lower) / self.H_delta - 1

    def _transform_N(self, value):
        return 2 * (value - self.N_lower) / self.N_delta - 1


class MLPEmbedding(nn.Module):
    """
    Generic MLP for embedding features of any dimensionality.

    Builds a configurable MLP: Linear → ReLU → ... → Linear based on
    the provided MLPEmbedConfig. Used internally by semantic wrapper classes.
    """

    def __init__(self, config: MLPEmbedConfig, device):
        super().__init__()
        self.config = config
        self.device = device

        layers = []

        # Input layer: input_dim → hidden_dim
        layers.append(nn.Linear(config.input_dim, config.hidden_dim, device=device))
        layers.append(nn.ReLU())

        # Additional hidden layers
        for _ in range(config.num_layers - 1):
            layers.append(
                nn.Linear(config.hidden_dim, config.hidden_dim, device=device)
            )
            layers.append(nn.ReLU())

        # Output layer: hidden_dim → output_dim
        layers.append(nn.Linear(config.hidden_dim, config.output_dim, device=device))

        self.mlp = nn.Sequential(*layers)

    def forward(self, x):
        """
        Embed input features to higher dimensional space.

        Args:
            x: Tensor of shape [n, input_dim]

        Returns:
            Embedded features of shape [n, output_dim]
        """
        return self.mlp(x)


class FeatureEmbedding(nn.Module):
    """
    Embed 2D assignment features to higher dimensional space.

    Maps [assignment_indicator, already_assigned] → feature_dim using
    a configurable MLP: Linear → ReLU → ... → Linear.
    """

    def __init__(self, config: FeatureEmbedConfig, device):
        super().__init__()
        mlp_config = MLPEmbedConfig(
            input_dim=2,
            output_dim=config.output_dim,
            hidden_dim=config.hidden_dim,
            num_layers=config.num_layers,
        )
        self.mlp = MLPEmbedding(mlp_config, device)

    def forward(self, features):
        """
        Embed 2D features to higher dimensional space.

        Args:
            features: Tensor of shape [n, 2] with assignment indicators

        Returns:
            Embedded features of shape [n, output_dim]
        """
        return self.mlp(features)


class ShiftEmbedding(nn.Module):
    """
    Embed 2D chemical shifts to higher dimensional space.

    Maps [H, N] → shift_dim using a configurable MLP: Linear → ReLU → ... → Linear.
    This embedding is shared between Peak and Residue shift features.
    """

    def __init__(self, config: ShiftEmbedConfig, device):
        super().__init__()
        mlp_config = MLPEmbedConfig(
            input_dim=2,
            output_dim=config.output_dim,
            hidden_dim=config.hidden_dim,
            num_layers=config.num_layers,
        )
        self.mlp = MLPEmbedding(mlp_config, device)

    def forward(self, shifts):
        """
        Embed 2D shifts to higher dimensional space.

        Args:
            shifts: Tensor of shape [n, 2] with [H, N] chemical shifts

        Returns:
            Embedded shifts of shape [n, output_dim]
        """
        return self.mlp(shifts)


class NoeEmbedding(nn.Module):
    """
    Embed 3D NOE features to higher dimensional space.

    Maps [N, H', H"] → shift_dim using a configurable MLP: Linear → ReLU → ... → Linear.
    Uses separate weights from ShiftEmbedding since NOE structure is different (3D vs 2D).
    """

    def __init__(self, config: ShiftEmbedConfig, device):
        super().__init__()
        mlp_config = MLPEmbedConfig(
            input_dim=3,
            output_dim=config.output_dim,
            hidden_dim=config.hidden_dim,
            num_layers=config.num_layers,
        )
        self.mlp = MLPEmbedding(mlp_config, device)

    def forward(self, noe_features):
        """
        Embed 3D NOE features to higher dimensional space.

        Args:
            noe_features: Tensor of shape [n, 3] with [N, H', H"] NOE shifts

        Returns:
            Embedded NOE features of shape [n, output_dim]
        """
        return self.mlp(noe_features)


class EmbedFeatures(nn.Module):
    """
    Apply all feature and shift embeddings to a heterogeneous graph.

    This module transforms all node features from their raw dimensions to learned
    embeddings:
    - Residue.x: [x, y, z, H, N] → [x, y, z, embedded_shifts...]
    - Peak.x: [H, N] → [embedded_shifts...]
    - Noe.x: [N, H', H"] → [embedded_noe...]
    - All .f attributes: [2] → [feature_dim]

    Coordinates remain unchanged (still 3D).
    """

    def __init__(self, device, config: ModelConfig):
        super().__init__()
        self.device = device
        self.config = config

        # Create embedding modules
        self.res_feature_embed = FeatureEmbedding(config.feature_embed, device)
        self.peak_feature_embed = FeatureEmbedding(config.feature_embed, device)
        self.noe_feature_embed = FeatureEmbedding(config.feature_embed, device)
        # use the same embedding for residue and peak shifts
        self.shift_embed = ShiftEmbedding(config.shift_embed, device)
        self.noe_embed = NoeEmbedding(config.shift_embed, device)

    def forward(self, data):
        """
        Embed all features and shifts in the graph.

        Args:
            data: HeteroData graph with standardized shifts

        Returns:
            HeteroData graph with embedded features
        """
        # Embed Residue shifts and concatenate with coordinates
        res_coords = data["Residue"].x[:, 0:3]  # [num_res, 3]
        res_shifts = data["Residue"].x[:, 3:5]  # [num_res, 2]
        res_shifts_embedded = self.shift_embed(res_shifts)  # [num_res, shift_dim]
        data["Residue"].x = torch.cat([res_coords, res_shifts_embedded], dim=-1)

        # Embed Peak shifts
        peak_shifts = data["Peak"].x  # [num_peaks, 2]
        data["Peak"].x = self.shift_embed(peak_shifts)  # [num_peaks, shift_dim]

        # Embed NOE features
        noe_features = data["Noe"].x  # [num_noes, 3]
        data["Noe"].x = self.noe_embed(noe_features)  # [num_noes, shift_dim]

        # Embed all .f attributes (assignment features)
        data["Residue"].f = self.res_feature_embed(data["Residue"].f)
        data["Peak"].f = self.peak_feature_embed(data["Peak"].f)
        data["Noe"].f = self.noe_feature_embed(data["Noe"].f)

        return data


class ChemicalShiftNorm(nn.Module):
    """
    Per-graph normalization with separate normalization for each node type.

    Applied AFTER embeddings, so dimensions are:
    - Residue.x: [coords (3D), embedded_shifts (shift_dim)]
    - Peak.x: [embedded_shifts (shift_dim)]
    - Noe.x: [embedded_noe (shift_dim)]

    For each graph independently:
    - Standardizes Residue XYZ coordinates to zero mean, unit variance
    - Standardizes Residue embedded shifts to zero mean, unit variance
    - Standardizes Peak embedded shifts to zero mean, unit variance
    - Standardizes NOE embedded features to zero mean, unit variance

    Each node type's shifts are normalized independently, allowing the network
    to learn different representations for predicted vs observed shifts.

    Implementation unbatches graphs, normalizes each separately, then re-batches
    to avoid in-place operations that break gradients.
    """

    def __init__(self, eps=1e-5):
        super().__init__()
        self.eps = eps

    def forward(self, data):
        # Check if data is batched
        is_batched = hasattr(data["Residue"], "batch")

        if not is_batched:
            # Single graph - normalize directly
            return self._normalize_single_graph(data)
        else:
            # Batched graphs - unbatch, normalize each, re-batch
            graph_list = data.to_data_list()
            normalized_list = [self._normalize_single_graph(g) for g in graph_list]
            return Batch.from_data_list(normalized_list)

    def _normalize_single_graph(self, data):
        """
        Normalize a single graph.

        Returns a new graph with normalized features (no in-place operations).
        After embeddings, dimensions are:
        - Residue.x: [coords (3D), embedded_shifts (shift_dim)]
        - Peak.x: [embedded_shifts (shift_dim)]
        - Noe.x: [embedded_noe (shift_dim)]
        """
        # Normalize Residue coordinates
        coords = data["Residue"].x[:, 0:3]
        coords_mean = coords.mean(dim=0, keepdim=True)
        coords_std = coords.std(dim=0, keepdim=True) + self.eps
        normalized_coords = (coords - coords_mean) / coords_std

        # Normalize Residue shifts
        residue_shifts = data["Residue"].x[:, 3:]  # [shift_dim]
        res_shift_mean = residue_shifts.mean(dim=0, keepdim=True)
        res_shift_std = residue_shifts.std(dim=0, keepdim=True) + self.eps
        normalized_residue_shifts = (residue_shifts - res_shift_mean) / res_shift_std

        # Normalize Peak shifts
        peak_shifts = data["Peak"].x  # [shift_dim]
        peak_shift_mean = peak_shifts.mean(dim=0, keepdim=True)
        peak_shift_std = peak_shifts.std(dim=0, keepdim=True) + self.eps
        normalized_peak_shifts = (peak_shifts - peak_shift_mean) / peak_shift_std

        # Normalize NOE shifts
        noe_shifts = data["Noe"].x  # [shift_dim]
        noe_shift_mean = noe_shifts.mean(dim=0, keepdim=True)
        noe_shift_std = noe_shifts.std(dim=0, keepdim=True) + self.eps
        normalized_noe_shifts = (noe_shifts - noe_shift_mean) / noe_shift_std

        # Reconstruct tensors
        data["Residue"].x = torch.cat(
            [normalized_coords, normalized_residue_shifts], dim=1
        )
        data["Peak"].x = normalized_peak_shifts
        data["Noe"].x = normalized_noe_shifts

        return data


class NMRNet(nn.Module):
    """
    Complete NMR GNN model combining message passing with prediction heads.

    Stacks NMRLayer(s) for graph message passing, then uses ValueCalc and
    PolicyCalc heads to predict state value and action probabilities.

    Applies ChemicalShiftNorm after each layer to ensure consistent normalization
    of chemical shifts across all node types.
    """

    def __init__(self, device, config: ModelConfig):
        super().__init__()
        self.config = config
        self.device = device

        # Prediction heads - pass config for dimension calculations
        self.value = ValueCalc(device, config)
        self.policy = PolicyCalc(device, config)

        # Standardization and embedding layers
        self.standardize = StandardizeShifts()
        self.embed_features = EmbedFeatures(device, config)

        # Build sequential stack: [NMRLayer, ChemicalShiftNorm, NMRLayer, ChemicalShiftNorm, ...]
        layers = []
        for i in range(config.num_nmr_layers):
            layers.append(NMRLayer(device, config))
            layers.append(ChemicalShiftNorm())

        self.nmr = nn.Sequential(*layers)

    def forward(self, data):
        # Standardize raw shifts to [-1, 1] range
        data = self.standardize(data)
        # Embed shifts and features to higher dimensions
        data = self.embed_features(data)
        # Message passing layers with normalization
        data = self.nmr(data)
        # Prediction heads
        value = self.value.calc_value(data)
        policy = self.policy.calc_policy(data)
        return value, policy
