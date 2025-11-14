"""
Pair-based graph message passing components for assigned peak-residue pairs.

This module implements direct message passing between Peak and Residue nodes
that have been assigned to each other, without intermediate triple nodes.

Architecture:
1. Message Operations (2 classes): Process edges directly with MLPs
2. Pair Composition (1 class): Wire both message passing directions together

This is simpler than triple processing since pairs connect exactly 2 nodes
via edges, while triples simulate hypergraphs with 3 nodes.

Pair Type:
- AssignedPair: (Peak, Residue) pairs connected by "assigned_to" edges
  Updates shifts and features in both directions

Node Type Naming:
- Residue: Protein residues with coordinates [x,y,z] and shifts [H,N]
- Peak: Observed chemical shifts [H,N]

Edge Naming:
- Bidirectional edges: ("Peak", "assigned_to", "Residue")
  * Used for Peak → Residue message passing with flow='source_to_target'
  * Used for Residue → Peak message passing with flow='target_to_source'
"""

import torch
import torch.nn as nn
from torch_geometric.nn import MessagePassing


# ============================================================================
# MLP Architecture Constants
# ============================================================================

# AssignedPeakToResidueMessage MLP
PEAK_TO_RESIDUE_INPUT_SIZE = 6  # shift_diffs(2) + peak_features(2) + residue_features(2)
PEAK_TO_RESIDUE_OUTPUT_SIZE = 3  # shift_weight(1) + feature_deltas(2) for residue

# AssignedResidueToPeakMessage MLP
RESIDUE_TO_PEAK_INPUT_SIZE = 6  # shift_diffs(2) + residue_features(2) + peak_features(2)
RESIDUE_TO_PEAK_OUTPUT_SIZE = 3  # shift_weight(1) + feature_deltas(2) for peak


# ============================================================================
# SECTION 1: Message Passing Operations
# ============================================================================


class AssignedPeakToResidueMessage(MessagePassing):
    """
    Message passing from assigned Peak to Residue nodes.

    Uses edge type: ("Peak", "assigned_to", "Residue") with flow='source_to_target'

    Message computation:
        1. Extract peak shifts, residue shifts, and features
        2. Compute shift differences (peak - residue)
        3. Concatenate shift diffs and features
        4. Pass through MLP to get residue deltas
        5. Apply deltas to residue shifts and features (aggregated)

    Equivariance:
        All calculations use shift differences (never absolute shifts)
    """

    def __init__(self, device, config, hidden_size: int = None, num_layers: int = None):
        """
        Initialize AssignedPeakToResidueMessage.

        Args:
            device: torch device (CPU or CUDA)
            config: ModelConfig with dimension settings
            hidden_size: Number of hidden units in MLP (default: from config)
            num_layers: Number of hidden layers (default: from config)
        """
        super().__init__(aggr="mean", flow="source_to_target")
        self.device = device
        self.config = config
        self.edge_type = ("Peak", "assigned_to", "Residue")

        # Use config defaults if not specified
        if hidden_size is None:
            hidden_size = config.mlp.hidden_size
        if num_layers is None:
            num_layers = config.mlp.num_layers

        # Calculate input/output sizes from config
        shift_dim = config.shift_embed.output_dim
        feature_dim = config.feature_embed.output_dim
        # Input: shift_diffs (shift_dim) + peak_features (feature_dim) + residue_features (feature_dim)
        input_size = shift_dim + 2 * feature_dim
        # Output: shift_weight (shift_dim) + feature_deltas (feature_dim) for residue
        output_size = shift_dim + feature_dim

        # Store dimensions
        self.shift_dim = shift_dim
        self.feature_dim = feature_dim

        # Build MLP
        layers = []

        # First hidden layer
        layers.append(nn.Linear(input_size, hidden_size))
        layers.append(nn.LayerNorm(hidden_size))
        layers.append(nn.ReLU())

        # Additional hidden layers
        for _ in range(num_layers - 1):
            layers.append(nn.Linear(hidden_size, hidden_size))
            layers.append(nn.LayerNorm(hidden_size))
            layers.append(nn.ReLU())

        # Output layer
        layers.append(nn.Linear(hidden_size, output_size))

        self.mlp = nn.Sequential(*layers).to(device)

    def forward(self, data):
        """
        Execute message passing from Peak to Residue nodes.

        Args:
            data: HeteroData graph with Peak and Residue nodes

        Returns:
            Updated HeteroData with Residue.x and Residue.f modified
        """
        # Get edge indices
        edge_index = data[self.edge_type].edge_index

        # Extract peak shifts [n_peaks, shift_dim] and features [n_peaks, feature_dim]
        peak_shifts = data["Peak"].x
        peak_features = data["Peak"].f

        # Extract residue shifts [n_residues, shift_dim] and features [n_residues, feature_dim]
        residue_shifts = data["Residue"].x[:, 3:]  # Shifts start at index 3
        residue_features = data["Residue"].f

        # Determine sizes for message passing
        num_residues = data["Residue"].x.size(0)
        num_peaks = data["Peak"].x.size(0)

        # Propagate updates to residues
        shift_updates, feature_updates = self.propagate(
            edge_index,
            peak_shifts=peak_shifts,
            peak_features=peak_features,
            residue_shifts=residue_shifts,
            residue_features=residue_features,
            size=(num_peaks, num_residues)
        )

        # Apply updates to residue nodes (assemble new tensors)
        old_x = data["Residue"].x
        new_coords = old_x[:, 0:3]  # Keep coordinates unchanged
        new_shifts = old_x[:, 3:] + shift_updates
        data["Residue"].x = torch.cat([new_coords, new_shifts], dim=-1)
        data["Residue"].f = data["Residue"].f + feature_updates

        return data

    def message(self, peak_shifts_j, peak_features_j, residue_shifts_i, residue_features_i):
        """
        Compute messages from peak (source) to residue (target).

        Args:
            peak_shifts_j: Peak shifts [n_edges, 2] (source)
            peak_features_j: Peak features [n_edges, 2] (source)
            residue_shifts_i: Residue shifts [n_edges, 2] (target)
            residue_features_i: Residue features [n_edges, 2] (target)

        Returns:
            Tuple of (shift_deltas, feature_deltas) for residues
        """
        # Compute shift difference vector (peak - residue) for equivariance
        # shift_diff is a 2D vector [H, N]
        shift_diff = peak_shifts_j - residue_shifts_i  # [n_edges, 2]

        # Concatenate features for MLP input (6 total)
        mlp_input = torch.cat([
            shift_diff,          # [n_edges, 2]
            peak_features_j,     # [n_edges, 2]
            residue_features_i,  # [n_edges, 2]
        ], dim=-1)  # [n_edges, 6]

        # Apply MLP to get scalar weight and feature deltas
        mlp_output = self.mlp(mlp_input)  # [n_edges, 3]

        # Parse output: shift_weight (shift_dim) + feature_deltas (feature_dim)
        shift_weight = mlp_output[:, :self.shift_dim]  # [n_edges, shift_dim]
        feature_deltas = mlp_output[:, self.shift_dim:]  # [n_edges, feature_dim]

        # Compute shift deltas by element-wise multiplication with shift difference
        # This ensures movement is always in the direction of the difference
        shift_deltas = shift_diff * shift_weight  # [n_edges, shift_dim]

        # Concatenate shift and feature deltas for aggregation
        # PyG will automatically apply mean aggregation (aggr='mean')
        return torch.cat([shift_deltas, feature_deltas], dim=-1)  # [n_edges, shift_dim + feature_dim]

    def update(self, aggr_out):
        """
        Split aggregated updates back into shift and feature components.

        Args:
            aggr_out: Aggregated tensor [n_nodes, shift_dim + feature_dim]

        Returns:
            Tuple of (shift_updates, feature_updates)
        """
        shift_updates = aggr_out[:, :self.shift_dim]  # [n_nodes, shift_dim]
        feature_updates = aggr_out[:, self.shift_dim:]  # [n_nodes, feature_dim]
        return shift_updates, feature_updates


class AssignedResidueToPeakMessage(MessagePassing):
    """
    Message passing from Residue to assigned Peak nodes.

    Uses edge type: ("Peak", "assigned_to", "Residue") with flow='target_to_source'

    Message computation:
        1. Extract residue shifts, peak shifts, and features
        2. Compute shift differences (residue - peak)
        3. Concatenate shift diffs and features
        4. Pass through MLP to get peak deltas
        5. Apply deltas to peak shifts and features (aggregated)

    Equivariance:
        All calculations use shift differences (never absolute shifts)
    """

    def __init__(self, device, config, hidden_size: int = None, num_layers: int = None):
        """
        Initialize AssignedResidueToPeakMessage.

        Args:
            device: torch device (CPU or CUDA)
            config: ModelConfig with dimension settings
            hidden_size: Number of hidden units in MLP (default: from config)
            num_layers: Number of hidden layers (default: from config)
        """
        super().__init__(aggr="mean", flow="target_to_source")
        self.device = device
        self.config = config
        self.edge_type = ("Peak", "assigned_to", "Residue")

        # Use config defaults if not specified
        if hidden_size is None:
            hidden_size = config.mlp.hidden_size
        if num_layers is None:
            num_layers = config.mlp.num_layers

        # Calculate input/output sizes from config
        shift_dim = config.shift_embed.output_dim
        feature_dim = config.feature_embed.output_dim
        # Input: shift_diffs (shift_dim) + residue_features (feature_dim) + peak_features (feature_dim)
        input_size = shift_dim + 2 * feature_dim
        # Output: shift_weight (shift_dim) + feature_deltas (feature_dim) for peak
        output_size = shift_dim + feature_dim

        # Store dimensions
        self.shift_dim = shift_dim
        self.feature_dim = feature_dim

        # Build MLP
        layers = []

        # First hidden layer
        layers.append(nn.Linear(input_size, hidden_size))
        layers.append(nn.LayerNorm(hidden_size))
        layers.append(nn.ReLU())

        # Additional hidden layers
        for _ in range(num_layers - 1):
            layers.append(nn.Linear(hidden_size, hidden_size))
            layers.append(nn.LayerNorm(hidden_size))
            layers.append(nn.ReLU())

        # Output layer
        layers.append(nn.Linear(hidden_size, output_size))

        self.mlp = nn.Sequential(*layers).to(device)

    def forward(self, data):
        """
        Execute message passing from Residue to Peak nodes.

        Args:
            data: HeteroData graph with Peak and Residue nodes

        Returns:
            Updated HeteroData with Peak.x and Peak.f modified
        """
        # Get edge indices
        edge_index = data[self.edge_type].edge_index

        # Extract peak shifts [n_peaks, shift_dim] and features [n_peaks, feature_dim]
        peak_shifts = data["Peak"].x
        peak_features = data["Peak"].f

        # Extract residue shifts [n_residues, shift_dim] and features [n_residues, feature_dim]
        residue_shifts = data["Residue"].x[:, 3:]  # Shifts start at index 3
        residue_features = data["Residue"].f

        # Determine sizes for message passing
        num_residues = data["Residue"].x.size(0)
        num_peaks = data["Peak"].x.size(0)

        # Propagate updates to peaks (with reversed flow, size is (target, source))
        shift_updates, feature_updates = self.propagate(
            edge_index,
            residue_shifts=residue_shifts,
            residue_features=residue_features,
            peak_shifts=peak_shifts,
            peak_features=peak_features,
            size=(num_peaks, num_residues)
        )

        # Apply updates to peak nodes (assemble new tensors)
        data["Peak"].x = data["Peak"].x + shift_updates
        data["Peak"].f = data["Peak"].f + feature_updates

        return data

    def message(self, residue_shifts_j, residue_features_j, peak_shifts_i, peak_features_i):
        """
        Compute messages from residue (source in reversed flow) to peak (target).

        Args:
            residue_shifts_j: Residue shifts [n_edges, shift_dim] (source)
            residue_features_j: Residue features [n_edges, feature_dim] (source)
            peak_shifts_i: Peak shifts [n_edges, shift_dim] (target)
            peak_features_i: Peak features [n_edges, feature_dim] (target)

        Returns:
            Tuple of (shift_deltas, feature_deltas) for peaks
        """
        # Compute shift difference vector (residue - peak) for equivariance
        shift_diff = residue_shifts_j - peak_shifts_i  # [n_edges, shift_dim]

        # Concatenate features for MLP input
        mlp_input = torch.cat([
            shift_diff,            # [n_edges, shift_dim]
            residue_features_j,    # [n_edges, feature_dim]
            peak_features_i,       # [n_edges, feature_dim]
        ], dim=-1)

        # Apply MLP to get weight vector and feature deltas
        mlp_output = self.mlp(mlp_input)

        # Parse output: shift_weight (shift_dim) + feature_deltas (feature_dim)
        shift_weight = mlp_output[:, :self.shift_dim]  # [n_edges, shift_dim]
        feature_deltas = mlp_output[:, self.shift_dim:]  # [n_edges, feature_dim]

        # Compute shift deltas by element-wise multiplication with shift difference
        # This ensures movement is always in the direction of the difference
        shift_deltas = shift_diff * shift_weight  # [n_edges, shift_dim]

        # Concatenate shift and feature deltas for aggregation
        # PyG will automatically apply mean aggregation (aggr='mean')
        return torch.cat([shift_deltas, feature_deltas], dim=-1)  # [n_edges, shift_dim + feature_dim]

    def update(self, aggr_out):
        """
        Split aggregated updates back into shift and feature components.

        Args:
            aggr_out: Aggregated tensor [n_nodes, shift_dim + feature_dim]

        Returns:
            Tuple of (shift_updates, feature_updates)
        """
        shift_updates = aggr_out[:, :self.shift_dim]  # [n_nodes, shift_dim]
        feature_updates = aggr_out[:, self.shift_dim:]  # [n_nodes, feature_dim]
        return shift_updates, feature_updates


# ============================================================================
# SECTION 2: Pair Composition Class
# ============================================================================


class AssignedPair(nn.Module):
    """
    Wire together bidirectional message passing for assigned peak-residue pairs.

    This composition class processes assigned (Peak, Residue) pairs by running
    message passing in both directions: Peak → Residue and Residue → Peak.

    Components:
        - AssignedPeakToResidueMessage: Peak → Residue updates
        - AssignedResidueToPeakMessage: Residue → Peak updates

    Forward pass sequence:
        peak_to_residue → residue_to_peak
    """

    def __init__(self, device, config, hidden_size: int = None, num_layers: int = None):
        """
        Initialize AssignedPair.

        Args:
            device: torch device (CPU or CUDA)
            config: ModelConfig with dimension settings
            hidden_size: Hidden units in message MLPs (default: from config)
            num_layers: Hidden layers in message MLPs (default: from config)
        """
        super().__init__()

        # Instantiate both message passing operations - pass config
        self.peak_to_residue = AssignedPeakToResidueMessage(device, config, hidden_size, num_layers)
        self.residue_to_peak = AssignedResidueToPeakMessage(device, config, hidden_size, num_layers)

    def forward(self, data):
        """
        Execute bidirectional message passing.

        Args:
            data: HeteroData graph

        Returns:
            Updated HeteroData graph
        """
        data = self.peak_to_residue(data)
        data = self.residue_to_peak(data)
        return data
