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
  Updates features in both directions

NEW ATTRIBUTE STRUCTURE (after refactoring):
- Uses .x for all feature operations (embedded shifts + flags)
- NO shift difference calculations (removed translation invariance)
- Network learns from absolute shift embeddings directly

Node Type Naming:
- Residue: Protein residues with coordinates .xyz and shifts .shifts [H,N]
- Peak: Observed chemical shifts .shifts [H,N]

Edge Naming:
- Bidirectional edges: ("Peak", "assigned_to", "Residue")
  * Used for Peak → Residue message passing with flow='source_to_target'
  * Used for Residue → Peak message passing with flow='target_to_source'
"""

import torch
import torch.nn as nn
from torch_geometric.nn import MessagePassing

from .mlp import MLP


# ============================================================================
# SECTION 1: Message Passing Operations
# ============================================================================


class AssignedPeakToResidueMessage(MessagePassing):
    """
    Message passing from assigned Peak to Residue nodes.

    Uses edge type: ("Peak", "assigned_to", "Residue") with flow='source_to_target'

    Message computation:
        1. Extract peak and residue embedded features from .x
        2. Concatenate absolute features (NO shift differences)
        3. Pass through MLP to get residue feature deltas
        4. Apply deltas to residue features (aggregated)

    IMPORTANT CHANGES (Task 1.7 - Remove Translation Invariance):
    - Uses ABSOLUTE embedded shift features (not shift differences)
    - Network can learn any necessary invariances from raw feature embeddings
    - Only updates .x features (feature portion only)
    """

    def __init__(self, device, config):
        """
        Initialize AssignedPeakToResidueMessage.

        Args:
            device: torch device (CPU or CUDA)
            config: ModelConfig with dimension settings
        """
        super().__init__(aggr="mean", flow="source_to_target")
        self.device = device
        self.config = config
        self.edge_type = ("Peak", "assigned_to", "Residue")

        # Calculate input/output sizes from config
        # NOTE: With unified architecture, .x is [embed_dim], not split into shift+feature
        embed_dim = config.shared.embed_dim

        # Input: peak features (embed_dim) + residue features (embed_dim)
        # NO SHIFT DIFFERENCES - using absolute embedded features
        input_size = 2 * embed_dim

        # Output: updates to unified embedding (embed_dim) for residue
        output_size = embed_dim

        # Store dimensions (for backward compatibility)
        self.shift_dim = embed_dim
        self.feature_dim = embed_dim

        # Pre-normalization layers for inputs (pre-norm pattern)
        self.norm_peak = nn.LayerNorm(embed_dim).to(device)
        self.norm_residue = nn.LayerNorm(embed_dim).to(device)

        # Build MLP (no internal LayerNorm - follows pre-norm pattern)
        self.mlp = MLP(input_size, output_size, config.message_mlp, device)

    def forward(self, data):
        """
        Execute message passing from Peak to Residue nodes.

        Args:
            data: HeteroData graph with Peak and Residue nodes

        Returns:
            Updated HeteroData with Residue.x modified (feature portion only)
        """
        # Get edge indices
        edge_index = data[self.edge_type].edge_index

        # Extract embedded features [n, shift_dim + feature_dim] from .x
        # After EmbedFeatures: Peak.x = [embedded_shifts (shift_dim), embedded_flags (feature_dim)]
        # After EmbedFeatures: Residue.x = [embedded_shifts (shift_dim), embedded_flags (feature_dim)]
        peak_features = data["Peak"].x
        residue_features = data["Residue"].x

        # Determine sizes for message passing
        num_residues = data["Residue"].x.size(0)
        num_peaks = data["Peak"].x.size(0)

        # Propagate updates to residues
        feature_updates = self.propagate(
            edge_index,
            peak_features=peak_features,
            residue_features=residue_features,
            size=(num_peaks, num_residues)
        )

        # Update the entire unified .x embedding
        # NOTE: With unified architecture, .x contains a single embedding (not split)
        # We update the full embedding with feature_updates
        data["Residue"].x = data["Residue"].x + feature_updates

        return data

    def message(self, peak_features_j, residue_features_i):
        """
        Compute messages from peak (source) to residue (target).

        Uses ABSOLUTE embedded features (no shift differences) with pre-normalization.

        Args:
            peak_features_j: Peak features [n_edges, shift_dim + feature_dim] (source)
            residue_features_i: Residue features [n_edges, shift_dim + feature_dim] (target)

        Returns:
            Feature deltas [n_edges, feature_dim] for residues
        """
        # Apply pre-normalization to inputs
        peak_features_norm = self.norm_peak(peak_features_j)
        residue_features_norm = self.norm_residue(residue_features_i)

        # Concatenate normalized ABSOLUTE features for MLP input
        mlp_input = torch.cat([
            peak_features_norm,      # [n_edges, shift_dim + feature_dim]
            residue_features_norm,   # [n_edges, shift_dim + feature_dim]
        ], dim=-1)

        # Apply MLP to get feature deltas
        feature_deltas = self.mlp(mlp_input)  # [n_edges, feature_dim]

        return feature_deltas

    def update(self, aggr_out):
        """
        Return aggregated feature updates.

        Args:
            aggr_out: Aggregated tensor [n_nodes, feature_dim]

        Returns:
            Feature updates [n_nodes, feature_dim]
        """
        return aggr_out


class AssignedResidueToPeakMessage(MessagePassing):
    """
    Message passing from Residue to assigned Peak nodes.

    Uses edge type: ("Peak", "assigned_to", "Residue") with flow='target_to_source'

    Message computation:
        1. Extract residue and peak embedded features from .x
        2. Concatenate absolute features (NO shift differences)
        3. Pass through MLP to get peak feature deltas
        4. Apply deltas to peak features (aggregated)

    IMPORTANT CHANGES (Task 1.7 - Remove Translation Invariance):
    - Uses ABSOLUTE embedded shift features (not shift differences)
    - Network can learn any necessary invariances from raw feature embeddings
    - Only updates .x features (feature portion only)
    """

    def __init__(self, device, config):
        """
        Initialize AssignedResidueToPeakMessage.

        Args:
            device: torch device (CPU or CUDA)
            config: ModelConfig with dimension settings
        """
        super().__init__(aggr="mean", flow="target_to_source")
        self.device = device
        self.config = config
        self.edge_type = ("Peak", "assigned_to", "Residue")

        # Calculate input/output sizes from config
        # NOTE: With unified architecture, .x is [embed_dim], not split into shift+feature
        embed_dim = config.shared.embed_dim

        # Input: residue features (embed_dim) + peak features (embed_dim)
        # NO SHIFT DIFFERENCES - using absolute embedded features
        input_size = 2 * embed_dim

        # Output: updates to unified embedding (embed_dim) for peak
        output_size = embed_dim

        # Store dimensions (for backward compatibility)
        self.shift_dim = embed_dim
        self.feature_dim = embed_dim

        # Pre-normalization layers for inputs (pre-norm pattern)
        self.norm_residue = nn.LayerNorm(embed_dim).to(device)
        self.norm_peak = nn.LayerNorm(embed_dim).to(device)

        # Build MLP (no internal LayerNorm - follows pre-norm pattern)
        self.mlp = MLP(input_size, output_size, config.message_mlp, device)

    def forward(self, data):
        """
        Execute message passing from Residue to Peak nodes.

        Args:
            data: HeteroData graph with Peak and Residue nodes

        Returns:
            Updated HeteroData with Peak.x modified (feature portion only)
        """
        # Get edge indices
        edge_index = data[self.edge_type].edge_index

        # Extract embedded features [n, shift_dim + feature_dim] from .x
        peak_features = data["Peak"].x
        residue_features = data["Residue"].x

        # Determine sizes for message passing
        num_residues = data["Residue"].x.size(0)
        num_peaks = data["Peak"].x.size(0)

        # Propagate updates to peaks (with reversed flow, size is (target, source))
        feature_updates = self.propagate(
            edge_index,
            residue_features=residue_features,
            peak_features=peak_features,
            size=(num_peaks, num_residues)
        )

        # Update the entire unified .x embedding
        # NOTE: With unified architecture, .x contains a single embedding (not split)
        # We update the full embedding with feature_updates
        data["Peak"].x = data["Peak"].x + feature_updates

        return data

    def message(self, residue_features_j, peak_features_i):
        """
        Compute messages from residue (source in reversed flow) to peak (target).

        Uses ABSOLUTE embedded features (no shift differences) with pre-normalization.

        Args:
            residue_features_j: Residue features [n_edges, shift_dim + feature_dim] (source)
            peak_features_i: Peak features [n_edges, shift_dim + feature_dim] (target)

        Returns:
            Feature deltas [n_edges, feature_dim] for peaks
        """
        # Apply pre-normalization to inputs
        residue_features_norm = self.norm_residue(residue_features_j)
        peak_features_norm = self.norm_peak(peak_features_i)

        # Concatenate normalized ABSOLUTE features for MLP input
        mlp_input = torch.cat([
            residue_features_norm,   # [n_edges, shift_dim + feature_dim]
            peak_features_norm,      # [n_edges, shift_dim + feature_dim]
        ], dim=-1)

        # Apply MLP to get feature deltas
        feature_deltas = self.mlp(mlp_input)  # [n_edges, feature_dim]

        return feature_deltas

    def update(self, aggr_out):
        """
        Return aggregated feature updates.

        Args:
            aggr_out: Aggregated tensor [n_nodes, feature_dim]

        Returns:
            Feature updates [n_nodes, feature_dim]
        """
        return aggr_out


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

    def __init__(self, device, config):
        """
        Initialize AssignedPair.

        Args:
            device: torch device (CPU or CUDA)
            config: ModelConfig with dimension settings
        """
        super().__init__()

        # Instantiate both message passing operations
        self.peak_to_residue = AssignedPeakToResidueMessage(device, config)
        self.residue_to_peak = AssignedResidueToPeakMessage(device, config)

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
