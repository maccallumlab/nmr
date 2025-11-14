"""
Modular triple-based graph message passing components.

This module implements a three-stage message passing pipeline using explicit,
modular components instead of conditional logic:

Architecture:
1. Gather Operations (5 classes): Extract features from source nodes to triple nodes
2. Update Operations (2 classes): Compute deltas via MLPs on triple nodes
3. Scatter Operations (5 classes): Propagate deltas back to source nodes
4. Triple Composition (4 classes): Wire gather/update/scatter for each triple type
5. Layer Orchestration (in network.py): Call all 4 triple types explicitly

This architecture eliminates runtime conditionals by making all behavioral choices
explicit at construction time.

Triple Types:
- ResidueResidueNoeTriple: (Residue, Residue, Noe) - Updates coordinates + shifts
- ResiduePeakNoeTriple: (Residue, Peak, Noe) - Updates shifts only
- PeakResidueNoeTriple: (Peak, Residue, Noe) - Updates shifts only
- PeakPeakNoeTriple: (Peak, Peak, Noe) - Updates shifts only

Node Type Naming:
- Residue: Protein residues with coordinates [x,y,z] and shifts [H,N]
- Peak: Observed chemical shifts [H,N]
- Noe: NOE constraints [N, H', H"]

Edge Naming Conventions:
- Bidirectional propagation edges: (source_node, "prop_first"/"prop_second"/"prop_noe", triple_type)
  * Used for gather with default flow='source_to_target'
  * Used for scatter with flow='target_to_source'
"""

import torch
import torch.nn as nn
from torch_geometric.nn import MessagePassing


# ============================================================================
# MLP Architecture Dimension Functions
# ============================================================================
# These functions compute input/output sizes for MLP architectures based on
# configurable shift_dim and feature_dim parameters


def residue_update_input_size(shift_dim, feature_dim):
    """
    Calculate input size for ResidueUpdate MLP.

    Components:
    - dist_squared: 1
    - shift differences: 3*shift_dim pairwise differences
      (diff_first_to_second, diff_first_to_noe, diff_second_to_noe)
    - features: 3*feature_dim (first_features, second_features, noe_features)

    Returns: 1 + 3*shift_dim + 3*feature_dim
    """
    return 1 + 3 * shift_dim + 3 * feature_dim


def residue_update_output_size(shift_dim, feature_dim):
    """
    Calculate output size for ResidueUpdate MLP.

    Components:
    - coord_weights: 2 (w_coord_first, w_coord_second)
    - shift_weights: 6 (w1-w6, scalar weights for each shift difference)
    - feature_deltas: 3*feature_dim (delta_first_features, delta_second_features, delta_noe_features)

    Returns: coord_weights(2) + shift_weights(6) + feature_deltas(3*feature_dim)
    """
    return 2 + 6 + 3 * feature_dim


def peak_update_input_size(shift_dim, feature_dim):
    """
    Calculate input size for PeakUpdate MLP.

    Components:
    - shift differences: 3*shift_dim pairwise differences
      (diff_first_to_second, diff_first_to_noe, diff_second_to_noe)
    - features: 3*feature_dim (first_features, second_features, noe_features)

    Returns: 3*shift_dim + 3*feature_dim
    Note: NO dist_squared (peaks have no coordinates).
    """
    return 3 * shift_dim + 3 * feature_dim


def peak_update_output_size(shift_dim, feature_dim):
    """
    Calculate output size for PeakUpdate MLP.

    Components:
    - shift_weights: 6 (w1-w6, scalar weights for each shift difference)
    - feature_deltas: 3*feature_dim (delta_first_features, delta_second_features, delta_noe_features)

    Returns: shift_weights(6) + feature_deltas(3*feature_dim)
    Note: NO coord_weights (peaks have no coordinates).
    """
    return 6 + 3 * feature_dim


# ============================================================================
# SECTION 1: Gather Operations
# ============================================================================
# These classes extract features from source nodes to triple nodes using
# PyTorch Geometric MessagePassing with aggr="mean"


class FirstResidueGather(MessagePassing):
    """
    Extract coordinates, shifts, and features from residues in first position.

    Uses edge type: ("Residue", "prop_first", triple_type)
    Sets attributes on triple nodes:
        - first_coords: [n, 3] coordinates
        - first_shifts: [n, shift_dim] chemical shift embeddings
        - first_features: [n, feature_dim] assignment feature embeddings
    """

    def __init__(self, triple_type: str):
        """
        Initialize FirstResidueGather.

        Args:
            triple_type: Name of target triple node type
        """
        super().__init__(aggr="mean")
        self.triple_type = triple_type
        self.edge_type = ("Residue", "prop_first", triple_type)

    def forward(self, data):
        """
        Extract features from Residue nodes to triple nodes.

        Args:
            data: HeteroData graph with Residue nodes and gather edges

        Returns:
            Updated HeteroData with first_coords, first_shifts, first_features set
        """
        # Extract coordinates [n, 3] from Residue.x[:, 0:3]
        coords = data["Residue"].x[:, 0:3]

        # Extract shifts [n, shift_dim] from Residue.x[:, 3:]
        shifts = data["Residue"].x[:, 3:]

        # Extract features [n, feature_dim] from Residue.f
        features = data["Residue"].f

        # Get edge indices for this gather operation
        edge_index = data[self.edge_type].edge_index

        # Determine number of target nodes (triple nodes)
        num_triples = data[self.triple_type].x.size(0)

        # Propagate coordinates with explicit size
        data[self.triple_type].first_coords = self.propagate(
            edge_index, x=coords, size=(coords.size(0), num_triples)
        )

        # Propagate shifts
        data[self.triple_type].first_shifts = self.propagate(
            edge_index, x=shifts, size=(shifts.size(0), num_triples)
        )

        # Propagate features
        data[self.triple_type].first_features = self.propagate(
            edge_index, x=features, size=(features.size(0), num_triples)
        )

        return data

    def message(self, x_j):
        """Pass through features from source nodes."""
        return x_j


class FirstPeakGather(MessagePassing):
    """
    Extract shifts and features from peaks in first position.

    Uses edge type: ("Peak", "prop_first", triple_type)
    Sets attributes on triple nodes:
        - first_shifts: [n, shift_dim] chemical shift embeddings
        - first_features: [n, feature_dim] assignment feature embeddings
    Note: Peaks have NO coordinates
    """

    def __init__(self, triple_type: str):
        """
        Initialize FirstPeakGather.

        Args:
            triple_type: Name of target triple node type
        """
        super().__init__(aggr="mean")
        self.triple_type = triple_type
        self.edge_type = ("Peak", "prop_first", triple_type)

    def forward(self, data):
        """
        Extract features from Peak nodes to triple nodes.

        Args:
            data: HeteroData graph with Peak nodes and gather edges

        Returns:
            Updated HeteroData with first_shifts, first_features set
        """
        # Extract shifts [n, shift_dim] from Peak.x
        shifts = data["Peak"].x

        # Extract features [n, feature_dim] from Peak.f
        features = data["Peak"].f

        # Get edge indices for this gather operation
        edge_index = data[self.edge_type].edge_index

        # Determine number of target nodes (triple nodes)
        num_triples = data[self.triple_type].x.size(0)

        # Propagate shifts with explicit size
        data[self.triple_type].first_shifts = self.propagate(
            edge_index, x=shifts, size=(shifts.size(0), num_triples)
        )

        # Propagate features
        data[self.triple_type].first_features = self.propagate(
            edge_index, x=features, size=(features.size(0), num_triples)
        )

        return data

    def message(self, x_j):
        """Pass through features from source nodes."""
        return x_j


class SecondResidueGather(MessagePassing):
    """
    Extract coordinates, shifts, and features from residues in second position.

    Uses edge type: ("Residue", "prop_second", triple_type)
    Sets attributes on triple nodes:
        - second_coords: [n, 3] coordinates
        - second_shifts: [n, shift_dim] chemical shift embeddings
        - second_features: [n, feature_dim] assignment feature embeddings
    """

    def __init__(self, triple_type: str):
        """
        Initialize SecondResidueGather.

        Args:
            triple_type: Name of target triple node type
        """
        super().__init__(aggr="mean")
        self.triple_type = triple_type
        self.edge_type = ("Residue", "prop_second", triple_type)

    def forward(self, data):
        """
        Extract features from Residue nodes to triple nodes.

        Args:
            data: HeteroData graph with Residue nodes and gather edges

        Returns:
            Updated HeteroData with second_coords, second_shifts, second_features set
        """
        # Extract coordinates [n, 3] from Residue.x[:, 0:3]
        coords = data["Residue"].x[:, 0:3]

        # Extract shifts [n, shift_dim] from Residue.x[:, 3:]
        shifts = data["Residue"].x[:, 3:]

        # Extract features [n, feature_dim] from Residue.f
        features = data["Residue"].f

        # Get edge indices for this gather operation
        edge_index = data[self.edge_type].edge_index

        # Determine number of target nodes (triple nodes)
        num_triples = data[self.triple_type].x.size(0)

        # Propagate coordinates with explicit size
        data[self.triple_type].second_coords = self.propagate(
            edge_index, x=coords, size=(coords.size(0), num_triples)
        )

        # Propagate shifts
        data[self.triple_type].second_shifts = self.propagate(
            edge_index, x=shifts, size=(shifts.size(0), num_triples)
        )

        # Propagate features
        data[self.triple_type].second_features = self.propagate(
            edge_index, x=features, size=(features.size(0), num_triples)
        )

        return data

    def message(self, x_j):
        """Pass through features from source nodes."""
        return x_j


class SecondPeakGather(MessagePassing):
    """
    Extract shifts and features from peaks in second position.

    Uses edge type: ("Peak", "prop_second", triple_type)
    Sets attributes on triple nodes:
        - second_shifts: [n, shift_dim] chemical shift embeddings
        - second_features: [n, feature_dim] assignment feature embeddings
    Note: Peaks have NO coordinates
    """

    def __init__(self, triple_type: str):
        """
        Initialize SecondPeakGather.

        Args:
            triple_type: Name of target triple node type
        """
        super().__init__(aggr="mean")
        self.triple_type = triple_type
        self.edge_type = ("Peak", "prop_second", triple_type)

    def forward(self, data):
        """
        Extract features from Peak nodes to triple nodes.

        Args:
            data: HeteroData graph with Peak nodes and gather edges

        Returns:
            Updated HeteroData with second_shifts, second_features set
        """
        # Extract shifts [n, shift_dim] from Peak.x
        shifts = data["Peak"].x

        # Extract features [n, feature_dim] from Peak.f
        features = data["Peak"].f

        # Get edge indices for this gather operation
        edge_index = data[self.edge_type].edge_index

        # Determine number of target nodes (triple nodes)
        num_triples = data[self.triple_type].x.size(0)

        # Propagate shifts with explicit size
        data[self.triple_type].second_shifts = self.propagate(
            edge_index, x=shifts, size=(shifts.size(0), num_triples)
        )

        # Propagate features
        data[self.triple_type].second_features = self.propagate(
            edge_index, x=features, size=(features.size(0), num_triples)
        )

        return data

    def message(self, x_j):
        """Pass through features from source nodes."""
        return x_j


class NoeGather(MessagePassing):
    """
    Extract NOE shifts and features from NOE constraint nodes.

    Uses edge type: ("Noe", "prop_noe", triple_type)
    Sets attributes on triple nodes:
        - noe_shifts: [n, shift_dim] NOE shift embeddings
        - noe_features: [n, feature_dim] NOE feature embeddings
    """

    def __init__(self, triple_type: str):
        """
        Initialize NoeGather.

        Args:
            triple_type: Name of target triple node type
        """
        super().__init__(aggr="mean")
        self.triple_type = triple_type
        self.edge_type = ("Noe", "prop_noe", triple_type)

    def forward(self, data):
        """
        Extract features from Noe nodes to triple nodes.

        Args:
            data: HeteroData graph with Noe nodes and gather edges

        Returns:
            Updated HeteroData with noe_shifts, noe_features set
        """
        # Extract NOE shifts [n, shift_dim] from Noe.x (embedded from original 3D NOE features)
        shifts = data["Noe"].x

        # Extract features [n, feature_dim] from Noe.f
        features = data["Noe"].f

        # Get edge indices for this gather operation
        edge_index = data[self.edge_type].edge_index

        # Determine number of target nodes (triple nodes)
        num_triples = data[self.triple_type].x.size(0)

        # Propagate shifts with explicit size
        data[self.triple_type].noe_shifts = self.propagate(
            edge_index, x=shifts, size=(shifts.size(0), num_triples)
        )

        # Propagate features
        data[self.triple_type].noe_features = self.propagate(
            edge_index, x=features, size=(features.size(0), num_triples)
        )

        return data

    def message(self, x_j):
        """Pass through features from source nodes."""
        return x_j


# ============================================================================
# SECTION 2: Update Operations
# ============================================================================
# These classes compute deltas on triple nodes using MLPs (standard nn.Module)


def _create_empty_deltas(device, num_triples=0, shift_dim=2, feature_dim=2):
    """
    Create empty delta tensors for triple nodes with zero instances.

    This helper function is used when a triple set is empty to ensure all
    delta attributes are properly initialized with correct shapes.

    Args:
        device: torch device (CPU or CUDA)
        num_triples: Number of triple nodes (typically 0 for empty sets)
        shift_dim: Dimension of shift embeddings (default: 2 for backward compatibility)
        feature_dim: Dimension of feature embeddings (default: 2 for backward compatibility)

    Returns:
        Dictionary with 8 delta tensors:
            - delta_first_coords: [num_triples, 3]
            - delta_first_shifts: [num_triples, shift_dim]
            - delta_first_features: [num_triples, feature_dim]
            - delta_second_coords: [num_triples, 3]
            - delta_second_shifts: [num_triples, shift_dim]
            - delta_second_features: [num_triples, feature_dim]
            - delta_noe_shifts: [num_triples, shift_dim]
            - delta_noe_features: [num_triples, feature_dim]
    """
    return {
        'delta_first_coords': torch.zeros((num_triples, 3), dtype=torch.float32, device=device),
        'delta_first_shifts': torch.zeros((num_triples, shift_dim), dtype=torch.float32, device=device),
        'delta_first_features': torch.zeros((num_triples, feature_dim), dtype=torch.float32, device=device),
        'delta_second_coords': torch.zeros((num_triples, 3), dtype=torch.float32, device=device),
        'delta_second_shifts': torch.zeros((num_triples, shift_dim), dtype=torch.float32, device=device),
        'delta_second_features': torch.zeros((num_triples, feature_dim), dtype=torch.float32, device=device),
        'delta_noe_shifts': torch.zeros((num_triples, shift_dim), dtype=torch.float32, device=device),
        'delta_noe_features': torch.zeros((num_triples, feature_dim), dtype=torch.float32, device=device),
    }


class ResidueUpdate(nn.Module):
    """
    Compute deltas for ResidueResidueNoeTriple using MLP.

    MLP Architecture:
        Input: variable (dist_squared + 3*shift_dim + 3*feature_dim)
        Hidden: configurable layers with ReLU activations
        Output: variable (2 coord weights + 6*shift_dim + 3*feature_dim)

    Equivariance:
        - Calculations are based on distances and differences between shifts
        - Coordinate deltas are computed as difference * learned_weights to maintain
          rotation/translation equivariance.
    """

    def __init__(self, triple_type: str, device, config):
        """
        Initialize ResidueUpdate.

        Args:
            triple_type: Name of triple node type (should be "ResidueResidueNoeTriple")
            device: torch device (CPU or CUDA)
            config: ModelConfig with shift_embed and feature_embed dimensions
        """
        super().__init__()
        self.triple_type = triple_type
        self.device = device
        self.config = config

        # Get MLP configuration from config
        hidden_size = config.mlp.hidden_size
        num_layers = config.mlp.num_layers

        # Calculate input/output sizes from config
        shift_dim = config.shift_embed.output_dim
        feature_dim = config.feature_embed.output_dim
        input_size = residue_update_input_size(shift_dim, feature_dim)
        output_size = residue_update_output_size(shift_dim, feature_dim)

        # Store dimensions for later use
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
        layers.append(nn.LayerNorm(output_size))

        self.mlp = nn.Sequential(*layers).to(device)

    def forward(self, data):
        """
        Compute deltas for triple nodes.

        After embeddings, dimensions are:
        - first_shifts: [n, shift_dim]
        - second_shifts: [n, shift_dim]
        - noe_shifts: [n, shift_dim]
        - first_features: [n, feature_dim]
        - second_features: [n, feature_dim]
        - noe_features: [n, feature_dim]

        Args:
            data: HeteroData with gathered attributes on triple nodes

        Returns:
            Updated HeteroData with delta attributes set
        """
        # Get gathered attributes (dimensions now variable)
        first_coords = data[self.triple_type].first_coords  # [n, 3]
        first_shifts = data[self.triple_type].first_shifts  # [n, shift_dim]
        first_features = data[self.triple_type].first_features  # [n, feature_dim]
        second_coords = data[self.triple_type].second_coords  # [n, 3]
        second_shifts = data[self.triple_type].second_shifts  # [n, shift_dim]
        second_features = data[self.triple_type].second_features  # [n, feature_dim]
        noe_shifts = data[self.triple_type].noe_shifts  # [n, shift_dim]
        noe_features = data[self.triple_type].noe_features  # [n, feature_dim]

        num_triples = first_shifts.size(0)

        # Handle empty triple sets using helper function
        if num_triples == 0:
            deltas = _create_empty_deltas(self.device, num_triples=0, shift_dim=self.shift_dim, feature_dim=self.feature_dim)
            for key, value in deltas.items():
                setattr(data[self.triple_type], key, value)
            return data

        # Calculate relative distance and dist_squared for equivariance
        rel_dist, dist_squared = calc_res_distance(first_coords, second_coords)

        # Compute shift differences BEFORE MLP input (for translation equivariance)
        diff_first_to_second = second_shifts - first_shifts  # [n, shift_dim]
        diff_first_to_noe = noe_shifts - first_shifts  # [n, shift_dim]
        diff_second_to_noe = noe_shifts - second_shifts  # [n, shift_dim]

        # Concatenate all features for MLP input
        # Input: dist_squared + 3*shift_dim (differences) + 3*feature_dim
        mlp_input = torch.cat(
            [
                dist_squared,  # [n, 1]
                diff_first_to_second,  # [n, shift_dim]
                diff_first_to_noe,  # [n, shift_dim]
                diff_second_to_noe,  # [n, shift_dim]
                first_features,  # [n, feature_dim]
                second_features,  # [n, feature_dim]
                noe_features,  # [n, feature_dim]
            ],
            dim=-1,
        )

        # Apply MLP to get deltas
        mlp_output = self.mlp(mlp_input)

        # Parse output:
        # 2 coord weights + 6 scalar shift weights + 3*feature_dim feature deltas
        coord_weight_first = 0 * mlp_output[:, 0:1]  # [n, 1] - still zeroed out
        coord_weight_second = 0 * mlp_output[:, 1:2]  # [n, 1] - still zeroed out

        # Shift weights: 6 scalars
        w1 = mlp_output[:, 2:3]  # [n, 1] - weight for first -> second
        w2 = mlp_output[:, 3:4]  # [n, 1] - weight for second -> first
        w3 = mlp_output[:, 4:5]  # [n, 1] - weight for first -> noe
        w4 = mlp_output[:, 5:6]  # [n, 1] - weight for second -> noe
        w5 = mlp_output[:, 6:7]  # [n, 1] - weight for noe -> first
        w6 = mlp_output[:, 7:8]  # [n, 1] - weight for noe -> second

        # Feature deltas: 3 vectors of feature_dim each
        feature_deltas = mlp_output[:, 8:]  # [n, 3*feature_dim]

        # Compute equivariant coordinate deltas: rel_dist * learned_weights
        delta_first_coords = rel_dist * coord_weight_first  # [n, 3]
        delta_second_coords = -rel_dist * coord_weight_second  # [n, 3]

        # Compute shift deltas using scalar weights × difference vectors
        # The scalar weights [n, 1] broadcast with difference vectors [n, shift_dim]
        #
        # Position 1 shifts - receives contributions from w1 and w3
        delta_first_shifts = w1 * diff_first_to_second + w3 * diff_first_to_noe  # [n, shift_dim]

        # Position 2 shifts - receives contributions from w2 and w4
        # Note: diff_first_to_second = second - first, so -diff_first_to_second = first - second
        delta_second_shifts = -w2 * diff_first_to_second + w4 * diff_second_to_noe  # [n, shift_dim]

        # NOE shifts - receives contributions from w5 and w6
        delta_noe_shifts = -w5 * diff_first_to_noe + -w6 * diff_second_to_noe  # [n, shift_dim]

        # Compute feature deltas (3 vectors of feature_dim each)
        delta_first_features = feature_deltas[:, 0*self.feature_dim:1*self.feature_dim]  # [n, feature_dim]
        delta_second_features = feature_deltas[:, 1*self.feature_dim:2*self.feature_dim]  # [n, feature_dim]
        delta_noe_features = feature_deltas[:, 2*self.feature_dim:3*self.feature_dim]  # [n, feature_dim]

        # Set delta attributes on triple nodes
        data[self.triple_type].delta_first_coords = delta_first_coords
        data[self.triple_type].delta_first_shifts = delta_first_shifts
        data[self.triple_type].delta_first_features = delta_first_features
        data[self.triple_type].delta_second_coords = delta_second_coords
        data[self.triple_type].delta_second_shifts = delta_second_shifts
        data[self.triple_type].delta_second_features = delta_second_features
        data[self.triple_type].delta_noe_shifts = delta_noe_shifts
        data[self.triple_type].delta_noe_features = delta_noe_features

        return data


class PeakUpdate(nn.Module):
    """
    Compute deltas for Peak-based triples using MLP.

    MLP Architecture:
        Input: variable (3*shift_dim + 3*feature_dim, NO dist_squared)
        Hidden: configurable layers with ReLU activations
        Output: variable (6*shift_dim + 3*feature_dim, NO coord weights)

    Coordinate Handling:
        Peaks have no coordinates, so all coordinate deltas are zero.

    Equivariance:
        Calculations are based on differences between shifts
    """

    def __init__(self, triple_type: str, device, config):
        """
        Initialize PeakUpdate.

        Args:
            triple_type: Name of triple node type
            device: torch device (CPU or CUDA)
            config: ModelConfig with shift_embed and feature_embed dimensions
        """
        super().__init__()
        self.triple_type = triple_type
        self.device = device
        self.config = config

        # Get MLP configuration from config
        hidden_size = config.mlp.hidden_size
        num_layers = config.mlp.num_layers

        # Calculate input/output sizes from config
        shift_dim = config.shift_embed.output_dim
        feature_dim = config.feature_embed.output_dim
        input_size = peak_update_input_size(shift_dim, feature_dim)
        output_size = peak_update_output_size(shift_dim, feature_dim)

        # Store dimensions for later use
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
        layers.append(nn.LayerNorm(output_size))

        self.mlp = nn.Sequential(*layers).to(device)

    def forward(self, data):
        """
        Compute deltas for triple nodes.

        After embeddings, dimensions are:
        - first_shifts: [n, shift_dim]
        - second_shifts: [n, shift_dim]
        - noe_shifts: [n, shift_dim]
        - first_features: [n, feature_dim]
        - second_features: [n, feature_dim]
        - noe_features: [n, feature_dim]

        Args:
            data: HeteroData with gathered attributes on triple nodes

        Returns:
            Updated HeteroData with delta attributes set
        """
        # Get gathered attributes (peaks have NO coordinates)
        first_shifts = data[self.triple_type].first_shifts  # [n, shift_dim]
        first_features = data[self.triple_type].first_features  # [n, feature_dim]
        second_shifts = data[self.triple_type].second_shifts  # [n, shift_dim]
        second_features = data[self.triple_type].second_features  # [n, feature_dim]
        noe_shifts = data[self.triple_type].noe_shifts  # [n, shift_dim]
        noe_features = data[self.triple_type].noe_features  # [n, feature_dim]

        num_triples = first_shifts.size(0)

        # Handle empty triple sets using helper function
        if num_triples == 0:
            deltas = _create_empty_deltas(self.device, num_triples=0, shift_dim=self.shift_dim, feature_dim=self.feature_dim)
            for key, value in deltas.items():
                setattr(data[self.triple_type], key, value)
            return data

        # Compute shift differences BEFORE MLP input (for translation equivariance)
        diff_first_to_second = second_shifts - first_shifts  # [n, shift_dim]
        diff_first_to_noe = noe_shifts - first_shifts  # [n, shift_dim]
        diff_second_to_noe = noe_shifts - second_shifts  # [n, shift_dim]

        # Concatenate all features for MLP input (NO distance calculations for peaks)
        # Input: 3*shift_dim (differences) + 3*feature_dim
        mlp_input = torch.cat(
            [
                diff_first_to_second,  # [n, shift_dim]
                diff_first_to_noe,  # [n, shift_dim]
                diff_second_to_noe,  # [n, shift_dim]
                first_features,  # [n, feature_dim]
                second_features,  # [n, feature_dim]
                noe_features,  # [n, feature_dim]
            ],
            dim=-1,
        )

        # Apply MLP to get outputs
        mlp_output = self.mlp(mlp_input)

        # Parse output:
        # 6 scalar shift weights + 3*feature_dim feature deltas (NO coord weights)
        w1 = mlp_output[:, 0:1]  # [n, 1] - weight for first -> second
        w2 = mlp_output[:, 1:2]  # [n, 1] - weight for second -> first
        w3 = mlp_output[:, 2:3]  # [n, 1] - weight for first -> noe
        w4 = mlp_output[:, 3:4]  # [n, 1] - weight for second -> noe
        w5 = mlp_output[:, 4:5]  # [n, 1] - weight for noe -> first
        w6 = mlp_output[:, 5:6]  # [n, 1] - weight for noe -> second

        # Feature deltas: 3 vectors of feature_dim each
        feature_deltas = mlp_output[:, 6:]  # [n, 3*feature_dim]

        # Create ZERO coordinate deltas (peaks have no coordinates)
        delta_first_coords = torch.zeros((num_triples, 3), dtype=torch.float32, device=self.device)
        delta_second_coords = torch.zeros((num_triples, 3), dtype=torch.float32, device=self.device)

        # Compute shift deltas using scalar weights × difference vectors
        # The scalar weights [n, 1] broadcast with difference vectors [n, shift_dim]
        #
        # Position 1 shifts - receives contributions from w1 and w3
        delta_first_shifts = w1 * diff_first_to_second + w3 * diff_first_to_noe  # [n, shift_dim]

        # Position 2 shifts - receives contributions from w2 and w4
        # Note: diff_first_to_second = second - first, so -diff_first_to_second = first - second
        delta_second_shifts = -w2 * diff_first_to_second + w4 * diff_second_to_noe  # [n, shift_dim]

        # NOE shifts - receives contributions from w5 and w6
        delta_noe_shifts = -w5 * diff_first_to_noe + -w6 * diff_second_to_noe  # [n, shift_dim]

        # Compute feature deltas (3 vectors of feature_dim each)
        delta_first_features = feature_deltas[:, 0*self.feature_dim:1*self.feature_dim]  # [n, feature_dim]
        delta_second_features = feature_deltas[:, 1*self.feature_dim:2*self.feature_dim]  # [n, feature_dim]
        delta_noe_features = feature_deltas[:, 2*self.feature_dim:3*self.feature_dim]  # [n, feature_dim]

        # Set delta attributes on triple nodes
        data[self.triple_type].delta_first_coords = delta_first_coords
        data[self.triple_type].delta_first_shifts = delta_first_shifts
        data[self.triple_type].delta_first_features = delta_first_features
        data[self.triple_type].delta_second_coords = delta_second_coords
        data[self.triple_type].delta_second_shifts = delta_second_shifts
        data[self.triple_type].delta_second_features = delta_second_features
        data[self.triple_type].delta_noe_shifts = delta_noe_shifts
        data[self.triple_type].delta_noe_features = delta_noe_features

        return data


# ============================================================================
# SECTION 3: Scatter Operations
# ============================================================================
# These classes propagate deltas from triple nodes back to source nodes using
# PyTorch Geometric MessagePassing with aggr="mean"


class FirstResidueScatter(MessagePassing):
    """
    Propagate deltas from triple nodes to residues in first position.

    Uses edge type: ("Residue", "prop_first", triple_type) with reversed flow
    Reads delta attributes from triple nodes:
        - delta_first_coords: [n, 3] coordinate deltas
        - delta_first_shifts: [n, shift_dim] shift deltas
        - delta_first_features: [n, feature_dim] feature deltas
    Updates Residue nodes:
        - Residue.x[:, 0:3] with coordinate deltas
        - Residue.x[:, 3:] with shift deltas
        - Residue.f with feature deltas
    """

    def __init__(self, triple_type: str):
        """
        Initialize FirstResidueScatter.

        Args:
            triple_type: Name of source triple node type
        """
        super().__init__(aggr="mean", flow="target_to_source")
        self.triple_type = triple_type
        self.edge_type = ("Residue", "prop_first", triple_type)

    def forward(self, data):
        """
        Propagate deltas from triple nodes to Residue nodes.

        Args:
            data: HeteroData with delta attributes on triple nodes

        Returns:
            Updated HeteroData with Residue.x and Residue.f modified
        """
        # Get delta attributes from triple nodes
        delta_coords = data[self.triple_type].delta_first_coords  # [n, 3]
        delta_shifts = data[self.triple_type].delta_first_shifts  # [n, shift_dim]
        delta_features = data[self.triple_type].delta_first_features  # [n, feature_dim]

        # Get edge indices for this scatter operation
        edge_index = data[self.edge_type].edge_index

        # Determine number of target nodes (Residue nodes - targets when using reversed flow)
        num_residues = data["Residue"].x.size(0)

        # Propagate coordinate deltas (with reversed flow, size is (target, source))
        coord_updates = self.propagate(
            edge_index, x=delta_coords, size=(num_residues, delta_coords.size(0))
        )

        # Propagate shift deltas
        shift_updates = self.propagate(
            edge_index, x=delta_shifts, size=(num_residues, delta_shifts.size(0))
        )

        # Propagate feature deltas
        feature_updates = self.propagate(
            edge_index, x=delta_features, size=(num_residues, delta_features.size(0))
        )

        # Assemble complete new tensors (avoid in-place updates for gradient preservation)
        # Update Residue.x by concatenating updated coords and shifts
        old_x = data["Residue"].x
        new_coords = old_x[:, 0:3] + coord_updates
        new_shifts = old_x[:, 3:] + shift_updates
        data["Residue"].x = torch.cat([new_coords, new_shifts], dim=-1)

        # Update Residue.f
        data["Residue"].f = data["Residue"].f + feature_updates

        return data

    def message(self, x_j):
        """Pass through deltas from triple nodes."""
        return x_j


class FirstPeakScatter(MessagePassing):
    """
    Propagate deltas from triple nodes to peaks in first position.

    Uses edge type: ("Peak", "prop_first", triple_type) with reversed flow
    Reads delta attributes from triple nodes:
        - delta_first_shifts: [n, shift_dim] shift deltas
        - delta_first_features: [n, feature_dim] feature deltas
    Updates Peak nodes:
        - Peak.x with shift deltas
        - Peak.f with feature deltas
    Note: Peaks have NO coordinates
    """

    def __init__(self, triple_type: str):
        """
        Initialize FirstPeakScatter.

        Args:
            triple_type: Name of source triple node type
        """
        super().__init__(aggr="mean", flow="target_to_source")
        self.triple_type = triple_type
        self.edge_type = ("Peak", "prop_first", triple_type)

    def forward(self, data):
        """
        Propagate deltas from triple nodes to Peak nodes.

        Args:
            data: HeteroData with delta attributes on triple nodes

        Returns:
            Updated HeteroData with Peak.x and Peak.f modified
        """
        # Get delta attributes from triple nodes (no coords for peaks)
        delta_shifts = data[self.triple_type].delta_first_shifts  # [n, shift_dim]
        delta_features = data[self.triple_type].delta_first_features  # [n, feature_dim]

        # Get edge indices for this scatter operation
        edge_index = data[self.edge_type].edge_index

        # Determine number of target nodes (Peak nodes - targets when using reversed flow)
        num_peaks = data["Peak"].x.size(0)

        # Propagate shift deltas (with reversed flow, size is (target, source))
        shift_updates = self.propagate(
            edge_index, x=delta_shifts, size=(num_peaks, delta_shifts.size(0))
        )

        # Propagate feature deltas
        feature_updates = self.propagate(
            edge_index, x=delta_features, size=(num_peaks, delta_features.size(0))
        )

        # Update Peak.x and Peak.f (assemble new tensors)
        data["Peak"].x = data["Peak"].x + shift_updates
        data["Peak"].f = data["Peak"].f + feature_updates

        return data

    def message(self, x_j):
        """Pass through deltas from triple nodes."""
        return x_j


class SecondResidueScatter(MessagePassing):
    """
    Propagate deltas from triple nodes to residues in second position.

    Uses edge type: ("Residue", "prop_second", triple_type) with reversed flow
    Reads delta attributes from triple nodes:
        - delta_second_coords: [n, 3] coordinate deltas
        - delta_second_shifts: [n, shift_dim] shift deltas
        - delta_second_features: [n, feature_dim] feature deltas
    Updates Residue nodes:
        - Residue.x[:, 0:3] with coordinate deltas
        - Residue.x[:, 3:] with shift deltas
        - Residue.f with feature deltas
    """

    def __init__(self, triple_type: str):
        """
        Initialize SecondResidueScatter.

        Args:
            triple_type: Name of source triple node type
        """
        super().__init__(aggr="mean", flow="target_to_source")
        self.triple_type = triple_type
        self.edge_type = ("Residue", "prop_second", triple_type)

    def forward(self, data):
        """
        Propagate deltas from triple nodes to Residue nodes.

        Args:
            data: HeteroData with delta attributes on triple nodes

        Returns:
            Updated HeteroData with Residue.x and Residue.f modified
        """
        # Get delta attributes from triple nodes
        delta_coords = data[self.triple_type].delta_second_coords  # [n, 3]
        delta_shifts = data[self.triple_type].delta_second_shifts  # [n, shift_dim]
        delta_features = data[self.triple_type].delta_second_features  # [n, feature_dim]

        # Get edge indices for this scatter operation
        edge_index = data[self.edge_type].edge_index

        # Determine number of target nodes (Residue nodes - targets when using reversed flow)
        num_residues = data["Residue"].x.size(0)

        # Propagate coordinate deltas (with reversed flow, size is (target, source))
        coord_updates = self.propagate(
            edge_index, x=delta_coords, size=(num_residues, delta_coords.size(0))
        )

        # Propagate shift deltas
        shift_updates = self.propagate(
            edge_index, x=delta_shifts, size=(num_residues, delta_shifts.size(0))
        )

        # Propagate feature deltas
        feature_updates = self.propagate(
            edge_index, x=delta_features, size=(num_residues, delta_features.size(0))
        )

        # Assemble complete new tensors (avoid in-place updates for gradient preservation)
        # Update Residue.x by concatenating updated coords and shifts
        old_x = data["Residue"].x
        new_coords = old_x[:, 0:3] + coord_updates
        new_shifts = old_x[:, 3:] + shift_updates
        data["Residue"].x = torch.cat([new_coords, new_shifts], dim=-1)

        # Update Residue.f
        data["Residue"].f = data["Residue"].f + feature_updates

        return data

    def message(self, x_j):
        """Pass through deltas from triple nodes."""
        return x_j


class SecondPeakScatter(MessagePassing):
    """
    Propagate deltas from triple nodes to peaks in second position.

    Uses edge type: ("Peak", "prop_second", triple_type) with reversed flow
    Reads delta attributes from triple nodes:
        - delta_second_shifts: [n, shift_dim] shift deltas
        - delta_second_features: [n, feature_dim] feature deltas
    Updates Peak nodes:
        - Peak.x with shift deltas
        - Peak.f with feature deltas
    Note: Peaks have NO coordinates
    """

    def __init__(self, triple_type: str):
        """
        Initialize SecondPeakScatter.

        Args:
            triple_type: Name of source triple node type
        """
        super().__init__(aggr="mean", flow="target_to_source")
        self.triple_type = triple_type
        self.edge_type = ("Peak", "prop_second", triple_type)

    def forward(self, data):
        """
        Propagate deltas from triple nodes to Peak nodes.

        Args:
            data: HeteroData with delta attributes on triple nodes

        Returns:
            Updated HeteroData with Peak.x and Peak.f modified
        """
        # Get delta attributes from triple nodes (no coords for peaks)
        delta_shifts = data[self.triple_type].delta_second_shifts  # [n, shift_dim]
        delta_features = data[self.triple_type].delta_second_features  # [n, feature_dim]

        # Get edge indices for this scatter operation
        edge_index = data[self.edge_type].edge_index

        # Determine number of target nodes (Peak nodes - targets when using reversed flow)
        num_peaks = data["Peak"].x.size(0)

        # Propagate shift deltas (with reversed flow, size is (target, source))
        shift_updates = self.propagate(
            edge_index, x=delta_shifts, size=(num_peaks, delta_shifts.size(0))
        )

        # Propagate feature deltas
        feature_updates = self.propagate(
            edge_index, x=delta_features, size=(num_peaks, delta_features.size(0))
        )

        # Update Peak.x and Peak.f (assemble new tensors)
        data["Peak"].x = data["Peak"].x + shift_updates
        data["Peak"].f = data["Peak"].f + feature_updates

        return data

    def message(self, x_j):
        """Pass through deltas from triple nodes."""
        return x_j


class NoeScatter(MessagePassing):
    """
    Propagate deltas from triple nodes to NOE constraint nodes.

    Uses edge type: ("Noe", "prop_noe", triple_type) with reversed flow
    Reads delta attributes from triple nodes:
        - delta_noe_shifts: [n, shift_dim] NOE shift deltas
        - delta_noe_features: [n, feature_dim] feature deltas
    Updates Noe nodes:
        - Noe.x with shift deltas
        - Noe.f with feature deltas
    """

    def __init__(self, triple_type: str):
        """
        Initialize NoeScatter.

        Args:
            triple_type: Name of source triple node type
        """
        super().__init__(aggr="mean", flow="target_to_source")
        self.triple_type = triple_type
        self.edge_type = ("Noe", "prop_noe", triple_type)

    def forward(self, data):
        """
        Propagate deltas from triple nodes to Noe nodes.

        Args:
            data: HeteroData with delta attributes on triple nodes

        Returns:
            Updated HeteroData with Noe.x and Noe.f modified
        """
        # Get delta attributes from triple nodes
        delta_shifts = data[self.triple_type].delta_noe_shifts  # [n, shift_dim]
        delta_features = data[self.triple_type].delta_noe_features  # [n, feature_dim]

        # Get edge indices for this scatter operation
        edge_index = data[self.edge_type].edge_index

        # Determine number of target nodes (Noe nodes - targets when using reversed flow)
        num_noes = data["Noe"].x.size(0)

        # Propagate shift deltas (with reversed flow, size is (target, source))
        shift_updates = self.propagate(
            edge_index, x=delta_shifts, size=(num_noes, delta_shifts.size(0))
        )

        # Propagate feature deltas
        feature_updates = self.propagate(
            edge_index, x=delta_features, size=(num_noes, delta_features.size(0))
        )

        # Update Noe.x and Noe.f (assemble new tensors)
        data["Noe"].x = data["Noe"].x + shift_updates
        data["Noe"].f = data["Noe"].f + feature_updates

        return data

    def message(self, x_j):
        """Pass through deltas from triple nodes."""
        return x_j


# ============================================================================
# SECTION 4: Triple Composition Classes
# ============================================================================
# These classes wire together gather, update, and scatter operations for each
# triple type


class ResidueResidueNoeTriple(nn.Module):
    """
    Wire together components for ResidueResidueNoeTriple.

    This triple type handles (Residue, Residue, Noe) relationships and is the
    only triple type that updates coordinates (equivariant).

    Components:
        - FirstResidueGather: Extract from first residue
        - SecondResidueGather: Extract from second residue
        - NoeGather: Extract from NOE constraint
        - ResidueUpdate: Compute coordinate and shift deltas
        - FirstResidueScatter: Propagate to first residue
        - SecondResidueScatter: Propagate to second residue
        - NoeScatter: Propagate to NOE constraint

    Forward pass sequence:
        gather_first → gather_second → gather_noe → update →
        scatter_first → scatter_second → scatter_noe
    """

    def __init__(self, device, config):
        """
        Initialize ResidueResidueNoeTriple.

        Args:
            device: torch device (CPU or CUDA)
            config: ModelConfig with dimension settings
        """
        super().__init__()
        triple_type = "ResidueResidueNoeTriple"

        # Instantiate gather operations
        self.first_gather = FirstResidueGather(triple_type)
        self.second_gather = SecondResidueGather(triple_type)
        self.noe_gather = NoeGather(triple_type)

        # Instantiate update operation
        self.update = ResidueUpdate(triple_type, device, config)

        # Instantiate scatter operations
        self.first_scatter = FirstResidueScatter(triple_type)
        self.second_scatter = SecondResidueScatter(triple_type)
        self.noe_scatter = NoeScatter(triple_type)

    def forward(self, data):
        """
        Execute gather → update → scatter pipeline.

        Args:
            data: HeteroData graph

        Returns:
            Updated HeteroData graph
        """
        data = self.first_gather(data)
        data = self.second_gather(data)
        data = self.noe_gather(data)
        data = self.update(data)
        data = self.first_scatter(data)
        data = self.second_scatter(data)
        data = self.noe_scatter(data)
        return data


class ResiduePeakNoeTriple(nn.Module):
    """
    Wire together components for ResiduePeakNoeTriple.

    This triple type handles (Residue, Peak, Noe) relationships. Updates shifts
    only, NO coordinate updates.

    Components:
        - FirstResidueGather: Extract from first residue
        - SecondPeakGather: Extract from second peak
        - NoeGather: Extract from NOE constraint
        - PeakUpdate: Compute shift deltas only
        - FirstResidueScatter: Propagate to first residue
        - SecondPeakScatter: Propagate to second peak
        - NoeScatter: Propagate to NOE constraint

    Forward pass sequence:
        gather_first → gather_second → gather_noe → update →
        scatter_first → scatter_second → scatter_noe
    """

    def __init__(self, device, config):
        """
        Initialize ResiduePeakNoeTriple.

        Args:
            device: torch device (CPU or CUDA)
            config: ModelConfig with dimension settings
        """
        super().__init__()
        triple_type = "ResiduePeakNoeTriple"

        # Instantiate gather operations
        self.first_gather = FirstResidueGather(triple_type)
        self.second_gather = SecondPeakGather(triple_type)
        self.noe_gather = NoeGather(triple_type)

        # Instantiate update operation
        self.update = PeakUpdate(triple_type, device, config)

        # Instantiate scatter operations
        self.first_scatter = FirstResidueScatter(triple_type)
        self.second_scatter = SecondPeakScatter(triple_type)
        self.noe_scatter = NoeScatter(triple_type)

    def forward(self, data):
        """
        Execute gather → update → scatter pipeline.

        Args:
            data: HeteroData graph

        Returns:
            Updated HeteroData graph
        """
        data = self.first_gather(data)
        data = self.second_gather(data)
        data = self.noe_gather(data)
        data = self.update(data)
        data = self.first_scatter(data)
        data = self.second_scatter(data)
        data = self.noe_scatter(data)
        return data


class PeakResidueNoeTriple(nn.Module):
    """
    Wire together components for PeakResidueNoeTriple.

    This triple type handles (Peak, Residue, Noe) relationships. Updates shifts
    only, NO coordinate updates.

    Components:
        - FirstPeakGather: Extract from first peak
        - SecondResidueGather: Extract from second residue
        - NoeGather: Extract from NOE constraint
        - PeakUpdate: Compute shift deltas only
        - FirstPeakScatter: Propagate to first peak
        - SecondResidueScatter: Propagate to second residue
        - NoeScatter: Propagate to NOE constraint

    Forward pass sequence:
        gather_first → gather_second → gather_noe → update →
        scatter_first → scatter_second → scatter_noe
    """

    def __init__(self, device, config):
        """
        Initialize PeakResidueNoeTriple.

        Args:
            device: torch device (CPU or CUDA)
            config: ModelConfig with dimension settings
        """
        super().__init__()
        triple_type = "PeakResidueNoeTriple"

        # Instantiate gather operations
        self.first_gather = FirstPeakGather(triple_type)
        self.second_gather = SecondResidueGather(triple_type)
        self.noe_gather = NoeGather(triple_type)

        # Instantiate update operation
        self.update = PeakUpdate(triple_type, device, config)

        # Instantiate scatter operations
        self.first_scatter = FirstPeakScatter(triple_type)
        self.second_scatter = SecondResidueScatter(triple_type)
        self.noe_scatter = NoeScatter(triple_type)

    def forward(self, data):
        """
        Execute gather → update → scatter pipeline.

        Args:
            data: HeteroData graph

        Returns:
            Updated HeteroData graph
        """
        data = self.first_gather(data)
        data = self.second_gather(data)
        data = self.noe_gather(data)
        data = self.update(data)
        data = self.first_scatter(data)
        data = self.second_scatter(data)
        data = self.noe_scatter(data)
        return data


class PeakPeakNoeTriple(nn.Module):
    """
    Wire together components for PeakPeakNoeTriple.

    This triple type handles (Peak, Peak, Noe) relationships. Updates shifts
    only, NO coordinate updates.

    Components:
        - FirstPeakGather: Extract from first peak
        - SecondPeakGather: Extract from second peak
        - NoeGather: Extract from NOE constraint
        - PeakUpdate: Compute shift deltas only
        - FirstPeakScatter: Propagate to first peak
        - SecondPeakScatter: Propagate to second peak
        - NoeScatter: Propagate to NOE constraint

    Forward pass sequence:
        gather_first → gather_second → gather_noe → update →
        scatter_first → scatter_second → scatter_noe
    """

    def __init__(self, device, config):
        """
        Initialize PeakPeakNoeTriple.

        Args:
            device: torch device (CPU or CUDA)
            config: ModelConfig with dimension settings
        """
        super().__init__()
        triple_type = "PeakPeakNoeTriple"

        # Instantiate gather operations
        self.first_gather = FirstPeakGather(triple_type)
        self.second_gather = SecondPeakGather(triple_type)
        self.noe_gather = NoeGather(triple_type)

        # Instantiate update operation
        self.update = PeakUpdate(triple_type, device, config)

        # Instantiate scatter operations
        self.first_scatter = FirstPeakScatter(triple_type)
        self.second_scatter = SecondPeakScatter(triple_type)
        self.noe_scatter = NoeScatter(triple_type)

    def forward(self, data):
        """
        Execute gather → update → scatter pipeline.

        Args:
            data: HeteroData graph

        Returns:
            Updated HeteroData graph
        """
        data = self.first_gather(data)
        data = self.second_gather(data)
        data = self.noe_gather(data)
        data = self.update(data)
        data = self.first_scatter(data)
        data = self.second_scatter(data)
        data = self.noe_scatter(data)
        return data


# ============================================================================
# SECTION 5: Helper Functions
# ============================================================================
# Calculation utilities used by update operations


def calc_noe_difference(x1, x2, noe):
    """
    Calculate shift differences between NOE and residue/peak shifts.

    Args:
        x1: First node shifts [n, 2] (H, N)
        x2: Second node shifts [n, 2] (H, N)
        noe: NOE shifts [n, 3] (N, H', H")

    Returns:
        Tuple of (diff_N, diff_H1, diff_H2) each [n, 1]
    """
    NOE_N1 = slice(0, 1)
    NOE_H1 = slice(1, 2)
    NOE_H2 = slice(2, 3)
    SHIFT_H = slice(0, 1)  # H is at index 0
    SHIFT_N = slice(1, 2)  # N is at index 1

    diff_N = noe[:, NOE_N1] - x1[:, SHIFT_N]  # N [n, 1]
    diff_H1 = noe[:, NOE_H1] - x1[:, SHIFT_H]  # H' [n, 1]
    diff_H2 = noe[:, NOE_H2] - x2[:, SHIFT_H]  # H" [n, 1]
    return diff_N, diff_H1, diff_H2


def calc_shift_difference(x1, x2):
    """
    Calculate shift differences between two nodes.

    Args:
        x1: First node shifts [n, 2] (H, N)
        x2: Second node shifts [n, 2] (H, N)

    Returns:
        Tuple of (diff_N, diff_H) each [n, 1]
    """
    SHIFT_H = slice(0, 1)  # H is at index 0
    SHIFT_N = slice(1, 2)  # N is at index 1

    diff_N = x1[:, SHIFT_N] - x2[:, SHIFT_N]  # N [n, 1]
    diff_H = x1[:, SHIFT_H] - x2[:, SHIFT_H]  # H [n, 1]
    return diff_N, diff_H


def calc_res_distance(x1, x2):
    """
    Calculate relative distance and squared distance between residues.

    Args:
        x1: First residue coordinates [n, 3]
        x2: Second residue coordinates [n, 3]

    Returns:
        Tuple of (rel_dist, dist_squared)
            rel_dist: Relative distance vector [n, 3]
            dist_squared: Squared distance scalar [n, 1]
    """
    rel_dist = x1 - x2  # [n, 3]
    dist_squared = torch.norm(rel_dist, dim=-1, keepdim=True) ** 2  # [n, 1]
    return rel_dist, dist_squared
