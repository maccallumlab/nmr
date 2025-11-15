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
- ResidueResidueNoeTriple: (Residue, Residue, Noe)
- ResiduePeakNoeTriple: (Residue, Peak, Noe)
- PeakResidueNoeTriple: (Peak, Residue, Noe)
- PeakPeakNoeTriple: (Peak, Peak, Noe)

Node Type Naming:
- Residue:
    Protein residues with coordinates .xyz [x,y,z], shifts .shifts [H,N], and features .x
- Peak: Observed chemical shifts .shifts [H,N] and features .x
- Noe: NOE constraints .shifts [N, H', H"] and features .x

NEW ATTRIBUTE STRUCTURE (after refactoring):
- Raw data attributes (IMMUTABLE, set once during construction):
  * .xyz: coordinates (Residue only)
  * .shifts: shift values (all node types)
  * .flags: assignment status (Residue and Peak only)
- Working feature attributes (updated during message passing):
  * .x: embedded features created by EmbedFeatures layer

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


def residue_update_input_size(feature_dim):
    """
    Calculate input size for ResidueUpdate MLP.

    Components:
    - dist_squared: 1
    - features: 3*feature_dim (first_features, second_features, noe_features)

    Returns: 1 + 3*feature_dim
    """
    return 1 + 3 * feature_dim


def residue_update_output_size(feature_dim):
    """
    Calculate output size for ResidueUpdate MLP.

    Components:
    - feature_deltas: 3*feature_dim (delta_first_features, delta_second_features, delta_noe_features)

    Returns: 3*feature_dim

    Note: No coordinate updates or shift updates - only .x features are modified
    """
    return 3 * feature_dim


def peak_update_input_size(feature_dim):
    """
    Calculate input size for PeakUpdate MLP.

    Components:
    - features: 3*feature_dim (first_features, second_features, noe_features)

    Returns: 3*feature_dim
    Note: NO dist_squared (peaks have no coordinates).
    """
    return 3 * feature_dim


def peak_update_output_size(feature_dim):
    """
    Calculate output size for PeakUpdate MLP.

    Components:
    - feature_deltas: 3*feature_dim (delta_first_features, delta_second_features, delta_noe_features)

    Returns: 3*feature_dim
    Note: No coordinate or shift updates - only .x features are modified
    """
    return 3 * feature_dim


# ============================================================================
# SECTION 1: Gather Operations
# ============================================================================
# These classes extract features from source nodes to triple nodes using
# PyTorch Geometric MessagePassing with aggr="mean"


class FirstResidueGather(MessagePassing):
    """
    Extract coordinates and features from residues in first position.

    Uses edge type: ("Residue", "prop_first", triple_type)
    Sets attributes on triple nodes:
        - first_coords: [n, 3] coordinates (from .xyz)
        - first_features: [n, feature_dim] assignment features (from .x)
    """

    def __init__(self, triple_type: str):
        """
        Initialize FirstResidueGather.

        Args:
            triple_type: Name of target triple node type
            config: ModelConfig with dimension settings
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
        # Extract coordinates [n, 3] from IMMUTABLE .xyz attribute
        coords = data["Residue"].xyz

        # Extract unified embedded features [n, embed_dim] from .x
        features = data["Residue"].x  # [n, embed_dim]

        # Get edge indices for this gather operation
        edge_index = data[self.edge_type].edge_index

        # Determine number of target nodes (triple nodes)
        num_triples = data[self.triple_type].x.size(0)

        # Propagate coordinates with explicit size
        data[self.triple_type].first_coords = self.propagate(
            edge_index, x=coords, size=(coords.size(0), num_triples)
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
        - first_features: [n, feature_dim] assignment features (from .x)
    """

    def __init__(self, triple_type: str):
        """
        Initialize FirstPeakGather.

        Args:
            triple_type: Name of target triple node type
            config: ModelConfig with dimension settings
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
            Updated HeteroData with first_features set
        """
        # Extract unified embedded features [n, embed_dim] from .x
        features = data["Peak"].x  # [n, embed_dim]

        # Get edge indices for this gather operation
        edge_index = data[self.edge_type].edge_index

        # Determine number of target nodes (triple nodes)
        num_triples = data[self.triple_type].x.size(0)

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
    Extract coordinates and features from residues in second position.

    Uses edge type: ("Residue", "prop_second", triple_type)
    Sets attributes on triple nodes:
        - second_coords: [n, 3] coordinates (from .xyz)
        - second_features: [n, feature_dim] assignment features (from .x)
    """

    def __init__(self, triple_type: str):
        """
        Initialize SecondResidueGather.

        Args:
            triple_type: Name of target triple node type
            config: ModelConfig with dimension settings
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
            Updated HeteroData with second_coords and second_features set
        """
        # Extract coordinates [n, 3] from IMMUTABLE .xyz attribute
        coords = data["Residue"].xyz

        # Extract unified embedded features [n, embed_dim] from .x
        # NOTE: With unified architecture, we use the full .x for both shifts and features
        features = data["Residue"].x  # [n, embed_dim]

        # Get edge indices for this gather operation
        edge_index = data[self.edge_type].edge_index

        # Determine number of target nodes (triple nodes)
        num_triples = data[self.triple_type].x.size(0)

        # Propagate coordinates with explicit size
        data[self.triple_type].second_coords = self.propagate(
            edge_index, x=coords, size=(coords.size(0), num_triples)
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
    Extract features from peaks in second position.

    Uses edge type: ("Peak", "prop_second", triple_type)
    Sets attributes on triple nodes:
        - second_features: [n, feature_dim] assignment features (from .x)
    """

    def __init__(self, triple_type: str):
        """
        Initialize SecondPeakGather.

        Args:
            triple_type: Name of target triple node type
            config: ModelConfig with dimension settings
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
            Updated HeteroData with second_features set
        """
        # Extract unified embedded features [n, embed_dim] from .x
        features = data["Peak"].x  # [n, embed_dim]

        # Get edge indices for this gather operation
        edge_index = data[self.edge_type].edge_index

        # Determine number of target nodes (triple nodes)
        num_triples = data[self.triple_type].x.size(0)

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
    Extract NOE features from NOE constraint nodes.

    Uses edge type: ("Noe", "prop_noe", triple_type)
    Sets attributes on triple nodes:
        - noe_features: [n, feature_dim] NOE feature embeddings
    """

    def __init__(self, triple_type: str):
        """
        Initialize NoeGather.

        Args:
            triple_type: Name of target triple node type
            config: ModelConfig with dimension settings
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
            Updated HeteroData with noe_features set
        """
        # Extract embedded NOE features from .x
        features = data["Noe"].x  # [n, embed_dim]

        # Get edge indices for this gather operation
        edge_index = data[self.edge_type].edge_index

        # Determine number of target nodes (triple nodes)
        num_triples = data[self.triple_type].x.size(0)

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


def _create_empty_deltas(device, num_triples=0, feature_dim=2):
    """
    Create empty delta tensors for triple nodes with zero instances.

    This helper function is used when a triple set is empty to ensure all
    delta attributes are properly initialized with correct shapes.

    Args:
        device: torch device (CPU or CUDA)
        num_triples: Number of triple nodes (typically 0 for empty sets)
        feature_dim: Dimension of feature embeddings (default: 2 for backward compatibility)

    Returns:
        Dictionary with 3 delta tensors:
            - delta_first_features: [num_triples, feature_dim]
            - delta_second_features: [num_triples, feature_dim]
            - delta_noe_features: [num_triples, feature_dim]
    """
    return {
        'delta_first_features': torch.zeros((num_triples, feature_dim), dtype=torch.float32, device=device),
        'delta_second_features': torch.zeros((num_triples, feature_dim), dtype=torch.float32, device=device),
        'delta_noe_features': torch.zeros((num_triples, feature_dim), dtype=torch.float32, device=device),
    }


class ResidueUpdate(nn.Module):
    """
    Compute deltas for ResidueResidueNoeTriple using MLP.

    MLP Architecture:
        Input: variable (dist_squared + 3*feature_dim)
        Hidden: configurable layers with ReLU activations
        Output: variable (3*feature_dim)
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
        feature_dim = config.embed.embed_dim
        input_size = residue_update_input_size(feature_dim)
        output_size = residue_update_output_size(feature_dim)

        # Store dimensions for later use
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
        first_features = data[self.triple_type].first_features  # [n, feature_dim]
        second_coords = data[self.triple_type].second_coords  # [n, 3]
        second_features = data[self.triple_type].second_features  # [n, feature_dim]
        noe_features = data[self.triple_type].noe_features  # [n, feature_dim]

        num_triples = first_features.size(0)

        # Handle empty triple sets using helper function
        if num_triples == 0:
            deltas = _create_empty_deltas(self.device, num_triples=0, feature_dim=self.feature_dim)
            for key, value in deltas.items():
                setattr(data[self.triple_type], key, value)
            return data

        # Calculate relative distance and dist_squared for distance-based attention
        dist_squared = calc_res_distance(first_coords, second_coords)

        # Concatenate all features for MLP input
        # Input: dist_squared + 3*feature_dim
        mlp_input = torch.cat(
            [
                dist_squared,  # [n, 1]
                first_features,  # [n, feature_dim]
                second_features,  # [n, feature_dim]
                noe_features,  # [n, feature_dim]
            ],
            dim=-1,
        )

        # Apply MLP to get deltas
        mlp_output = self.mlp(mlp_input)

        # Parse output: 3*feature_dim feature deltas
        # No coordinate or shift deltas - only update .x features
        delta_first_features = mlp_output[:, 0*self.feature_dim:1*self.feature_dim]  # [n, feature_dim]
        delta_second_features = mlp_output[:, 1*self.feature_dim:2*self.feature_dim]  # [n, feature_dim]
        delta_noe_features = mlp_output[:, 2*self.feature_dim:3*self.feature_dim]  # [n, feature_dim]

        # Set delta attributes on triple nodes (features only)
        data[self.triple_type].delta_first_features = delta_first_features
        data[self.triple_type].delta_second_features = delta_second_features
        data[self.triple_type].delta_noe_features = delta_noe_features

        return data


class PeakUpdate(nn.Module):
    """
    Compute deltas for Peak-based triples using MLP.

    MLP Architecture:
        Input: variable (3*feature_dim)
        Hidden: configurable layers with ReLU activations
        Output: variable (3*feature_dim)
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
        feature_dim = config.embed.embed_dim
        input_size = peak_update_input_size(feature_dim)
        output_size = peak_update_output_size(feature_dim)

        # Store dimensions for later use
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
        - first_features: [n, feature_dim]
        - second_features: [n, feature_dim]
        - noe_features: [n, feature_dim]

        Args:
            data: HeteroData with gathered attributes on triple nodes

        Returns:
            Updated HeteroData with delta attributes set
        """
        # Get gathered attributes (peaks have NO coordinates)
        first_features = data[self.triple_type].first_features  # [n, feature_dim]
        second_features = data[self.triple_type].second_features  # [n, feature_dim]
        noe_features = data[self.triple_type].noe_features  # [n, feature_dim]

        num_triples = first_features.size(0)

        # Handle empty triple sets using helper function
        if num_triples == 0:
            deltas = _create_empty_deltas(self.device, num_triples=0, feature_dim=self.feature_dim)
            for key, value in deltas.items():
                setattr(data[self.triple_type], key, value)
            return data

        # Concatenate all features for MLP input (NO distance calculations for peaks)
        # Input: 3*shift_dim (ABSOLUTE shifts) + 3*feature_dim
        # NO SHIFT DIFFERENCES - network learns from absolute shift values
        mlp_input = torch.cat(
            [
                first_features,  # [n, feature_dim]
                second_features,  # [n, feature_dim]
                noe_features,  # [n, feature_dim]
            ],
            dim=-1,
        )

        # Apply MLP to get outputs
        mlp_output = self.mlp(mlp_input)

        # Parse output: 3*feature_dim feature deltas
        # No shift updates - only update .x features
        delta_first_features = mlp_output[:, 0*self.feature_dim:1*self.feature_dim]  # [n, feature_dim]
        delta_second_features = mlp_output[:, 1*self.feature_dim:2*self.feature_dim]  # [n, feature_dim]
        delta_noe_features = mlp_output[:, 2*self.feature_dim:3*self.feature_dim]  # [n, feature_dim]

        # Set delta attributes on triple nodes (features only)
        data[self.triple_type].delta_first_features = delta_first_features
        data[self.triple_type].delta_second_features = delta_second_features
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
        - delta_first_features: [n, feature_dim] feature deltas
    Updates Residue nodes:
        - Residue.x with feature deltas (only updates feature portion)
    """

    def __init__(self, triple_type: str):
        """
        Initialize FirstResidueScatter.

        Args:
            triple_type: Name of source triple node type
            config: ModelConfig with dimension settings
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
            Updated HeteroData with Residue.x modified (features only)
        """
        # Get delta attributes from triple nodes (features only)
        delta_features = data[self.triple_type].delta_first_features  # [n, feature_dim]

        # Get edge indices for this scatter operation
        edge_index = data[self.edge_type].edge_index

        # Determine number of target nodes (Residue nodes - targets when using reversed flow)
        num_residues = data["Residue"].x.size(0)

        # Propagate feature deltas (with reversed flow, size is (target, source))
        feature_updates = self.propagate(
            edge_index, x=delta_features, size=(num_residues, delta_features.size(0))
        )
        data["Residue"].x = data["Residue"].x + feature_updates

        return data

    def message(self, x_j):
        """Pass through deltas from triple nodes."""
        return x_j


class FirstPeakScatter(MessagePassing):
    """
    Propagate deltas from triple nodes to peaks in first position.

    Uses edge type: ("Peak", "prop_first", triple_type) with reversed flow
    Reads delta attributes from triple nodes:
        - delta_first_features: [n, feature_dim] feature deltas
    Updates Peak nodes:
        - Peak.x with feature deltas (only updates feature portion)
    """

    def __init__(self, triple_type: str):
        """
        Initialize FirstPeakScatter.

        Args:
            triple_type: Name of source triple node type
            config: ModelConfig with dimension settings
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
            Updated HeteroData with Peak.x modified (features only)
        """
        # Get delta attributes from triple nodes (features only)
        delta_features = data[self.triple_type].delta_first_features  # [n, feature_dim]

        # Get edge indices for this scatter operation
        edge_index = data[self.edge_type].edge_index

        # Determine number of target nodes (Peak nodes - targets when using reversed flow)
        num_peaks = data["Peak"].x.size(0)

        # Propagate feature deltas (with reversed flow, size is (target, source))
        feature_updates = self.propagate(
            edge_index, x=delta_features, size=(num_peaks, delta_features.size(0))
        )
        data["Peak"].x = data["Peak"].x + feature_updates

        return data

    def message(self, x_j):
        """Pass through deltas from triple nodes."""
        return x_j


class SecondResidueScatter(MessagePassing):
    """
    Propagate deltas from triple nodes to residues in second position.

    Uses edge type: ("Residue", "prop_second", triple_type) with reversed flow
    Reads delta attributes from triple nodes:
        - delta_second_features: [n, feature_dim] feature deltas
    Updates Residue nodes:
        - Residue.x with feature deltas (only updates feature portion)
    """

    def __init__(self, triple_type: str):
        """
        Initialize SecondResidueScatter.

        Args:
            triple_type: Name of source triple node type
            config: ModelConfig with dimension settings
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
            Updated HeteroData with Residue.x modified (features only)
        """
        # Get delta attributes from triple nodes (features only)
        delta_features = data[self.triple_type].delta_second_features  # [n, feature_dim]

        # Get edge indices for this scatter operation
        edge_index = data[self.edge_type].edge_index

        # Determine number of target nodes (Residue nodes - targets when using reversed flow)
        num_residues = data["Residue"].x.size(0)

        # Propagate feature deltas (with reversed flow, size is (target, source))
        feature_updates = self.propagate(
            edge_index, x=delta_features, size=(num_residues, delta_features.size(0))
        )
        data["Residue"].x = data["Residue"].x + feature_updates

        return data

    def message(self, x_j):
        """Pass through deltas from triple nodes."""
        return x_j


class SecondPeakScatter(MessagePassing):
    """
    Propagate deltas from triple nodes to peaks in second position.

    Uses edge type: ("Peak", "prop_second", triple_type) with reversed flow
    Reads delta attributes from triple nodes:
        - delta_second_features: [n, feature_dim] feature deltas
    Updates Peak nodes:
        - Peak.x with feature deltas (only updates feature portion)
    """

    def __init__(self, triple_type: str):
        """
        Initialize SecondPeakScatter.

        Args:
            triple_type: Name of source triple node type
            config: ModelConfig with dimension settings
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
            Updated HeteroData with Peak.x modified (features only)
        """
        # Get delta attributes from triple nodes (features only)
        delta_features = data[self.triple_type].delta_second_features  # [n, feature_dim]

        # Get edge indices for this scatter operation
        edge_index = data[self.edge_type].edge_index

        # Determine number of target nodes (Peak nodes - targets when using reversed flow)
        num_peaks = data["Peak"].x.size(0)

        # Propagate feature deltas (with reversed flow, size is (target, source))
        feature_updates = self.propagate(
            edge_index, x=delta_features, size=(num_peaks, delta_features.size(0))
        )
        data["Peak"].x = data["Peak"].x + feature_updates

        return data

    def message(self, x_j):
        """Pass through deltas from triple nodes."""
        return x_j


class NoeScatter(MessagePassing):
    """
    Propagate deltas from triple nodes to NOE constraint nodes.

    Uses edge type: ("Noe", "prop_noe", triple_type) with reversed flow
    Reads delta attributes from triple nodes:
        - delta_noe_features: [n, feature_dim] feature deltas
    Updates Noe nodes:
        - Noe.x with feature deltas (entire .x is updated since NOEs have no flags)
    """

    def __init__(self, triple_type: str):
        """
        Initialize NoeScatter.

        Args:
            triple_type: Name of source triple node type
            config: ModelConfig with dimension settings
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
            Updated HeteroData with Noe.x modified
        """
        # Get delta attributes from triple nodes (features only)
        delta_features = data[self.triple_type].delta_noe_features  # [n, feature_dim]

        # Get edge indices for this scatter operation
        edge_index = data[self.edge_type].edge_index

        # Determine number of target nodes (Noe nodes - targets when using reversed flow)
        num_noes = data["Noe"].x.size(0)

        # Propagate feature deltas (with reversed flow, size is (target, source))
        feature_updates = self.propagate(
            edge_index, x=delta_features, size=(num_noes, delta_features.size(0))
        )
        data["Noe"].x = data["Noe"].x + feature_updates

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

    This triple type handles (Residue, Residue, Noe) relationships.
    Updates only .x features (no coordinate or shift updates).

    Components:
        - FirstResidueGather: Extract from first residue
        - SecondResidueGather: Extract from second residue
        - NoeGather: Extract from NOE constraint
        - ResidueUpdate: Compute feature deltas
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

        # Instantiate gather operations (pass config for dimensions)
        self.first_gather = FirstResidueGather(triple_type)
        self.second_gather = SecondResidueGather(triple_type)
        self.noe_gather = NoeGather(triple_type)

        # Instantiate update operation
        self.update = ResidueUpdate(triple_type, device, config)

        # Instantiate scatter operations (pass config for dimensions)
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

    This triple type handles (Residue, Peak, Noe) relationships. Updates
    features only, NO coordinate or shift updates.

    Components:
        - FirstResidueGather: Extract from first residue
        - SecondPeakGather: Extract from second peak
        - NoeGather: Extract from NOE constraint
        - PeakUpdate: Compute feature deltas only
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

        # Instantiate gather operations (pass config for dimensions)
        self.first_gather = FirstResidueGather(triple_type)
        self.second_gather = SecondPeakGather(triple_type)
        self.noe_gather = NoeGather(triple_type)

        # Instantiate update operation
        self.update = PeakUpdate(triple_type, device, config)

        # Instantiate scatter operations (pass config for dimensions)
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

    This triple type handles (Peak, Residue, Noe) relationships. Updates
    features only, NO coordinate or shift updates.

    Components:
        - FirstPeakGather: Extract from first peak
        - SecondResidueGather: Extract from second residue
        - NoeGather: Extract from NOE constraint
        - PeakUpdate: Compute feature deltas only
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

        # Instantiate gather operations (pass config for dimensions)
        self.first_gather = FirstPeakGather(triple_type)
        self.second_gather = SecondResidueGather(triple_type)
        self.noe_gather = NoeGather(triple_type)

        # Instantiate update operation
        self.update = PeakUpdate(triple_type, device, config)

        # Instantiate scatter operations (pass config for dimensions)
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

    This triple type handles (Peak, Peak, Noe) relationships. Updates
    features only, NO coordinate or shift updates.

    Components:
        - FirstPeakGather: Extract from first peak
        - SecondPeakGather: Extract from second peak
        - NoeGather: Extract from NOE constraint
        - PeakUpdate: Compute feature deltas only
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

        # Instantiate gather operations (pass config for dimensions)
        self.first_gather = FirstPeakGather(triple_type)
        self.second_gather = SecondPeakGather(triple_type)
        self.noe_gather = NoeGather(triple_type)

        # Instantiate update operation
        self.update = PeakUpdate(triple_type, device, config)

        # Instantiate scatter operations (pass config for dimensions)
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


def calc_res_distance(x1, x2):
    """
    Calculate relative distance and squared distance between residues.

    Uses IMMUTABLE .xyz coordinates (not .x attributes).

    Args:
        x1: First residue coordinates [n, 3] (from .xyz)
        x2: Second residue coordinates [n, 3] (from .xyz)

    Returns:
        Tuple of (rel_dist, dist_squared)
            rel_dist: Relative distance vector [n, 3]
            dist_squared: Squared distance scalar [n, 1]
    """
    rel_dist = x1 - x2  # [n, 3]
    dist_squared = torch.norm(rel_dist, dim=-1, keepdim=True) ** 2  # [n, 1]
    return dist_squared
