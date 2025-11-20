"""
Modular triple-based graph message passing components.

This module implements a three-stage message passing pipeline using explicit,
modular components instead of conditional logic:

Architecture:
1. Gather Operations (2 classes): Extract features from source nodes to triple nodes
   - GatherToTriple: Generic parameterized gather (handles Residue/Peak, first/second)
   - NoeGather: Specialized NOE constraint gather
2. Update Operations (2 classes): Compute deltas via MLPs on triple nodes
3. Scatter Operations (2 classes): Propagate deltas back to source nodes
   - ScatterFromTriple: Generic parameterized scatter (handles Residue/Peak, first/second)
   - NoeScatter: Specialized NOE constraint scatter
4. Triple Composition (4 classes): Wire gather/update/scatter for each triple type
5. Layer Orchestration (in network.py): Call all 4 triple types explicitly

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

Attribute Structure
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

from .mlp import MLP


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


class GatherToTriple(MessagePassing):
    """
    Generic gather operation that extracts features from source nodes to triple nodes.

    Supports both Residue and Peak node types in first or second position.
    When has_coords=True (Residue nodes), propagates both coordinates and features.
    When has_coords=False (Peak nodes), propagates features only.

    Uses edge type: (node_type, f"prop_{position}", triple_type)
    Sets attributes on triple nodes:
        - {position}_coords: [n, 3] coordinates (only if has_coords=True)
        - {position}_features: [n, feature_dim] embedded features
    """

    def __init__(self, node_type: str, triple_type: str, position: str, has_coords: bool):
        """
        Initialize GatherToTriple.

        Args:
            node_type: Source node type ("Residue" or "Peak")
            triple_type: Target triple node type
            position: Position in triple ("first" or "second")
            has_coords: Whether to propagate coordinates (True for Residue, False for Peak)
        """
        super().__init__(aggr="mean")
        self.node_type = node_type
        self.triple_type = triple_type
        self.position = position
        self.has_coords = has_coords
        self.edge_type = (node_type, f"prop_{position}", triple_type)

    def forward(self, data):
        """
        Extract features from source nodes to triple nodes.

        Args:
            data: HeteroData graph with source nodes and gather edges

        Returns:
            Updated HeteroData with {position}_coords (if has_coords) and {position}_features set
        """
        # Extract embedded features from .x
        features = data[self.node_type].x  # [n, embed_dim]

        # Get edge indices for this gather operation
        edge_index = data[self.edge_type].edge_index

        # Determine number of target nodes (triple nodes)
        num_triples = data[self.triple_type].x.size(0)

        # Propagate coordinates if this node type has them (Residue only)
        if self.has_coords:
            coords = data[self.node_type].xyz  # [n, 3]
            setattr(
                data[self.triple_type],
                f"{self.position}_coords",
                self.propagate(edge_index, x=coords, size=(coords.size(0), num_triples))
            )

        # Propagate features (all node types)
        setattr(
            data[self.triple_type],
            f"{self.position}_features",
            self.propagate(edge_index, x=features, size=(features.size(0), num_triples))
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

        # Calculate input/output sizes from config
        feature_dim = config.shared.embed_dim
        input_size = residue_update_input_size(feature_dim)
        output_size = residue_update_output_size(feature_dim)

        # Store dimensions for later use
        self.feature_dim = feature_dim

        # Pre-normalization layers for inputs (pre-norm pattern)
        self.norm_first_features = nn.LayerNorm(feature_dim).to(device)
        self.norm_second_features = nn.LayerNorm(feature_dim).to(device)
        self.norm_noe_features = nn.LayerNorm(feature_dim).to(device)

        # Build MLP (no internal LayerNorm - follows pre-norm pattern)
        self.mlp = MLP(input_size, output_size, config.message_mlp, device)

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

        # Apply pre-normalization to gathered features
        first_features_norm = self.norm_first_features(first_features)
        second_features_norm = self.norm_second_features(second_features)
        noe_features_norm = self.norm_noe_features(noe_features)

        # Concatenate all features for MLP input
        # Input: dist_squared + 3*feature_dim (normalized features)
        mlp_input = torch.cat(
            [
                dist_squared,  # [n, 1]
                first_features_norm,  # [n, feature_dim]
                second_features_norm,  # [n, feature_dim]
                noe_features_norm,  # [n, feature_dim]
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

        # Calculate input/output sizes from config
        feature_dim = config.shared.embed_dim
        input_size = peak_update_input_size(feature_dim)
        output_size = peak_update_output_size(feature_dim)

        # Store dimensions for later use
        self.feature_dim = feature_dim

        # Pre-normalization layers for inputs (pre-norm pattern)
        self.norm_first_features = nn.LayerNorm(feature_dim).to(device)
        self.norm_second_features = nn.LayerNorm(feature_dim).to(device)
        self.norm_noe_features = nn.LayerNorm(feature_dim).to(device)

        # Build MLP (no internal LayerNorm - follows pre-norm pattern)
        self.mlp = MLP(input_size, output_size, config.message_mlp, device)

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

        # Apply pre-normalization to gathered features
        first_features_norm = self.norm_first_features(first_features)
        second_features_norm = self.norm_second_features(second_features)
        noe_features_norm = self.norm_noe_features(noe_features)

        # Concatenate all features for MLP input (NO distance calculations for peaks)
        # Input: 3*feature_dim (normalized features)
        mlp_input = torch.cat(
            [
                first_features_norm,  # [n, feature_dim]
                second_features_norm,  # [n, feature_dim]
                noe_features_norm,  # [n, feature_dim]
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


class ScatterFromTriple(MessagePassing):
    """
    Generic scatter operation that propagates deltas from triple nodes back to source nodes.

    Supports both Residue and Peak node types in first or second position.
    All scatter operations are identical - they only differ in edge type and target node type.

    Uses edge type: (node_type, f"prop_{position}", triple_type) with reversed flow
    Reads delta attributes from triple nodes:
        - delta_{position}_features: [n, feature_dim] feature deltas
    Updates target nodes:
        - {node_type}.x with feature deltas
    """

    def __init__(self, node_type: str, triple_type: str, position: str):
        """
        Initialize ScatterFromTriple.

        Args:
            node_type: Target node type ("Residue" or "Peak")
            triple_type: Source triple node type
            position: Position in triple ("first" or "second")
        """
        super().__init__(aggr="mean", flow="target_to_source")
        self.node_type = node_type
        self.triple_type = triple_type
        self.position = position
        self.edge_type = (node_type, f"prop_{position}", triple_type)

    def forward(self, data):
        """
        Propagate deltas from triple nodes to target nodes.

        Args:
            data: HeteroData with delta attributes on triple nodes

        Returns:
            Updated HeteroData with {node_type}.x modified
        """
        # Get delta attributes from triple nodes
        delta_features = getattr(data[self.triple_type], f"delta_{self.position}_features")

        # Get edge indices for this scatter operation
        edge_index = data[self.edge_type].edge_index

        # Determine number of target nodes (targets when using reversed flow)
        num_targets = data[self.node_type].x.size(0)

        # Propagate feature deltas (with reversed flow, size is (target, source))
        feature_updates = self.propagate(
            edge_index, x=delta_features, size=(num_targets, delta_features.size(0))
        )
        data[self.node_type].x = data[self.node_type].x + feature_updates

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
        - GatherToTriple (first, Residue, has_coords=True): Extract from first residue
        - GatherToTriple (second, Residue, has_coords=True): Extract from second residue
        - NoeGather: Extract from NOE constraint
        - ResidueUpdate: Compute feature deltas
        - ScatterFromTriple (first, Residue): Propagate to first residue
        - ScatterFromTriple (second, Residue): Propagate to second residue
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
        self.first_gather = GatherToTriple("Residue", triple_type, "first", has_coords=True)
        self.second_gather = GatherToTriple("Residue", triple_type, "second", has_coords=True)
        self.noe_gather = NoeGather(triple_type)

        # Instantiate update operation
        self.update = ResidueUpdate(triple_type, device, config)

        # Instantiate scatter operations
        self.first_scatter = ScatterFromTriple("Residue", triple_type, "first")
        self.second_scatter = ScatterFromTriple("Residue", triple_type, "second")
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
        - GatherToTriple (first, Residue, has_coords=True): Extract from first residue
        - GatherToTriple (second, Peak, has_coords=False): Extract from second peak
        - NoeGather: Extract from NOE constraint
        - PeakUpdate: Compute feature deltas only
        - ScatterFromTriple (first, Residue): Propagate to first residue
        - ScatterFromTriple (second, Peak): Propagate to second peak
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
        self.first_gather = GatherToTriple("Residue", triple_type, "first", has_coords=True)
        self.second_gather = GatherToTriple("Peak", triple_type, "second", has_coords=False)
        self.noe_gather = NoeGather(triple_type)

        # Instantiate update operation
        self.update = PeakUpdate(triple_type, device, config)

        # Instantiate scatter operations
        self.first_scatter = ScatterFromTriple("Residue", triple_type, "first")
        self.second_scatter = ScatterFromTriple("Peak", triple_type, "second")
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
        - GatherToTriple (first, Peak, has_coords=False): Extract from first peak
        - GatherToTriple (second, Residue, has_coords=True): Extract from second residue
        - NoeGather: Extract from NOE constraint
        - PeakUpdate: Compute feature deltas only
        - ScatterFromTriple (first, Peak): Propagate to first peak
        - ScatterFromTriple (second, Residue): Propagate to second residue
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
        self.first_gather = GatherToTriple("Peak", triple_type, "first", has_coords=False)
        self.second_gather = GatherToTriple("Residue", triple_type, "second", has_coords=True)
        self.noe_gather = NoeGather(triple_type)

        # Instantiate update operation
        self.update = PeakUpdate(triple_type, device, config)

        # Instantiate scatter operations
        self.first_scatter = ScatterFromTriple("Peak", triple_type, "first")
        self.second_scatter = ScatterFromTriple("Residue", triple_type, "second")
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
        - GatherToTriple (first, Peak, has_coords=False): Extract from first peak
        - GatherToTriple (second, Peak, has_coords=False): Extract from second peak
        - NoeGather: Extract from NOE constraint
        - PeakUpdate: Compute feature deltas only
        - ScatterFromTriple (first, Peak): Propagate to first peak
        - ScatterFromTriple (second, Peak): Propagate to second peak
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
        self.first_gather = GatherToTriple("Peak", triple_type, "first", has_coords=False)
        self.second_gather = GatherToTriple("Peak", triple_type, "second", has_coords=False)
        self.noe_gather = NoeGather(triple_type)

        # Instantiate update operation
        self.update = PeakUpdate(triple_type, device, config)

        # Instantiate scatter operations
        self.first_scatter = ScatterFromTriple("Peak", triple_type, "first")
        self.second_scatter = ScatterFromTriple("Peak", triple_type, "second")
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
