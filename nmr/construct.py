"""
Graph construction utilities for NMR assignment.

This module constructs heterogeneous graphs from NMR data for chemical shift assignment
using Graph Neural Networks. The graph structure includes:

Node Types:
- Residue: Protein residues with raw data (.xyz, .shifts, .flags) and working features (.x)
- Peak: Observed chemical shifts with raw data (.shifts, .flags) and working features (.x)
- Noe: NOE distance constraints with raw data (.shifts) and working features (.x)
- Triple nodes: Four types representing different relationship configurations:
  * ResidueResidueNoeTriple: (Residue, Residue, Noe) - Updates coordinates and shifts
  * ResiduePeakNoeTriple: (Residue, Peak, Noe) - Updates shifts only
  * PeakResidueNoeTriple: (Peak, Residue, Noe) - Updates shifts only
  * PeakPeakNoeTriple: (Peak, Peak, Noe) - Updates shifts only
- Value aggregation nodes: VALUE_NOE, VALUE_SHIFT, VALUE_RES

Attribute Structure:
- Raw data attributes (IMMUTABLE, set once during construction):
  * Residue.xyz [n, 3]: cartesian coordinates
  * Residue.shifts [n, 2]: predicted chemical shifts
  * Residue.flags [n, 1]: assignment status (previously assigned)
  * Peak.shifts [n, 2]: observed chemical shifts
  * Peak.flags [n, 2]: assignment status (to be assigned, previously assigned)
  * Noe.shifts [n, 3]: NOE shift values
- Working feature attributes (updated during message passing):
  * .x for all node types: embedded features created by EmbedFeatures layer

Edge Types:
- Bidirectional propagation edges: Used for both gather (source → triple) and scatter (triple → source)
  * (source_node, "prop_first", triple_type) - First position connections
  * (source_node, "prop_second", triple_type) - Second position connections
  * ("Noe", "prop_noe", triple_type) - NOE constraint connections
  Direction is controlled by flow parameter in MessagePassing layers
- Value aggregation edges: Aggregate features for value prediction
  * (node_type, "aggregate", value_node_type)
"""

from typing import Any, Dict, Tuple

import torch
from torch_geometric.data import HeteroData
from time import time

from nmr.models.network import ModelConfig


def construct_graph(
    history: Dict[str, Any], device: torch.device | str, config: ModelConfig
) -> HeteroData:
    """
    Constructs a complete heterogeneous graph from a history state.

    Args:
        history: Dictionary containing coordinates, shifts, NOEs, and assignment state
        device: Device to place tensors on ('cpu' or 'cuda')
        config: ModelConfig containing embed_dim for .x initialization

    Returns:
        HeteroData graph with all nodes and edges constructed
    """
    data = construct_node_data(history, device, config)
    data = construct_edges(data, device)
    return data


def construct_node_data(
    histories: Dict[str, Any], device: torch.device | str, config
) -> HeteroData:
    """
    Builds graph nodes from input histories.

    Creates nodes for Residue, Peak, Noe, triple types, and value aggregation nodes.

    Args:
        histories: Dictionary containing coordinates, shifts, NOEs, and assignment state
        device: Device to place tensors on ('cpu' or 'cuda')
        config: ModelConfig containing embed_dim for .x initialization

    Returns:
        HeteroData graph with all nodes constructed
    """
    data = HeteroData()
    data = _construct_data_nodes(data, histories, device, config)
    data = _construct_triple_nodes(data, device, config)
    data = _construct_value_nodes(data, device, config)
    data = _construct_node_features(data, histories)
    data.shift_to_assign = torch.tensor(
        [int(histories["shift_to_assign"])], dtype=torch.long, device=device
    )
    return data


def construct_edges(data: HeteroData, device: torch.device | str) -> HeteroData:
    """
    Constructs all edge indices for the heterogeneous graph.

    Creates bidirectional propagation edges, value aggregation edges, and policy edges
    for all triple configurations, plus transformer attention edges.

    Args:
        data: HeteroData graph with nodes already constructed
        device: Device to place tensors on ('cpu' or 'cuda')

    Returns:
        HeteroData graph with all edges constructed
    """
    num_noe = len(data["Noe"].shifts)
    num_peak = len(data["Peak"].shifts)
    num_residue = len(data["Residue"].xyz)

    # Add all edge types for each triple configuration
    _add_triple_edges(
        data, ("Residue", "Residue", "Noe"), num_noe, num_peak, num_residue, device
    )
    _add_triple_edges(
        data, ("Residue", "Peak", "Noe"), num_noe, num_peak, num_residue, device
    )
    _add_triple_edges(
        data, ("Peak", "Residue", "Noe"), num_noe, num_peak, num_residue, device
    )
    _add_triple_edges(
        data, ("Peak", "Peak", "Noe"), num_noe, num_peak, num_residue, device
    )
    _add_value_aggregation_edges(data, num_noe, num_peak, num_residue, device)
    _add_transformer_attention_edges(data, num_noe, num_peak, num_residue, device)
    return data


def _construct_data_nodes(
    data: HeteroData,
    histories: Dict[str, Any],
    device: torch.device | str,
    config: ModelConfig,
) -> HeteroData:
    """
    Constructs Noe, Peak, and Residue nodes with raw data attributes.

    Creates IMMUTABLE raw data attributes:
    - Residue.xyz [n, 3]: coordinates
    - Residue.shifts [n, 2]: predicted shift values
    - Peak.shifts [n, 2]: observed shift values
    - Noe.shifts [n, 3]: NOE shift values

    Note: .flags attributes are set in _construct_node_features based on assignment state

    Args:
        data: HeteroData graph to add nodes to
        histories: Dictionary containing coordinates, shifts, and NOEs
        device: Device to place tensors on
        config: ModelConfig containing embed_dim for .x initialization

    Returns:
        HeteroData with data nodes added
    """
    # Extract coordinates from histories
    # histories["coordinates"] is [x, y, z, H, N] format
    coords = torch.tensor(histories["coordinates"], dtype=torch.float32, device=device)

    # Residue raw data attributes (IMMUTABLE after this initialization)
    # Extract and normalize coordinates to zero mean, unit variance
    raw_coords = coords[:, 0:3]  # [n, 3]
    coords_mean = raw_coords.mean(dim=0, keepdim=True)
    coords_std = raw_coords.std(dim=0, keepdim=True) + 1e-5
    data["Residue"].xyz = (
        raw_coords - coords_mean
    ) / coords_std  # Normalized coordinates
    data["Residue"].shifts = coords[
        :, 3:5
    ]  # [n, 2] - predicted shifts [H, N] (raw, not normalized yet)

    # Peak raw data attributes (IMMUTABLE)
    peak_shifts = torch.tensor(
        histories["obs_chemical_shifts"], dtype=torch.float32, device=device
    )  # [n, 2] - observed shifts [H, N]
    data["Peak"].shifts = peak_shifts

    # NOE raw data attributes (IMMUTABLE)
    noe_shifts = torch.tensor(
        histories["noes"], dtype=torch.float32, device=device
    )  # [n, 3] - NOE shifts [N, H', H"]
    data["Noe"].shifts = noe_shifts

    # Initialize .x attributes to zeros so PyG can batch/unbatch them
    # EmbedFeatures layer will overwrite these with actual embeddings
    embed_dim = config.embed.embed_dim
    data["Residue"].x = torch.zeros(
        (len(coords), embed_dim), dtype=torch.float32, device=device
    )
    data["Peak"].x = torch.zeros(
        (len(peak_shifts), embed_dim), dtype=torch.float32, device=device
    )
    data["Noe"].x = torch.zeros(
        (len(noe_shifts), embed_dim), dtype=torch.float32, device=device
    )

    return data


def _construct_triple_nodes(
    data: HeteroData, device: torch.device | str, config: ModelConfig
) -> HeteroData:
    """
    Constructs triple nodes for all four triple types.

    Triple nodes act as intermediaries for message passing between residues, peaks, and NOEs.
    Each triple type represents a different configuration of source nodes.

    Args:
        data: HeteroData graph to add triple nodes to
        device: Device to place tensors on
        config: ModelConfig containing embed_dim for .x initialization

    Returns:
        HeteroData with triple nodes added
    """
    num_noe = len(data["Noe"].shifts)
    num_peak = len(data["Peak"].shifts)
    num_residue = len(data["Residue"].xyz)
    embed_dim = config.embed.embed_dim

    # ResidueResidueNoeTriple: All combinations of (residue_i, residue_j, noe_k)
    num_res_res_noe = num_residue * num_residue * num_noe
    data["ResidueResidueNoeTriple"].x = torch.zeros(
        (num_res_res_noe, embed_dim), dtype=torch.float32, device=device
    )

    # ResiduePeakNoeTriple: All combinations of (residue_i, peak_j, noe_k)
    num_res_peak_noe = num_residue * num_peak * num_noe
    data["ResiduePeakNoeTriple"].x = torch.zeros(
        (num_res_peak_noe, embed_dim), dtype=torch.float32, device=device
    )

    # PeakResidueNoeTriple: All combinations of (peak_i, residue_j, noe_k)
    num_peak_res_noe = num_peak * num_residue * num_noe
    data["PeakResidueNoeTriple"].x = torch.zeros(
        (num_peak_res_noe, embed_dim), dtype=torch.float32, device=device
    )

    # PeakPeakNoeTriple: All combinations of (peak_i, peak_j, noe_k)
    num_peak_peak_noe = num_peak * num_peak * num_noe
    data["PeakPeakNoeTriple"].x = torch.zeros(
        (num_peak_peak_noe, embed_dim), dtype=torch.float32, device=device
    )

    return data


def _construct_value_nodes(
    data: HeteroData, device: torch.device | str, config: ModelConfig
) -> HeteroData:
    """
    Constructs value aggregation nodes for value prediction head.

    Value nodes aggregate information from Noe, Peak, and Residue nodes
    to predict the quality of the current assignment state.

    Args:
        data: HeteroData graph to add value nodes to
        device: Device to place tensors on
        config: ModelConfig containing embed_dim for .x initialization

    Returns:
        HeteroData with value nodes added
    """
    embed_dim = config.embed.embed_dim
    data["VALUE_NOE"].x = torch.zeros(
        (1, embed_dim), dtype=torch.float32, device=device
    )
    data["VALUE_SHIFT"].x = torch.zeros(
        (1, embed_dim), dtype=torch.float32, device=device
    )
    data["VALUE_RES"].x = torch.zeros(
        (1, embed_dim), dtype=torch.float32, device=device
    )
    return data


def _construct_node_features(data: HeteroData, histories: Dict[str, Any]) -> HeteroData:
    """
    Initializes node .flags attributes based on assignment state.

    Sets IMMUTABLE flag attributes:
    - Residue.flags [n, 1]: previously assigned flag
    - Peak.flags [n, 2]: (to be assigned flag, previously assigned flag)
    - NOE nodes have NO flags

    Args:
        data: HeteroData graph with nodes
        histories: Dictionary containing assignment state

    Returns:
        HeteroData with node flags initialized
    """
    num_residues = data["Residue"].xyz.shape[0]
    num_peaks = data["Peak"].shifts.shape[0]
    device = data["Residue"].xyz.device

    # Initialize flags tensors
    data["Residue"].flags = torch.zeros(
        (num_residues, 1), dtype=torch.float32, device=device
    )
    data["Peak"].flags = torch.zeros((num_peaks, 2), dtype=torch.float32, device=device)

    # Mark the peak being assigned
    shift_to_assign = int(histories["shift_to_assign"])
    data["Peak"].flags[shift_to_assign, 0] = 1.0  # to be assigned flag

    # Mark assigned peaks and residues
    for shift_idx, residue_idx in histories["assignments"].items():
        shift_idx = int(shift_idx)
        residue_idx = int(residue_idx)

        # Update flags
        data["Peak"].flags[shift_idx, 1] = 1.0  # previously assigned flag
        data["Residue"].flags[residue_idx, 0] = 1.0  # previously assigned flag

    # Add edges for existing assignments (peak -> residue mappings)
    # Always add the edge type, even if empty, for consistent graph structure
    if histories["assignments"]:
        peak_indices = list(histories["assignments"].keys())
        residue_indices = list(histories["assignments"].values())
        data["Peak", "assigned_to", "Residue"].edge_index = torch.tensor(
            [peak_indices, residue_indices],
            dtype=torch.long,
            device=device,
        )
    else:
        # Create empty edge_index with shape [2, 0]
        data["Peak", "assigned_to", "Residue"].edge_index = torch.empty(
            (2, 0), dtype=torch.long, device=device
        )

    return data


def _get_triple_edges(
    num_noe: int, source1: int, source2: int, device: torch.device | str
) -> torch.Tensor:
    """
    Generates edge indices for connecting triple nodes to their source nodes.

    Each triple node represents a combination of (NOE, source1, source2). This function
    creates the edge connectivity to link each triple back to its three constituent nodes.

    Args:
        num_noe: Number of NOE nodes
        source1: Number of nodes in first source type (Residue or Peak)
        source2: Number of nodes in second source type (Residue or Peak)
        device: Device to place tensors on

    Returns:
        Tensor of shape [2, 3, num_edges] where:
            - dim 0: [source_indices, triple_indices]
            - dim 1: [noe_position, first_position, second_position]
            - dim 2: all num_noe * source1 * source2 edge combinations

        For example, edges[0, 1, k] gives the source1 node index for triple k,
        and edges[1, 1, k] gives the triple node index k.
    """
    total_edges = num_noe * source1 * source2

    # Create source node indices for all triple combinations
    # Each triple (i, j, k) corresponds to edge index: i*(source1*source2) + j*source2 + k

    # NOE indices: [0,0,...,0, 1,1,...,1, 2,2,...,2, ...] (each repeated source1*source2 times)
    noe_indices = torch.arange(num_noe, device=device).repeat_interleave(
        source1 * source2
    )

    # Source1 indices: [0,0,...,0, 1,1,...,1, ...] (each repeated source2 times, pattern repeats num_noe times)
    source1_indices = (
        torch.arange(source1, device=device).repeat_interleave(source2).repeat(num_noe)
    )

    # Source2 indices: [0,1,2,...,source2-1, 0,1,2,...,source2-1, ...] (cycles continuously)
    source2_indices = torch.arange(source2, device=device).repeat(num_noe * source1)

    # Stack into source_nodes tensor [3, total_edges]
    source_nodes = torch.stack([noe_indices, source1_indices, source2_indices], dim=0)

    # Create target indices [0, 1, 2, ..., total_edges-1] repeated 3 times
    target_nodes = torch.arange(total_edges, device=device).unsqueeze(0).repeat(3, 1)

    return torch.stack([source_nodes, target_nodes], dim=0)


def _add_value_aggregation_edges(
    data: HeteroData,
    num_noe: int,
    num_peak: int,
    num_residue: int,
    device: torch.device | str,
) -> None:
    """
    Adds value aggregation edges from data nodes to value nodes.

    These edges allow the value prediction head to aggregate information
    from all Noe, Peak, and Residue nodes.

    Args:
        data: HeteroData graph to add edges to
        num_noe: Number of NOE nodes
        num_peak: Number of peak nodes
        num_residue: Number of residue nodes
        device: Device to place tensors on
    """
    peak = torch.arange(0, num_peak, device=device)
    noe = torch.arange(0, num_noe, device=device)
    resid = torch.arange(0, num_residue, device=device)

    peak_repeats = torch.zeros(num_peak, device=device).long()
    noe_repeats = torch.zeros(num_noe, device=device).long()
    resid_repeats = torch.zeros(num_residue, device=device).long()

    data["Peak", "SHIFT_extract", "VALUE_SHIFT"].edge_index = torch.stack(
        (peak, peak_repeats), dim=0
    )
    data["Noe", "aggregate", "VALUE_NOE"].edge_index = torch.stack(
        (noe, noe_repeats), dim=0
    )
    data["Residue", "RES_extract", "VALUE_RES"].edge_index = torch.stack(
        (resid, resid_repeats), dim=0
    )


def _add_triple_edges(
    data: HeteroData,
    triple_type: Tuple[str, str, str],
    num_noe: int,
    num_peak: int,
    num_residue: int,
    device: torch.device | str,
) -> None:
    """
    Constructs bidirectional propagation edges for a single triple type.

    Creates edges that can be traversed in both directions for message passing:
    - Gather direction: source → triple (default flow='source_to_target')
    - Scatter direction: triple → source (with flow='target_to_source')

    Triple types use descriptive names:
    - ResidueResidueNoeTriple: Residue-Residue-Noe configuration
    - ResiduePeakNoeTriple: Residue-Peak-Noe configuration
    - PeakResidueNoeTriple: Peak-Residue-Noe configuration
    - PeakPeakNoeTriple: Peak-Peak-Noe configuration

    Edge naming convention:
    - (source_node, "prop_first", triple_name) - First position bidirectional edges
    - (source_node, "prop_second", triple_name) - Second position bidirectional edges
    - ("Noe", "prop_noe", triple_name) - NOE bidirectional edges

    Args:
        data: HeteroData graph to add edges to
        triple_type: Tuple of (source1, source2, source3) node type names
        num_noe: Number of NOE nodes
        num_peak: Number of peak nodes
        num_residue: Number of residue nodes
        device: Device to place tensors on
    """
    source1, source2, _ = triple_type

    # Determine source counts based on node types
    source1_count = num_residue if source1 == "Residue" else num_peak
    source2_count = num_residue if source2 == "Residue" else num_peak

    # Generate triple name from types
    triple_name = f"{source1}{source2}NoeTriple"

    # Get edge indices for all combinations
    # edges shape: [2, 3, num_edges] where 2 = [source, target], 3 = [noe, first, second]
    edges = _get_triple_edges(num_noe, source1_count, source2_count, device)

    # Bidirectional propagation edges: source → triple (can be reversed for scatter)
    # First position (source1 ↔ triple)
    data[source1, "prop_first", triple_name].edge_index = torch.stack(
        [edges[0, 1], edges[1, 1]], dim=0
    )

    # Second position (source2 ↔ triple)
    data[source2, "prop_second", triple_name].edge_index = torch.stack(
        [edges[0, 2], edges[1, 2]], dim=0
    )

    # NOE position (Noe ↔ triple)
    data["Noe", "prop_noe", triple_name].edge_index = torch.stack(
        [edges[0, 0], edges[1, 0]], dim=0
    )


def _add_transformer_attention_edges(
    data: HeteroData,
    num_noe: int,
    num_peak: int,
    num_residue: int,
    device: torch.device | str,
) -> None:
    """
    Adds transformer attention edges for all attention operations.

    Creates fully connected edges for:
    - Residue self-attention
    - Peak self-attention
    - Residue-Peak cross-attention (both directions)
    - Residue-NOE cross-attention (both directions)
    - Peak-NOE cross-attention (both directions)

    Args:
        data: HeteroData graph to add edges to
        num_noe: Number of NOE nodes
        num_peak: Number of peak nodes
        num_residue: Number of residue nodes
        device: Device to place tensors on
    """
    # 1. Residue self-attention: all-to-all Residue connections
    if num_residue > 0:
        res_sources = torch.arange(num_residue, device=device).repeat_interleave(num_residue)
        res_targets = torch.arange(num_residue, device=device).repeat(num_residue)
        data["Residue", "res_res_attn", "Residue"].edge_index = torch.stack(
            [res_sources, res_targets], dim=0
        )

    # 2. Peak self-attention: all-to-all Peak connections
    if num_peak > 0:
        peak_sources = torch.arange(num_peak, device=device).repeat_interleave(num_peak)
        peak_targets = torch.arange(num_peak, device=device).repeat(num_peak)
        data["Peak", "peak_peak_attn", "Peak"].edge_index = torch.stack(
            [peak_sources, peak_targets], dim=0
        )

    # 3. Peak → Residue cross-attention: all Peaks to all Residues
    if num_peak > 0 and num_residue > 0:
        peak_sources = torch.arange(num_peak, device=device).repeat_interleave(num_residue)
        res_targets = torch.arange(num_residue, device=device).repeat(num_peak)
        data["Peak", "peak_res_attn", "Residue"].edge_index = torch.stack(
            [peak_sources, res_targets], dim=0
        )

    # 4. Residue → Peak cross-attention: all Residues to all Peaks
    if num_residue > 0 and num_peak > 0:
        res_sources = torch.arange(num_residue, device=device).repeat_interleave(num_peak)
        peak_targets = torch.arange(num_peak, device=device).repeat(num_residue)
        data["Residue", "res_peak_attn", "Peak"].edge_index = torch.stack(
            [res_sources, peak_targets], dim=0
        )

    # 5. Residue → NOE cross-attention: all Residues to all NOEs
    if num_residue > 0 and num_noe > 0:
        res_sources = torch.arange(num_residue, device=device).repeat_interleave(num_noe)
        noe_targets = torch.arange(num_noe, device=device).repeat(num_residue)
        data["Residue", "res_noe_attn", "Noe"].edge_index = torch.stack(
            [res_sources, noe_targets], dim=0
        )

    # 6. Peak → NOE cross-attention: all Peaks to all NOEs
    if num_peak > 0 and num_noe > 0:
        peak_sources = torch.arange(num_peak, device=device).repeat_interleave(num_noe)
        noe_targets = torch.arange(num_noe, device=device).repeat(num_peak)
        data["Peak", "peak_noe_attn", "Noe"].edge_index = torch.stack(
            [peak_sources, noe_targets], dim=0
        )

    # 7. NOE → Residue cross-attention: all NOEs to all Residues
    if num_noe > 0 and num_residue > 0:
        noe_sources = torch.arange(num_noe, device=device).repeat_interleave(num_residue)
        res_targets = torch.arange(num_residue, device=device).repeat(num_noe)
        data["Noe", "noe_res_attn", "Residue"].edge_index = torch.stack(
            [noe_sources, res_targets], dim=0
        )

    # 8. NOE → Peak cross-attention: all NOEs to all Peaks
    if num_noe > 0 and num_peak > 0:
        noe_sources = torch.arange(num_noe, device=device).repeat_interleave(num_peak)
        peak_targets = torch.arange(num_peak, device=device).repeat(num_noe)
        data["Noe", "noe_peak_attn", "Peak"].edge_index = torch.stack(
            [noe_sources, peak_targets], dim=0
        )
