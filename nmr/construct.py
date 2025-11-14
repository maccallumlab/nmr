"""
Graph construction utilities for NMR assignment.

This module constructs heterogeneous graphs from NMR data for chemical shift assignment
using Graph Neural Networks. The graph structure includes:

Node Types:
- Residue: Protein residues with coordinates [x,y,z] and predicted shifts [H,N]
- Peak: Observed chemical shifts [H,N]
- Noe: NOE distance constraints [N, H', H"]
- Triple nodes: Four types representing different relationship configurations:
  * ResidueResidueNoeTriple: (Residue, Residue, Noe) - Updates coordinates and shifts
  * ResiduePeakNoeTriple: (Residue, Peak, Noe) - Updates shifts only
  * PeakResidueNoeTriple: (Peak, Residue, Noe) - Updates shifts only
  * PeakPeakNoeTriple: (Peak, Peak, Noe) - Updates shifts only
- Value aggregation nodes: VALUE_NOE, VALUE_SHIFT, VALUE_RES

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


def construct_graph(history: Dict[str, Any], device: torch.device | str) -> HeteroData:
    """
    Constructs a complete heterogeneous graph from a history state.

    Args:
        history: Dictionary containing coordinates, shifts, NOEs, and assignment state
        device: Device to place tensors on ('cpu' or 'cuda')

    Returns:
        HeteroData graph with all nodes and edges constructed
    """
    data = construct_node_data(history, device)
    data = construct_edges(data, device)
    return data


def construct_node_data(
    histories: Dict[str, Any], device: torch.device | str
) -> HeteroData:
    """
    Builds graph nodes from input histories.

    Creates nodes for Residue, Peak, Noe, triple types, and value aggregation nodes.

    Args:
        histories: Dictionary containing coordinates, shifts, NOEs, and assignment state
        device: Device to place tensors on ('cpu' or 'cuda')

    Returns:
        HeteroData graph with all nodes constructed
    """
    data = HeteroData()
    data = _construct_data_nodes(data, histories, device)
    data = _construct_triple_nodes(data, device)
    data = _construct_value_nodes(data, device)
    data = _construct_node_features(data, histories)
    data.shift_to_assign = torch.tensor(
        [int(histories["shift_to_assign"])], dtype=torch.long, device=device
    )
    return data


def construct_edges(data: HeteroData, device: torch.device | str) -> HeteroData:
    """
    Constructs all edge indices for the heterogeneous graph.

    Creates bidirectional propagation edges, value aggregation edges, and policy edges
    for all triple configurations.

    Args:
        data: HeteroData graph with nodes already constructed
        device: Device to place tensors on ('cpu' or 'cuda')

    Returns:
        HeteroData graph with all edges constructed
    """
    num_noe = len(data["Noe"].x)
    num_peak = len(data["Peak"].x)
    num_residue = len(data["Residue"].x)

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
    return data


def _construct_data_nodes(
    data: HeteroData, histories: Dict[str, Any], device: torch.device | str
) -> HeteroData:
    """
    Constructs Noe, Peak, and Residue nodes with initial features.

    Args:
        data: HeteroData graph to add nodes to
        histories: Dictionary containing coordinates, shifts, and NOEs
        device: Device to place tensors on

    Returns:
        HeteroData with data nodes added
    """
    data["Noe"].x = torch.tensor(histories["noes"], dtype=torch.float32, device=device)
    data["Noe"].f = torch.zeros(
        (len(histories["noes"]), 2), dtype=torch.float32, device=device
    )
    data["Peak"].x = torch.tensor(
        histories["obs_chemical_shifts"], dtype=torch.float32, device=device
    )
    data["Peak"].f = torch.zeros(
        (len(histories["obs_chemical_shifts"]), 2),
        dtype=torch.float32,
        device=device,
    )
    # Residue.x = [coordinates (3), predicted_shifts (2)] = [5 features total]
    # Coordinates already contains [x, y, z, H, N] as 5 features
    data["Residue"].x = torch.tensor(
        histories["coordinates"], dtype=torch.float32, device=device
    )
    data["Residue"].f = torch.zeros(
        (len(histories["coordinates"]), 2), dtype=torch.float32, device=device
    )
    return data


def _construct_triple_nodes(data: HeteroData, device: torch.device | str) -> HeteroData:
    """
    Constructs triple nodes for all four triple types.

    Triple nodes act as intermediaries for message passing between residues, peaks, and NOEs.
    Each triple type represents a different configuration of source nodes.

    Args:
        data: HeteroData graph to add triple nodes to
        device: Device to place tensors on

    Returns:
        HeteroData with triple nodes added
    """
    num_noe = len(data["Noe"].x)
    num_peak = len(data["Peak"].x)
    num_residue = len(data["Residue"].x)

    # ResidueResidueNoeTriple: All combinations of (residue_i, residue_j, noe_k)
    num_res_res_noe = num_residue * num_residue * num_noe
    data["ResidueResidueNoeTriple"].x = torch.zeros(
        (num_res_res_noe, 1), dtype=torch.float32, device=device
    )
    data["ResidueResidueNoeTriple"].f = torch.zeros(
        (num_res_res_noe, 2), dtype=torch.float32, device=device
    )

    # ResiduePeakNoeTriple: All combinations of (residue_i, peak_j, noe_k)
    num_res_peak_noe = num_residue * num_peak * num_noe
    data["ResiduePeakNoeTriple"].x = torch.zeros(
        (num_res_peak_noe, 1), dtype=torch.float32, device=device
    )
    data["ResiduePeakNoeTriple"].f = torch.zeros(
        (num_res_peak_noe, 2), dtype=torch.float32, device=device
    )

    # PeakResidueNoeTriple: All combinations of (peak_i, residue_j, noe_k)
    num_peak_res_noe = num_peak * num_residue * num_noe
    data["PeakResidueNoeTriple"].x = torch.zeros(
        (num_peak_res_noe, 1), dtype=torch.float32, device=device
    )
    data["PeakResidueNoeTriple"].f = torch.zeros(
        (num_peak_res_noe, 2), dtype=torch.float32, device=device
    )

    # PeakPeakNoeTriple: All combinations of (peak_i, peak_j, noe_k)
    num_peak_peak_noe = num_peak * num_peak * num_noe
    data["PeakPeakNoeTriple"].x = torch.zeros(
        (num_peak_peak_noe, 1), dtype=torch.float32, device=device
    )
    data["PeakPeakNoeTriple"].f = torch.zeros(
        (num_peak_peak_noe, 2), dtype=torch.float32, device=device
    )

    return data


def _construct_value_nodes(data: HeteroData, device: torch.device | str) -> HeteroData:
    """
    Constructs value aggregation nodes for value prediction head.

    Value nodes aggregate information from Noe, Peak, and Residue nodes
    to predict the quality of the current assignment state.

    Args:
        data: HeteroData graph to add value nodes to
        device: Device to place tensors on

    Returns:
        HeteroData with value nodes added
    """
    data["VALUE_NOE"].x = torch.zeros((1, 1), dtype=torch.float32, device=device)
    data["VALUE_SHIFT"].x = torch.zeros((1, 1), dtype=torch.float32, device=device)
    data["VALUE_RES"].x = torch.zeros((1, 1), dtype=torch.float32, device=device)
    return data


def _construct_node_features(data: HeteroData, histories: Dict[str, Any]) -> HeteroData:
    """
    Initializes node features based on assignment state.

    Sets feature flags on Peak and Residue nodes to indicate which nodes
    are currently being assigned or have been assigned.

    Args:
        data: HeteroData graph with nodes
        histories: Dictionary containing assignment state

    Returns:
        HeteroData with node features initialized
    """
    # Mark the peak being assigned
    shift_to_assign = int(histories["shift_to_assign"])
    data["Peak"].f[shift_to_assign, 0] = 1.0

    # Mark assigned peaks and residues
    for shift_idx, residue_idx in histories["assignments"].items():
        data["Peak"].f[int(shift_idx), 1] = 1.0
        data["Residue"].f[int(residue_idx), 1] = 1.0

    # Add edges for existing assignments (peak -> residue mappings)
    # Always add the edge type, even if empty, for consistent graph structure
    if histories["assignments"]:
        peak_indices = list(histories["assignments"].keys())
        residue_indices = list(histories["assignments"].values())
        data["Peak", "assigned_to", "Residue"].edge_index = torch.tensor(
            [peak_indices, residue_indices],
            dtype=torch.long,
            device=data["Peak"].x.device,
        )
    else:
        # Create empty edge_index with shape [2, 0]
        data["Peak", "assigned_to", "Residue"].edge_index = torch.empty(
            (2, 0), dtype=torch.long, device=data["Peak"].x.device
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
    noe_indices = torch.arange(num_noe, device=device).repeat_interleave(source1 * source2)

    # Source1 indices: [0,0,...,0, 1,1,...,1, ...] (each repeated source2 times, pattern repeats num_noe times)
    source1_indices = torch.arange(source1, device=device).repeat_interleave(source2).repeat(num_noe)

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
