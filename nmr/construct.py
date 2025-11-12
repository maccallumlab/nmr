"""Graph construction utilities for NMR assignment."""

from itertools import product
from typing import Any, Dict, Tuple

import numpy as np
import torch
from torch_geometric.data import HeteroData


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

    Args:
        data: HeteroData graph with nodes already constructed
        device: Device to place tensors on ('cpu' or 'cuda')

    Returns:
        HeteroData graph with all edges constructed
    """
    num_noe = len(data["NOE"].x)
    num_shift = len(data["SHIFT"].x)
    num_res = len(data["RES"].x)

    # Add all edge types
    _add_triple_edges(
        data, 0, ("RES", "RES", "NOE"), num_noe, num_shift, num_res, device
    )
    _add_triple_edges(
        data, 1, ("RES", "SHIFT", "NOE"), num_noe, num_shift, num_res, device
    )
    _add_triple_edges(
        data, 2, ("SHIFT", "RES", "NOE"), num_noe, num_shift, num_res, device
    )
    _add_triple_edges(
        data, 3, ("SHIFT", "SHIFT", "NOE"), num_noe, num_shift, num_res, device
    )
    _add_value_aggregation_edges(data, num_noe, num_shift, num_res, device)

    return data


def _construct_data_nodes(
    data: HeteroData, histories: Dict[str, Any], device: torch.device | str
) -> HeteroData:
    """Constructs NOE, SHIFT, and RES nodes with initial features."""
    data["NOE"].x = torch.tensor(histories["noes"], dtype=torch.float32, device=device)
    data["NOE"].f = torch.zeros(
        (len(histories["noes"]), 2), dtype=torch.float32, device=device
    )
    data["SHIFT"].x = torch.tensor(
        histories["obs_chemical_shifts"], dtype=torch.float32, device=device
    )
    data["SHIFT"].f = torch.zeros(
        (len(histories["obs_chemical_shifts"]), 2),
        dtype=torch.float32,
        device=device,
    )
    data["RES"].x = torch.tensor(
        histories["coordinates"], dtype=torch.float32, device=device
    )
    data["RES"].f = torch.zeros(
        (len(histories["coordinates"]), 2), dtype=torch.float32, device=device
    )
    return data


def _construct_node_features(data: HeteroData, histories: Dict[str, Any]) -> HeteroData:
    """Sets node features based on assignment status."""
    for i in range(len(data["SHIFT"].f)):
        # is the shift being assigned right now?
        if i == histories["shift_to_assign"]:
            data["SHIFT"].f[i, 0] = 1
        # has the shift already been assigned?
        if i in histories["assignments"].keys():
            data["SHIFT"].f[i, 1] = 1

    for i in range(len(data["RES"].f)):
        # has the residue been assigned?
        if i in histories["assignments"].values():
            data["RES"].f[i] = 1
    return data


def _construct_triple_nodes(data: HeteroData, device: torch.device | str) -> HeteroData:
    """
    Generates triple node types. Nothing is ever stored in these - they are simply placeholders to construct/use in edge types.
    """
    data["TRIPLE0"].x = torch.zeros((1, 1), dtype=torch.float32, device=device)
    data["TRIPLE1"].x = torch.zeros((1, 1), dtype=torch.float32, device=device)
    data["TRIPLE2"].x = torch.zeros((1, 1), dtype=torch.float32, device=device)
    data["TRIPLE3"].x = torch.zeros((1, 1), dtype=torch.float32, device=device)
    return data


def _construct_value_nodes(data: HeteroData, device: torch.device | str) -> HeteroData:
    """Constructs value aggregation nodes for NOE, SHIFT, and RES."""
    data["VALUE_NOE"].x = torch.zeros(
        1, 1, dtype=torch.float32, device=device
    )  # has to be num graphs batched?
    data["VALUE_SHIFT"].x = torch.zeros(1, 1, dtype=torch.float32, device=device)
    data["VALUE_RES"].x = torch.zeros(1, 1, dtype=torch.float32, device=device)
    return data


def _get_triple_edges(
    num_noe: int, source1: int, source2: int, device: torch.device | str
) -> torch.Tensor:
    """
    Grabs all index combinations from the three ranges (residue, shift, noe) and orders these into source and target indices for the edges.
    Combinations occur along columns into a triple node.

    Args:
        num_noe: Number of NOE nodes
        source1: Number of nodes for first source type
        source2: Number of nodes for second source type
        device: Device to place tensors on

    Returns:
        Stacked tensor of source and target node indices
    """
    combo = list(product(range(num_noe), range(source1), range(source2)))
    combo_tensor = torch.tensor(combo, device=device)

    # Switches tensor dimension for source (columns set up for each edge [0,0,0] --> [0],[0],[0])
    source_nodes = torch.transpose(combo_tensor, 0, 1)
    # Repeats target edges for number of occurences in source (3 for the triple in this instance, to be assigned to each incoming node type)
    target_nodes = torch.tensor(range(len(source_nodes[0])), device=device).repeat(
        len(source_nodes), 1
    )

    return torch.stack([source_nodes, target_nodes], dim=0)


def _get_pairwise_edges(
    data: HeteroData, num_res: int, device: torch.device | str
) -> torch.Tensor:
    """
    Grabs index combinations between shifts and residues and orders these into source and target indices for the edges.

    Args:
        data: HeteroData graph with shift features
        num_res: Number of residue nodes
        device: Device to place tensors on

    Returns:
        Stacked tensor of residue and shift indices for pairwise edges
    """
    # shift = torch.arange(0, num_shift)
    shift = np.nonzero(data["SHIFT"].f[:, 0])[
        0
    ]  # need to specify if it's the shift being assigned
    resid = torch.arange(0, num_res, device=device)

    shift_repeats = shift.repeat_interleave(num_res)
    # resid_repeats = resid.repeat(num_shift)
    resid_repeats = resid
    return torch.stack((resid_repeats, shift_repeats), dim=0)


def _add_value_aggregation_edges(
    data: HeteroData,
    num_noe: int,
    num_shift: int,
    num_res: int,
    device: torch.device | str,
) -> None:
    """
    Adds value aggregation edges from data nodes to value nodes.

    Args:
        data: HeteroData graph to add edges to
        num_noe: Number of NOE nodes
        num_shift: Number of shift nodes
        num_res: Number of residue nodes
        device: Device to place tensors on
    """
    shift = torch.arange(0, num_shift, device=device)
    noe = torch.arange(0, num_noe, device=device)
    resid = torch.arange(0, num_res, device=device)

    shift_repeats = torch.zeros(num_shift, device=device).long()
    noe_repeats = torch.zeros(num_noe, device=device).long()
    resid_repeats = torch.zeros(num_res, device=device).long()

    data["SHIFT", "SHIFT_extract", "VALUE_SHIFT"].edge_index = torch.stack(
        (shift, shift_repeats), dim=0
    )
    data["NOE", "NOE_extract", "VALUE_NOE"].edge_index = torch.stack(
        (noe, noe_repeats), dim=0
    )
    data["RES", "RES_extract", "VALUE_RES"].edge_index = torch.stack(
        (resid, resid_repeats), dim=0
    )


def _add_triple_edges(
    data: HeteroData,
    triple_num: int,
    triple_type: Tuple[str, str, str],
    num_noe: int,
    num_shift: int,
    num_res: int,
    device: torch.device | str,
) -> None:
    """
    Constructs all edges for a single triple type.

    Args:
        data: HeteroData graph to add edges to
        triple_num: Triple type number (0-3)
        triple_type: Tuple of (source1, source2, source3) node type names
        num_noe: Number of NOE nodes
        num_shift: Number of shift nodes
        num_res: Number of residue nodes
        device: Device to place tensors on
    """
    source1, source2, source3 = triple_type

    # Need to make sure I'm using the right node range (shift and residue number could vary)
    num_node1 = num_res if source1 == "RES" else num_shift
    num_node2 = num_res if source2 == "RES" else num_shift

    # Triple in
    edges_in = _get_triple_edges(num_noe, num_node1, num_node2, device)
    data[f"{source3}", "NOE_extract", f"TRIPLE{triple_num}"].edge_index = edges_in[:, 0]
    data[f"{source1}", "NH1_extract", f"TRIPLE{triple_num}"].edge_index = edges_in[:, 1]
    data[f"{source2}", "NH2_extract", f"TRIPLE{triple_num}"].edge_index = edges_in[:, 2]

    # Triple out (reverse in/out ordering)
    edges_out = torch.stack([edges_in[1], edges_in[0]], dim=0)
    data[f"TRIPLE{triple_num}", "NOE_add", f"{source3}"].edge_index = edges_out[:, 0]
    data[f"TRIPLE{triple_num}", "NH1_add", f"{source1}"].edge_index = edges_out[:, 1]
    data[f"TRIPLE{triple_num}", "NH2_add", f"{source2}"].edge_index = edges_out[:, 2]

    # Only residue nodes will have coordinate edges
    if source1 == "RES":
        data[f"TRIPLE{triple_num}", "res1_add", "RES"].edge_index = edges_out[:, 1]
    if source2 == "RES":
        data[f"TRIPLE{triple_num}", "res2_add", "RES"].edge_index = edges_out[:, 2]

    # Edges for self loop
    data[f"TRIPLE{triple_num}", "update", f"TRIPLE{triple_num}"].edge_index = (
        torch.stack([edges_in[1, 0], edges_in[1, 0]], dim=0)
    )

    if triple_num == 0:
        data["RES", "pair", "SHIFT"].edge_index = _get_pairwise_edges(
            data, num_res, device
        )  # pairwise edges - no message passing
