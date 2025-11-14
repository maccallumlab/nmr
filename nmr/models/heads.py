"""
Value and policy prediction heads for the NMR GNN.

Contains components for aggregating graph information and predicting
value estimates and policy distributions.
"""

import torch
import torch.nn as nn
from torch_geometric.nn import MessagePassing


class BatchMessagePass(MessagePassing):
    """
    Message passing for aggregating node features to batch-level representations.
    Used by value and policy heads to collect information across the graph.
    """

    def __init__(self, aggr, device, config):
        super().__init__(aggr=aggr)
        self.device = device
        self.config = config

        # After embeddings:
        # - NOE: shift_dim (embedded from 3D)
        # - Peak: shift_dim (embedded from 2D)
        # - Residue: 3 (coords) + shift_dim
        # Note: NOE and Peak both have shift_dim, so they share the same reduction layer
        shift_dim = config.shift_embed.output_dim
        self.shift_reduce = nn.Linear(shift_dim, 1, device=self.device)
        self.res_reduce = nn.Linear(3 + shift_dim, 1, device=self.device)

    def forward(self, x_source, x_target, edge_index):
        # SIZE (n, m) (source, target)
        return self.propagate(
            edge_index=edge_index,
            x=(x_source, x_target),
            size=(x_source.size(0), x_target.size(0)),
        )

    def message(self, x_j):
        dim = x_j.size(1)
        shift_dim = self.config.shift_embed.output_dim

        # Check dimension to determine node type
        if dim == shift_dim:
            # NOE and Peak both have shift_dim after embedding, use same reduction
            return self.shift_reduce(x_j)
        elif dim == 3 + shift_dim:
            # Residue nodes: 3 coords + shift_dim
            return self.res_reduce(x_j)
        else:
            raise ValueError(f"Unexpected feature dimension: {dim}")

    def update(self, aggr_out, x):
        return aggr_out


class ValueCalc(nn.Module):
    """
    Value function estimation head.

    Aggregates information from Noe, Peak, and Residue nodes to predict
    a scalar value representing the quality of the current state.
    """

    def __init__(self, device, config):
        super().__init__()
        self.device = device
        self.config = config

        self.batch_message = BatchMessagePass(aggr="mean", device=self.device, config=config)
        self.hidden = 64

        self.testmlp = nn.Sequential(
            nn.Linear(3, self.hidden, device=self.device),
            nn.LayerNorm(self.hidden, device=self.device),
            nn.ReLU(),
            nn.Linear(self.hidden, 1, device=self.device),
        )

    def get_aggr(self, x_source, x_target, edge_index):
        aggr = self.batch_message(x_source, x_target, edge_index)
        return aggr

    def calc_value(self, data):
        """
        Calculate value estimate for the current state.

        Aggregates features from Noe, Peak, and Residue nodes using value aggregation edges.

        Args:
            data: HeteroData graph with Noe, Peak, Residue nodes and value aggregation edges

        Returns:
            Value tensor [batch_size, 1]
        """
        noe = self.get_aggr(
            data["Noe"].x,
            data["VALUE_NOE"].x,
            data["Noe", "aggregate", "VALUE_NOE"].edge_index,
        )
        shift = self.get_aggr(
            data["Peak"].x,
            data["VALUE_SHIFT"].x,
            data["Peak", "SHIFT_extract", "VALUE_SHIFT"].edge_index,
        )
        resid = self.get_aggr(
            data["Residue"].x,
            data["VALUE_RES"].x,
            data["Residue", "RES_extract", "VALUE_RES"].edge_index,
        )

        concat_aggr = torch.cat((noe, shift, resid), dim=-1)
        value = self.testmlp(concat_aggr)
        return value


class PolicyCalc(nn.Module):
    """
    Policy prediction head.

    Computes action probabilities (which residue to assign to the current shift)
    based on pairwise distances between residue and shift features.
    """

    def __init__(self, device, config):
        super().__init__()
        self.device = device
        self.config = config
        # After embeddings, residue shifts start at index 3 and go to the end
        # No need for fixed slice since we'll use [:, 3:] dynamically

    def calc_policy(self, data):
        """
        Calculate policy logits for each graph in the batch.

        For each graph, computes the full pairwise distance matrix between all peaks
        and all residues. Returns logits (not probabilities).

        The returned logits have shape [num_peaks, num_residues] where:
        - logits[peak_i, residue_j] = logit for assigning peak i to residue j
        - Higher logit = closer in chemical shift space (negative squared distance)

        For training:
        - Use logits[shift_to_assign, :] with action as target for current assignment
        - Use logits[assigned_peak, :] with assigned_residue as target for previous assignments

        Args:
            data: HeteroData (single or batched) with Residue nodes and Peak nodes

        Returns:
            List of logit tensors (negative squared distances), one per graph in batch
            Shape per graph: [num_peaks, num_residues]
        """
        # Check if data is batched or single
        # Batched data has a 'batch' attribute on nodes
        is_batched = hasattr(data["Residue"], 'batch')

        if is_batched:
            # Use PyG's batching mechanism
            data_unbatched = data.to_data_list()
        else:
            # Wrap single graph in a list
            data_unbatched = [data]

        # Compute the policy logits for each item
        policies = []
        for item in data_unbatched:
            # Get all peak shift embeddings: shape [num_peaks, shift_dim]
            peak_features = item["Peak"].x

            # Get all residue shift embeddings: shape [num_residues, shift_dim]
            residue_features = item["Residue"].x[:, 3:]  # Extract shift dimensions starting at index 3

            # Compute pairwise squared distances using broadcasting
            # peak_features.unsqueeze(1): [num_peaks, 1, shift_dim]
            # residue_features.unsqueeze(0): [1, num_residues, shift_dim]
            # diff: [num_peaks, num_residues, shift_dim]
            diff = peak_features.unsqueeze(1) - residue_features.unsqueeze(0)
            squared_distances = torch.sum(diff ** 2, dim=-1)  # [num_peaks, num_residues]

            # Use negative squared distances as logits (closer = higher logit)
            logits = -squared_distances

            policies.append(logits)

        return policies
