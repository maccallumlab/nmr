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

    def __init__(self, aggr, device):
        super().__init__(aggr=aggr)
        self.device = device

        self.noe_reduce = nn.Linear(3, 1, device=self.device)
        self.shift_reduce = nn.Linear(2, 1, device=self.device)
        self.res_reduce = nn.Linear(5, 1, device=self.device)

    def forward(self, x_source, x_target, edge_index):
        # SIZE (n, m) (source, target)
        return self.propagate(
            edge_index=edge_index,
            x=(x_source, x_target),
            size=(x_source.size(0), x_target.size(0)),
        )

    def message(self, x_j):
        if len(x_j[0]) == 3:
            return self.noe_reduce(x_j)
        if len(x_j[0]) == 2:
            return self.shift_reduce(x_j)
        if len(x_j[0]) == 5:
            return self.res_reduce(x_j)

    def update(self, aggr_out, x):
        return aggr_out


class ValueCalc(nn.Module):
    """
    Value function estimation head.

    Aggregates information from NOE, SHIFT, and RES nodes to predict
    a scalar value representing the quality of the current state.
    """

    def __init__(self, device):
        super().__init__()
        self.device = device

        self.batch_message = BatchMessagePass(aggr="mean", device=self.device)
        self.hidden = 64

        self.testmlp = nn.Sequential(
            nn.Linear(3, self.hidden, device=self.device),
            nn.ReLU(),
            nn.Linear(self.hidden, 1, device=self.device),
        )

    def get_aggr(self, x_source, x_target, edge_index):
        aggr = self.batch_message(x_source, x_target, edge_index)
        return aggr

    def calc_value(self, data):
        noe = self.get_aggr(
            data["NOE"].x,
            data["VALUE_NOE"].x,
            data["NOE", "NOE_extract", "VALUE_NOE"].edge_index,
        )
        shift = self.get_aggr(
            data["SHIFT"].x,
            data["VALUE_SHIFT"].x,
            data["SHIFT", "SHIFT_extract", "VALUE_SHIFT"].edge_index,
        )
        resid = self.get_aggr(
            data["RES"].x,
            data["VALUE_RES"].x,
            data["RES", "RES_extract", "VALUE_RES"].edge_index,
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

    def __init__(self, device):
        super().__init__()
        self.RES_NH = slice(3, 5)
        self.device = device

    def calc_policy(self, data):
        """
        Calculate policy logits for each graph in the batch.

        For each graph, computes squared distances between the shift being assigned
        and all residues. Returns logits (not probabilities).

        Args:
            data: Batched HeteroData with RES nodes, SHIFT nodes, and ("RES", "pair", "SHIFT") edges

        Returns:
            List of logit tensors (negative squared distances), one per graph in batch
        """
        # Unbatch the data
        data_unbatched = data.to_data_list()

        # Compute the policy logits for each item
        policies = []
        for item in data_unbatched:
            # Extract the shift being assigned and all residues
            # The pairwise edges already connect only the shift being assigned to all residues
            edge_index = item["RES", "pair", "SHIFT"].edge_index

            # Get residue NH features (H1, N15)
            resid_features = item["RES"].x[edge_index[0]][:, self.RES_NH]

            # Get shift NH features (H1, N15)
            shift_features = item["SHIFT"].x[edge_index[1]]

            # Compute squared pairwise distances (one per residue)
            # squared_distance = (H1_res - H1_shift)^2 + (N15_res - N15_shift)^2
            squared_distances = torch.sum((resid_features - shift_features) ** 2, dim=-1)

            # Use negative squared distances as logits (closer = higher logit)
            logits = -squared_distances

            policies.append(logits.unsqueeze(0))  # Add batch dimension for consistency

        return policies
