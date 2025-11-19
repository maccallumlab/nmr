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

    All node types now have the same .x dimension (embed_dim):
    - NOE.x: [embed_dim]
    - Peak.x: [embed_dim]
    - Residue.x: [embed_dim]
    """

    def __init__(self, aggr, embed_dim: int, device):
        """
        Initialize BatchMessagePass with explicit parameters.

        Args:
            aggr: Aggregation method ('mean', 'sum', etc.)
            embed_dim: Input feature dimension
            device: torch device (CPU or CUDA)
        """
        super().__init__(aggr=aggr)
        self.device = device
        self.embed_dim = embed_dim
        self.reduce = nn.Linear(embed_dim, 1, device=self.device)

    def forward(self, x_source, x_target, edge_index):
        # SIZE (n, m) (source, target)
        return self.propagate(
            edge_index=edge_index,
            x=(x_source, x_target),
            size=(x_source.size(0), x_target.size(0)),
        )

    def message(self, x_j):
        # All node types use the same reduction
        return self.reduce(x_j)

    def update(self, aggr_out, x):
        return aggr_out


class ValueCalc(nn.Module):
    """
    Value function estimation head.

    Aggregates information from Noe, Peak, and Residue nodes to predict
    a scalar value representing the quality of the current state.
    """

    def __init__(self, embed_dim: int, value_mlp_config, device):
        """
        Initialize ValueCalc with explicit parameters.

        Args:
            embed_dim: Input feature dimension
            value_mlp_config: MLPConfig for value MLP
            device: torch device (CPU or CUDA)
        """
        super().__init__()
        self.device = device
        self.embed_dim = embed_dim

        self.batch_message = BatchMessagePass(aggr="mean", embed_dim=embed_dim, device=self.device)

        self.testmlp = nn.Sequential(
            nn.Linear(3, value_mlp_config.hidden_size, device=self.device),
            nn.LayerNorm(value_mlp_config.hidden_size, device=self.device),
            nn.ReLU(),
            nn.Linear(value_mlp_config.hidden_size, 1, device=self.device),
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
    based on pairwise dot products between residue and peak embedded features.

    All node types now have the same .x dimension (embed_dim):
    - Peak.x: [embed_dim]
    - Residue.x: [embed_dim]

    Policy is computed by comparing the entire embedded feature vectors using dot products.
    """

    def __init__(self, embed_dim: int, device):
        """
        Initialize PolicyCalc with explicit parameters.

        Args:
            embed_dim: Input feature dimension
            device: torch device (CPU or CUDA)
        """
        super().__init__()
        self.device = device
        self.embed_dim = embed_dim

    def calc_policy(self, data):
        """
        Calculate policy logits for each graph in the batch.

        For each graph, computes the full pairwise dot product matrix between all peaks
        and all residues using their embedded features.

        Returns logits (not probabilities).

        The returned logits have shape [num_peaks, num_residues] where:
        - logits[peak_i, residue_j] = logit for assigning peak i to residue j
        - Higher logit = more aligned in embedded feature space (dot product)

        For training:
        - Use logits[shift_to_assign, :] with action as target for current assignment
        - Use logits[assigned_peak, :] with assigned_residue as target for previous assignments

        Args:
            data: HeteroData (single or batched) with Residue nodes and Peak nodes

        Returns:
            List of logit tensors (dot products), one per graph in batch
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
            # Get peak embedded features
            # Peak.x = [embed_dim]
            peak_features = item["Peak"].x  # [num_peaks, embed_dim]

            # Get residue embedded features
            # Residue.x = [embed_dim]
            residue_features = item["Residue"].x  # [num_residues, embed_dim]

            # Compute pairwise dot products using matrix multiplication
            # peak_features: [num_peaks, embed_dim]
            # residue_features.T: [embed_dim, num_residues]
            # logits: [num_peaks, num_residues]
            logits = torch.matmul(peak_features, residue_features.T)

            policies.append(logits)

        return policies
