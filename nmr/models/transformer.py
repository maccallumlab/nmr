"""
Transformer-based attention mechanisms for NMR chemical shift assignment.

This module implements a transformer-based alternative to triple-based message passing,
using GATv2-like pairwise attention mechanisms to address scalability issues.

Architecture Overview:
- Self-attention within node types (residue, peak, NOE)
- Cross-attention between residue and peak nodes
- NOE four-stream update architecture
- Information transfer from NOE back to nodes

Key Features:
- Multi-head attention support
- Distance-aware attention for residues (using cartesian coordinates)
- Feature-only attention for peaks and NOEs
- Residual connections throughout
- Handles empty node sets gracefully

Attribute Structure (Post-Refactoring):
- Raw data attributes (IMMUTABLE during forward pass):
  * .xyz: coordinates (Residue only) [n, 3]
  * .shifts: shift values (all node types) [n, 2 or 3]
  * .flags: assignment status (Residue and Peak only)
- Working feature attributes (updated during message passing):
  * .x: embedded features [n, embed_dim]

Node Types:
- Residue: Protein residues with coordinates .xyz, shifts .shifts [H,N], and features .x
- Peak: Observed chemical shifts .shifts [H,N] and features .x
- Noe: NOE constraints .shifts [N, H', H"] and features .x
"""

from dataclasses import dataclass

import torch
import torch.nn as nn
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import softmax


# ============================================================================
# Helper Functions
# ============================================================================


def calc_res_distance(xyz_i, xyz_j):
    """
    Calculate Euclidean distance between residues.

    Args:
        xyz_i: Coordinates of target residues [num_edges, 3]
        xyz_j: Coordinates of source residues [num_edges, 3]

    Returns:
        Distance [num_edges, 1]
    """
    rel_dist = xyz_i - xyz_j  # [num_edges, 3]
    distance = torch.norm(rel_dist, dim=-1, keepdim=True)  # [num_edges, 1]
    return distance


# ============================================================================
# SECTION 1: Configuration Dataclasses
# ============================================================================


@dataclass
class AttentionConfig:
    """Configuration for attention mechanisms."""

    num_heads: int = 4  # Number of attention heads
    attention_dim: int = 64  # Dimension of attention space per head
    edge_feature_dim: int = 0  # Dimension of edge features (0 for no edge features, 1 for dist_squared)


# ============================================================================
# SECTION 2: Basic Attention Mechanisms
# ============================================================================


class GATv2AttentionCore(MessagePassing):
    """
    Core GATv2 attention computation (tensor-based, no HeteroData dependencies).

    This class implements the pure attention calculation mechanism from the GATv2 paper,
    working directly with tensors rather than HeteroData structures. It can be reused
    in different contexts where GATv2-style attention is needed.

    GATv2 Formula:
        alpha_ij = softmax_j(att^T * LeakyReLU(W_l*x_i + W_r*x_j))

    Args:
        in_channels: Dimension of input node embeddings
        out_channels: Dimension of output features
        head_dim: Dimension per attention head
        heads: Number of attention heads (default: 1)
        negative_slope: LeakyReLU negative slope (default: 0.2)
        device: torch device (CPU or CUDA)

    Returns:
        Delta (attention output before residual): [num_dest_nodes, out_channels]

    References:
        "How Attentive are Graph Attention Networks?" (Brody et al., 2021)
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        head_dim: int,
        heads: int = 1,
        negative_slope: float = 0.2,
        device=None,
    ):
        """Initialize GATv2AttentionCore module."""
        # Initialize MessagePassing with add aggregation (attention weights already normalized)
        super().__init__(aggr="add", node_dim=0)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.head_dim = head_dim
        self.heads = heads
        self.negative_slope = negative_slope
        self.device = device

        # GATv2: Separate linear transformations for destination (target) and source nodes
        self.lin_dest = nn.Linear(in_channels, heads * head_dim, bias=False, device=device)
        self.lin_source = nn.Linear(in_channels, heads * head_dim, bias=False, device=device)

        # Attention parameter: shape (1, heads, head_dim)
        self.att = nn.Parameter(torch.empty(1, heads, head_dim, device=device))

        # Output projection: project concatenated heads back to out_channels
        self.out_proj = nn.Linear(heads * head_dim, out_channels, device=device)

        self.reset_parameters()

    def reset_parameters(self):
        """Initialize parameters using Glorot/Xavier initialization."""
        nn.init.xavier_uniform_(self.lin_dest.weight)
        nn.init.xavier_uniform_(self.lin_source.weight)
        nn.init.xavier_uniform_(self.att)
        nn.init.xavier_uniform_(self.out_proj.weight)
        if self.out_proj.bias is not None:
            nn.init.zeros_(self.out_proj.bias)

    def forward(self, x_source: torch.Tensor, x_dest: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """
        Apply GATv2 attention to compute feature updates (delta).

        Args:
            x_source: Source node features [num_source_nodes, in_channels]
            x_dest: Destination node features [num_dest_nodes, in_channels]
            edge_index: Edge indices [2, num_edges] (source->dest)

        Returns:
            Delta for destination nodes [num_dest_nodes, out_channels]
        """
        H, C = self.heads, self.head_dim

        # Apply linear transformations and reshape for multi-head attention
        x_dest_transformed = self.lin_dest(x_dest).view(-1, H, C)  # [num_dest_nodes, heads, head_dim]
        x_source_transformed = self.lin_source(x_source).view(-1, H, C)  # [num_source_nodes, heads, head_dim]

        # Use PyG message passing to compute attention-weighted aggregation
        # PyG convention: x=(source, dest) tuple → x_j comes from x_source, x_i comes from x_dest
        size = (x_source.size(0), x_dest.size(0))
        out = self.propagate(edge_index, x=(x_source_transformed, x_dest_transformed), size=size)

        # Flatten multi-head output: [num_dest_nodes, heads * head_dim]
        out = out.view(-1, self.heads * self.head_dim)

        # Apply output projection: [num_dest_nodes, heads * head_dim] -> [num_dest_nodes, out_channels]
        out = self.out_proj(out)

        return out  # Return delta (no residual connection)

    def message(self, x_i, x_j, index, size_i):
        """
        Compute attention-weighted messages for each edge (GATv2 paper implementation).

        Args:
            x_i: Target node features [num_edges, heads, head_dim]
            x_j: Source node features [num_edges, heads, head_dim]
            index: Target node indices for each edge [num_edges]
            size_i: Number of target nodes

        Returns:
            Attention-weighted source features [num_edges, heads, head_dim]
        """
        # GATv2 attention: add transformed source and target features
        x = x_i + x_j  # [num_edges, heads, head_dim]

        # Apply LeakyReLU nonlinearity
        x = torch.nn.functional.leaky_relu(x, self.negative_slope)

        # Compute attention scores: element-wise multiply with att vector, then sum
        alpha = (x * self.att).sum(dim=-1)  # [num_edges, heads]

        # Apply softmax per target node (handles batched graphs correctly)
        alpha = softmax(alpha, index, num_nodes=size_i)

        # Apply attention weights to source features
        return x_j * alpha.unsqueeze(-1)  # [num_edges, heads, head_dim]


class GATv2Attention(nn.Module):
    """
    GATv2-style attention wrapper for HeteroData graphs.

    This wrapper class handles:
    - Node and edge type navigation in HeteroData
    - Empty node/edge set handling
    - Residual connections

    Delegates the core attention computation to GATv2AttentionCore.

    Can handle:
    - Self-attention: source_type == dest_type (e.g., "Peak" -> "Peak")
    - Cross-attention: source_type != dest_type (e.g., "Peak" -> "Residue")

    References:
        "How Attentive are Graph Attention Networks?" (Brody et al., 2021)
    """

    def __init__(
        self,
        source_type: str,
        dest_type: str,
        in_channels: int,
        out_channels: int,
        head_dim: int,
        heads: int = 1,
        negative_slope: float = 0.2,
        edge_name: str = "self_attn",
        device=None,
    ):
        """
        Initialize GATv2Attention wrapper.

        Args:
            source_type: Type of source nodes ("Peak", "Noe", or "Residue")
            dest_type: Type of destination nodes ("Peak", "Noe", or "Residue")
            in_channels: Dimension of input node embeddings (.x attribute) for both types
            out_channels: Dimension of output features (.x attribute after attention)
            head_dim: Dimension per attention head
            heads: Number of attention heads (default: 1)
            negative_slope: LeakyReLU negative slope (default: 0.2)
            edge_name: Name for edge type (default: "self_attn")
            device: torch device (CPU or CUDA)
        """
        super().__init__()

        self.source_type = source_type
        self.dest_type = dest_type
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.head_dim = head_dim
        self.heads = heads
        self.device = device

        # Edge type for attention (can be self or cross-attention)
        self.edge_type = (source_type, edge_name, dest_type)

        # Core attention computation
        self.core = GATv2AttentionCore(
            in_channels=in_channels,
            out_channels=out_channels,
            head_dim=head_dim,
            heads=heads,
            negative_slope=negative_slope,
            device=device,
        )

    def forward(self, data):
        """
        Apply attention to nodes in HeteroData graph.

        Args:
            data: HeteroData graph with node features in .x attribute

        Returns:
            Updated HeteroData with attention output in dest_type .x attribute
            Shape: [num_dest_nodes, out_channels]
        """
        # Handle empty node sets (check both source and dest)
        if data[self.source_type].x.size(0) == 0 or data[self.dest_type].x.size(0) == 0:
            return data

        # Get input features from source and destination node types
        x_source = data[self.source_type].x  # [num_source_nodes, in_channels]
        x_dest = data[self.dest_type].x  # [num_dest_nodes, in_channels]

        # Get edge indices
        edge_index = data[self.edge_type].edge_index  # [2, num_edges]

        # Handle case with no edges
        if edge_index.size(1) == 0:
            return data

        # Compute attention delta using core
        delta = self.core.forward(x_source, x_dest, edge_index)

        # Apply residual connection: x_new = x_old + attention_output
        data[self.dest_type].x = x_dest + delta

        return data


class ResidueSelfAttentionTransformer(MessagePassing):
    """
    Distance-aware GATv2-style self-attention mechanism for Residue nodes.

    This class extends the standard GATv2 attention by incorporating spatial information
    through the Euclidean distance between residues. The distance is transformed
    by a learnable linear layer and added to the feature combination before computing attention.

    GATv2 + Distance Formula:
        alpha_ij = softmax_j(att^T * LeakyReLU(W_dest*x_i + W_source*x_j + W_dist*d_ij))

    Where:
        - x_i, x_j: node features for target and source residues
        - d_ij: Euclidean distance between residue coordinates
        - W_dest, W_source, W_dist: learnable linear transformations

    Key features:
    - Specialized for Residue nodes only (no node_type parameter needed)
    - Uses .xyz attribute for 3D coordinates [num_nodes, 3]
    - Multi-head attention support
    - Handles empty node sets and missing edges gracefully

    References:
        "How Attentive are Graph Attention Networks?" (Brody et al., 2021)
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        head_dim: int,
        heads: int = 1,
        negative_slope: float = 0.2,
        device=None,
    ):
        """
        Initialize ResidueSelfAttentionTransformer module.

        Args:
            in_channels: Dimension of input node embeddings (.x attribute)
            out_channels: Dimension of output features (.x attribute after attention)
            head_dim: Dimension per attention head
            heads: Number of attention heads (default: 1)
            negative_slope: LeakyReLU negative slope (default: 0.2)
            device: torch device (CPU or CUDA)
        """
        # Initialize MessagePassing with add aggregation
        super().__init__(aggr="add", node_dim=0)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.head_dim = head_dim
        self.heads = heads
        self.negative_slope = negative_slope
        self.device = device

        # Fixed node type for residues
        self.node_type = "Residue"
        self.edge_type = ("Residue", "self_attn", "Residue")

        # GATv2 transformations for destination (target) and source nodes
        self.lin_dest = nn.Linear(
            in_channels, heads * head_dim, bias=False, device=device
        )
        self.lin_source = nn.Linear(
            in_channels, heads * head_dim, bias=False, device=device
        )

        # Distance transformation: distance (1D) to feature space
        self.lin_dist = nn.Linear(1, heads * head_dim, bias=False, device=device)

        # Attention parameter: shape (1, heads, head_dim)
        self.att = nn.Parameter(torch.empty(1, heads, head_dim, device=device))

        # Output projection: concatenated heads back to out_channels
        self.out_proj = nn.Linear(heads * head_dim, out_channels, device=device)

        self.reset_parameters()

    def reset_parameters(self):
        """Initialize parameters using Glorot/Xavier initialization."""
        nn.init.xavier_uniform_(self.lin_dest.weight)
        nn.init.xavier_uniform_(self.lin_source.weight)
        nn.init.xavier_uniform_(self.lin_dist.weight)
        nn.init.xavier_uniform_(self.att)
        nn.init.xavier_uniform_(self.out_proj.weight)
        if self.out_proj.bias is not None:
            nn.init.zeros_(self.out_proj.bias)

    def forward(self, data):
        """
        Apply distance-aware self-attention to Residue nodes.

        Args:
            data: HeteroData graph with:
                - data["Residue"].x: node features [num_nodes, in_channels]
                - data["Residue"].xyz: coordinates [num_nodes, 3]

        Returns:
            Updated HeteroData with attention output in data["Residue"].x
            Shape: [num_nodes, out_channels]
        """
        H, C = self.heads, self.head_dim

        # Handle empty node sets
        if data["Residue"].x.size(0) == 0:
            return data

        # Get input features and coordinates
        x = data["Residue"].x  # [num_nodes, in_channels]
        xyz = data["Residue"].xyz  # [num_nodes, 3]

        # Get edge indices
        edge_index = data[self.edge_type].edge_index  # [2, num_edges]

        # Handle case with no edges
        if edge_index.size(1) == 0:
            return data

        # Apply linear transformations and reshape for multi-head attention
        x_dest = self.lin_dest(x).view(-1, H, C)  # [num_nodes, heads, head_dim]
        x_source = self.lin_source(x).view(-1, H, C)  # [num_nodes, heads, head_dim]

        # Use PyG message passing with coordinates
        # PyG will automatically index xyz by edge_index, providing xyz_i and xyz_j in message()
        out = self.propagate(edge_index, x=(x_source, x_dest), xyz=xyz, size=None)

        # Flatten multi-head output: [num_nodes, heads * head_dim]
        out = out.view(-1, self.heads * self.head_dim)

        # Apply output projection: [num_nodes, heads * head_dim] -> [num_nodes, out_channels]
        out = self.out_proj(out)

        # Apply residual connection: x_new = x_old + attention_output
        data["Residue"].x = x + out

        return data

    def message(self, x_i, x_j, xyz_i, xyz_j, index, size_i):
        """
        Compute attention-weighted messages with distance awareness.

        PyTorch Geometric automatically indexes the inputs:
        - x_i, xyz_i: indexed from x_dest and xyz (second tuple element), target nodes
        - x_j, xyz_j: indexed from x_source and xyz (first tuple element), source nodes

        Args:
            x_i: Target node features [num_edges, heads, head_dim]
                 (from x_dest, second element of propagate x tuple)
            x_j: Source node features [num_edges, heads, head_dim]
                 (from x_source, first element of propagate x tuple)
            xyz_i: Target node coordinates [num_edges, 3]
            xyz_j: Source node coordinates [num_edges, 3]
            index: Target node indices for each edge [num_edges]
            size_i: Number of target nodes

        Returns:
            Attention-weighted source features [num_edges, heads, head_dim]
        """
        # Compute Euclidean distance between residues
        distance = calc_res_distance(xyz_i, xyz_j)  # [num_edges, 1]

        # Transform distance to feature space
        dist_features = self.lin_dist(distance)  # [num_edges, heads * head_dim]
        dist_features = dist_features.view(
            -1, self.heads, self.head_dim
        )  # [num_edges, heads, head_dim]

        # GATv2 + Distance: add transformed features before nonlinearity
        # This is the key innovation: x = W_dest*x_i + W_source*x_j + W_dist*d_ij
        x = x_i + x_j + dist_features  # [num_edges, heads, head_dim]

        # Apply LeakyReLU nonlinearity (critical for GATv2's dynamic attention)
        x = torch.nn.functional.leaky_relu(x, self.negative_slope)

        # Compute attention scores: element-wise multiply with attention vector, then sum
        # att: [1, heads, head_dim]
        # x: [num_edges, heads, head_dim]
        alpha = (x * self.att).sum(dim=-1)  # [num_edges, heads]

        # Apply softmax per target node (handles batched graphs correctly)
        alpha = softmax(alpha, index, num_nodes=size_i)

        # Apply attention weights to source features
        return x_j * alpha.unsqueeze(-1)  # [num_edges, heads, head_dim]
