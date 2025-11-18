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

# ============================================================================
# SECTION 2: Basic Attention Mechanisms
# ============================================================================


class AttentionCore(MessagePassing):
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
        """Initialize AttentionCore module."""
        # Initialize MessagePassing with add aggregation (attention weights already normalized)
        super().__init__(aggr="add", node_dim=0)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.head_dim = head_dim
        self.heads = heads
        self.negative_slope = negative_slope
        self.device = device

        # Pre-normalization layers for inputs (pre-norm pattern)
        self.norm_source = nn.LayerNorm(in_channels, device=device)
        self.norm_dest = nn.LayerNorm(in_channels, device=device)

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
        Apply GATv2 attention to compute feature updates (delta) with pre-normalization.

        Args:
            x_source: Source node features [num_source_nodes, in_channels]
            x_dest: Destination node features [num_dest_nodes, in_channels]
            edge_index: Edge indices [2, num_edges] (source->dest)

        Returns:
            Delta for destination nodes [num_dest_nodes, out_channels]
        """
        H, C = self.heads, self.head_dim

        # Apply pre-normalization to inputs
        x_source_norm = self.norm_source(x_source)
        x_dest_norm = self.norm_dest(x_dest)

        # Apply linear transformations and reshape for multi-head attention
        x_dest_transformed = self.lin_dest(x_dest_norm).view(-1, H, C)  # [num_dest_nodes, heads, head_dim]
        x_source_transformed = self.lin_source(x_source_norm).view(-1, H, C)  # [num_source_nodes, heads, head_dim]

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


class SpatialAttentionCore(MessagePassing):
    """
    Core GATv2 + Distance attention computation (tensor-based, no HeteroData dependencies).

    This class extends the pure GATv2 attention mechanism with spatial awareness by
    incorporating Euclidean distance between nodes. It works directly with tensors
    rather than HeteroData structures, making it reusable in different contexts where
    distance-aware attention is needed.

    GATv2 + Distance Formula:
        alpha_ij = softmax_j(att^T * LeakyReLU(W_dest*x_i + W_source*x_j + W_dist*d_ij))

    Where:
        - x_i, x_j: node features for target and source nodes
        - d_ij: Euclidean distance between node coordinates
        - W_dest, W_source, W_dist: learnable linear transformations
        - att: learnable attention parameter vector

    Key Differences from AttentionCore:
        - **Input**: Accepts coordinate tensors (xyz_source, xyz_dest) in addition to features
        - **Distance calculation**: Computes Euclidean distance in message() method
        - **Distance transformation**: Learnable lin_dist layer projects distance (1D) to attention space
        - **Combined features**: Adds distance features to node features before computing attention

    Key Differences from ResidueSelfAttentionTransformer:
        - **Tensor-based**: Works with tensors, not HeteroData
        - **No node type assumptions**: Doesn't assume specific node types or graph structure
        - **Reusable**: Can be wrapped for different use cases (self-attention, cross-attention, etc.)
        - **Core computation only**: No residual connections or graph navigation (handled by wrappers)

    Args:
        in_channels: Dimension of input node embeddings
        out_channels: Dimension of output features
        head_dim: Dimension per attention head
        heads: Number of attention heads (default: 1)
        negative_slope: LeakyReLU negative slope (default: 0.2)
        device: torch device (CPU or CUDA)

    Returns:
        Delta (attention output before residual): [num_dest_nodes, out_channels]

    Example - Basic Usage:
        >>> # Create spatial attention core
        >>> core = SpatialAttentionCore(
        ...     in_channels=64,
        ...     out_channels=64,
        ...     head_dim=16,
        ...     heads=4,
        ...     device='cpu'
        ... )
        >>>
        >>> # Prepare inputs
        >>> x_source = torch.randn(100, 64)  # 100 source nodes
        >>> x_dest = torch.randn(50, 64)     # 50 destination nodes
        >>> xyz_source = torch.randn(100, 3)  # source coordinates
        >>> xyz_dest = torch.randn(50, 3)     # destination coordinates
        >>> edge_index = torch.randint(0, 100, (2, 200))  # 200 edges
        >>>
        >>> # Compute attention
        >>> delta = core.forward(x_source, x_dest, xyz_source, xyz_dest, edge_index)
        >>> # delta shape: [50, 64]
        >>>
        >>> # Apply residual connection (done by wrapper)
        >>> x_dest_new = x_dest + delta

    Example - Self-Attention:
        >>> # For self-attention, source and dest are the same
        >>> x = torch.randn(100, 64)
        >>> xyz = torch.randn(100, 3)
        >>> edge_index = torch.randint(0, 100, (2, 500))
        >>>
        >>> delta = core.forward(x, x, xyz, xyz, edge_index)
        >>> x_new = x + delta

    Example - Cross-Attention:
        >>> # For cross-attention, source and dest are different
        >>> x_residues = torch.randn(50, 64)
        >>> xyz_residues = torch.randn(50, 3)
        >>> x_peaks = torch.randn(30, 64)
        >>> xyz_peaks = torch.randn(30, 3)
        >>> edge_index = torch.randint(0, 50, (2, 100))  # residues -> peaks
        >>>
        >>> delta = core.forward(x_residues, x_peaks, xyz_residues, xyz_peaks, edge_index)
        >>> x_peaks_new = x_peaks + delta

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
        """Initialize SpatialAttentionCore module."""
        # Initialize MessagePassing with add aggregation (attention weights already normalized)
        super().__init__(aggr="add", node_dim=0)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.head_dim = head_dim
        self.heads = heads
        self.negative_slope = negative_slope
        self.device = device

        # Pre-normalization layers for inputs (pre-norm pattern)
        self.norm_source = nn.LayerNorm(in_channels, device=device)
        self.norm_dest = nn.LayerNorm(in_channels, device=device)

        # GATv2: Separate linear transformations for destination (target) and source nodes
        self.lin_dest = nn.Linear(in_channels, heads * head_dim, bias=False, device=device)
        self.lin_source = nn.Linear(in_channels, heads * head_dim, bias=False, device=device)

        # Distance transformation: distance (1D) to feature space
        self.lin_dist = nn.Linear(1, heads * head_dim, bias=False, device=device)

        # Attention parameter: shape (1, heads, head_dim)
        self.att = nn.Parameter(torch.empty(1, heads, head_dim, device=device))

        # Output projection: project concatenated heads back to out_channels
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

    def forward(
        self,
        x_source: torch.Tensor,
        x_dest: torch.Tensor,
        xyz_source: torch.Tensor,
        xyz_dest: torch.Tensor,
        edge_index: torch.Tensor,
    ) -> torch.Tensor:
        """
        Apply GATv2 + Distance attention to compute feature updates (delta) with pre-normalization.

        Args:
            x_source: Source node features [num_source_nodes, in_channels]
            x_dest: Destination node features [num_dest_nodes, in_channels]
            xyz_source: Source node coordinates [num_source_nodes, 3]
            xyz_dest: Destination node coordinates [num_dest_nodes, 3]
            edge_index: Edge indices [2, num_edges] (source->dest)

        Returns:
            Delta for destination nodes [num_dest_nodes, out_channels]
        """
        H, C = self.heads, self.head_dim

        # Apply pre-normalization to inputs
        x_source_norm = self.norm_source(x_source)
        x_dest_norm = self.norm_dest(x_dest)

        # Apply linear transformations and reshape for multi-head attention
        x_dest_transformed = self.lin_dest(x_dest_norm).view(-1, H, C)  # [num_dest_nodes, heads, head_dim]
        x_source_transformed = self.lin_source(x_source_norm).view(-1, H, C)  # [num_source_nodes, heads, head_dim]

        # Use PyG message passing to compute attention-weighted aggregation
        # PyG convention: x=(source, dest) tuple → x_j comes from x_source, x_i comes from x_dest
        # Also pass coordinates as xyz=(source, dest) tuple → xyz_j from xyz_source, xyz_i from xyz_dest
        size = (x_source.size(0), x_dest.size(0))
        out = self.propagate(
            edge_index,
            x=(x_source_transformed, x_dest_transformed),
            xyz=(xyz_source, xyz_dest),
            size=size,
        )

        # Flatten multi-head output: [num_dest_nodes, heads * head_dim]
        out = out.view(-1, self.heads * self.head_dim)

        # Apply output projection: [num_dest_nodes, heads * head_dim] -> [num_dest_nodes, out_channels]
        out = self.out_proj(out)

        return out  # Return delta (no residual connection)

    def message(self, x_i, x_j, xyz_i, xyz_j, index, size_i):
        """
        Compute attention-weighted messages with distance awareness (GATv2 + Distance).

        PyTorch Geometric automatically indexes the inputs:
        - x_i, xyz_i: indexed from x_dest and xyz_dest (second tuple element), target nodes
        - x_j, xyz_j: indexed from x_source and xyz_source (first tuple element), source nodes

        Args:
            x_i: Target node features [num_edges, heads, head_dim]
            x_j: Source node features [num_edges, heads, head_dim]
            xyz_i: Target node coordinates [num_edges, 3]
            xyz_j: Source node coordinates [num_edges, 3]
            index: Target node indices for each edge [num_edges]
            size_i: Number of target nodes

        Returns:
            Attention-weighted source features [num_edges, heads, head_dim]
        """
        # Compute Euclidean distance between nodes
        distance = calc_res_distance(xyz_i, xyz_j)  # [num_edges, 1]

        # Transform distance to feature space
        dist_features = self.lin_dist(distance)  # [num_edges, heads * head_dim]
        dist_features = dist_features.view(-1, self.heads, self.head_dim)  # [num_edges, heads, head_dim]

        # GATv2 + Distance: add transformed features before nonlinearity
        # This combines node features with spatial information
        x = x_i + x_j + dist_features  # [num_edges, heads, head_dim]

        # Apply LeakyReLU nonlinearity (critical for GATv2's dynamic attention)
        x = torch.nn.functional.leaky_relu(x, self.negative_slope)

        # Compute attention scores: element-wise multiply with attention vector, then sum
        alpha = (x * self.att).sum(dim=-1)  # [num_edges, heads]

        # Apply softmax per target node (handles batched graphs correctly)
        alpha = softmax(alpha, index, num_nodes=size_i)

        # Apply attention weights to source features
        return x_j * alpha.unsqueeze(-1)  # [num_edges, heads, head_dim]


class MonoAxialAttention(nn.Module):
    """
    GATv2-style attention wrapper for HeteroData graphs with conditional spatial awareness.

    This wrapper class handles:
    - Node and edge type navigation in HeteroData
    - Empty node/edge set handling
    - Residual connections with optional projection
    - Conditional spatial attention based on node types

    Attention Mechanism Selection:
    - **Residue-to-Residue**: Uses SpatialAttentionCore (distance-aware attention)
      - Incorporates Euclidean distance between node coordinates (.xyz attribute)
      - Enables biologically meaningful spatial relationships in protein structures
    - **All other cases**: Uses AttentionCore (feature-only attention)
      - Peak-to-Peak, Noe-to-Noe, or any cross-attention with non-Residue types
      - Standard GATv2 attention without spatial information

    Can handle:
    - Self-attention: source_type == dest_type (e.g., "Peak" -> "Peak", "Residue" -> "Residue")
    - Cross-attention: source_type != dest_type (e.g., "Peak" -> "Residue")
    - Channel transformation: in_channels != out_channels with learned projection

    Requirements:
    - Residue nodes must have .xyz attribute (3D coordinates) when using spatial attention
    - All nodes must have .x attribute (feature embeddings)

    Example - Residue Self-Attention (Spatial):
        >>> # Distance-aware attention for residues
        >>> attn = MonoAxialAttention(
        ...     source_type="Residue",
        ...     dest_type="Residue",
        ...     in_channels=64,
        ...     out_channels=64,
        ...     head_dim=16,
        ...     heads=4
        ... )
        >>> # data["Residue"].xyz must exist with shape [num_residues, 3]
        >>> updated_data = attn(data)

    Example - Peak Self-Attention (Non-Spatial):
        >>> # Feature-only attention for peaks
        >>> attn = MonoAxialAttention(
        ...     source_type="Peak",
        ...     dest_type="Peak",
        ...     in_channels=64,
        ...     out_channels=64,
        ...     head_dim=16,
        ...     heads=4
        ... )
        >>> updated_data = attn(data)

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
        Initialize MonoAxialAttention wrapper.

        Args:
            source_type: Type of source nodes ("Peak", "Noe", or "Residue")
            dest_type: Type of destination nodes ("Peak", "Noe", or "Residue")
            in_channels: Dimension of input node embeddings (.x attribute)
            out_channels: Dimension of output node embeddings (.x attribute)
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
        # Use SpatialAttentionCore when both source and dest are Residue nodes (with xyz coordinates)
        # Otherwise use standard AttentionCore (for Peak, Noe, or mixed-type attention)
        if source_type == "Residue" and dest_type == "Residue":
            self.core = SpatialAttentionCore(
                in_channels=in_channels,
                out_channels=out_channels,
                head_dim=head_dim,
                heads=heads,
                negative_slope=negative_slope,
                device=device,
            )
        else:
            self.core = AttentionCore(
                in_channels=in_channels,
                out_channels=out_channels,
                head_dim=head_dim,
                heads=heads,
                negative_slope=negative_slope,
                device=device,
            )

        # Projection layer for residual connection
        # Use linear projection when dimensions don't match, identity otherwise
        if in_channels != out_channels:
            self.projection = nn.Linear(in_channels, out_channels, device=device)
        else:
            self.projection = nn.Identity()

        self.reset_parameters()

    def reset_parameters(self):
        """Initialize parameters using Glorot/Xavier initialization."""
        # Initialize projection layer if it's a Linear layer
        if isinstance(self.projection, nn.Linear):
            nn.init.xavier_uniform_(self.projection.weight)
            if self.projection.bias is not None:
                nn.init.zeros_(self.projection.bias)

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
        # Pass xyz coordinates if using SpatialAttentionCore (Residue-to-Residue attention)
        if isinstance(self.core, SpatialAttentionCore):
            xyz_source = data[self.source_type].xyz  # [num_source_nodes, 3]
            xyz_dest = data[self.dest_type].xyz  # [num_dest_nodes, 3]
            delta = self.core.forward(x_source, x_dest, xyz_source, xyz_dest, edge_index)
        else:
            delta = self.core.forward(x_source, x_dest, edge_index)

        # Apply residual connection with projection: x_new = projection(x_old) + attention_output
        data[self.dest_type].x = self.projection(x_dest) + delta

        return data


class BiAxialAttention(nn.Module):
    """
    Biaxial attention mechanism using dual attention with feature combination.

    This module updates destination node features by attending to two source node types
    simultaneously, enabling the model to capture cross-modal relationships between
    different node type combinations. The dual attention mechanism combines information
    from both sources through a learnable MLP before applying a residual update.

    Architecture Overview:
        The module employs two parallel attention cores with conditional spatial awareness:
        1. Source 1 attention: aggregates information from source_type_1 nodes to dest_type
        2. Source 2 attention: aggregates information from source_type_2 nodes to dest_type

        Each attention stream independently selects between:
        - SpatialAttentionCore: When both source and dest are "Residue" (distance-aware)
        - AttentionCore: For all other node type combinations (feature-only)

        These attention outputs are combined with a linear transformation of the destination
        features through a combination MLP, then applied as a residual update.

    Attention Mechanism Selection:
        - **Residue-to-Residue**: Uses SpatialAttentionCore (distance-aware attention)
          - Incorporates Euclidean distance between node coordinates (.xyz attribute)
          - Enables biologically meaningful spatial relationships in protein structures
        - **All other cases**: Uses AttentionCore (feature-only attention)
          - Peak-to-Peak, Noe-to-Noe, or any combination with non-Residue types
          - Standard GATv2 attention without spatial information

    Dual Attention Mechanism:
        - Source 1 Attention: Destination nodes query Source 1 nodes using edge_type_1
        - Source 2 Attention: Destination nodes query Source 2 nodes using edge_type_2

        Each stream can independently use spatial or non-spatial attention based on
        node types. For example, one stream can be Residue->Residue (spatial) while
        the other is Peak->Residue (non-spatial).

    Edge Type Naming Convention:
        Edge types follow the pattern (source_type, edge_name, dest_type):
        - (source_type_1, edge_name_1, dest_type)
        - (source_type_2, edge_name_2, dest_type)

        Default edge names are "biaxial_attn_1" and "biaxial_attn_2", but can be
        customized via constructor parameters.

    Feature Combination Logic:
        1. Transform destination features: dest_transformed = linear(dest.x)
        2. Concatenate three components: combined = [delta_1, delta_2, dest_transformed]
        3. Apply MLP: delta = MLP(combined)
           - Architecture: Linear -> LayerNorm -> ReLU -> Linear
           - Input dimension: channels * 3
           - Hidden dimension: configurable (default: channels * 2)
           - Output dimension: channels
        4. Apply residual update: dest.x = dest.x + delta (true residual connection)

    Empty Set Handling:
        The module handles edge cases gracefully:
        - Empty destination nodes: early return without modification
        - Empty source_type_1 or source_type_2 nodes: early return without modification
        - Empty edge sets (zero edges): early return without modification

        All empty set checks occur at the start of forward() to fail fast.

    Dimension Flow:
        Input:
            dest.x: [num_dests, channels]
            source_1.x: [num_src1, channels]
            source_2.x: [num_src2, channels]

        Attention Outputs:
            delta_1: [num_dests, channels]
            delta_2: [num_dests, channels]
            dest_transformed: [num_dests, channels]

        Combination:
            combined: [num_dests, channels * 3]
            delta: [num_dests, channels]

        Output:
            dest.x: [num_dests, channels]

    Integration:
        This module is designed to integrate seamlessly with the existing attention
        infrastructure:
        - Uses AttentionCore for attention computation (reusability)
        - Compatible with nn.ModuleList and nn.Sequential
        - Follows the same initialization pattern as other attention modules
        - Works with both CPU and CUDA devices
        - Handles batched graphs through PyG's batching mechanism

    Args:
        source_type_1: Type of first source node type (e.g., "Residue")
        source_type_2: Type of second source node type (e.g., "Peak")
        dest_type: Type of destination node type (e.g., "Noe")
        channels: Dimension of node embeddings (.x attribute) for all node types.
            Used for both input and output dimensions (assumes in_channels == out_channels).
        head_dim: Dimension per attention head (for multi-head attention)
        heads: Number of attention heads (default: 1)
        negative_slope: LeakyReLU negative slope for attention computation (default: 0.2)
        edge_name_1: Edge type name for first source (default: "biaxial_attn_1")
        edge_name_2: Edge type name for second source (default: "biaxial_attn_2")
        hidden_size: Hidden dimension for combination MLP (default: channels * 2)
        device: torch device for computation (CPU or CUDA)

    Example - Generic Usage:
        >>> # Create a biaxial attention module for any node type combination
        >>> module = BiAxialAttention(
        ...     source_type_1="NodeTypeA",
        ...     source_type_2="NodeTypeB",
        ...     dest_type="NodeTypeC",
        ...     channels=64,
        ...     head_dim=16,
        ...     heads=4,
        ...     device='cpu'
        ... )
        >>> updated_data = module(data)
        >>> # Destination features are updated in-place: data["NodeTypeC"].x

    Example - NMR-Specific Usage:
        >>> # Use NOE as destination, Residue and Peak as sources
        >>> module = BiAxialAttention(
        ...     source_type_1="Residue",
        ...     source_type_2="Peak",
        ...     dest_type="Noe",
        ...     channels=64,
        ...     head_dim=16,
        ...     heads=4,
        ...     device='cpu'
        ... )
        >>> updated_data = module(data)

    Example - Residue-to-Residue Dual Attention (Spatial):
        >>> # Both streams use spatial attention for Residue-to-Residue
        >>> module = BiAxialAttention(
        ...     source_type_1="Residue",
        ...     source_type_2="Residue",
        ...     dest_type="Residue",
        ...     channels=64,
        ...     head_dim=16,
        ...     heads=4,
        ...     device='cpu'
        ... )
        >>> # data["Residue"].xyz must exist with shape [num_residues, 3]
        >>> updated_data = module(data)

    Requirements:
        - Residue nodes must have .xyz attribute (3D coordinates) when using spatial attention
        - All nodes must have .x attribute (feature embeddings)

    References:
        This module extends the attention mechanism to handle dual attention
        over any combination of heterogeneous node types.
    """

    def __init__(
        self,
        source_type_1: str,
        source_type_2: str,
        dest_type: str,
        channels: int,
        head_dim: int,
        heads: int = 1,
        negative_slope: float = 0.2,
        edge_name_1: str = None,
        edge_name_2: str = None,
        hidden_size: int = None,
        device=None,
    ):
        """
        Initialize BiAxialAttention module.

        Creates two AttentionCore instances for dual attention from two source types,
        a linear transformation layer for destination features, and a combination MLP for
        merging all information streams.

        Args:
            source_type_1: Type of first source node type (e.g., "Residue").
            source_type_2: Type of second source node type (e.g., "Peak").
            dest_type: Type of destination node type (e.g., "Noe").
            channels: Dimension of node embeddings (.x attribute) for all node types.
                Used for both input and output (assumes in_channels == out_channels).
                Must match the feature dimension of all source and destination nodes.
            head_dim: Dimension per attention head. Total attention dimension per core
                is heads * head_dim.
            heads: Number of attention heads for multi-head attention (default: 1).
                Higher values allow the model to attend to different representation
                subspaces simultaneously.
            negative_slope: LeakyReLU negative slope for attention computation (default: 0.2).
                Controls the slope for negative values in the attention scoring function.
            edge_name_1: Edge type name for first source (default: "biaxial_attn_1").
            edge_name_2: Edge type name for second source (default: "biaxial_attn_2").
            hidden_size: Hidden dimension for combination MLP (default: channels * 2).
                If None, defaults to channels * 2 for sufficient representational capacity.
            device: torch device (CPU or CUDA) for parameter initialization and computation.
                All parameters and computations will use this device.
        """
        super().__init__()

        # Store node types
        self.source_type_1 = source_type_1
        self.source_type_2 = source_type_2
        self.dest_type = dest_type

        # Store configuration
        self.channels = channels
        self.head_dim = head_dim
        self.heads = heads
        self.device = device

        # Default hidden size for combination MLP
        if hidden_size is None:
            hidden_size = channels * 2

        # Generate default edge names if not provided
        if edge_name_1 is None:
            edge_name_1 = "biaxial_attn_1"
        if edge_name_2 is None:
            edge_name_2 = "biaxial_attn_2"

        # Construct edge type tuples dynamically
        # Convention: (source_type, edge_name, dest_type)
        self.edge_type_1 = (source_type_1, edge_name_1, dest_type)
        self.edge_type_2 = (source_type_2, edge_name_2, dest_type)

        # Dual attention cores for both sources
        # Use SpatialAttentionCore when both source and dest are Residue nodes
        # Otherwise use standard AttentionCore (for Peak, Noe, or mixed-type attention)
        if source_type_1 == "Residue" and dest_type == "Residue":
            self.attention_1 = SpatialAttentionCore(
                in_channels=channels,
                out_channels=channels,
                head_dim=head_dim,
                heads=heads,
                negative_slope=negative_slope,
                device=device,
            )
        else:
            self.attention_1 = AttentionCore(
                in_channels=channels,
                out_channels=channels,
                head_dim=head_dim,
                heads=heads,
                negative_slope=negative_slope,
                device=device,
            )

        if source_type_2 == "Residue" and dest_type == "Residue":
            self.attention_2 = SpatialAttentionCore(
                in_channels=channels,
                out_channels=channels,
                head_dim=head_dim,
                heads=heads,
                negative_slope=negative_slope,
                device=device,
            )
        else:
            self.attention_2 = AttentionCore(
                in_channels=channels,
                out_channels=channels,
                head_dim=head_dim,
                heads=heads,
                negative_slope=negative_slope,
                device=device,
            )

        # Destination feature transformation layer
        # Projects destination features to output dimension for combination
        self.dest_linear = nn.Linear(channels, channels, device=device)

        # Pre-normalization layers for inputs (pre-norm pattern)
        self.norm_source_1 = nn.LayerNorm(channels, device=device)
        self.norm_source_2 = nn.LayerNorm(channels, device=device)
        self.norm_dest = nn.LayerNorm(channels, device=device)

        # Combination MLP: merges attention outputs with destination features
        # Architecture: Linear -> ReLU -> Linear (no internal LayerNorm - pre-norm pattern)
        # Input: delta_1 + delta_2 + dest_transformed = channels * 3
        # Output: channels (for residual application)
        mlp_input_size = channels * 3
        self.combine_mlp = nn.Sequential(
            nn.Linear(mlp_input_size, hidden_size, device=device),
            nn.ReLU(),
            nn.Linear(hidden_size, channels, device=device),
        )

        self.reset_parameters()

    def reset_parameters(self):
        """
        Initialize parameters using Glorot/Xavier initialization.

        Initializes:
        - Destination linear layer: xavier_uniform for weights, zeros for biases
        - MLP layers: xavier_uniform for weights, zeros for biases

        Note: AttentionCore instances initialize their own parameters
        during construction via their reset_parameters() method.
        """
        # Initialize destination linear layer
        nn.init.xavier_uniform_(self.dest_linear.weight)
        if self.dest_linear.bias is not None:
            nn.init.zeros_(self.dest_linear.bias)

        # Initialize MLP layers
        for module in self.combine_mlp:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, data):
        """
        Apply biaxial attention to destination nodes in HeteroData graph.

        Performs the following steps:
        1. Validate that all required node types are non-empty
        2. Extract features from source_type_1, source_type_2, and dest_type nodes
        3. Compute attention from source_type_1 (source_1 -> dest)
        4. Compute attention from source_type_2 (source_2 -> dest)
        5. Transform destination features with linear layer
        6. Concatenate all three components (delta_1 + delta_2 + dest_transformed)
        7. Apply combination MLP to compute final delta
        8. Apply residual update to destination features (in-place modification)

        Args:
            data: HeteroData graph containing:
                - data[dest_type].x: Destination features [num_dests, channels]
                - data[source_type_1].x: Source 1 features [num_src1, channels]
                - data[source_type_2].x: Source 2 features [num_src2, channels]
                - data[edge_type_1].edge_index: Edge indices [2, num_edges_1]
                - data[edge_type_2].edge_index: Edge indices [2, num_edges_2]

        Returns:
            HeteroData: Updated graph with modified destination features.
                - data[dest_type].x: Updated features [num_dests, channels]
                - All other node features remain unchanged

        Note:
            If any node type is empty (size 0) or if either edge set is empty,
            the function returns the input data unchanged without error.
        """
        # Handle empty node sets (early return)
        # Check each node type independently to fail fast
        if data[self.dest_type].x.size(0) == 0:
            return data
        if data[self.source_type_1].x.size(0) == 0:
            return data
        if data[self.source_type_2].x.size(0) == 0:
            return data

        # Extract features from all node types
        dest_x = data[self.dest_type].x  # [num_dests, channels]
        source_x_1 = data[self.source_type_1].x  # [num_src1, channels]
        source_x_2 = data[self.source_type_2].x  # [num_src2, channels]

        # Get edge indices for both attention mechanisms
        edge_index_1 = data[self.edge_type_1].edge_index  # [2, num_edges_1]
        edge_index_2 = data[self.edge_type_2].edge_index  # [2, num_edges_2]

        # Handle empty edge sets
        # Both edge sets must be non-empty for biaxial attention to work
        if edge_index_1.size(1) == 0 or edge_index_2.size(1) == 0:
            return data

        # Apply pre-normalization to inputs (pre-norm pattern)
        # Note: AttentionCore also has its own normalization layers
        source_x_1_norm = self.norm_source_1(source_x_1)
        source_x_2_norm = self.norm_source_2(source_x_2)
        dest_x_norm = self.norm_dest(dest_x)

        # Compute attention from first source: source_type_1 (source) -> dest_type (dest)
        # Pass xyz coordinates if using SpatialAttentionCore (Residue-to-Residue attention)
        if isinstance(self.attention_1, SpatialAttentionCore):
            xyz_source_1 = data[self.source_type_1].xyz  # [num_src1, 3]
            xyz_dest = data[self.dest_type].xyz  # [num_dests, 3]
            delta_1 = self.attention_1(source_x_1_norm, dest_x_norm, xyz_source_1, xyz_dest, edge_index_1)
        else:
            delta_1 = self.attention_1(source_x_1_norm, dest_x_norm, edge_index_1)

        # Compute attention from second source: source_type_2 (source) -> dest_type (dest)
        # Pass xyz coordinates if using SpatialAttentionCore (Residue-to-Residue attention)
        if isinstance(self.attention_2, SpatialAttentionCore):
            xyz_source_2 = data[self.source_type_2].xyz  # [num_src2, 3]
            # Reuse xyz_dest if already extracted, otherwise extract it
            if not isinstance(self.attention_1, SpatialAttentionCore):
                xyz_dest = data[self.dest_type].xyz  # [num_dests, 3]
            delta_2 = self.attention_2(source_x_2_norm, dest_x_norm, xyz_source_2, xyz_dest, edge_index_2)
        else:
            delta_2 = self.attention_2(source_x_2_norm, dest_x_norm, edge_index_2)

        # Transform destination features to output dimension
        dest_transformed = self.dest_linear(dest_x_norm)  # [num_dests, channels]

        # Concatenate all three components for combination MLP
        # Feature combination captures interactions between both source types
        combined = torch.cat([delta_1, delta_2, dest_transformed], dim=-1)  # [num_dests, channels * 3]

        # Apply combination MLP to compute final delta
        # MLP learns to weight and combine the three information streams
        delta = self.combine_mlp(combined)  # [num_dests, channels]

        # Apply residual update to original destination features (in-place modification)
        data[self.dest_type].x = dest_x + delta

        return data
