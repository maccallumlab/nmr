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

Attribute Structure:
- Raw data attributes (immutable during forward pass):
  * .xyz: coordinates (Residue only) [n, 3]
  * .shifts: shift values (all node types) [n, 2 or 3]
  * .flags: assignment status (Residue and Peak only)
- Working features (updated during message passing):
  * .x: embedded features [n, embed_dim]

Node Types:
- Residue: Protein residues with coordinates .xyz, shifts .shifts [H,N], and features .x
- Peak: Observed chemical shifts .shifts [H,N] and features .x
- Noe: NOE constraints .shifts [N, H', H"] and features .x
"""

import torch
import torch.nn as nn
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import softmax

from .config import AttentionConfig, MLPConfig, ModelConfig
from .mlp import MLP


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
        embed_dim: Input/output feature dimension
        attention_config: AttentionConfig containing attention_dim and num_heads
        device: torch device (CPU or CUDA)

    Returns:
        Delta (attention output before residual): [num_dest_nodes, out_channels]

    References:
        "How Attentive are Graph Attention Networks?" (Brody et al., 2021)
    """

    def __init__(self, embed_dim: int, attention_config: AttentionConfig, device):
        """Initialize AttentionCore module with explicit parameters."""
        # Initialize MessagePassing with add aggregation (attention weights already normalized)
        super().__init__(aggr="add", node_dim=0)

        # Store parameters
        self.device = device
        self.in_channels = embed_dim
        self.out_channels = embed_dim
        self.head_dim = attention_config.attention_dim
        self.heads = attention_config.num_heads
        self.negative_slope = 0.2  # Fixed value (not user-configurable)

        # Pre-normalization layers for inputs (pre-norm pattern)
        self.norm_source = nn.LayerNorm(embed_dim, device=device)
        self.norm_dest = nn.LayerNorm(embed_dim, device=device)

        # GATv2: Separate linear transformations for destination (target) and source nodes
        self.lin_dest = nn.Linear(embed_dim, self.heads * self.head_dim, bias=False, device=device)
        self.lin_source = nn.Linear(embed_dim, self.heads * self.head_dim, bias=False, device=device)

        # Attention parameter: shape (1, heads, head_dim)
        self.att = nn.Parameter(torch.empty(1, self.heads, self.head_dim, device=device))

        # Output projection: project concatenated heads back to out_channels
        self.out_proj = nn.Linear(self.heads * self.head_dim, embed_dim, device=device)

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
        - Accepts coordinate tensors (xyz_source, xyz_dest) in addition to features
        - Computes Euclidean distance in message() method
        - Learnable lin_dist layer projects distance (1D) to attention space
        - Adds distance features to node features before computing attention

    Args:
        embed_dim: Input/output feature dimension
        attention_config: AttentionConfig containing attention_dim and num_heads
        device: torch device (CPU or CUDA)

    Returns:
        Delta (attention output before residual): [num_dest_nodes, out_channels]

    Example:
        >>> core = SpatialAttentionCore(embed_dim=128, attention_config=config, device=device)
        >>> delta = core(x_source, x_dest, xyz_source, xyz_dest, edge_index)
        >>> x_new = x_dest + delta  # Apply residual (done by wrapper)

    References:
        "How Attentive are Graph Attention Networks?" (Brody et al., 2021)
    """

    def __init__(self, embed_dim: int, attention_config: AttentionConfig, device):
        """Initialize SpatialAttentionCore module with explicit parameters."""
        # Initialize MessagePassing with add aggregation (attention weights already normalized)
        super().__init__(aggr="add", node_dim=0)

        # Store parameters
        self.device = device
        self.in_channels = embed_dim
        self.out_channels = embed_dim
        self.head_dim = attention_config.attention_dim
        self.heads = attention_config.num_heads
        self.negative_slope = 0.2  # Fixed value (not user-configurable)

        # Pre-normalization layers for inputs (pre-norm pattern)
        self.norm_source = nn.LayerNorm(embed_dim, device=device)
        self.norm_dest = nn.LayerNorm(embed_dim, device=device)

        # GATv2: Separate linear transformations for destination (target) and source nodes
        self.lin_dest = nn.Linear(embed_dim, self.heads * self.head_dim, bias=False, device=device)
        self.lin_source = nn.Linear(embed_dim, self.heads * self.head_dim, bias=False, device=device)

        # Distance transformation: distance (1D) to feature space
        self.lin_dist = nn.Linear(1, self.heads * self.head_dim, bias=False, device=device)

        # Attention parameter: shape (1, heads, head_dim)
        self.att = nn.Parameter(torch.empty(1, self.heads, self.head_dim, device=device))

        # Output projection: project concatenated heads back to out_channels
        self.out_proj = nn.Linear(self.heads * self.head_dim, embed_dim, device=device)

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

    Example:
        >>> # Residue self-attention (spatial)
        >>> attn = MonoAxialAttention("Residue", "Residue", 64, 64, head_dim=16, heads=4)
        >>> updated_data = attn(data)
        >>>
        >>> # Peak self-attention (non-spatial)
        >>> attn = MonoAxialAttention("Peak", "Peak", 64, 64, head_dim=16, heads=4)
        >>> updated_data = attn(data)

    References:
        "How Attentive are Graph Attention Networks?" (Brody et al., 2021)
    """

    def __init__(
        self,
        source_type: str,
        dest_type: str,
        edge_name: str,
        embed_dim: int,
        attention_config: AttentionConfig,
        device,
    ):
        """
        Initialize MonoAxialAttention wrapper with explicit parameters.

        Args:
            source_type: Type of source nodes ("Peak", "Noe", or "Residue")
            dest_type: Type of destination nodes ("Peak", "Noe", or "Residue")
            edge_name: Name for edge type (e.g., "self_attn", "noe_res_attn")
            embed_dim: Input/output feature dimension
            attention_config: AttentionConfig containing attention_dim and num_heads
            device: torch device (CPU or CUDA)
        """
        super().__init__()

        # Store parameters
        self.device = device
        self.source_type = source_type
        self.dest_type = dest_type
        self.in_channels = embed_dim
        self.out_channels = embed_dim
        self.head_dim = attention_config.attention_dim
        self.heads = attention_config.num_heads

        # Edge type for attention (can be self or cross-attention)
        self.edge_type = (source_type, edge_name, dest_type)

        # Core attention computation
        # Use SpatialAttentionCore when both source and dest are Residue nodes (with xyz coordinates)
        # Otherwise use standard AttentionCore (for Peak, Noe, or mixed-type attention)
        if source_type == "Residue" and dest_type == "Residue":
            self.core = SpatialAttentionCore(embed_dim, attention_config, device)
        else:
            self.core = AttentionCore(embed_dim, attention_config, device)

        # Projection layer for residual connection
        # Use linear projection when dimensions don't match, identity otherwise
        if embed_dim != embed_dim:  # This will always be False, but keeping structure
            self.projection = nn.Linear(embed_dim, embed_dim, device=device)
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
        Two parallel attention streams aggregate information from both source types to the
        destination. Each stream uses SpatialAttentionCore for Residue-to-Residue attention
        (distance-aware) or AttentionCore otherwise (feature-only). Outputs are combined via
        MLP with destination features and applied as a residual update.

    Mechanism Details:
        - Uses SpatialAttentionCore for Residue-to-Residue (distance-aware with .xyz)
        - Uses AttentionCore for all other combinations (feature-only GATv2)
        - Edge types: (source_type, edge_name, dest_type) with default names
          "biaxial_attn_1" and "biaxial_attn_2"

    Feature Combination:
        Concatenates [delta_1, delta_2, dest_transformed] → MLP(channels*3 → hidden → channels)
        → residual update: dest.x = dest.x + delta

    Note: Returns early without modification if any node type or edge set is empty.

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

    Example:
        >>> # NMR-specific: Residue and Peak → NOE
        >>> module = BiAxialAttention("Residue", "Peak", "Noe", 64, head_dim=16, heads=4)
        >>> updated_data = module(data)

    Requirements:
        - Residue nodes must have .xyz attribute (3D coordinates) when using spatial attention
        - All nodes must have .x attribute (feature embeddings)
    """

    def __init__(
        self,
        source_type_1: str,
        source_type_2: str,
        dest_type: str,
        edge_name_1: str,
        edge_name_2: str,
        embed_dim: int,
        attention_config: AttentionConfig,
        combine_mlp_config: MLPConfig,
        device,
    ):
        """
        Initialize BiAxialAttention module with explicit parameters.

        Creates two AttentionCore instances for dual attention from two source types,
        a linear transformation layer for destination features, and a combination MLP for
        merging all information streams.

        Args:
            source_type_1: Type of first source node type (e.g., "Residue").
            source_type_2: Type of second source node type (e.g., "Peak").
            dest_type: Type of destination node type (e.g., "Noe").
            edge_name_1: Edge type name for first source (e.g., "res_noe_attn").
            edge_name_2: Edge type name for second source (e.g., "peak_noe_attn").
            embed_dim: Input/output feature dimension
            attention_config: AttentionConfig containing attention_dim and num_heads
            combine_mlp_config: MLPConfig for combination MLP
            device: torch device (CPU or CUDA) for parameter initialization and computation.
        """
        super().__init__()

        # Store parameters
        self.device = device
        self.source_type_1 = source_type_1
        self.source_type_2 = source_type_2
        self.dest_type = dest_type
        self.channels = embed_dim
        self.head_dim = attention_config.attention_dim
        self.heads = attention_config.num_heads

        # Construct edge type tuples
        # Convention: (source_type, edge_name, dest_type)
        self.edge_type_1 = (source_type_1, edge_name_1, dest_type)
        self.edge_type_2 = (source_type_2, edge_name_2, dest_type)

        # Dual attention cores for both sources
        # Use SpatialAttentionCore when both source and dest are Residue nodes
        # Otherwise use standard AttentionCore (for Peak, Noe, or mixed-type attention)
        if source_type_1 == "Residue" and dest_type == "Residue":
            self.attention_1 = SpatialAttentionCore(embed_dim, attention_config, device)
        else:
            self.attention_1 = AttentionCore(embed_dim, attention_config, device)

        if source_type_2 == "Residue" and dest_type == "Residue":
            self.attention_2 = SpatialAttentionCore(embed_dim, attention_config, device)
        else:
            self.attention_2 = AttentionCore(embed_dim, attention_config, device)

        # Destination feature transformation layer
        # Projects destination features to output dimension for combination
        self.dest_linear = nn.Linear(embed_dim, embed_dim, device=device)

        # Pre-normalization layers
        self.norm_source_1 = nn.LayerNorm(embed_dim, device=device)
        self.norm_source_2 = nn.LayerNorm(embed_dim, device=device)
        self.norm_dest = nn.LayerNorm(embed_dim, device=device)

        # Combination MLP
        mlp_input_size = embed_dim * 3
        self.combine_mlp = MLP(mlp_input_size, embed_dim, combine_mlp_config, device)

        self.reset_parameters()

    def reset_parameters(self):
        """Initialize parameters using Glorot/Xavier initialization."""
        # Initialize destination linear layer
        nn.init.xavier_uniform_(self.dest_linear.weight)
        if self.dest_linear.bias is not None:
            nn.init.zeros_(self.dest_linear.bias)

        # MLP initialization is handled by MLP class constructor

    def forward(self, data):
        """
        Apply biaxial attention to destination nodes.

        Args:
            data: HeteroData graph with node features (.x) and edge indices

        Returns:
            Updated HeteroData with modified dest_type.x features
        """
        # Handle empty node sets
        if data[self.dest_type].x.size(0) == 0:
            return data
        if data[self.source_type_1].x.size(0) == 0:
            return data
        if data[self.source_type_2].x.size(0) == 0:
            return data

        # Extract features
        dest_x = data[self.dest_type].x
        source_x_1 = data[self.source_type_1].x
        source_x_2 = data[self.source_type_2].x

        # Get edge indices
        edge_index_1 = data[self.edge_type_1].edge_index
        edge_index_2 = data[self.edge_type_2].edge_index

        # Handle empty edge sets
        if edge_index_1.size(1) == 0 or edge_index_2.size(1) == 0:
            return data

        # Apply pre-normalization
        source_x_1_norm = self.norm_source_1(source_x_1)
        source_x_2_norm = self.norm_source_2(source_x_2)
        dest_x_norm = self.norm_dest(dest_x)

        # Compute attention from first source
        if isinstance(self.attention_1, SpatialAttentionCore):
            xyz_source_1 = data[self.source_type_1].xyz
            xyz_dest = data[self.dest_type].xyz
            delta_1 = self.attention_1(source_x_1_norm, dest_x_norm, xyz_source_1, xyz_dest, edge_index_1)
        else:
            delta_1 = self.attention_1(source_x_1_norm, dest_x_norm, edge_index_1)

        # Compute attention from second source
        if isinstance(self.attention_2, SpatialAttentionCore):
            xyz_source_2 = data[self.source_type_2].xyz
            if not isinstance(self.attention_1, SpatialAttentionCore):
                xyz_dest = data[self.dest_type].xyz
            delta_2 = self.attention_2(source_x_2_norm, dest_x_norm, xyz_source_2, xyz_dest, edge_index_2)
        else:
            delta_2 = self.attention_2(source_x_2_norm, dest_x_norm, edge_index_2)

        # Combine attention outputs with destination features
        dest_transformed = self.dest_linear(dest_x_norm)
        combined = torch.cat([delta_1, delta_2, dest_transformed], dim=-1)
        delta = self.combine_mlp(combined)

        # Apply residual update
        data[self.dest_type].x = dest_x + delta

        return data


class TriAxialAttention(nn.Module):
    """
    Triaxial attention mechanism using triple attention with feature combination.

    This module updates destination node features by attending to three source node types
    simultaneously, enabling the model to capture complex multi-modal relationships between
    different node type combinations. The triple attention mechanism combines information
    from all three sources through a learnable MLP before applying a residual update.

    Architecture Overview:
        Three parallel attention streams aggregate information from all source types to the
        destination. Each stream uses SpatialAttentionCore for Residue-to-Residue attention
        (distance-aware) or AttentionCore otherwise (feature-only). Outputs are combined via
        MLP with destination features and applied as a residual update.

    Mechanism Details:
        - Uses SpatialAttentionCore for Residue-to-Residue (distance-aware with .xyz)
        - Uses AttentionCore for all other combinations (feature-only GATv2)
        - Edge types: (source_type, edge_name, dest_type) with default names
          "triaxial_attn_1", "triaxial_attn_2", and "triaxial_attn_3"

    Feature Combination:
        Concatenates [delta_1, delta_2, delta_3, dest_transformed] →
        MLP(channels*4 → hidden → channels) → residual update: dest.x = dest.x + delta

    Note: Returns early without modification if any node type or edge set is empty.

    Args:
        source_type_1: Type of first source node type (e.g., "Residue")
        source_type_2: Type of second source node type (e.g., "Residue")
        source_type_3: Type of third source node type (e.g., "Peak")
        dest_type: Type of destination node type (e.g., "Noe")
        channels: Dimension of node embeddings (.x attribute) for all node types.
            Used for both input and output dimensions (assumes in_channels == out_channels).
        head_dim: Dimension per attention head (for multi-head attention)
        heads: Number of attention heads (default: 1)
        negative_slope: LeakyReLU negative slope for attention computation (default: 0.2)
        edge_name_1: Edge type name for first source (default: "triaxial_attn_1")
        edge_name_2: Edge type name for second source (default: "triaxial_attn_2")
        edge_name_3: Edge type name for third source (default: "triaxial_attn_3")
        hidden_size: Hidden dimension for combination MLP (default: channels * 2)
        device: torch device for computation (CPU or CUDA)

    Example:
        >>> # NMR-specific: Two Residue streams + Peak → NOE
        >>> module = TriAxialAttention("Residue", "Residue", "Peak", "Noe", 64, head_dim=16, heads=4)
        >>> updated_data = module(data)

    Requirements:
        - Residue nodes must have .xyz attribute (3D coordinates) when using spatial attention
        - All nodes must have .x attribute (feature embeddings)
    """

    def __init__(
        self,
        source_type_1: str,
        source_type_2: str,
        source_type_3: str,
        dest_type: str,
        edge_name_1: str,
        edge_name_2: str,
        edge_name_3: str,
        embed_dim: int,
        attention_config: AttentionConfig,
        combine_mlp_config: MLPConfig,
        device,
    ):
        """
        Initialize TriAxialAttention module with explicit parameters.

        Creates three AttentionCore/SpatialAttentionCore instances for triple attention from
        three source types, a linear transformation layer for destination features, and a
        combination MLP for merging all information streams.

        Args:
            source_type_1: Type of first source node type (e.g., "Residue").
            source_type_2: Type of second source node type (e.g., "Residue").
            source_type_3: Type of third source node type (e.g., "Peak").
            dest_type: Type of destination node type (e.g., "Noe").
            edge_name_1: Edge type name for first source (e.g., "triaxial_attn_1").
            edge_name_2: Edge type name for second source (e.g., "triaxial_attn_2").
            edge_name_3: Edge type name for third source (e.g., "triaxial_attn_3").
            embed_dim: Input/output feature dimension
            attention_config: AttentionConfig containing attention_dim and num_heads
            combine_mlp_config: MLPConfig for combination MLP
            device: torch device (CPU or CUDA) for parameter initialization and computation.
        """
        super().__init__()

        # Store parameters
        self.device = device
        self.source_type_1 = source_type_1
        self.source_type_2 = source_type_2
        self.source_type_3 = source_type_3
        self.dest_type = dest_type
        self.channels = embed_dim
        self.head_dim = attention_config.attention_dim
        self.heads = attention_config.num_heads

        # Construct edge type tuples
        # Convention: (source_type, edge_name, dest_type)
        self.edge_type_1 = (source_type_1, edge_name_1, dest_type)
        self.edge_type_2 = (source_type_2, edge_name_2, dest_type)
        self.edge_type_3 = (source_type_3, edge_name_3, dest_type)

        # Triple attention cores for all three sources
        # Use SpatialAttentionCore when both source and dest are Residue nodes
        # Otherwise use standard AttentionCore (for Peak, Noe, or mixed-type attention)
        if source_type_1 == "Residue" and dest_type == "Residue":
            self.attention_1 = SpatialAttentionCore(embed_dim, attention_config, device)
        else:
            self.attention_1 = AttentionCore(embed_dim, attention_config, device)

        if source_type_2 == "Residue" and dest_type == "Residue":
            self.attention_2 = SpatialAttentionCore(embed_dim, attention_config, device)
        else:
            self.attention_2 = AttentionCore(embed_dim, attention_config, device)

        if source_type_3 == "Residue" and dest_type == "Residue":
            self.attention_3 = SpatialAttentionCore(embed_dim, attention_config, device)
        else:
            self.attention_3 = AttentionCore(embed_dim, attention_config, device)

        # Destination feature transformation layer
        # Projects destination features to output dimension for combination
        self.dest_linear = nn.Linear(embed_dim, embed_dim, device=device)

        # Pre-normalization layers for inputs (pre-norm pattern)
        self.norm_source_1 = nn.LayerNorm(embed_dim, device=device)
        self.norm_source_2 = nn.LayerNorm(embed_dim, device=device)
        self.norm_source_3 = nn.LayerNorm(embed_dim, device=device)
        self.norm_dest = nn.LayerNorm(embed_dim, device=device)

        # Combination MLP: merges attention outputs with destination features
        # Input: delta_1 + delta_2 + delta_3 + dest_transformed = channels * 4
        # Output: channels (for residual application)
        mlp_input_size = embed_dim * 4
        self.combine_mlp = MLP(mlp_input_size, embed_dim, combine_mlp_config, device)

        self.reset_parameters()

    def reset_parameters(self):
        """Initialize parameters using Glorot/Xavier initialization."""
        # Initialize destination linear layer
        nn.init.xavier_uniform_(self.dest_linear.weight)
        if self.dest_linear.bias is not None:
            nn.init.zeros_(self.dest_linear.bias)

        # MLP initialization is handled by MLP class constructor

    def forward(self, data):
        """
        Apply triaxial attention to destination nodes.

        Args:
            data: HeteroData graph with node features (.x) and edge indices

        Returns:
            Updated HeteroData with modified dest_type.x features
        """
        # Handle empty node sets
        if data[self.dest_type].x.size(0) == 0:
            return data
        if data[self.source_type_1].x.size(0) == 0:
            return data
        if data[self.source_type_2].x.size(0) == 0:
            return data
        if data[self.source_type_3].x.size(0) == 0:
            return data

        # Extract features
        dest_x = data[self.dest_type].x
        source_x_1 = data[self.source_type_1].x
        source_x_2 = data[self.source_type_2].x
        source_x_3 = data[self.source_type_3].x

        # Get edge indices
        edge_index_1 = data[self.edge_type_1].edge_index
        edge_index_2 = data[self.edge_type_2].edge_index
        edge_index_3 = data[self.edge_type_3].edge_index

        # Handle empty edge sets
        if edge_index_1.size(1) == 0 or edge_index_2.size(1) == 0 or edge_index_3.size(1) == 0:
            return data

        # Apply pre-normalization
        source_x_1_norm = self.norm_source_1(source_x_1)
        source_x_2_norm = self.norm_source_2(source_x_2)
        source_x_3_norm = self.norm_source_3(source_x_3)
        dest_x_norm = self.norm_dest(dest_x)

        # Compute attention from first source
        if isinstance(self.attention_1, SpatialAttentionCore):
            xyz_source_1 = data[self.source_type_1].xyz
            xyz_dest = data[self.dest_type].xyz
            delta_1 = self.attention_1(source_x_1_norm, dest_x_norm, xyz_source_1, xyz_dest, edge_index_1)
        else:
            delta_1 = self.attention_1(source_x_1_norm, dest_x_norm, edge_index_1)

        # Compute attention from second source
        if isinstance(self.attention_2, SpatialAttentionCore):
            xyz_source_2 = data[self.source_type_2].xyz
            if not isinstance(self.attention_1, SpatialAttentionCore):
                xyz_dest = data[self.dest_type].xyz
            delta_2 = self.attention_2(source_x_2_norm, dest_x_norm, xyz_source_2, xyz_dest, edge_index_2)
        else:
            delta_2 = self.attention_2(source_x_2_norm, dest_x_norm, edge_index_2)

        # Compute attention from third source
        if isinstance(self.attention_3, SpatialAttentionCore):
            xyz_source_3 = data[self.source_type_3].xyz
            if not isinstance(self.attention_1, SpatialAttentionCore) and not isinstance(
                self.attention_2, SpatialAttentionCore
            ):
                xyz_dest = data[self.dest_type].xyz
            delta_3 = self.attention_3(source_x_3_norm, dest_x_norm, xyz_source_3, xyz_dest, edge_index_3)
        else:
            delta_3 = self.attention_3(source_x_3_norm, dest_x_norm, edge_index_3)

        # Combine attention outputs with destination features
        dest_transformed = self.dest_linear(dest_x_norm)
        combined = torch.cat([delta_1, delta_2, delta_3, dest_transformed], dim=-1)
        delta = self.combine_mlp(combined)

        # Apply residual update
        data[self.dest_type].x = dest_x + delta

        return data
