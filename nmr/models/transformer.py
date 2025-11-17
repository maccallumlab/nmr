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


class MonoAxialAttention(nn.Module):
    """
    GATv2-style attention wrapper for HeteroData graphs.

    This wrapper class handles:
    - Node and edge type navigation in HeteroData
    - Empty node/edge set handling
    - Residual connections with optional projection

    Delegates the core attention computation to AttentionCore.

    Can handle:
    - Self-attention: source_type == dest_type (e.g., "Peak" -> "Peak")
    - Cross-attention: source_type != dest_type (e.g., "Peak" -> "Residue")
    - Channel transformation: in_channels != out_channels with learned projection

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
        The module employs two parallel AttentionCore instances:
        1. Source 1 attention: aggregates information from source_type_1 nodes to dest_type
        2. Source 2 attention: aggregates information from source_type_2 nodes to dest_type

        These attention outputs are combined with a linear transformation of the destination
        features through a combination MLP, then applied as a residual update.

    Dual Attention Mechanism:
        - Source 1 Attention: Destination nodes query Source 1 nodes using edge_type_1
        - Source 2 Attention: Destination nodes query Source 2 nodes using edge_type_2

        This architecture is general and works with any node type combination.

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
        # Both cores use the same architecture but operate on different edge types
        self.attention_1 = AttentionCore(
            in_channels=channels,
            out_channels=channels,
            head_dim=head_dim,
            heads=heads,
            negative_slope=negative_slope,
            device=device,
        )

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

        # Combination MLP: merges attention outputs with destination features
        # Architecture: Linear -> LayerNorm -> ReLU -> Linear
        # Input: delta_1 + delta_2 + dest_transformed = channels * 3
        # Output: channels (for residual application)
        mlp_input_size = channels * 3
        self.combine_mlp = nn.Sequential(
            nn.Linear(mlp_input_size, hidden_size, device=device),
            nn.LayerNorm(hidden_size, device=device),
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

        # Compute attention from first source: source_type_1 (source) -> dest_type (dest)
        # AttentionCore signature: forward(x_source, x_dest, edge_index)
        delta_1 = self.attention_1(source_x_1, dest_x, edge_index_1)  # [num_dests, channels]

        # Compute attention from second source: source_type_2 (source) -> dest_type (dest)
        delta_2 = self.attention_2(source_x_2, dest_x, edge_index_2)  # [num_dests, channels]

        # Transform destination features to output dimension
        dest_transformed = self.dest_linear(dest_x)  # [num_dests, channels]

        # Concatenate all three components for combination MLP
        # Feature combination captures interactions between both source types
        combined = torch.cat([delta_1, delta_2, dest_transformed], dim=-1)  # [num_dests, channels * 3]

        # Apply combination MLP to compute final delta
        # MLP learns to weight and combine the three information streams
        delta = self.combine_mlp(combined)  # [num_dests, channels]

        # Apply residual update to original destination features (in-place modification)
        data[self.dest_type].x = dest_x + delta

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
