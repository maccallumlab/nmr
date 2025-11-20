# transformer-attention Specification

## Purpose
TBD - created by archiving change update-monoaxial-spatial-attention. Update Purpose after archive.
## Requirements
### Requirement: MonoAxialAttention class structure
The `MonoAxialAttention` class SHALL be implemented in `nmr/models/transformer.py` as a HeteroData wrapper for GATv2-style attention with the following structure:
- Accept parameters: `source_type`, `dest_type`, `in_channels`, `out_channels`, `head_dim`, `heads`, `negative_slope`, `edge_name`, and `device`
- Support both self-attention (`source_type == dest_type`) and cross-attention (`source_type != dest_type`)
- Include a projection layer for residual connections when `in_channels != out_channels`
- Support attention across any node type combination (Residue, Peak, Noe)

#### Scenario: Instantiate MonoAxialAttention for Peak self-attention
```python
from nmr.models.transformer import MonoAxialAttention

attention = MonoAxialAttention(
    source_type="Peak",
    dest_type="Peak",
    in_channels=64,
    out_channels=64,
    head_dim=16,
    heads=4,
    device='cpu'
)

# Verify attributes exist
assert hasattr(attention, 'core')
assert hasattr(attention, 'projection')
assert attention.source_type == "Peak"
assert attention.dest_type == "Peak"
```

#### Scenario: Instantiate for cross-attention (Peak to Residue)
```python
attention = MonoAxialAttention(
    source_type="Peak",
    dest_type="Residue",
    in_channels=64,
    out_channels=64,
    head_dim=16,
    heads=4,
    edge_name="cross_attn",
    device='cpu'
)

assert attention.edge_type == ("Peak", "cross_attn", "Residue")
```

### Requirement: Conditional spatial attention for Residue nodes
The `MonoAxialAttention` class SHALL use `SpatialAttentionCore` when BOTH `source_type` and `dest_type` are "Residue", and SHALL use `AttentionCore` for all other node type combinations.

#### Scenario: Use SpatialAttentionCore for Residue-to-Residue attention
```python
from nmr.models.transformer import MonoAxialAttention, SpatialAttentionCore, AttentionCore

# Residue-to-Residue should use SpatialAttentionCore
residue_attention = MonoAxialAttention(
    source_type="Residue",
    dest_type="Residue",
    in_channels=64,
    out_channels=64,
    head_dim=16,
    heads=4,
    device='cpu'
)

assert isinstance(residue_attention.core, SpatialAttentionCore)
```

#### Scenario: Use AttentionCore for non-Residue attention
```python
# Peak-to-Peak should use AttentionCore (no spatial awareness)
peak_attention = MonoAxialAttention(
    source_type="Peak",
    dest_type="Peak",
    in_channels=64,
    out_channels=64,
    head_dim=16,
    heads=4,
    device='cpu'
)

assert isinstance(peak_attention.core, AttentionCore)

# Peak-to-Residue should use AttentionCore (destination is Residue but source is not)
cross_attention = MonoAxialAttention(
    source_type="Peak",
    dest_type="Residue",
    in_channels=64,
    out_channels=64,
    head_dim=16,
    heads=4,
    device='cpu'
)

assert isinstance(cross_attention.core, AttentionCore)
```

### Requirement: Forward method with coordinate passing
The `forward()` method SHALL:
- Accept a HeteroData graph with node features in `.x` attribute
- When using `SpatialAttentionCore`, extract `.xyz` coordinates from source and destination nodes
- Pass coordinates to `SpatialAttentionCore.forward()` along with features and edge indices
- When using `AttentionCore`, pass only features and edge indices
- Return the updated HeteroData graph with attention output in destination node `.x` attribute

#### Scenario: Residue-to-Residue attention uses coordinates
```python
import torch
from torch_geometric.data import HeteroData
from nmr.models.transformer import MonoAxialAttention

# Create test data with Residue nodes
data = HeteroData()
num_residues = 10
data["Residue"].x = torch.randn(num_residues, 64)
data["Residue"].xyz = torch.randn(num_residues, 3)

# Create self-attention edges
edge_index = torch.combinations(torch.arange(num_residues), r=2, with_replacement=True).t()
data[("Residue", "self_attn", "Residue")].edge_index = edge_index

# Create attention module
attention = MonoAxialAttention(
    source_type="Residue",
    dest_type="Residue",
    in_channels=64,
    out_channels=64,
    head_dim=16,
    heads=4,
    device='cpu'
)

# Forward pass should use xyz coordinates
output = attention(data)

# Verify output shape
assert output["Residue"].x.shape == (num_residues, 64)
```

#### Scenario: Peak attention ignores coordinates (feature-only)
```python
# Create test data with Peak nodes (no xyz)
data = HeteroData()
num_peaks = 10
data["Peak"].x = torch.randn(num_peaks, 64)

edge_index = torch.combinations(torch.arange(num_peaks), r=2, with_replacement=True).t()
data[("Peak", "self_attn", "Peak")].edge_index = edge_index

attention = MonoAxialAttention(
    source_type="Peak",
    dest_type="Peak",
    in_channels=64,
    out_channels=64,
    head_dim=16,
    heads=4,
    device='cpu'
)

# Forward pass should not require xyz
output = attention(data)
assert output["Peak"].x.shape == (num_peaks, 64)
```

### Requirement: Distance affects Residue attention
When using Residue-to-Residue attention, the output SHALL be affected by the Euclidean distance between node coordinates, demonstrating distance-aware attention behavior.

#### Scenario: Different coordinates produce different attention outputs
```python
import torch
from torch_geometric.data import HeteroData
from nmr.models.transformer import MonoAxialAttention

# Create two scenarios with same features but different coordinates
num_residues = 5
x = torch.randn(num_residues, 64)
edge_index = torch.tensor([[0, 1, 2], [1, 2, 3]])

# Scenario 1: All residues at same position (distance = 0)
data_same = HeteroData()
data_same["Residue"].x = x.clone()
data_same["Residue"].xyz = torch.zeros(num_residues, 3)
data_same[("Residue", "self_attn", "Residue")].edge_index = edge_index

# Scenario 2: Residues at different positions (varying distances)
data_diff = HeteroData()
data_diff["Residue"].x = x.clone()
data_diff["Residue"].xyz = torch.randn(num_residues, 3)
data_diff[("Residue", "self_attn", "Residue")].edge_index = edge_index

# Create attention module
attention = MonoAxialAttention(
    source_type="Residue",
    dest_type="Residue",
    in_channels=64,
    out_channels=64,
    head_dim=16,
    heads=4,
    device='cpu'
)

# Set seed for reproducibility
torch.manual_seed(42)
attention.reset_parameters()

output_same = attention(data_same)
output_diff = attention(data_diff)

# Outputs should differ because distance affects attention
assert not torch.allclose(output_same["Residue"].x, output_diff["Residue"].x, atol=1e-5)
```

### Requirement: Backward compatibility for non-Residue nodes
The behavior of `MonoAxialAttention` for non-Residue node types (Peak, Noe) SHALL remain unchanged from the original implementation using `AttentionCore`.

#### Scenario: Peak attention behavior unchanged
```python
import torch
from torch_geometric.data import HeteroData
from nmr.models.transformer import MonoAxialAttention

# This test verifies that Peak attention still works as before
data = HeteroData()
num_peaks = 8
data["Peak"].x = torch.randn(num_peaks, 64)
edge_index = torch.combinations(torch.arange(num_peaks), r=2, with_replacement=True).t()
data[("Peak", "self_attn", "Peak")].edge_index = edge_index

attention = MonoAxialAttention(
    source_type="Peak",
    dest_type="Peak",
    in_channels=64,
    out_channels=64,
    head_dim=16,
    heads=4,
    device='cpu'
)

# Should work without xyz coordinates
output = attention(data)
assert output["Peak"].x.shape == (num_peaks, 64)
assert not torch.isnan(output["Peak"].x).any()
assert not torch.isinf(output["Peak"].x).any()
```

### Requirement: Gradient flow through spatial path
When using Residue-to-Residue attention, gradients SHALL flow through both node features (.x) and coordinates (.xyz), enabling end-to-end training.

#### Scenario: Gradients backpropagate through coordinates
```python
import torch
from torch_geometric.data import HeteroData
from nmr.models.transformer import MonoAxialAttention

data = HeteroData()
num_residues = 6

# Create inputs with gradient tracking
input_features = torch.randn(num_residues, 64, requires_grad=True)
input_xyz = torch.randn(num_residues, 3, requires_grad=True)

data["Residue"].x = input_features
data["Residue"].xyz = input_xyz

edge_index = torch.combinations(torch.arange(num_residues), r=2, with_replacement=True).t()
data[("Residue", "self_attn", "Residue")].edge_index = edge_index

attention = MonoAxialAttention(
    source_type="Residue",
    dest_type="Residue",
    in_channels=64,
    out_channels=64,
    head_dim=16,
    heads=4,
    device='cpu'
)

# Forward and backward pass
output = attention(data)
loss = output["Residue"].x.sum()
loss.backward()

# Verify gradients exist on both features and coordinates
assert input_features.grad is not None
assert input_xyz.grad is not None
assert torch.any(input_features.grad != 0)
assert torch.any(input_xyz.grad != 0)
```

### Requirement: Empty node and edge set handling
The `forward()` method SHALL handle empty node sets and empty edge sets gracefully for both spatial and non-spatial attention modes without raising errors.

#### Scenario: Empty Residue node set with spatial attention
```python
import torch
from torch_geometric.data import HeteroData
from nmr.models.transformer import MonoAxialAttention

data = HeteroData()
data["Residue"].x = torch.zeros(0, 64)
data["Residue"].xyz = torch.zeros(0, 3)
data[("Residue", "self_attn", "Residue")].edge_index = torch.zeros(2, 0, dtype=torch.long)

attention = MonoAxialAttention(
    source_type="Residue",
    dest_type="Residue",
    in_channels=64,
    out_channels=64,
    head_dim=16,
    heads=4,
    device='cpu'
)

# Should not crash
output = attention(data)
assert output["Residue"].x.shape == (0, 64)
```

### Requirement: Documentation
The `MonoAxialAttention` class docstring SHALL document:
- Conditional use of SpatialAttentionCore for Residue-to-Residue attention
- Use of AttentionCore for all other node type combinations
- Requirements for .xyz attribute when using Residue nodes
- Support for both self-attention and cross-attention
- Residual connection behavior with optional projection

#### Scenario: Docstring is comprehensive
```python
from nmr.models.transformer import MonoAxialAttention

# Verify docstring exists and mentions spatial attention
assert MonoAxialAttention.__doc__ is not None
assert len(MonoAxialAttention.__doc__) > 200

doc = MonoAxialAttention.__doc__
# Should mention both attention cores
assert 'SpatialAttentionCore' in doc or 'spatial' in doc.lower()
assert 'AttentionCore' in doc or 'attention' in doc.lower()
# Should mention Residue nodes
assert 'Residue' in doc
```

### Requirement: BiAxialAttention conditional spatial attention
The `BiAxialAttention` class SHALL conditionally use `SpatialAttentionCore` for each attention stream (attention_1 and attention_2) when the corresponding source type and destination type are both "Residue". For all other node type combinations, it SHALL use `AttentionCore`.

#### Scenario: Both attention streams use SpatialAttentionCore for Residue-to-Residue
```python
from nmr.models.transformer import BiAxialAttention, SpatialAttentionCore

# When both sources and dest are Residue, both streams use spatial attention
attention = BiAxialAttention(
    source_type_1="Residue",
    source_type_2="Residue",
    dest_type="Residue",
    channels=64,
    head_dim=16,
    heads=4,
    device='cpu'
)

assert isinstance(attention.attention_1, SpatialAttentionCore)
assert isinstance(attention.attention_2, SpatialAttentionCore)
```

#### Scenario: Mixed attention types - one spatial, one non-spatial
```python
from nmr.models.transformer import BiAxialAttention, SpatialAttentionCore, AttentionCore

# First source is Residue (spatial), second is Peak (non-spatial), dest is Residue
attention = BiAxialAttention(
    source_type_1="Residue",
    source_type_2="Peak",
    dest_type="Residue",
    channels=64,
    head_dim=16,
    heads=4,
    device='cpu'
)

# Only attention_1 should use spatial core (Residue->Residue)
assert isinstance(attention.attention_1, SpatialAttentionCore)
# attention_2 uses non-spatial core (Peak->Residue)
assert isinstance(attention.attention_2, AttentionCore)
```

#### Scenario: No spatial attention for non-Residue nodes
```python
# When neither source is paired with Residue dest, use non-spatial cores
attention = BiAxialAttention(
    source_type_1="Peak",
    source_type_2="Peak",
    dest_type="Noe",
    channels=64,
    head_dim=16,
    heads=4,
    device='cpu'
)

assert isinstance(attention.attention_1, AttentionCore)
assert isinstance(attention.attention_2, AttentionCore)
```

### Requirement: BiAxialAttention forward method with coordinate passing
The `BiAxialAttention.forward()` method SHALL:
- Check each attention core type independently using `isinstance(core, SpatialAttentionCore)`
- When an attention core is `SpatialAttentionCore`, extract `.xyz` coordinates from the corresponding source and destination nodes and pass them to the core's forward method
- When an attention core is `AttentionCore`, pass only features and edge indices
- Handle both streams independently (one can be spatial while the other is non-spatial)

#### Scenario: Dual Residue-to-Residue attention uses coordinates for both streams
```python
import torch
from torch_geometric.data import HeteroData
from nmr.models.transformer import BiAxialAttention

# Create test data with all Residue nodes
data = HeteroData()
num_residues = 10
data["Residue"].x = torch.randn(num_residues, 64)
data["Residue"].xyz = torch.randn(num_residues, 3)

# Create edges for both attention streams (same source type for simplicity)
edge_index = torch.combinations(torch.arange(num_residues), r=2, with_replacement=True).t()
data[("Residue", "biaxial_attn_1", "Residue")].edge_index = edge_index
data[("Residue", "biaxial_attn_2", "Residue")].edge_index = edge_index

# Create dual-attention module
attention = BiAxialAttention(
    source_type_1="Residue",
    source_type_2="Residue",
    dest_type="Residue",
    channels=64,
    head_dim=16,
    heads=4,
    device='cpu'
)

# Forward pass should use xyz coordinates for both streams
output = attention(data)

# Verify output shape
assert output["Residue"].x.shape == (num_residues, 64)
```

#### Scenario: Mixed spatial and non-spatial streams
```python
import torch
from torch_geometric.data import HeteroData
from nmr.models.transformer import BiAxialAttention

# Create test data with Residue and Peak nodes
data = HeteroData()
num_residues = 8
num_peaks = 12
num_noes = 5

data["Residue"].x = torch.randn(num_residues, 64)
data["Residue"].xyz = torch.randn(num_residues, 3)
data["Peak"].x = torch.randn(num_peaks, 64)
data["Noe"].x = torch.randn(num_noes, 64)

# Create edges: Residue->Noe (spatial) and Peak->Noe (non-spatial)
edge_index_1 = torch.randint(0, num_residues, (2, 20))
edge_index_2 = torch.randint(0, num_peaks, (2, 20))
data[("Residue", "biaxial_attn_1", "Noe")].edge_index = edge_index_1
data[("Peak", "biaxial_attn_2", "Noe")].edge_index = edge_index_2

# Note: Residue->Noe uses AttentionCore because dest is not Residue
attention = BiAxialAttention(
    source_type_1="Residue",
    source_type_2="Peak",
    dest_type="Noe",
    channels=64,
    head_dim=16,
    heads=4,
    device='cpu'
)

# Both cores are non-spatial since dest is Noe
output = attention(data)
assert output["Noe"].x.shape == (num_noes, 64)
```

### Requirement: Distance affects BiAxial Residue attention
When using Residue-to-Residue dual-attention, the output SHALL be affected by the Euclidean distance between node coordinates for streams using `SpatialAttentionCore`.

#### Scenario: Different coordinates produce different dual-attention outputs
```python
import torch
from torch_geometric.data import HeteroData
from nmr.models.transformer import BiAxialAttention

# Create two scenarios with same features but different coordinates
num_residues = 6
x = torch.randn(num_residues, 64)
edge_index = torch.tensor([[0, 1, 2], [1, 2, 3]])

# Scenario 1: All residues at same position (distance = 0)
data_same = HeteroData()
data_same["Residue"].x = x.clone()
data_same["Residue"].xyz = torch.zeros(num_residues, 3)
data_same[("Residue", "biaxial_attn_1", "Residue")].edge_index = edge_index
data_same[("Residue", "biaxial_attn_2", "Residue")].edge_index = edge_index

# Scenario 2: Residues at different positions (varying distances)
data_diff = HeteroData()
data_diff["Residue"].x = x.clone()
data_diff["Residue"].xyz = torch.randn(num_residues, 3)
data_diff[("Residue", "biaxial_attn_1", "Residue")].edge_index = edge_index
data_diff[("Residue", "biaxial_attn_2", "Residue")].edge_index = edge_index

# Create dual-attention module
attention = BiAxialAttention(
    source_type_1="Residue",
    source_type_2="Residue",
    dest_type="Residue",
    channels=64,
    head_dim=16,
    heads=4,
    device='cpu'
)

# Set seed for reproducibility
torch.manual_seed(42)
attention.reset_parameters()

output_same = attention(data_same)
output_diff = attention(data_diff)

# Outputs should differ because distance affects both attention streams
assert not torch.allclose(output_same["Residue"].x, output_diff["Residue"].x, atol=1e-5)
```

### Requirement: BiAxialAttention gradient flow through spatial paths
When using Residue-to-Residue dual-attention, gradients SHALL flow through both node features (.x) and coordinates (.xyz) for streams using `SpatialAttentionCore`.

#### Scenario: Gradients backpropagate through coordinates in dual-attention
```python
import torch
from torch_geometric.data import HeteroData
from nmr.models.transformer import BiAxialAttention

data = HeteroData()
num_residues = 6

# Create inputs with gradient tracking
input_features = torch.randn(num_residues, 64, requires_grad=True)
input_xyz = torch.randn(num_residues, 3, requires_grad=True)

data["Residue"].x = input_features
data["Residue"].xyz = input_xyz

edge_index = torch.combinations(torch.arange(num_residues), r=2, with_replacement=True).t()
data[("Residue", "biaxial_attn_1", "Residue")].edge_index = edge_index
data[("Residue", "biaxial_attn_2", "Residue")].edge_index = edge_index

attention = BiAxialAttention(
    source_type_1="Residue",
    source_type_2="Residue",
    dest_type="Residue",
    channels=64,
    head_dim=16,
    heads=4,
    device='cpu'
)

# Forward and backward pass
output = attention(data)
loss = output["Residue"].x.sum()
loss.backward()

# Verify gradients exist on both features and coordinates
assert input_features.grad is not None
assert input_xyz.grad is not None
assert torch.any(input_features.grad != 0)
assert torch.any(input_xyz.grad != 0)
```

### Requirement: BiAxialAttention backward compatibility
The behavior of `BiAxialAttention` for non-Residue node types SHALL remain unchanged from the original implementation using `AttentionCore`.

#### Scenario: Peak-to-Noe dual-attention behavior unchanged
```python
import torch
from torch_geometric.data import HeteroData
from nmr.models.transformer import BiAxialAttention

# This test verifies that non-Residue attention still works as before
data = HeteroData()
num_peaks = 8
num_noes = 5

data["Peak"].x = torch.randn(num_peaks, 64)
data["Noe"].x = torch.randn(num_noes, 64)

edge_index_1 = torch.randint(0, num_peaks, (2, 15))
edge_index_2 = torch.randint(0, num_peaks, (2, 15))
data[("Peak", "biaxial_attn_1", "Noe")].edge_index = edge_index_1
data[("Peak", "biaxial_attn_2", "Noe")].edge_index = edge_index_2

attention = BiAxialAttention(
    source_type_1="Peak",
    source_type_2="Peak",
    dest_type="Noe",
    channels=64,
    head_dim=16,
    heads=4,
    device='cpu'
)

# Should work without xyz coordinates
output = attention(data)
assert output["Noe"].x.shape == (num_noes, 64)
assert not torch.isnan(output["Noe"].x).any()
assert not torch.isinf(output["Noe"].x).any()
```

### Requirement: BiAxialAttention documentation
The `BiAxialAttention` class docstring SHALL document:
- Conditional use of `SpatialAttentionCore` for Residue-to-Residue attention in each stream
- Use of `AttentionCore` for all other node type combinations
- Requirements for .xyz attribute when using Residue nodes
- Independent handling of each attention stream (mixed spatial/non-spatial allowed)

#### Scenario: Docstring documents spatial attention behavior
```python
from nmr.models.transformer import BiAxialAttention

# Verify docstring exists and mentions spatial attention
assert BiAxialAttention.__doc__ is not None
assert len(BiAxialAttention.__doc__) > 300

doc = BiAxialAttention.__doc__
# Should mention spatial attention mechanism
assert 'SpatialAttentionCore' in doc or 'spatial' in doc.lower()
# Should mention Residue nodes
assert 'Residue' in doc
```

### Requirement: NMRTransformerLayer class structure
The `NMRTransformerLayer` class SHALL be implemented in `nmr/models/network.py` with the following structure:
- Accept parameters: `device` and `config` (ModelConfig instance)
- Instantiate one `AssignedPair` module for assigned Peak-Residue pairs
- Instantiate three `BiAxialAttention` modules for dual-source attention operations
- Instantiate two `MonoAxialAttention` modules for single-source attention operations
- Execute attention operations in a defined sequence in the `forward()` method

#### Scenario: Instantiate NMRTransformerLayer
```python
from nmr.models.network import NMRTransformerLayer, ModelConfig

config = ModelConfig(
    num_nmr_layers=1,
    layer_type="transformer"
)

layer = NMRTransformerLayer(device='cpu', config=config)

# Verify all attention modules are instantiated
assert hasattr(layer, 'assigned_pair')
assert hasattr(layer, 'residue_from_residue_peak')
assert hasattr(layer, 'peak_from_peak_residue')
assert hasattr(layer, 'noe_from_residue_peak')
assert hasattr(layer, 'residue_from_noe')
assert hasattr(layer, 'peak_from_noe')
```

#### Scenario: Layer accepts HeteroData and returns updated graph
```python
import torch
from torch_geometric.data import HeteroData
from nmr.models.network import NMRTransformerLayer, ModelConfig

# Create test graph
data = HeteroData()
data["Residue"].x = torch.randn(10, 128)
data["Residue"].xyz = torch.randn(10, 3)
data["Peak"].x = torch.randn(10, 128)
data["Noe"].x = torch.randn(5, 128)

# Add required edges (simplified for test)
data[("Residue", "res_res_attn", "Residue")].edge_index = torch.randint(0, 10, (2, 20))
data[("Peak", "peak_res_attn", "Residue")].edge_index = torch.randint(0, 10, (2, 15))
data[("Peak", "peak_peak_attn", "Peak")].edge_index = torch.randint(0, 10, (2, 20))
data[("Residue", "res_peak_attn", "Peak")].edge_index = torch.randint(0, 10, (2, 15))
data[("Residue", "res_noe_attn", "Noe")].edge_index = torch.randint(0, 5, (2, 10))
data[("Peak", "peak_noe_attn", "Noe")].edge_index = torch.randint(0, 5, (2, 10))
data[("Noe", "noe_res_attn", "Residue")].edge_index = torch.randint(0, 10, (2, 10))
data[("Noe", "noe_peak_attn", "Peak")].edge_index = torch.randint(0, 10, (2, 10))
data[("Peak", "assigned_to", "Residue")].edge_index = torch.randint(0, 10, (2, 5))

config = ModelConfig(layer_type="transformer")
layer = NMRTransformerLayer(device='cpu', config=config)

output = layer(data)

# Verify output is HeteroData and features are updated
assert isinstance(output, HeteroData)
assert output["Residue"].x.shape == (10, 128)
assert output["Peak"].x.shape == (10, 128)
assert output["Noe"].x.shape == (5, 128)
```

### Requirement: NMRTransformerLayer attention sequence
The `forward()` method of `NMRTransformerLayer` SHALL execute attention operations in this exact order:
1. `assigned_pair` (AssignedPair module)
2. `residue_from_residue_peak` (BiAxialAttention: Residue + Peak → Residue)
3. `peak_from_peak_residue` (BiAxialAttention: Peak + Residue → Peak)
4. `noe_from_residue_peak` (BiAxialAttention: Residue + Peak → Noe)
5. `residue_from_noe` (MonoAxialAttention: Noe → Residue)
6. `peak_from_noe` (MonoAxialAttention: Noe → Peak)

#### Scenario: Verify execution order by tracing updates
```python
import torch
from torch_geometric.data import HeteroData
from nmr.models.network import NMRTransformerLayer, ModelConfig

# Create minimal test graph
data = HeteroData()
data["Residue"].x = torch.zeros(5, 128)
data["Residue"].xyz = torch.randn(5, 3)
data["Peak"].x = torch.zeros(5, 128)
data["Noe"].x = torch.zeros(3, 128)

# Add minimal edges
data[("Residue", "res_res_attn", "Residue")].edge_index = torch.tensor([[0], [1]])
data[("Peak", "peak_res_attn", "Residue")].edge_index = torch.tensor([[0], [1]])
data[("Peak", "peak_peak_attn", "Peak")].edge_index = torch.tensor([[0], [1]])
data[("Residue", "res_peak_attn", "Peak")].edge_index = torch.tensor([[0], [1]])
data[("Residue", "res_noe_attn", "Noe")].edge_index = torch.tensor([[0], [0]])
data[("Peak", "peak_noe_attn", "Noe")].edge_index = torch.tensor([[0], [0]])
data[("Noe", "noe_res_attn", "Residue")].edge_index = torch.tensor([[0], [1]])
data[("Noe", "noe_peak_attn", "Peak")].edge_index = torch.tensor([[0], [1]])
data[("Peak", "assigned_to", "Residue")].edge_index = torch.tensor([[0], [0]])

config = ModelConfig(layer_type="transformer")
layer = NMRTransformerLayer(device='cpu', config=config)

output = layer(data)

# After forward pass, all node features should be non-zero (updated by attention)
assert torch.any(output["Residue"].x != 0)
assert torch.any(output["Peak"].x != 0)
assert torch.any(output["Noe"].x != 0)
```

### Requirement: NMRTransformerLayer edge type naming
The `NMRTransformerLayer` SHALL use the following edge type names for attention operations:
- Residue self-attention: `("Residue", "res_res_attn", "Residue")`
- Peak to Residue cross-attention: `("Peak", "peak_res_attn", "Residue")`
- Peak self-attention: `("Peak", "peak_peak_attn", "Peak")`
- Residue to Peak cross-attention: `("Residue", "res_peak_attn", "Peak")`
- Residue to Noe cross-attention: `("Residue", "res_noe_attn", "Noe")`
- Peak to Noe cross-attention: `("Peak", "peak_noe_attn", "Noe")`
- Noe to Residue cross-attention: `("Noe", "noe_res_attn", "Residue")`
- Noe to Peak cross-attention: `("Noe", "noe_peak_attn", "Peak")`

#### Scenario: Verify edge type names in module instantiation
```python
from nmr.models.network import NMRTransformerLayer, ModelConfig

config = ModelConfig(layer_type="transformer")
layer = NMRTransformerLayer(device='cpu', config=config)

# Check BiAxialAttention edge types
assert layer.residue_from_residue_peak.edge_type_1 == ("Residue", "res_res_attn", "Residue")
assert layer.residue_from_residue_peak.edge_type_2 == ("Peak", "peak_res_attn", "Residue")

assert layer.peak_from_peak_residue.edge_type_1 == ("Peak", "peak_peak_attn", "Peak")
assert layer.peak_from_peak_residue.edge_type_2 == ("Residue", "res_peak_attn", "Peak")

assert layer.noe_from_residue_peak.edge_type_1 == ("Residue", "res_noe_attn", "Noe")
assert layer.noe_from_residue_peak.edge_type_2 == ("Peak", "peak_noe_attn", "Noe")

# Check MonoAxialAttention edge types
assert layer.residue_from_noe.edge_type == ("Noe", "noe_res_attn", "Residue")
assert layer.peak_from_noe.edge_type == ("Noe", "noe_peak_attn", "Peak")
```

### Requirement: NMRTransformerLayer configuration usage
The `NMRTransformerLayer` SHALL use the following configuration from `ModelConfig`:
- `config.embed.embed_dim` for channel dimensions in all attention modules
- `config.attention.num_heads` and `config.attention.attention_dim` for multi-head attention configuration
- `config.mlp` for `AssignedPair` MLP configuration

#### Scenario: Configuration propagates to attention modules
```python
from nmr.models.network import NMRTransformerLayer, ModelConfig, EmbedConfig, MLPConfig
from nmr.models.transformer import AttentionConfig

embed_config = EmbedConfig(embed_dim=64)
attention_config = AttentionConfig(num_heads=8, attention_dim=16)
mlp_config = MLPConfig(hidden_size=128, num_layers=2)

config = ModelConfig(
    layer_type="transformer",
    embed=embed_config,
    attention=attention_config,
    mlp=mlp_config
)

layer = NMRTransformerLayer(device='cpu', config=config)

# Verify attention modules use correct configuration
assert layer.residue_from_residue_peak.channels == 64
assert layer.residue_from_residue_peak.heads == 8
assert layer.residue_from_residue_peak.head_dim == 16

assert layer.residue_from_noe.out_channels == 64
assert layer.residue_from_noe.heads == 8
```

### Requirement: NMRTransformerNet class structure
The `NMRTransformerNet` class SHALL be implemented in `nmr/models/network.py` with the following structure:
- Accept parameters: `device` and `config` (ModelConfig instance)
- Instantiate `EmbedFeatures` module for feature embedding
- Instantiate a stack of `NMRTransformerLayer` modules (count determined by `config.num_nmr_layers`)
- Instantiate `ValueCalc` and `PolicyCalc` prediction heads
- Execute embedding → transformer layers → value/policy heads in `forward()` method

#### Scenario: Instantiate NMRTransformerNet
```python
from nmr.models.network import NMRTransformerNet, ModelConfig

config = ModelConfig(
    num_nmr_layers=3,
    layer_type="transformer"
)

net = NMRTransformerNet(device='cpu', config=config)

# Verify components are instantiated
assert hasattr(net, 'embed_features')
assert hasattr(net, 'nmr')  # Sequential stack of layers
assert hasattr(net, 'value')
assert hasattr(net, 'policy')
assert len(net.nmr) == 3  # 3 transformer layers
```

#### Scenario: Forward pass produces value and policy outputs
```python
import torch
from torch_geometric.data import HeteroData
from nmr.models.network import NMRTransformerNet, ModelConfig

# Create test graph with raw features
data = HeteroData()
data["Residue"].shifts = torch.randn(10, 2)  # [H, N]
data["Residue"].flags = torch.randn(10, 1)
data["Residue"].xyz = torch.randn(10, 3)
data["Peak"].shifts = torch.randn(10, 2)  # [H, N]
data["Peak"].flags = torch.randn(10, 2)
data["Noe"].shifts = torch.randn(5, 3)  # [N, H', H"]

# Add required edges (simplified)
data[("Residue", "res_res_attn", "Residue")].edge_index = torch.randint(0, 10, (2, 20))
data[("Peak", "peak_res_attn", "Residue")].edge_index = torch.randint(0, 10, (2, 15))
data[("Peak", "peak_peak_attn", "Peak")].edge_index = torch.randint(0, 10, (2, 20))
data[("Residue", "res_peak_attn", "Peak")].edge_index = torch.randint(0, 10, (2, 15))
data[("Residue", "res_noe_attn", "Noe")].edge_index = torch.randint(0, 5, (2, 10))
data[("Peak", "peak_noe_attn", "Noe")].edge_index = torch.randint(0, 5, (2, 10))
data[("Noe", "noe_res_attn", "Residue")].edge_index = torch.randint(0, 10, (2, 10))
data[("Noe", "noe_peak_attn", "Peak")].edge_index = torch.randint(0, 10, (2, 10))
data[("Peak", "assigned_to", "Residue")].edge_index = torch.randint(0, 10, (2, 5))

config = ModelConfig(
    num_nmr_layers=2,
    layer_type="transformer"
)

net = NMRTransformerNet(device='cpu', config=config)

value, policy = net(data)

# Verify outputs have correct shapes
assert value.shape == (1,)  # Scalar value
assert policy.shape[0] == 10  # Policy over peaks (num_peaks)
```

### Requirement: NMRTransformerNet forward method sequence
The `forward()` method of `NMRTransformerNet` SHALL execute the following operations in order:
1. Embed features using `embed_features` (converts .shifts and .flags to .x embeddings)
2. Apply stacked `NMRTransformerLayer` modules via `nmr` sequential
3. Compute value prediction using `value.calc_value(data)`
4. Compute policy prediction using `policy.calc_policy(data)`
5. Return tuple `(value, policy)`

#### Scenario: Verify forward method execution sequence
```python
import torch
from torch_geometric.data import HeteroData
from nmr.models.network import NMRTransformerNet, ModelConfig

# Create test graph
data = HeteroData()
data["Residue"].shifts = torch.randn(8, 2)
data["Residue"].flags = torch.randn(8, 1)
data["Residue"].xyz = torch.randn(8, 3)
data["Peak"].shifts = torch.randn(8, 2)
data["Peak"].flags = torch.randn(8, 2)
data["Noe"].shifts = torch.randn(4, 3)

# Add required edges
data[("Residue", "res_res_attn", "Residue")].edge_index = torch.randint(0, 8, (2, 15))
data[("Peak", "peak_res_attn", "Residue")].edge_index = torch.randint(0, 8, (2, 12))
data[("Peak", "peak_peak_attn", "Peak")].edge_index = torch.randint(0, 8, (2, 15))
data[("Residue", "res_peak_attn", "Peak")].edge_index = torch.randint(0, 8, (2, 12))
data[("Residue", "res_noe_attn", "Noe")].edge_index = torch.randint(0, 4, (2, 8))
data[("Peak", "peak_noe_attn", "Noe")].edge_index = torch.randint(0, 4, (2, 8))
data[("Noe", "noe_res_attn", "Residue")].edge_index = torch.randint(0, 8, (2, 8))
data[("Noe", "noe_peak_attn", "Peak")].edge_index = torch.randint(0, 8, (2, 8))
data[("Peak", "assigned_to", "Residue")].edge_index = torch.randint(0, 8, (2, 4))

config = ModelConfig(num_nmr_layers=1, layer_type="transformer")
net = NMRTransformerNet(device='cpu', config=config)

# Initially, .x attributes should not exist
assert not hasattr(data["Residue"], 'x')
assert not hasattr(data["Peak"], 'x')
assert not hasattr(data["Noe"], 'x')

value, policy = net(data)

# After forward pass, .x attributes should be created by embedding
assert hasattr(data["Residue"], 'x')
assert hasattr(data["Peak"], 'x')
assert hasattr(data["Noe"], 'x')

# Outputs should have correct types and shapes
assert isinstance(value, torch.Tensor)
assert isinstance(policy, torch.Tensor)
assert value.ndim == 1
assert policy.ndim == 1
```

### Requirement: ModelConfig layer_type validation
The `ModelConfig.layer_type` field SHALL accept "triple" or "transformer" values. When `layer_type="transformer"`, `NMRNet.__init__()` SHALL instantiate `NMRTransformerLayer` instead of `NMRLayer`.

#### Scenario: ModelConfig layer_type controls architecture selection
```python
from nmr.models.network import NMRNet, ModelConfig, NMRLayer, NMRTransformerLayer

# Triple architecture (default)
config_triple = ModelConfig(num_nmr_layers=2, layer_type="triple")
net_triple = NMRNet(device='cpu', config=config_triple)
assert isinstance(net_triple.nmr[0], NMRLayer)

# Transformer architecture
config_transformer = ModelConfig(num_nmr_layers=2, layer_type="transformer")
net_transformer = NMRNet(device='cpu', config=config_transformer)
assert isinstance(net_transformer.nmr[0], NMRTransformerLayer)
```

### Requirement: Spatial attention for Residue-to-Residue operations
When `NMRTransformerLayer` executes Residue-to-Residue attention (via the `residue_from_residue_peak` BiAxialAttention module), the first attention stream SHALL use `SpatialAttentionCore` to incorporate Euclidean distance between residue coordinates.

#### Scenario: Residue-to-Residue stream uses spatial attention
```python
from nmr.models.network import NMRTransformerLayer, ModelConfig
from nmr.models.transformer import SpatialAttentionCore

config = ModelConfig(layer_type="transformer")
layer = NMRTransformerLayer(device='cpu', config=config)

# The first stream (Residue -> Residue) should use SpatialAttentionCore
assert isinstance(layer.residue_from_residue_peak.attention_1, SpatialAttentionCore)
```

#### Scenario: Distance affects Residue updates in transformer layer
```python
import torch
from torch_geometric.data import HeteroData
from nmr.models.network import NMRTransformerLayer, ModelConfig

# Create two scenarios with same features but different coordinates
num_residues = 6
x = torch.randn(num_residues, 128)
peak_x = torch.randn(num_residues, 128)
noe_x = torch.randn(3, 128)

edge_index_res = torch.tensor([[0, 1, 2], [1, 2, 3]])
edge_index_peak_res = torch.tensor([[0, 1], [1, 2]])
edge_index_peak = torch.tensor([[0, 1], [1, 2]])
edge_index_res_peak = torch.tensor([[0, 1], [1, 2]])
edge_index_res_noe = torch.tensor([[0, 1], [0, 1]])
edge_index_peak_noe = torch.tensor([[0, 1], [0, 1]])
edge_index_noe_res = torch.tensor([[0, 1], [0, 1]])
edge_index_noe_peak = torch.tensor([[0, 1], [0, 1]])
edge_index_assigned = torch.tensor([[0], [0]])

# Scenario 1: All residues at same position
data_same = HeteroData()
data_same["Residue"].x = x.clone()
data_same["Residue"].xyz = torch.zeros(num_residues, 3)
data_same["Peak"].x = peak_x.clone()
data_same["Noe"].x = noe_x.clone()
data_same[("Residue", "res_res_attn", "Residue")].edge_index = edge_index_res
data_same[("Peak", "peak_res_attn", "Residue")].edge_index = edge_index_peak_res
data_same[("Peak", "peak_peak_attn", "Peak")].edge_index = edge_index_peak
data_same[("Residue", "res_peak_attn", "Peak")].edge_index = edge_index_res_peak
data_same[("Residue", "res_noe_attn", "Noe")].edge_index = edge_index_res_noe
data_same[("Peak", "peak_noe_attn", "Noe")].edge_index = edge_index_peak_noe
data_same[("Noe", "noe_res_attn", "Residue")].edge_index = edge_index_noe_res
data_same[("Noe", "noe_peak_attn", "Peak")].edge_index = edge_index_noe_peak
data_same[("Peak", "assigned_to", "Residue")].edge_index = edge_index_assigned

# Scenario 2: Residues at different positions
data_diff = HeteroData()
data_diff["Residue"].x = x.clone()
data_diff["Residue"].xyz = torch.randn(num_residues, 3)
data_diff["Peak"].x = peak_x.clone()
data_diff["Noe"].x = noe_x.clone()
data_diff[("Residue", "res_res_attn", "Residue")].edge_index = edge_index_res
data_diff[("Peak", "peak_res_attn", "Residue")].edge_index = edge_index_peak_res
data_diff[("Peak", "peak_peak_attn", "Peak")].edge_index = edge_index_peak
data_diff[("Residue", "res_peak_attn", "Peak")].edge_index = edge_index_res_peak
data_diff[("Residue", "res_noe_attn", "Noe")].edge_index = edge_index_res_noe
data_diff[("Peak", "peak_noe_attn", "Noe")].edge_index = edge_index_peak_noe
data_diff[("Noe", "noe_res_attn", "Residue")].edge_index = edge_index_noe_res
data_diff[("Noe", "noe_peak_attn", "Peak")].edge_index = edge_index_noe_peak
data_diff[("Peak", "assigned_to", "Residue")].edge_index = edge_index_assigned

config = ModelConfig(layer_type="transformer")
layer = NMRTransformerLayer(device='cpu', config=config)

# Set seed for reproducibility
torch.manual_seed(42)
layer.reset_parameters() if hasattr(layer, 'reset_parameters') else None

output_same = layer(data_same)
output_diff = layer(data_diff)

# Outputs should differ because distance affects Residue attention
assert not torch.allclose(output_same["Residue"].x, output_diff["Residue"].x, atol=1e-5)
```

### Requirement: Empty node set handling in transformer layer
The `NMRTransformerLayer.forward()` method SHALL handle empty node sets gracefully without raising errors, leveraging the empty set handling already present in `MonoAxialAttention` and `BiAxialAttention`.

#### Scenario: Empty Residue node set
```python
import torch
from torch_geometric.data import HeteroData
from nmr.models.network import NMRTransformerLayer, ModelConfig

data = HeteroData()
data["Residue"].x = torch.zeros(0, 128)
data["Residue"].xyz = torch.zeros(0, 3)
data["Peak"].x = torch.randn(5, 128)
data["Noe"].x = torch.randn(3, 128)

# Add edges (will be empty for Residue-related operations)
data[("Residue", "res_res_attn", "Residue")].edge_index = torch.zeros(2, 0, dtype=torch.long)
data[("Peak", "peak_res_attn", "Residue")].edge_index = torch.zeros(2, 0, dtype=torch.long)
data[("Peak", "peak_peak_attn", "Peak")].edge_index = torch.randint(0, 5, (2, 8))
data[("Residue", "res_peak_attn", "Peak")].edge_index = torch.zeros(2, 0, dtype=torch.long)
data[("Residue", "res_noe_attn", "Noe")].edge_index = torch.zeros(2, 0, dtype=torch.long)
data[("Peak", "peak_noe_attn", "Noe")].edge_index = torch.randint(0, 3, (2, 6))
data[("Noe", "noe_res_attn", "Residue")].edge_index = torch.zeros(2, 0, dtype=torch.long)
data[("Noe", "noe_peak_attn", "Peak")].edge_index = torch.randint(0, 5, (2, 6))
data[("Peak", "assigned_to", "Residue")].edge_index = torch.zeros(2, 0, dtype=torch.long)

config = ModelConfig(layer_type="transformer")
layer = NMRTransformerLayer(device='cpu', config=config)

# Should not crash
output = layer(data)
assert output["Residue"].x.shape == (0, 128)
assert output["Peak"].x.shape == (5, 128)
assert output["Noe"].x.shape == (3, 128)
```

### Requirement: Gradient flow through transformer layer
When using `NMRTransformerLayer`, gradients SHALL flow backward through all attention operations to enable end-to-end training.

#### Scenario: Gradients backpropagate through layer
```python
import torch
from torch_geometric.data import HeteroData
from nmr.models.network import NMRTransformerLayer, ModelConfig

data = HeteroData()
num_residues = 6

# Create inputs with gradient tracking
input_features = torch.randn(num_residues, 128, requires_grad=True)
input_xyz = torch.randn(num_residues, 3, requires_grad=True)
peak_features = torch.randn(num_residues, 128, requires_grad=True)
noe_features = torch.randn(3, 128, requires_grad=True)

data["Residue"].x = input_features
data["Residue"].xyz = input_xyz
data["Peak"].x = peak_features
data["Noe"].x = noe_features

edge_index = torch.tensor([[0, 1, 2], [1, 2, 3]])
data[("Residue", "res_res_attn", "Residue")].edge_index = edge_index
data[("Peak", "peak_res_attn", "Residue")].edge_index = edge_index
data[("Peak", "peak_peak_attn", "Peak")].edge_index = edge_index
data[("Residue", "res_peak_attn", "Peak")].edge_index = edge_index
data[("Residue", "res_noe_attn", "Noe")].edge_index = torch.tensor([[0, 1], [0, 1]])
data[("Peak", "peak_noe_attn", "Noe")].edge_index = torch.tensor([[0, 1], [0, 1]])
data[("Noe", "noe_res_attn", "Residue")].edge_index = torch.tensor([[0, 1], [0, 1]])
data[("Noe", "noe_peak_attn", "Peak")].edge_index = torch.tensor([[0, 1], [0, 1]])
data[("Peak", "assigned_to", "Residue")].edge_index = torch.tensor([[0], [0]])

config = ModelConfig(layer_type="transformer")
layer = NMRTransformerLayer(device='cpu', config=config)

# Forward and backward pass
output = layer(data)
loss = output["Residue"].x.sum() + output["Peak"].x.sum() + output["Noe"].x.sum()
loss.backward()

# Verify gradients exist on inputs
assert input_features.grad is not None
assert input_xyz.grad is not None
assert peak_features.grad is not None
assert noe_features.grad is not None
assert torch.any(input_features.grad != 0)
```

### Requirement: Documentation for transformer layer and network
The `NMRTransformerLayer` and `NMRTransformerNet` classes SHALL have comprehensive docstrings documenting:
- Architecture overview and motivation (alternative to triple-based message passing)
- Attention operation sequence and information flow pattern
- Edge type requirements for graph construction
- Configuration usage from `ModelConfig`
- Comparison with triple-based architecture (NMRLayer and NMRNet)

#### Scenario: Docstrings are comprehensive
```python
from nmr.models.network import NMRTransformerLayer, NMRTransformerNet

# Verify docstrings exist and are informative
assert NMRTransformerLayer.__doc__ is not None
assert len(NMRTransformerLayer.__doc__) > 300
assert 'transformer' in NMRTransformerLayer.__doc__.lower() or 'attention' in NMRTransformerLayer.__doc__.lower()

assert NMRTransformerNet.__doc__ is not None
assert len(NMRTransformerNet.__doc__) > 200
```

### Requirement: TriAxialAttention class structure
The `TriAxialAttention` class SHALL be implemented in `nmr/models/transformer.py` as a HeteroData wrapper for triple-source GATv2-style attention with the following structure:
- Accept parameters: `source_type_1`, `source_type_2`, `source_type_3`, `dest_type`, `channels`, `head_dim`, `heads`, `negative_slope`, `edge_name_1`, `edge_name_2`, `edge_name_3`, `hidden_size`, and `device`
- Create three attention cores (one for each source type to destination type)
- Include pre-normalization layers for all three sources and destination
- Include a destination feature transformation layer
- Include a combination MLP that merges four components (delta_1 + delta_2 + delta_3 + dest_transformed)
- Support attention across any node type combination (Residue, Peak, Noe)

#### Scenario: Instantiate TriAxialAttention for triple Residue-to-Noe attention
```python
from nmr.models.transformer import TriAxialAttention

attention = TriAxialAttention(
    source_type_1="Residue",
    source_type_2="Residue",
    source_type_3="Residue",
    dest_type="Noe",
    channels=64,
    head_dim=16,
    heads=4,
    device='cpu'
)

# Verify attributes exist
assert hasattr(attention, 'attention_1')
assert hasattr(attention, 'attention_2')
assert hasattr(attention, 'attention_3')
assert hasattr(attention, 'dest_linear')
assert hasattr(attention, 'combine_mlp')
assert attention.source_type_1 == "Residue"
assert attention.source_type_2 == "Residue"
assert attention.source_type_3 == "Residue"
assert attention.dest_type == "Noe"
```

#### Scenario: Instantiate with custom edge names
```python
attention = TriAxialAttention(
    source_type_1="Peak",
    source_type_2="Residue",
    source_type_3="Noe",
    dest_type="Value",
    channels=64,
    head_dim=16,
    heads=4,
    edge_name_1="edge1",
    edge_name_2="edge2",
    edge_name_3="edge3",
    device='cpu'
)

assert attention.edge_type_1 == ("Peak", "edge1", "Value")
assert attention.edge_type_2 == ("Residue", "edge2", "Value")
assert attention.edge_type_3 == ("Noe", "edge3", "Value")
```

#### Scenario: Instantiate with custom hidden size
```python
attention = TriAxialAttention(
    source_type_1="Residue",
    source_type_2="Peak",
    source_type_3="Noe",
    dest_type="Noe",
    channels=64,
    head_dim=16,
    heads=4,
    hidden_size=256,  # Custom hidden dimension
    device='cpu'
)

# Verify MLP input is channels * 4 (3 deltas + dest_transformed)
assert attention.combine_mlp[0].in_features == 64 * 4
assert attention.combine_mlp[0].out_features == 256
assert attention.combine_mlp[2].out_features == 64
```

### Requirement: TriAxialAttention conditional spatial attention
The `TriAxialAttention` class SHALL conditionally use `SpatialAttentionCore` for each attention stream (attention_1, attention_2, and attention_3) when the corresponding source type and destination type are both "Residue". For all other node type combinations, it SHALL use `AttentionCore`.

#### Scenario: All three attention streams use SpatialAttentionCore for Residue-to-Residue
```python
from nmr.models.transformer import TriAxialAttention, SpatialAttentionCore

# When all three sources and dest are Residue, all streams use spatial attention
attention = TriAxialAttention(
    source_type_1="Residue",
    source_type_2="Residue",
    source_type_3="Residue",
    dest_type="Residue",
    channels=64,
    head_dim=16,
    heads=4,
    device='cpu'
)

assert isinstance(attention.attention_1, SpatialAttentionCore)
assert isinstance(attention.attention_2, SpatialAttentionCore)
assert isinstance(attention.attention_3, SpatialAttentionCore)
```

#### Scenario: Mixed attention types - some spatial, some non-spatial
```python
from nmr.models.transformer import TriAxialAttention, SpatialAttentionCore, AttentionCore

# First two sources are Residue (spatial), third is Peak (non-spatial), dest is Residue
attention = TriAxialAttention(
    source_type_1="Residue",
    source_type_2="Residue",
    source_type_3="Peak",
    dest_type="Residue",
    channels=64,
    head_dim=16,
    heads=4,
    device='cpu'
)

# First two should use spatial core (Residue->Residue)
assert isinstance(attention.attention_1, SpatialAttentionCore)
assert isinstance(attention.attention_2, SpatialAttentionCore)
# Third uses non-spatial core (Peak->Residue)
assert isinstance(attention.attention_3, AttentionCore)
```

#### Scenario: No spatial attention for non-Residue destination
```python
# When dest is not Residue, no streams use spatial attention
attention = TriAxialAttention(
    source_type_1="Residue",
    source_type_2="Residue",
    source_type_3="Residue",
    dest_type="Noe",
    channels=64,
    head_dim=16,
    heads=4,
    device='cpu'
)

assert isinstance(attention.attention_1, AttentionCore)
assert isinstance(attention.attention_2, AttentionCore)
assert isinstance(attention.attention_3, AttentionCore)
```

### Requirement: TriAxialAttention forward method with coordinate passing
The `TriAxialAttention.forward()` method SHALL:
- Check each attention core type independently using `isinstance(core, SpatialAttentionCore)`
- When an attention core is `SpatialAttentionCore`, extract `.xyz` coordinates from the corresponding source and destination nodes and pass them to the core's forward method
- When an attention core is `AttentionCore`, pass only features and edge indices
- Handle all three streams independently (any combination of spatial/non-spatial)
- Apply pre-normalization to all source and destination features before attention
- Concatenate four components: delta_1, delta_2, delta_3, and dest_transformed (each of dimension `channels`)
- Apply combination MLP to the concatenated features (input dimension: `channels * 4`)
- Apply residual update to destination features: `dest.x = dest.x + delta`

#### Scenario: Triple Residue-to-Residue attention uses coordinates for all streams
```python
import torch
from torch_geometric.data import HeteroData
from nmr.models.transformer import TriAxialAttention

# Create test data with all Residue nodes
data = HeteroData()
num_residues = 10
data["Residue"].x = torch.randn(num_residues, 64)
data["Residue"].xyz = torch.randn(num_residues, 3)

# Create edges for all three attention streams
edge_index = torch.combinations(torch.arange(num_residues), r=2, with_replacement=True).t()
data[("Residue", "triaxial_attn_1", "Residue")].edge_index = edge_index
data[("Residue", "triaxial_attn_2", "Residue")].edge_index = edge_index
data[("Residue", "triaxial_attn_3", "Residue")].edge_index = edge_index

# Create triple-attention module
attention = TriAxialAttention(
    source_type_1="Residue",
    source_type_2="Residue",
    source_type_3="Residue",
    dest_type="Residue",
    channels=64,
    head_dim=16,
    heads=4,
    device='cpu'
)

# Forward pass should use xyz coordinates for all three streams
output = attention(data)

# Verify output shape
assert output["Residue"].x.shape == (num_residues, 64)
```

#### Scenario: Mixed spatial and non-spatial streams with three different source types
```python
import torch
from torch_geometric.data import HeteroData
from nmr.models.transformer import TriAxialAttention

# Create test data with Residue, Peak, and Noe nodes
data = HeteroData()
num_residues = 8
num_peaks = 12
num_noes = 5
num_values = 6

data["Residue"].x = torch.randn(num_residues, 64)
data["Residue"].xyz = torch.randn(num_residues, 3)
data["Peak"].x = torch.randn(num_peaks, 64)
data["Noe"].x = torch.randn(num_noes, 64)
data["Value"].x = torch.randn(num_values, 64)

# Create edges: Residue->Value, Peak->Value, Noe->Value
edge_index_1 = torch.randint(0, num_residues, (2, 20))
edge_index_2 = torch.randint(0, num_peaks, (2, 20))
edge_index_3 = torch.randint(0, num_noes, (2, 20))
data[("Residue", "triaxial_attn_1", "Value")].edge_index = edge_index_1
data[("Peak", "triaxial_attn_2", "Value")].edge_index = edge_index_2
data[("Noe", "triaxial_attn_3", "Value")].edge_index = edge_index_3

# All cores are non-spatial since dest is Value
attention = TriAxialAttention(
    source_type_1="Residue",
    source_type_2="Peak",
    source_type_3="Noe",
    dest_type="Value",
    channels=64,
    head_dim=16,
    heads=4,
    device='cpu'
)

output = attention(data)
assert output["Value"].x.shape == (num_values, 64)
```

#### Scenario: MLP processes four-way concatenation
```python
import torch
from torch_geometric.data import HeteroData
from nmr.models.transformer import TriAxialAttention

data = HeteroData()
num_nodes = 5

# Single node type for simplicity
data["Residue"].x = torch.randn(num_nodes, 64)
data["Residue"].xyz = torch.randn(num_nodes, 3)

edge_index = torch.combinations(torch.arange(num_nodes), r=2, with_replacement=True).t()
data[("Residue", "triaxial_attn_1", "Residue")].edge_index = edge_index
data[("Residue", "triaxial_attn_2", "Residue")].edge_index = edge_index
data[("Residue", "triaxial_attn_3", "Residue")].edge_index = edge_index

attention = TriAxialAttention(
    source_type_1="Residue",
    source_type_2="Residue",
    source_type_3="Residue",
    dest_type="Residue",
    channels=64,
    head_dim=16,
    heads=4,
    device='cpu'
)

# Verify MLP processes channels * 4 input
assert attention.combine_mlp[0].in_features == 64 * 4  # delta_1 + delta_2 + delta_3 + dest_transformed

output = attention(data)
assert output["Residue"].x.shape == (num_nodes, 64)
```

### Requirement: Distance affects TriAxial Residue attention
When using Residue-to-Residue triple-attention, the output SHALL be affected by the Euclidean distance between node coordinates for streams using `SpatialAttentionCore`.

#### Scenario: Different coordinates produce different triple-attention outputs
```python
import torch
from torch_geometric.data import HeteroData
from nmr.models.transformer import TriAxialAttention

# Create two scenarios with same features but different coordinates
num_residues = 6
x = torch.randn(num_residues, 64)
edge_index = torch.tensor([[0, 1, 2], [1, 2, 3]])

# Scenario 1: All residues at same position (distance = 0)
data_same = HeteroData()
data_same["Residue"].x = x.clone()
data_same["Residue"].xyz = torch.zeros(num_residues, 3)
data_same[("Residue", "triaxial_attn_1", "Residue")].edge_index = edge_index
data_same[("Residue", "triaxial_attn_2", "Residue")].edge_index = edge_index
data_same[("Residue", "triaxial_attn_3", "Residue")].edge_index = edge_index

# Scenario 2: Residues at different positions (varying distances)
data_diff = HeteroData()
data_diff["Residue"].x = x.clone()
data_diff["Residue"].xyz = torch.randn(num_residues, 3)
data_diff[("Residue", "triaxial_attn_1", "Residue")].edge_index = edge_index
data_diff[("Residue", "triaxial_attn_2", "Residue")].edge_index = edge_index
data_diff[("Residue", "triaxial_attn_3", "Residue")].edge_index = edge_index

# Create triple-attention module
attention = TriAxialAttention(
    source_type_1="Residue",
    source_type_2="Residue",
    source_type_3="Residue",
    dest_type="Residue",
    channels=64,
    head_dim=16,
    heads=4,
    device='cpu'
)

# Set seed for reproducibility
torch.manual_seed(42)
attention.reset_parameters()

output_same = attention(data_same)
output_diff = attention(data_diff)

# Outputs should differ because distance affects all three attention streams
assert not torch.allclose(output_same["Residue"].x, output_diff["Residue"].x, atol=1e-5)
```

### Requirement: TriAxialAttention gradient flow through spatial paths
When using Residue-to-Residue triple-attention, gradients SHALL flow through both node features (.x) and coordinates (.xyz) for streams using `SpatialAttentionCore`.

#### Scenario: Gradients backpropagate through coordinates in triple-attention
```python
import torch
from torch_geometric.data import HeteroData
from nmr.models.transformer import TriAxialAttention

data = HeteroData()
num_residues = 6

# Create inputs with gradient tracking
input_features = torch.randn(num_residues, 64, requires_grad=True)
input_xyz = torch.randn(num_residues, 3, requires_grad=True)

data["Residue"].x = input_features
data["Residue"].xyz = input_xyz

edge_index = torch.combinations(torch.arange(num_residues), r=2, with_replacement=True).t()
data[("Residue", "triaxial_attn_1", "Residue")].edge_index = edge_index
data[("Residue", "triaxial_attn_2", "Residue")].edge_index = edge_index
data[("Residue", "triaxial_attn_3", "Residue")].edge_index = edge_index

attention = TriAxialAttention(
    source_type_1="Residue",
    source_type_2="Residue",
    source_type_3="Residue",
    dest_type="Residue",
    channels=64,
    head_dim=16,
    heads=4,
    device='cpu'
)

# Forward and backward pass
output = attention(data)
loss = output["Residue"].x.sum()
loss.backward()

# Verify gradients exist on both features and coordinates
assert input_features.grad is not None
assert input_xyz.grad is not None
assert torch.any(input_features.grad != 0)
assert torch.any(input_xyz.grad != 0)
```

### Requirement: TriAxialAttention empty set handling
The `forward()` method SHALL handle empty node sets and empty edge sets gracefully for all three attention streams without raising errors.

#### Scenario: Empty destination node set
```python
import torch
from torch_geometric.data import HeteroData
from nmr.models.transformer import TriAxialAttention

data = HeteroData()
data["Residue"].x = torch.zeros(0, 64)
data["Residue"].xyz = torch.zeros(0, 3)
data[("Residue", "triaxial_attn_1", "Residue")].edge_index = torch.zeros(2, 0, dtype=torch.long)
data[("Residue", "triaxial_attn_2", "Residue")].edge_index = torch.zeros(2, 0, dtype=torch.long)
data[("Residue", "triaxial_attn_3", "Residue")].edge_index = torch.zeros(2, 0, dtype=torch.long)

attention = TriAxialAttention(
    source_type_1="Residue",
    source_type_2="Residue",
    source_type_3="Residue",
    dest_type="Residue",
    channels=64,
    head_dim=16,
    heads=4,
    device='cpu'
)

# Should not crash
output = attention(data)
assert output["Residue"].x.shape == (0, 64)
```

#### Scenario: Empty source node set (one of three sources)
```python
import torch
from torch_geometric.data import HeteroData
from nmr.models.transformer import TriAxialAttention

data = HeteroData()
num_residues = 5
num_peaks = 5

data["Residue"].x = torch.randn(num_residues, 64)
data["Residue"].xyz = torch.randn(num_residues, 3)
data["Peak"].x = torch.randn(num_peaks, 64)
data["Noe"].x = torch.zeros(0, 64)  # Empty third source

edge_index_1 = torch.randint(0, num_residues, (2, 10))
edge_index_2 = torch.randint(0, num_peaks, (2, 10))
edge_index_3 = torch.zeros(2, 0, dtype=torch.long)

data[("Residue", "triaxial_attn_1", "Residue")].edge_index = edge_index_1
data[("Peak", "triaxial_attn_2", "Residue")].edge_index = edge_index_2
data[("Noe", "triaxial_attn_3", "Residue")].edge_index = edge_index_3

attention = TriAxialAttention(
    source_type_1="Residue",
    source_type_2="Peak",
    source_type_3="Noe",
    dest_type="Residue",
    channels=64,
    head_dim=16,
    heads=4,
    device='cpu'
)

# Should return early without crashing
output = attention(data)
assert output["Residue"].x.shape == (num_residues, 64)
```

#### Scenario: Empty edge set for one stream
```python
import torch
from torch_geometric.data import HeteroData
from nmr.models.transformer import TriAxialAttention

data = HeteroData()
num_nodes = 5

data["Residue"].x = torch.randn(num_nodes, 64)
data["Residue"].xyz = torch.randn(num_nodes, 3)

edge_index = torch.combinations(torch.arange(num_nodes), r=2, with_replacement=True).t()
data[("Residue", "triaxial_attn_1", "Residue")].edge_index = edge_index
data[("Residue", "triaxial_attn_2", "Residue")].edge_index = edge_index
data[("Residue", "triaxial_attn_3", "Residue")].edge_index = torch.zeros(2, 0, dtype=torch.long)  # No edges

attention = TriAxialAttention(
    source_type_1="Residue",
    source_type_2="Residue",
    source_type_3="Residue",
    dest_type="Residue",
    channels=64,
    head_dim=16,
    heads=4,
    device='cpu'
)

# Should return early without crashing when any edge set is empty
output = attention(data)
assert output["Residue"].x.shape == (num_nodes, 64)
```

### Requirement: TriAxialAttention backward compatibility
The implementation of `TriAxialAttention` SHALL NOT modify or break existing functionality of `MonoAxialAttention`, `BiAxialAttention`, `AttentionCore`, or `SpatialAttentionCore`.

#### Scenario: Existing classes unchanged after adding TriAxialAttention
```python
from nmr.models.transformer import (
    MonoAxialAttention,
    BiAxialAttention,
    AttentionCore,
    SpatialAttentionCore,
    TriAxialAttention
)

# Verify all classes are importable
assert MonoAxialAttention is not None
assert BiAxialAttention is not None
assert AttentionCore is not None
assert SpatialAttentionCore is not None
assert TriAxialAttention is not None

# Verify TriAxialAttention doesn't share mutable state with other classes
mono = MonoAxialAttention("Residue", "Residue", 64, 64, 16, 4, device='cpu')
tri = TriAxialAttention("Residue", "Residue", "Residue", "Residue", 64, 16, 4, device='cpu')

assert mono is not tri
assert mono.core is not tri.attention_1
```

#### Scenario: MonoAxialAttention and BiAxialAttention still work correctly
```python
import torch
from torch_geometric.data import HeteroData
from nmr.models.transformer import MonoAxialAttention, BiAxialAttention

# Test MonoAxialAttention still works
data = HeteroData()
num_nodes = 5
data["Peak"].x = torch.randn(num_nodes, 64)
edge_index = torch.combinations(torch.arange(num_nodes), r=2, with_replacement=True).t()
data[("Peak", "self_attn", "Peak")].edge_index = edge_index

mono_attn = MonoAxialAttention("Peak", "Peak", 64, 64, 16, 4, device='cpu')
output_mono = mono_attn(data)
assert output_mono["Peak"].x.shape == (num_nodes, 64)

# Test BiAxialAttention still works
data[("Peak", "biaxial_attn_1", "Peak")].edge_index = edge_index
data[("Peak", "biaxial_attn_2", "Peak")].edge_index = edge_index

bi_attn = BiAxialAttention("Peak", "Peak", "Peak", 64, 16, 4, device='cpu')
output_bi = bi_attn(data)
assert output_bi["Peak"].x.shape == (num_nodes, 64)
```

### Requirement: TriAxialAttention documentation
The `TriAxialAttention` class docstring SHALL document:
- Purpose: Triple-source attention mechanism extending the pattern from MonoAxialAttention and BiAxialAttention
- Conditional use of `SpatialAttentionCore` for Residue-to-Residue attention in each stream
- Use of `AttentionCore` for all other node type combinations
- Requirements for .xyz attribute when using Residue nodes
- Independent handling of each attention stream (mixed spatial/non-spatial allowed)
- Four-way feature combination logic (delta_1, delta_2, delta_3, dest_transformed)
- Edge type naming convention
- Empty set handling behavior
- At least 2-3 usage examples showing common patterns

#### Scenario: Docstring is comprehensive and follows BiAxialAttention pattern
```python
from nmr.models.transformer import TriAxialAttention

# Verify docstring exists and is substantial
assert TriAxialAttention.__doc__ is not None
assert len(TriAxialAttention.__doc__) > 400

doc = TriAxialAttention.__doc__

# Should mention key concepts
assert 'triaxial' in doc.lower() or 'triple' in doc.lower() or 'three' in doc.lower()
assert 'SpatialAttentionCore' in doc or 'spatial' in doc.lower()
assert 'Residue' in doc
assert 'attention' in doc.lower()

# Should mention the three-source pattern
assert 'source_type_1' in doc
assert 'source_type_2' in doc
assert 'source_type_3' in doc
```

#### Scenario: Docstring includes usage examples
```python
from nmr.models.transformer import TriAxialAttention

doc = TriAxialAttention.__doc__

# Should include example code blocks
assert 'Example' in doc or 'example' in doc or '>>>' in doc

# Should show how to instantiate the class
assert 'TriAxialAttention(' in doc
```

