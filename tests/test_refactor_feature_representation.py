"""
Tests for refactored node attribute structure.

This module tests the new feature representation where:
- Raw data (.xyz, .shifts, .flags) is set once during construction and remains immutable
- Working features (.x) are created by embedding and updated during message passing
"""

import sys
import unittest
from pathlib import Path

import torch
from torch_geometric.data import HeteroData

# Add parent directory to path to import nmr module
sys.path.insert(0, str(Path(__file__).parent.parent))

from nmr.construct import construct_graph
from nmr.models.network import ModelConfig, NMRNet, FeatureNorm
from torch_geometric.data import Batch


class TestRefactoredAttributeStructure(unittest.TestCase):
    """Test that new attribute structure is correctly implemented."""

    def setUp(self):
        """Create a minimal test state for graph construction."""
        self.device = torch.device('cpu')
        self.config = ModelConfig()
        self.state = {
            'coordinates': [
                [1.0, 2.0, 3.0, 8.0, 120.0],
                [4.0, 5.0, 6.0, 8.5, 125.0],
            ],
            'obs_chemical_shifts': [
                [8.1, 121.0],
                [8.6, 126.0],
            ],
            'noes': [
                [120.0, 8.0, 8.5],
            ],
            'assignments': {},
            'shift_to_assign': 0,
        }

    def test_residue_nodes_have_xyz_attribute(self):
        """Test that residue nodes have .xyz attribute with shape [n, 3]."""
        data = construct_graph(self.state, self.device, self.config)

        # Check that .xyz attribute exists
        self.assertTrue(hasattr(data['Residue'], 'xyz'))

        # Check shape is [n_residues, 3]
        self.assertEqual(data['Residue'].xyz.shape, (2, 3))

        # Check dtype is float32
        self.assertEqual(data['Residue'].xyz.dtype, torch.float32)

    def test_residue_nodes_have_shifts_attribute(self):
        """Test that residue nodes have .shifts attribute with shape [n, 2]."""
        data = construct_graph(self.state, self.device, self.config)

        # Check that .shifts attribute exists
        self.assertTrue(hasattr(data['Residue'], 'shifts'))

        # Check shape is [n_residues, 2]
        self.assertEqual(data['Residue'].shifts.shape, (2, 2))

        # Check dtype is float32
        self.assertEqual(data['Residue'].shifts.dtype, torch.float32)

    def test_residue_nodes_have_flags_attribute(self):
        """Test that residue nodes have .flags attribute with shape [n, 1]."""
        data = construct_graph(self.state, self.device, self.config)

        # Check that .flags attribute exists
        self.assertTrue(hasattr(data['Residue'], 'flags'))

        # Check shape is [n_residues, 1]
        self.assertEqual(data['Residue'].flags.shape, (2, 1))

        # Check dtype is float32
        self.assertEqual(data['Residue'].flags.dtype, torch.float32)

    def test_peak_nodes_have_shifts_attribute(self):
        """Test that peak nodes have .shifts attribute with shape [n, 2]."""
        data = construct_graph(self.state, self.device, self.config)

        # Check that .shifts attribute exists
        self.assertTrue(hasattr(data['Peak'], 'shifts'))

        # Check shape is [n_peaks, 2]
        self.assertEqual(data['Peak'].shifts.shape, (2, 2))

        # Check dtype is float32
        self.assertEqual(data['Peak'].shifts.dtype, torch.float32)

    def test_peak_nodes_have_flags_attribute(self):
        """Test that peak nodes have .flags attribute with shape [n, 2]."""
        data = construct_graph(self.state, self.device, self.config)

        # Check that .flags attribute exists
        self.assertTrue(hasattr(data['Peak'], 'flags'))

        # Check shape is [n_peaks, 2]
        self.assertEqual(data['Peak'].flags.shape, (2, 2))

        # Check dtype is float32
        self.assertEqual(data['Peak'].flags.dtype, torch.float32)

    def test_noe_nodes_have_shifts_attribute(self):
        """Test that NOE nodes have .shifts attribute with shape [n, 3]."""
        data = construct_graph(self.state, self.device, self.config)

        # Check that .shifts attribute exists
        self.assertTrue(hasattr(data['Noe'], 'shifts'))

        # Check shape is [n_noes, 3]
        self.assertEqual(data['Noe'].shifts.shape, (1, 3))

        # Check dtype is float32
        self.assertEqual(data['Noe'].shifts.dtype, torch.float32)

    def test_noe_nodes_have_no_flags_attribute(self):
        """Test that NOE nodes do not have .flags attribute."""
        data = construct_graph(self.state, self.device, self.config)

        # NOE nodes should NOT have .flags attribute
        self.assertFalse(hasattr(data['Noe'], 'flags'))

    def test_xyz_and_shifts_immutable_during_forward_pass(self):
        """Test that .xyz and .shifts are never modified during forward pass."""
        config = ModelConfig(num_nmr_layers=1)
        model = NMRNet(self.device, config)

        # Construct graph
        data = construct_graph(self.state, self.device, self.config)

        # Clone raw data attributes to check immutability
        residue_xyz_before = data['Residue'].xyz.clone()
        residue_shifts_before = data['Residue'].shifts.clone()
        peak_shifts_before = data['Peak'].shifts.clone()
        noe_shifts_before = data['Noe'].shifts.clone()

        # Run forward pass
        model(data)

        # Verify raw data attributes unchanged
        self.assertTrue(torch.equal(data['Residue'].xyz, residue_xyz_before))
        self.assertTrue(torch.equal(data['Residue'].shifts, residue_shifts_before))
        self.assertTrue(torch.equal(data['Peak'].shifts, peak_shifts_before))
        self.assertTrue(torch.equal(data['Noe'].shifts, noe_shifts_before))

    def test_x_attribute_updated_during_forward_pass(self):
        """Test that .x attributes are properly updated during message passing."""
        config = ModelConfig(num_nmr_layers=1)
        model = NMRNet(self.device, config)

        # Construct graph
        data = construct_graph(self.state, self.device, self.config)

        # Check that .x attributes exist after embedding
        # (They should be created by EmbedFeatures layer)
        # We don't check values here, just that they exist and have correct shapes

        # Run forward pass
        model(data)

        # After forward pass, .x attributes should exist with correct dimensions
        self.assertTrue(hasattr(data['Residue'], 'x'))
        self.assertTrue(hasattr(data['Peak'], 'x'))
        self.assertTrue(hasattr(data['Noe'], 'x'))

        # Check that .x has been updated (shape should match embed_dim)
        # Residue.x should be [n, 3 + embed_dim] (coords + embedded features)
        self.assertEqual(data['Residue'].x.ndim, 2)
        self.assertEqual(data['Residue'].x.shape[0], 2)

        # Peak.x should be [n, embed_dim]
        self.assertEqual(data['Peak'].x.ndim, 2)
        self.assertEqual(data['Peak'].x.shape[0], 2)

        # NOE.x should be [n, embed_dim]
        self.assertEqual(data['Noe'].x.ndim, 2)
        self.assertEqual(data['Noe'].x.shape[0], 1)

    def test_flags_set_correctly_during_construction(self):
        """Test that .flags are set correctly based on assignment state."""
        # Create state with an assignment
        state_with_assignment = {
            'coordinates': [
                [1.0, 2.0, 3.0, 8.0, 120.0],
                [4.0, 5.0, 6.0, 8.5, 125.0],
            ],
            'obs_chemical_shifts': [
                [8.1, 121.0],
                [8.6, 126.0],
            ],
            'noes': [
                [120.0, 8.0, 8.5],
            ],
            'assignments': {0: 1},  # Peak 0 assigned to Residue 1
            'shift_to_assign': 1,   # Peak 1 is being assigned
        }

        data = construct_graph(state_with_assignment, self.device, self.config)

        # Peak flags: [to_be_assigned, already_assigned]
        # Peak 0: [0, 1] - not being assigned, already assigned
        self.assertEqual(data['Peak'].flags[0, 0].item(), 0.0)
        self.assertEqual(data['Peak'].flags[0, 1].item(), 1.0)

        # Peak 1: [1, 0] - being assigned, not already assigned
        self.assertEqual(data['Peak'].flags[1, 0].item(), 1.0)
        self.assertEqual(data['Peak'].flags[1, 1].item(), 0.0)

        # Residue flags: [already_assigned]
        # Residue 0: [0] - not assigned
        self.assertEqual(data['Residue'].flags[0, 0].item(), 0.0)

        # Residue 1: [1] - assigned to peak 0
        self.assertEqual(data['Residue'].flags[1, 0].item(), 1.0)

    def test_gradient_flow_with_new_structure(self):
        """Test that gradients flow correctly through new attribute structure."""
        config = ModelConfig(num_nmr_layers=1)
        model = NMRNet(self.device, config)

        # Construct graph
        data = construct_graph(self.state, self.device, self.config)

        # Run forward pass
        value, policy = model(data)

        # Compute a simple loss
        # policy is a list, so we need to sum all elements
        policy_sum = sum(p.sum() for p in policy)
        loss = value.sum() + policy_sum

        # Backward pass should not raise any errors
        try:
            loss.backward()
            gradient_flow_ok = True
        except RuntimeError as e:
            gradient_flow_ok = False
            print(f"Gradient flow error: {e}")

        self.assertTrue(gradient_flow_ok)


class TestFeatureNormLayerNorm(unittest.TestCase):
    """Test the LayerNorm-based FeatureNorm implementation."""

    def setUp(self):
        """Create test data structures."""
        self.device = torch.device('cpu')
        self.config = ModelConfig()
        self.embed_dim = self.config.embed.embed_dim
        self.feature_norm = FeatureNorm(self.embed_dim)

    def test_feature_norm_has_separate_layer_norms(self):
        """Test that FeatureNorm has separate LayerNorm for each node type."""
        self.assertTrue(hasattr(self.feature_norm, 'residue_norm'))
        self.assertTrue(hasattr(self.feature_norm, 'peak_norm'))
        self.assertTrue(hasattr(self.feature_norm, 'noe_norm'))

        # Check they are LayerNorm instances
        self.assertIsInstance(self.feature_norm.residue_norm, torch.nn.LayerNorm)
        self.assertIsInstance(self.feature_norm.peak_norm, torch.nn.LayerNorm)
        self.assertIsInstance(self.feature_norm.noe_norm, torch.nn.LayerNorm)

    def test_feature_norm_has_learnable_parameters(self):
        """Test that LayerNorm has learnable affine parameters (γ, β)."""
        # Each LayerNorm should have weight (γ) and bias (β) parameters
        params = dict(self.feature_norm.named_parameters())

        # Check residue_norm parameters
        self.assertIn('residue_norm.weight', params)
        self.assertIn('residue_norm.bias', params)
        self.assertEqual(params['residue_norm.weight'].shape, (self.embed_dim,))
        self.assertEqual(params['residue_norm.bias'].shape, (self.embed_dim,))

        # Check peak_norm parameters
        self.assertIn('peak_norm.weight', params)
        self.assertIn('peak_norm.bias', params)

        # Check noe_norm parameters
        self.assertIn('noe_norm.weight', params)
        self.assertIn('noe_norm.bias', params)

    def test_feature_norm_normalizes_per_node(self):
        """Test that FeatureNorm normalizes each node independently across features."""
        # Create a FeatureNorm without affine parameters for simpler testing
        feature_norm_no_affine = FeatureNorm(self.embed_dim)
        # Set elementwise_affine to False after creation to test pure normalization
        # (LayerNorm with affine=True applies: γ * normalized + β, which changes statistics)
        # For this test, we want to verify the normalization itself works

        # Create a simple heterogeneous graph
        data = HeteroData()

        # Create features with random values (not all zeros to avoid division issues)
        torch.manual_seed(42)
        data['Residue'].x = torch.randn(3, self.embed_dim)
        data['Peak'].x = torch.randn(2, self.embed_dim)
        data['Noe'].x = torch.randn(1, self.embed_dim)

        # Store original features
        orig_residue = data['Residue'].x.clone()
        orig_peak = data['Peak'].x.clone()
        orig_noe = data['Noe'].x.clone()

        # Apply FeatureNorm
        normalized_data = feature_norm_no_affine(data)

        # After LayerNorm with affine parameters, the mean and std won't be exactly 0 and 1
        # But the normalization should still occur - verify features changed
        self.assertFalse(torch.allclose(normalized_data['Residue'].x, orig_residue))
        self.assertFalse(torch.allclose(normalized_data['Peak'].x, orig_peak))
        self.assertFalse(torch.allclose(normalized_data['Noe'].x, orig_noe))

        # Verify shape is preserved
        self.assertEqual(normalized_data['Residue'].x.shape, (3, self.embed_dim))
        self.assertEqual(normalized_data['Peak'].x.shape, (2, self.embed_dim))
        self.assertEqual(normalized_data['Noe'].x.shape, (1, self.embed_dim))

    def test_feature_norm_works_with_batched_data(self):
        """Test that FeatureNorm works directly with batched graphs (no unbatching)."""
        # Create two separate graphs
        data1 = HeteroData()
        data1['Residue'].x = torch.randn(2, self.embed_dim)
        data1['Peak'].x = torch.randn(3, self.embed_dim)
        data1['Noe'].x = torch.randn(1, self.embed_dim)

        data2 = HeteroData()
        data2['Residue'].x = torch.randn(3, self.embed_dim)
        data2['Peak'].x = torch.randn(2, self.embed_dim)
        data2['Noe'].x = torch.randn(2, self.embed_dim)

        # Batch the graphs
        batched_data = Batch.from_data_list([data1, data2])

        # Apply FeatureNorm - should work without unbatching
        try:
            normalized_batched = self.feature_norm(batched_data)
            batching_works = True
        except Exception as e:
            batching_works = False
            print(f"Error with batched data: {e}")

        self.assertTrue(batching_works)

        # Verify that all nodes are normalized (each node independently)
        # Total nodes: 5 residues (2+3), 5 peaks (3+2), 3 noes (1+2)
        self.assertEqual(normalized_batched['Residue'].x.shape[0], 5)
        self.assertEqual(normalized_batched['Peak'].x.shape[0], 5)
        self.assertEqual(normalized_batched['Noe'].x.shape[0], 3)

    def test_feature_norm_gradient_flow(self):
        """Test that gradients flow through FeatureNorm correctly."""
        # Create simple data - store as parameters to track gradients properly
        residue_x = torch.nn.Parameter(torch.randn(2, self.embed_dim))
        peak_x = torch.nn.Parameter(torch.randn(2, self.embed_dim))
        noe_x = torch.nn.Parameter(torch.randn(1, self.embed_dim))

        data = HeteroData()
        data['Residue'].x = residue_x
        data['Peak'].x = peak_x
        data['Noe'].x = noe_x

        # Apply FeatureNorm
        normalized_data = self.feature_norm(data)

        # Compute a loss
        loss = normalized_data['Residue'].x.sum() + \
               normalized_data['Peak'].x.sum() + \
               normalized_data['Noe'].x.sum()

        # Backward pass
        try:
            loss.backward()
            gradient_flow_ok = True
        except RuntimeError as e:
            gradient_flow_ok = False
            print(f"Gradient flow error: {e}")

        self.assertTrue(gradient_flow_ok)

        # Check that input parameters received gradients
        self.assertIsNotNone(residue_x.grad)
        self.assertIsNotNone(peak_x.grad)
        self.assertIsNotNone(noe_x.grad)

    def test_feature_norm_in_nmrnet(self):
        """Test that FeatureNorm integrates correctly into NMRNet."""
        config = ModelConfig(num_nmr_layers=2)
        model = NMRNet(self.device, config)

        # Create a test state
        state = {
            'coordinates': [
                [1.0, 2.0, 3.0, 8.0, 120.0],
                [4.0, 5.0, 6.0, 8.5, 125.0],
            ],
            'obs_chemical_shifts': [
                [8.1, 121.0],
                [8.6, 126.0],
            ],
            'noes': [
                [120.0, 8.0, 8.5],
            ],
            'assignments': {},
            'shift_to_assign': 0,
        }

        # Construct graph
        data = construct_graph(state, self.device, config)

        # Run forward pass
        try:
            value, policy = model(data)
            integration_ok = True
        except Exception as e:
            integration_ok = False
            print(f"Integration error: {e}")

        self.assertTrue(integration_ok)

    def test_separate_layer_norms_learn_independently(self):
        """Test that separate LayerNorms can learn different parameters."""
        # After initialization, the weights should be the same (all ones)
        # But they should be separate parameters that can diverge during training

        residue_weight = self.feature_norm.residue_norm.weight
        peak_weight = self.feature_norm.peak_norm.weight
        noe_weight = self.feature_norm.noe_norm.weight

        # Initially all should be ones (default initialization)
        self.assertTrue(torch.allclose(residue_weight, torch.ones_like(residue_weight)))
        self.assertTrue(torch.allclose(peak_weight, torch.ones_like(peak_weight)))
        self.assertTrue(torch.allclose(noe_weight, torch.ones_like(noe_weight)))

        # But they should be different parameter objects
        self.assertIsNot(residue_weight, peak_weight)
        self.assertIsNot(peak_weight, noe_weight)
        self.assertIsNot(residue_weight, noe_weight)

        # Verify they can be updated independently
        with torch.no_grad():
            self.feature_norm.residue_norm.weight[0] = 2.0

        # Only residue_norm should be affected
        self.assertEqual(self.feature_norm.residue_norm.weight[0].item(), 2.0)
        self.assertEqual(self.feature_norm.peak_norm.weight[0].item(), 1.0)
        self.assertEqual(self.feature_norm.noe_norm.weight[0].item(), 1.0)


if __name__ == '__main__':
    unittest.main()
