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
from nmr.models.network import ModelConfig, NMRNet
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
        # Residue.x should be [n, embed_dim] (embedded features only)
        self.assertEqual(data['Residue'].x.ndim, 2)
        self.assertEqual(data['Residue'].x.shape[0], 2)
        self.assertEqual(data['Residue'].x.shape[1], self.config.embed.embed_dim)

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


if __name__ == '__main__':
    unittest.main()
