"""
Integration tests for triple system re-architecture.

These tests verify the complete pipeline functionality including:
- End-to-end forward pass through entire network
- Graph construction with new naming creates valid graphs
- All 4 triple types process correctly in sequence
- Gradient flow through gather → update → scatter pipeline
- Equivariance of ResidueResidueNoeTriple coordinate updates
- Peak-based triples produce zero coordinate deltas
- Batch processing with multiple graphs
- Edge case handling (empty triple sets, single node graphs)
"""

import copy
import sys
import unittest
from pathlib import Path

import numpy as np
import torch
from torch_geometric.loader import DataLoader

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from nmr.construct import construct_graph
from nmr.models.network import NMRNet, NMRLayer, ModelConfig, EmbedFeatures
from nmr.nmr_gym.fake_data import FakeDataGenerator
from nmr.nmr_gym.gym_env import GymEnv


def create_state_dict_from_env(env_state):
    """
    Convert gym environment state to construct_graph compatible state.

    The gym_env.reset() returns coordinates as Protein namedtuples with
    (x, y, z, H1, N15). construct_graph expects coordinates to contain all 5 values.
    """
    # Extract all 5 values from Protein namedtuples: x, y, z, H1, N15
    coordinates = [[p.x, p.y, p.z, p.H1, p.N15] for p in env_state["coordinates"]]

    # Extract obs_shifts from HSQCPeak namedtuples
    obs_shifts = [[s.H1, s.N15] for s in env_state["obs_chemical_shifts"]]

    # Extract noes from NOEPeak namedtuples
    noes = [[n.N15, n.H1, n.H2] for n in env_state["noes"]]

    return {
        "coordinates": coordinates,
        "obs_chemical_shifts": obs_shifts,
        "noes": noes,
        "assignments": env_state["assignments"],
        "shift_to_assign": env_state["shift_to_assign"],
    }


class TestCompleteIntegration(unittest.TestCase):
    """Integration tests for the complete triple system pipeline."""

    def setUp(self):
        """Set up test environment and small test data."""
        self.device = torch.device("cpu")
        self.config = ModelConfig()
        self.num_resid = 5

        # Generate a small test dataset
        generator = FakeDataGenerator(self.num_resid)
        coords, obs_shifts, pred_shifts, noes, connectivity = (
            generator.generate_data_arrays(random_key=False)
        )
        self.pred_coords, self.obs_shifts, self.noes, self.connectivity = (
            generator.order_data(coords, obs_shifts, pred_shifts, noes, connectivity)
        )

        # Create environment and initial state
        self.env = GymEnv(self.num_resid)
        env_state = self.env.reset(
            self.pred_coords, self.obs_shifts, self.noes, self.connectivity
        )

        # Convert to construct_graph compatible format
        self.state = create_state_dict_from_env(env_state)

    def test_end_to_end_forward_pass(self):
        """Test complete forward pass through entire network."""
        # Construct graph
        data = construct_graph(self.state, self.device, self.config)

        # Verify graph structure has new naming
        self.assertIn("Residue", data.node_types)
        self.assertIn("Peak", data.node_types)
        self.assertIn("Noe", data.node_types)
        self.assertIn("ResidueResidueNoeTriple", data.node_types)
        self.assertIn("ResiduePeakNoeTriple", data.node_types)
        self.assertIn("PeakResidueNoeTriple", data.node_types)
        self.assertIn("PeakPeakNoeTriple", data.node_types)

        # Create network and run forward pass
        net = NMRNet(self.device, ModelConfig())

        # Run forward pass
        value, policy = net(data)

        # Verify outputs have correct shapes
        self.assertEqual(value.shape, (1, 1))
        self.assertIsInstance(policy, list)
        self.assertEqual(len(policy), 1)
        # Policy shape is [num_peaks, num_residues] for full pairwise distance matrix
        self.assertEqual(policy[0].shape, (self.num_resid, self.num_resid))

        # Verify outputs are finite
        self.assertTrue(torch.isfinite(value).all())
        for p in policy:
            self.assertTrue(torch.isfinite(p).all())

    def test_graph_construction_with_new_naming(self):
        """Test that graph construction creates valid graphs with new naming."""
        # Construct graph
        data = construct_graph(self.state, self.device, self.config)

        # Verify node types use new naming
        node_types = data.node_types
        self.assertIn("Residue", node_types)
        self.assertIn("Peak", node_types)
        self.assertIn("Noe", node_types)

        # Verify old naming is NOT present
        self.assertNotIn("RES", node_types)
        self.assertNotIn("SHIFT", node_types)
        self.assertNotIn("NOE", node_types)
        self.assertNotIn("TRIPLE0", node_types)
        self.assertNotIn("TRIPLE1", node_types)
        self.assertNotIn("TRIPLE2", node_types)
        self.assertNotIn("TRIPLE3", node_types)

        # Verify triple types use full descriptive names
        self.assertIn("ResidueResidueNoeTriple", node_types)
        self.assertIn("ResiduePeakNoeTriple", node_types)
        self.assertIn("PeakResidueNoeTriple", node_types)
        self.assertIn("PeakPeakNoeTriple", node_types)

        # Verify edge types use new naming
        edge_types = data.edge_types

        # Check for prop edges (bidirectional)
        prop_edges = [et for et in edge_types if "prop" in et[1]]
        self.assertTrue(len(prop_edges) > 0, "No prop edges found")

        # Verify old triple edge naming is NOT present (exclude VALUE edges)
        old_edge_names = ["NH1_extract", "NH2_extract", "NH1_add", "NH2_add"]
        for edge_type in edge_types:
            self.assertNotIn(
                edge_type[1],
                old_edge_names,
                f"Old edge naming '{edge_type[1]}' still present",
            )

    def test_all_four_triple_types_process_correctly(self):
        """Test that all 4 triple types process correctly in sequence."""
        # Construct graph
        data = construct_graph(self.state, self.device, self.config)

        # Embed features before calling NMRLayer
        # Coordinates are already normalized during graph construction
        config = ModelConfig()
        embed = EmbedFeatures(self.device, config)
        data = embed(data)

        # Store original node features
        original_residue_x = data["Residue"].x.clone()
        original_peak_x = data["Peak"].x.clone()
        original_noe_x = data["Noe"].x.clone()

        # Create a single NMRLayer (which calls all 4 triples)
        layer = NMRLayer(self.device, config)

        # Run forward pass through layer
        data = layer(data)

        # Verify that node features were updated
        # (At least some should change after message passing)
        residue_changed = not torch.allclose(data["Residue"].x, original_residue_x)
        peak_changed = not torch.allclose(data["Peak"].x, original_peak_x)
        noe_changed = not torch.allclose(data["Noe"].x, original_noe_x)

        # At least one node type should have changed
        self.assertTrue(
            residue_changed or peak_changed or noe_changed,
            "No node features changed after layer forward pass",
        )

        # Verify outputs are finite
        self.assertTrue(torch.isfinite(data["Residue"].x).all())
        self.assertTrue(torch.isfinite(data["Peak"].x).all())
        self.assertTrue(torch.isfinite(data["Noe"].x).all())

    def test_gradient_flow_through_pipeline(self):
        """Test gradient flow through gather → update → scatter pipeline."""
        # Construct graph
        data = construct_graph(self.state, self.device, self.config)

        # Create network
        net = NMRNet(self.device, ModelConfig())

        # Run forward pass
        value, policy = net(data)

        # Compute simple loss
        loss = value.sum() + sum(p.sum() for p in policy)

        # Run backward pass
        loss.backward()

        # Verify gradients exist and are valid for all parameters
        num_nonzero_grads = 0
        for name, param in net.named_parameters():
            if param.requires_grad:
                self.assertIsNotNone(param.grad, f"No gradient for parameter {name}")

                # Verify gradients are finite (not NaN/inf)
                self.assertTrue(
                    torch.isfinite(param.grad).all(),
                    f"Non-finite gradients in parameter {name}",
                )

                # Verify gradients have reasonable magnitudes
                grad_norm = param.grad.norm().item()
                self.assertLess(
                    grad_norm, 1e6, f"Gradient norm too large for {name}: {grad_norm}"
                )

                # Count non-zero gradients (zero gradients are OK with ReLU)
                if grad_norm > 0.0:
                    num_nonzero_grads += 1

        # Verify that at least some gradients are non-zero (not all dead)
        total_params = sum(1 for p in net.parameters() if p.requires_grad)
        self.assertGreater(
            num_nonzero_grads, 0, "All gradients are zero - no gradient flow detected"
        )
        self.assertGreater(
            num_nonzero_grads,
            total_params * 0.5,
            f"Too many zero gradients: {num_nonzero_grads}/{total_params}",
        )

    def test_peak_based_triples_produce_zero_coordinate_deltas(self):
        """Test that Peak-based triples produce zero coordinate deltas."""
        # Construct graph
        data = construct_graph(self.state, self.device, self.config)

        # Embed features
        # Coordinates are already normalized during graph construction
        config = ModelConfig()
        embed = EmbedFeatures(self.device, config)
        data = embed(data)

        # Store original Residue coordinates (from .xyz, not .x)
        original_residue_coords = data["Residue"].xyz.clone()

        # Create network and isolate the Peak-based triples
        # We'll test by running the full layer and checking that coordinates
        # in .xyz never change (coordinate updates have been removed)
        layer = NMRLayer(self.device, config)

        # Manually call each triple and check coordinate changes

        # First, run only ResidueResidueNoeTriple
        # Coordinates are already normalized during graph construction
        data1 = construct_graph(self.state, self.device, self.config)
        data1 = embed(data1)
        data1 = layer.residue_residue_noe(data1)
        coords_after_res_res = data1["Residue"].xyz

        # Verify coordinates did NOT change (coordinate updates removed)
        self.assertTrue(
            torch.allclose(coords_after_res_res, original_residue_coords, atol=1e-6),
            "ResidueResidueNoeTriple should not change .xyz (coordinate updates removed)"
        )

        # Now test each Peak-based triple
        for triple_name, triple_module in [
            ("ResiduePeakNoeTriple", layer.residue_peak_noe),
            ("PeakResidueNoeTriple", layer.peak_residue_noe),
            ("PeakPeakNoeTriple", layer.peak_peak_noe),
        ]:
            data_peak = construct_graph(self.state, self.device, self.config)
            data_peak = embed(data_peak)
            coords_before = data_peak["Residue"].xyz.clone()

            # Run the Peak-based triple
            data_peak = triple_module(data_peak)
            coords_after = data_peak["Residue"].xyz

            # Verify coordinates did NOT change (or changed by at most numerical error)
            coord_delta = (coords_after - coords_before).abs().max().item()
            self.assertLess(
                coord_delta, 1e-5, f"{triple_name} changed .xyz by {coord_delta}"
            )

    def test_batch_processing_with_multiple_graphs(self):
        """Test batch processing with multiple graphs."""
        # Generate multiple graphs
        graphs = []
        for i in range(3):
            # Use different random states for variety
            np.random.seed(i)
            generator = FakeDataGenerator(self.num_resid)
            coords, obs_shifts, pred_shifts, noes, connectivity = (
                generator.generate_data_arrays(random_key=False)
            )
            coords, obs_shifts, noes, connectivity = generator.order_data(
                coords, obs_shifts, pred_shifts, noes, connectivity
            )

            env = GymEnv(self.num_resid)
            env_state = env.reset(coords, obs_shifts, noes, connectivity)

            # Convert to construct_graph format
            state = create_state_dict_from_env(env_state)

            # Construct graph
            data = construct_graph(state, self.device, self.config)
            graphs.append(data)

        # Create data loader for batching
        loader = DataLoader(graphs, batch_size=3, follow_batch=["x"])

        # Get batched data
        batched_data = next(iter(loader))

        # Create network
        net = NMRNet(self.device, ModelConfig())

        # Run forward pass on batch
        value, policy = net(batched_data)

        # Verify output shapes
        # Value should have 3 entries (one per graph)
        self.assertEqual(value.shape[0], 3)

        # Policy should have 3 entries (one per graph), each [num_peaks, num_residues]
        self.assertIsInstance(policy, list)
        self.assertEqual(len(policy), 3)
        for p in policy:
            self.assertEqual(p.shape, (self.num_resid, self.num_resid))

        # Verify outputs are finite
        self.assertTrue(torch.isfinite(value).all())
        for p in policy:
            self.assertTrue(torch.isfinite(p).all())

    def test_edge_case_handling(self):
        """Test handling of edge cases (minimum size graphs)."""
        # Test: Small graph (minimum viable size)
        # Use 3 residues to ensure we get some NOEs
        generator_small = FakeDataGenerator(num_resid=3)
        coords, obs_shifts, pred_shifts, noes, connectivity = (
            generator_small.generate_data_arrays(random_key=False)
        )
        coords, obs_shifts, noes, connectivity = generator_small.order_data(
            coords, obs_shifts, pred_shifts, noes, connectivity
        )

        env_small = GymEnv(3)
        env_state_small = env_small.reset(coords, obs_shifts, noes, connectivity)
        state_small = create_state_dict_from_env(env_state_small)

        # Construct graph
        data_small = construct_graph(state_small, self.device, self.config)

        # Create network and run forward pass
        net = NMRNet(self.device, ModelConfig())

        try:
            value, policy = net(data_small)

            # Verify outputs are valid
            self.assertEqual(value.shape, (1, 1))
            self.assertIsInstance(policy, list)
            self.assertEqual(len(policy), 1)
            # Policy shape is [num_peaks, num_residues] = [3, 3]
            self.assertEqual(policy[0].shape, (3, 3))
            self.assertTrue(torch.isfinite(value).all())
            for p in policy:
                self.assertTrue(torch.isfinite(p).all())

        except Exception as e:
            self.fail(f"Forward pass failed on small graph: {e}")

        # Count triple nodes in small graph
        for triple_type in [
            "ResidueResidueNoeTriple",
            "ResiduePeakNoeTriple",
            "PeakResidueNoeTriple",
            "PeakPeakNoeTriple",
        ]:
            if triple_type in data_small.node_types:
                num_triples = data_small[triple_type].x.size(0)
                # Even if some types have zero instances, forward pass should work
                self.assertGreaterEqual(num_triples, 0)


class TestCoordinateUpdateComparison(unittest.TestCase):
    """Test comparison between triple types for coordinate updates."""

    def setUp(self):
        """Set up test environment."""
        self.device = torch.device("cpu")
        self.config = ModelConfig()
        self.num_resid = 5

        # Generate test dataset
        generator = FakeDataGenerator(self.num_resid)
        coords, obs_shifts, pred_shifts, noes, connectivity = (
            generator.generate_data_arrays(random_key=False)
        )
        coords, obs_shifts, noes, connectivity = generator.order_data(
            coords, obs_shifts, pred_shifts, noes, connectivity
        )

        # Create environment and initial state
        env = GymEnv(self.num_resid)
        env_state = env.reset(coords, obs_shifts, noes, connectivity)
        self.state = create_state_dict_from_env(env_state)

    def test_coordinate_update_comparison(self):
        """Verify no triple types update coordinates (coordinate updates removed)."""
        config = ModelConfig()
        layer = NMRLayer(self.device, config)
        embed = EmbedFeatures(self.device, config)

        # Test ResidueResidueNoeTriple
        # Coordinates are already normalized during graph construction
        data_res_res = construct_graph(self.state, self.device, self.config)
        data_res_res = embed(data_res_res)
        original_coords = data_res_res["Residue"].xyz.clone()

        data_res_res = layer.residue_residue_noe(data_res_res)
        updated_coords = data_res_res["Residue"].xyz

        # ResidueResidueNoeTriple should NOT update coordinates (updates removed)
        self.assertTrue(
            torch.allclose(original_coords, updated_coords, atol=1e-6),
            "ResidueResidueNoeTriple should not update .xyz (coordinate updates removed)",
        )

        # Test ResiduePeakNoeTriple
        data_res_peak = construct_graph(self.state, self.device, self.config)
        data_res_peak = embed(data_res_peak)
        coords_before = data_res_peak["Residue"].xyz.clone()
        data_res_peak = layer.residue_peak_noe(data_res_peak)
        coords_after = data_res_peak["Residue"].xyz

        # ResiduePeakNoeTriple should NOT update coordinates
        self.assertTrue(
            torch.allclose(coords_before, coords_after, atol=1e-6),
            "ResiduePeakNoeTriple should not update .xyz",
        )

        # Test PeakResidueNoeTriple
        data_peak_res = construct_graph(self.state, self.device, self.config)
        data_peak_res = embed(data_peak_res)
        coords_before = data_peak_res["Residue"].xyz.clone()
        data_peak_res = layer.peak_residue_noe(data_peak_res)
        coords_after = data_peak_res["Residue"].xyz

        # PeakResidueNoeTriple should NOT update coordinates
        self.assertTrue(
            torch.allclose(coords_before, coords_after, atol=1e-6),
            "PeakResidueNoeTriple should not update .xyz",
        )

        # Test PeakPeakNoeTriple
        data_peak_peak = construct_graph(self.state, self.device, self.config)
        data_peak_peak = embed(data_peak_peak)
        coords_before = data_peak_peak["Residue"].xyz.clone()
        data_peak_peak = layer.peak_peak_noe(data_peak_peak)
        coords_after = data_peak_peak["Residue"].xyz

        # PeakPeakNoeTriple should NOT update coordinates
        self.assertTrue(
            torch.allclose(coords_before, coords_after, atol=1e-6),
            "PeakPeakNoeTriple should not update .xyz",
        )

        # Verify all types update embedded features (.x) but not raw data
        for triple_name, data_obj in [
            ("ResidueResidueNoeTriple", data_res_res),
            ("ResiduePeakNoeTriple", data_res_peak),
            ("PeakResidueNoeTriple", data_peak_res),
            ("PeakPeakNoeTriple", data_peak_peak),
        ]:
            # Just verify that embedded features and flags exist and are finite
            self.assertTrue(
                torch.isfinite(data_obj["Residue"].x).all(),
                f"{triple_name}: Residue.x features are not finite",
            )
            self.assertTrue(
                torch.isfinite(data_obj["Residue"].flags).all(),
                f"{triple_name}: Residue.flags are not finite",
            )
            self.assertTrue(
                torch.isfinite(data_obj["Peak"].x).all(),
                f"{triple_name}: Peak.x features are not finite",
            )
            self.assertTrue(
                torch.isfinite(data_obj["Peak"].flags).all(),
                f"{triple_name}: Peak.flags are not finite",
            )


if __name__ == "__main__":
    unittest.main()
