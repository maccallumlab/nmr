import unittest
import torch
import sys
from pathlib import Path

# Add parent directory to path to allow imports from nmr package
sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.train_gnn import split_train_test, evaluate_test_set
from nmr.construct import construct_graph
from nmr.models import NMRNet, ModelConfig
from torch_geometric.loader import DataLoader


class TestTrainTestSplit(unittest.TestCase):
    def setUp(self):
        """Create mock graphs for testing."""
        # Create simple mock state dicts
        self.device = "cpu"
        self.config = ModelConfig(num_nmr_layers=1, layer_type="triple")

        # Create 20 mock graphs with different state dicts
        self.graphs = []
        for i in range(20):
            # Create a simple test state dict with correct format
            state_dict = {
                "coordinates": [
                    [0.0 + i*0.1, 0.0, 0.0, 7.5 + i*0.1, 115.0 + i*0.1],
                    [1.0 + i*0.1, 0.0, 0.0, 8.0 + i*0.1, 120.0 + i*0.1],
                    [2.0 + i*0.1, 0.0, 0.0, 7.8 + i*0.1, 118.0 + i*0.1],
                ],
                "obs_chemical_shifts": [
                    [7.5 + i*0.1, 115.0 + i*0.1],
                    [8.0 + i*0.1, 120.0 + i*0.1],
                    [7.8 + i*0.1, 118.0 + i*0.1],
                ],
                "noes": [[115.0, 7.5, 8.0], [120.0, 8.0, 7.8]],
                "shift_to_assign": 0,
                "assignments": {},
            }

            graph = construct_graph(state_dict, self.device, self.config)
            # Attach mock targets
            graph.action = torch.tensor([i % 3], dtype=torch.long)
            graph.value = torch.tensor([float(i)], dtype=torch.float32)
            self.graphs.append(graph)

    def test_split_ratios(self):
        """Test that split ratios work correctly."""
        # Test 80/20 split
        train, test = split_train_test(self.graphs, train_ratio=0.8, seed=42)
        self.assertEqual(len(train), 16)
        self.assertEqual(len(test), 4)

        # Test 70/30 split
        train, test = split_train_test(self.graphs, train_ratio=0.7, seed=42)
        self.assertEqual(len(train), 14)
        self.assertEqual(len(test), 6)

        # Test 50/50 split
        train, test = split_train_test(self.graphs, train_ratio=0.5, seed=42)
        self.assertEqual(len(train), 10)
        self.assertEqual(len(test), 10)

    def test_seed_deterministic(self):
        """Test that seed produces deterministic splits."""
        train1, test1 = split_train_test(self.graphs, train_ratio=0.8, seed=42)
        train2, test2 = split_train_test(self.graphs, train_ratio=0.8, seed=42)

        # Check that splits are identical
        self.assertEqual(len(train1), len(train2))
        self.assertEqual(len(test1), len(test2))

        # Check that graph order is the same
        for g1, g2 in zip(train1, train2):
            self.assertTrue(torch.equal(g1.action, g2.action))
            self.assertTrue(torch.equal(g1.value, g2.value))

        for g1, g2 in zip(test1, test2):
            self.assertTrue(torch.equal(g1.action, g2.action))
            self.assertTrue(torch.equal(g1.value, g2.value))

    def test_different_seeds(self):
        """Test that different seeds produce different splits."""
        train1, test1 = split_train_test(self.graphs, train_ratio=0.8, seed=42)
        train2, test2 = split_train_test(self.graphs, train_ratio=0.8, seed=123)

        # Check that splits are different (with high probability)
        # Compare graph IDs (actions) to see if order is different
        actions1 = [g.action.item() for g in train1[:5]]
        actions2 = [g.action.item() for g in train2[:5]]

        # With high probability, at least one element should be different
        self.assertNotEqual(actions1, actions2)

    def test_train_test_disjoint(self):
        """Test that train and test sets are disjoint."""
        train, test = split_train_test(self.graphs, train_ratio=0.8, seed=42)

        # Check that all graphs are either in train or test, but not both
        self.assertEqual(len(train) + len(test), len(self.graphs))

        # Check that there's no overlap (using graph value as unique identifier)
        train_values = {g.value.item() for g in train}
        test_values = {g.value.item() for g in test}

        self.assertEqual(len(train_values & test_values), 0)

    def test_evaluate_no_gradient_update(self):
        """Test that test evaluation doesn't affect training gradients."""
        train, test = split_train_test(self.graphs, train_ratio=0.8, seed=42)

        # Create network and optimizer
        net = NMRNet(self.device, self.config)
        test_loader = DataLoader(test, batch_size=2, shuffle=False)

        # Get initial parameter values
        initial_params = {
            name: param.clone() for name, param in net.named_parameters()
        }

        # Run test evaluation
        net.train()
        test_metrics = evaluate_test_set(net, test_loader)

        # Check that parameters haven't changed
        for name, param in net.named_parameters():
            self.assertTrue(
                torch.equal(param, initial_params[name]),
                f"Parameter {name} changed during test evaluation"
            )

        # Check that model is back in train mode
        self.assertTrue(net.training)

        # Check that metrics are returned
        self.assertIn('loss', test_metrics)
        self.assertIn('accuracy', test_metrics)
        self.assertIsInstance(test_metrics['loss'], float)
        self.assertIsInstance(test_metrics['accuracy'], float)


if __name__ == "__main__":
    unittest.main()
