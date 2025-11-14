"""
Tests for terminology updates in the Triple System Re-Architecture.

This test file verifies that the graph construction uses the new naming conventions:
- Node types: Residue, Peak, Noe (instead of RES, SHIFT, NOE)
- Triple types: ResidueResidueNoeTriple, ResiduePeakNoeTriple, etc. (instead of TRIPLE0/1/2/3)
- Edge relations: prop_first/second/noe bidirectional edges (instead of NH1/NH2_extract/add)
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import unittest
import torch
from nmr.construct import construct_graph


class TestTerminologyUpdates(unittest.TestCase):
    """Test that graph construction uses new node and edge naming conventions."""

    def setUp(self):
        """Create a minimal test history for graph construction."""
        self.device = torch.device("cpu")
        self.history = {
            "coordinates": [[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]],
            "obs_chemical_shifts": [[8.5, 120.0], [7.8, 118.0]],
            "noes": [[120.0, 8.5, 7.8], [118.0, 7.8, 8.5]],
            "assignments": {},
            "shift_to_assign": 0,
        }

    def test_node_types_use_new_names(self):
        """Test that graph construction creates nodes with new names (Residue, Peak, Noe)."""
        data = construct_graph(self.history, self.device)

        # Verify new node types exist
        self.assertIn("Residue", data.node_types)
        self.assertIn("Peak", data.node_types)
        self.assertIn("Noe", data.node_types)

        # Verify old node types do not exist
        self.assertNotIn("RES", data.node_types)
        self.assertNotIn("SHIFT", data.node_types)
        self.assertNotIn("NOE", data.node_types)

    def test_triple_names_use_descriptive_names(self):
        """Test that triple node types use full descriptive names."""
        data = construct_graph(self.history, self.device)

        # Verify new triple node types exist
        self.assertIn("ResidueResidueNoeTriple", data.node_types)
        self.assertIn("ResiduePeakNoeTriple", data.node_types)
        self.assertIn("PeakResidueNoeTriple", data.node_types)
        self.assertIn("PeakPeakNoeTriple", data.node_types)

        # Verify old triple node types do not exist
        self.assertNotIn("ResResNoe", data.node_types)
        self.assertNotIn("ResShiftNoe", data.node_types)
        self.assertNotIn("ShiftResNoe", data.node_types)
        self.assertNotIn("ShiftShiftNoe", data.node_types)
        self.assertNotIn("TRIPLE0", data.node_types)
        self.assertNotIn("TRIPLE1", data.node_types)
        self.assertNotIn("TRIPLE2", data.node_types)
        self.assertNotIn("TRIPLE3", data.node_types)

    def test_edge_relations_use_prop_naming(self):
        """Test that edge relations use new naming (prop_first, prop_second, etc.)."""
        data = construct_graph(self.history, self.device)

        # Get all edge types as strings
        edge_types = [str(et) for et in data.edge_types]

        # Verify new edge relation names exist (bidirectional prop_* edges)
        prop_edges = [et for et in edge_types if "prop_first" in et or "prop_second" in et or "prop_noe" in et]

        self.assertGreater(len(prop_edges), 0, "Should have prop_* edges")

        # Verify old triple-related edge names do not exist (exclude value aggregation edges)
        old_triple_edges = [et for et in edge_types if
                           ("NH1_extract" in et or "NH2_extract" in et or
                            "NH1_add" in et or "NH2_add" in et or
                            "res1_add" in et or "res2_add" in et)]

        self.assertEqual(len(old_triple_edges), 0,
                        f"Should not have old triple edge names, found: {old_triple_edges}")

    def test_triple_names_dictionary_removed(self):
        """Test that TRIPLE_NAMES dictionary has been removed from construct module."""
        import nmr.construct as construct_module

        # TRIPLE_NAMES should not exist as an attribute
        self.assertFalse(hasattr(construct_module, "TRIPLE_NAMES"),
                        "TRIPLE_NAMES dictionary should be removed from construct.py")


if __name__ == "__main__":
    unittest.main()
