import unittest
from itertools import chain
import sys
from pathlib import Path

# Add parent directory to path to allow imports from nmr package
sys.path.insert(0, str(Path(__file__).parent.parent))

from nmr.nmr_gym.assignment_order import FractionalActivation



class TestFracAct(unittest.TestCase):

    # Test for activation loop
    # Expected to return the target noe peak divided by the total number of different/unique noe peak options
    def test_activation_loop(self):
        
        cases = [
            {'noe': [(0, 1), (0, 2)], 'noe_set': {0, 1, 2}, 'assign_order': [], 'target': 0, 'expected': 1/3, 'message': 'CASE 1: First assignment'},
            {'noe': [(0, 1), (0, 2)], 'noe_set': {1, 2}, 'assign_order': [0], 'target': 0, 'expected': 0, 'message': 'CASE 2: Target already assigned'},
            {'noe': [(0, 1), (0, 2)], 'noe_set': {2}, 'assign_order': [0, 1],'target': 2, 'expected': 1.0, 'message': 'CASE 3: Final assignment'},
            {'noe': [(0, 1), (0, 2)], 'noe_set': {}, 'assign_order': [0, 1, 2], 'target': 0, 'expected': 0, 'message': 'CASE 4: Everything already assigned'},
            {'noe': [(0, 1), (0, 2)], 'noe_set': {0, 1, 2}, 'assign_order': [], 'target': 3, 'expected': 0, 'message': 'CASE 5: Target not present in NOE'}
            ]

        for case in cases:
            with self.subTest(message=case['message']):
                frac_act = FractionalActivation(3, case['noe'])
                result = frac_act.activation_loop(case['target'], case['noe_set'], case['assign_order'])
                self.assertEqual(result, case['expected'])

    # According to the fractional activations calculated, should give the correct order
    # If it is a tie, then lowest index goes first
    def test_fractional_activation_for_assignment_order(self):
        
        cases = [
            {'restraints': ([[(0, 1), (0, 2)], [(2, 1), (1, 1)], [(1, 0), (3, 2)]]), 'num_resid': 4, 'expected': [1, 2, 0, 3], 'message': 'CASE 1: Acceptable order'},
            {'restraints': ([[(0, 1), (0, 2)], [(2, 1), (1, 1)], [(1, 0), (3, 2)]]), 'num_resid': 5, 'expected': [1, 2, 0, 3, 4], 'message': 'CASE 2: Higher resid count --> Missing residues added last'},
            # {'restraints': ([[(2, 3)], [(2, 1)], [(1, 0)], [(1, 3)], [(1, 0)], [(2, 3)], [(1, 3)], [(2, 0)]]), 'num_resid': 3, 'expected': [1, 3, 2, 0], 'message': 'CASE 2: Lower resid count --> Expected Failure'},
            {'restraints': ([[(0, 6)], [(1, 4)], [(1, 6)], [(2, 3)], [(2, 3)], [(1, 4)], [], [(0, 6)], [], [(4, 8)]]), 'num_resid': 10, 'expected': [1, 4, 6, 0, 2, 3, 8, 5, 7, 9], 'message': 'CASE 3: Handling blank NOES (true case)'},
            {'restraints': ([[(0, 3)], [(0, 1)], [(1, 2)], [(1, 3)], [(1, 2)], [(0, 3)], [(1, 3)], [(0, 2)]]), 'num_resid': 4, 'expected': [1, 3, 0, 2], 'message': 'CASE 4: Hand solved, single restraints'}
            ]

        for case in cases:
            with self.subTest(message=case['message']):
                frac_act = FractionalActivation(case['num_resid'], case['restraints'])
                result = frac_act.fractional_activation()
                self.assertEqual(result, case['expected'])



if __name__ == '__main__':
    unittest.main()