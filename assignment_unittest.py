#unittest for fractional activations
#has to be made into a class for it to work

import unittest

class TestFracAct(unittest.TestCase):

    def setUp(self):
        self.frac_act = FracAct()

    #test for activation loop
    #expected to return the noe peak divided by the total number of different/unique noe peaks

    def test_activation_loop(self):
        assign_order = []
        noe = [(0, 1), (0, 2)]
        target = 0

        result = self.frac_act.activation_loop(target, noe, assign_order)
        expected = 1/3

        self.assertEqual(result, expected)

    #test if each noe peak will be evaluated to give the fractional activation

    def test_fractional_activation_of_all_noe_peaks(self):
        num_resid = 3
        noes = [[(0, 1), (0, 2)]]

        result = self.frac_act.fractional_activation_summation(num_resid, noes)
        expected = [1/3, 1/3, 1/3]

        self.assertEqual(result, expected)

    #for more then one noe, the probabilities are summed

    def test_fractional_activation_summation_of_noes(self):
        num_resid = 4
        noes = [(0, 1), (0, 2)], [(2,1),(1,1)], [(1,0),(3,2)]

        result = self.frac_act.fractional_activation_summation(num_resid, noes)
        expected = [7/12, 13/12, 13/12, 1/4]

        self.assertEqual(result, expected)

    #according to the fractional activations calculated, should give the correct order
    #if it is a tie, then lowest value goes first

    def test_fractional_activation_for_assignment_order(self):
        num_resid = 4
        noes = [(0, 1), (0, 2)], [(2,1),(1,1)], [(1,0),(3,2)]

        result = self.frac_act.fractional_activation(num_resid, noes)
        expected = [1, 2, 0, 3]

        self.assertEqual(result, expected)

    #if number of residues is higher than the number of peaks, then it will loop through to give first/lowest value peak ex. will give 0 for every additional residue
    #if number of residues is lower than the number of peaks, then it will return the first value again (0, in this case)

    def test_fractional_activation_for_num_resid(self):
        num_resid = 5
        noes = [[(0, 1), (0, 2)]], [(2,1),(1,1)], [(1,0),(3,2)]

        result = self.frac_act.fractional_activation(num_resid, noes)
        expected = [1, 2, 0, 3]

        self.assertEqual(result, expected)