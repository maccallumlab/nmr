import unittest
import numpy as np
import sys
from pathlib import Path

# Add parent directory to path to allow imports from nmr package
sys.path.insert(0, str(Path(__file__).parent.parent))

from nmr.nmr_gym.energy import Energy



class TestEnergyMethods(unittest.TestCase):

    def setUp(self):
        self.coords = [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6], [0.7, 0.8, 0.9], [0.91, 0.92, 0.93], [0.95, 0.96, 0.97]] # x,y,z
        self.energy = Energy(self.coords, 0, 0)
        self.dist_grid = self.energy.calc_pdist()

    # Test for NOE activation - if activated, the expected value will be the lowest energy calculated
    def test_energy_activated(self):
        restraint = [(2, 1), (0, 3)]
        assignments = {2: 1, 0: 3, 1: 2, 3: 4}
        
        result = self.energy.calc_restraint_energy(assignments, restraint, self.dist_grid, 0.5)
        expected = 0.0

        self.assertEqual(result, expected)

    # Test if NOE not activated - if not activated, the expected value will be 0 (no energy calculated)
    def test_energy_not_activated(self):
        restraints = [[(2, 1), (0, 3)]]
        assignments = {0: 3}

        result = self.energy.get_total_energy(restraints, assignments, self.dist_grid)
        expected = 0

        self.assertEqual(result, expected)

    # Test for sum of all energies for one activated NOE, the expected will be single restraint calculation
    def test_energy_sum_not_activated(self):
        restraints = [[(2, 1), (0, 3)], [(1, 2)]]
        assignments = {2: 1, 0: 3, 1: 2}

        result = self.energy.get_total_energy(restraints, assignments, self.dist_grid)
        expected = 0.01019237886466845

        self.assertEqual(result, expected)

    # Test for sum of all energies for more then one activated NOE, the expected will be the sum of the two restraints
    def test_energy_sum_activated(self):
        restraints = [[(2, 1), (0, 3)], [(1, 2)]]
        assignments = {2: 1, 0: 3, 1: 2, 3: 0}

        result = self.energy.get_total_energy(restraints, assignments, self.dist_grid)
        expected = 0.01019237886466845*2

        self.assertEqual(result, expected)

    # Test for flat bottom restraint, distance above tolerance
    def test_flat_bottom(self):
        tolerance = 0.5
        x = 5 # distance
        result = self.energy.flat_bottom(x, tolerance)
        expected = 22 # x^2 - tolerance*x
        self.assertEqual(result, expected)



if __name__ == '__main__':
    unittest.main()

    '''
    calculations for the expected of each unittest

    coords = [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6], [0.7, 0.8, 0.9], [0.91, 0.92, 0.93], [0.95,0.96, 0.97]]  #x,y,z

    #calculation for test_energy_activated
    restraints = [[(2, 1),(0, 3)]]

    x = np.linalg.norm((coords[2])-np.array(coords[1]))

    expected_x = x**2 - 0.5*x

    print(expected_x)

    #calculation for test_energy_sum

    y = np.linalg.norm(np.array(coords[1])-np.array(coords[2]))

    expected_y = y**2 - 0.5*y

    print(expected_y + expected_x)

    '''
