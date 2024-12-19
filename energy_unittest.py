import unittest

class TestEnergyMethods(unittest.TestCase):

    def setUp(self):
        self.energy = Energy()

    #test for activated, if activated, the expected will be the energy calculation
    def test_energy_activated(self):

        #random data
        coords = [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6], [0.7, 0.8, 0.9], [0.91, 0.92, 0.93], [0.95,0.96, 0.97]]  #x,y,z
        restraints = [[(2, 1),(0, 3)]]
        assignments = {2: 1, 0: 3, 1: 2, 3: 4}

        result = self.energy.get_energy(restraints, coords, assignments)
        expected = 0.01019237886466845

        self.assertEqual(result, expected)

    #test for not activated, if not activated, the expected will be 0
    def test_energy_not_activated(self):

        #random data
        coords = [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6], [0.7,0.8,0.9], [0.91, 0.92, 0.93], [0.95,0.96, 0.97]]  #x,y,z
        restraints = [[(2, 1), (0, 3)]]
        assignments = {0:3}

        result = self.energy.get_energy(restraints, coords, assignments)
        expected = 0

        self.assertEqual(result, expected)

    #test for sum of all energies for more then one activated NOE, the expected will be the total energy
    #also works for NOE not activated, expected will be the same as test_energy_activated
    def test_energy_sum(self):
        coords = [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6], [0.7, 0.8, 0.9], [0.1, 0.92, 0.93], [0.95,0.96, 0.97]]  #x,y,z
        restraints = [(2, 1), (0, 3)], [(1, 2)]
        assignments = {2: 1, 0: 3, 1: 2}

        result = self.energy.get_energy(restraints, coords, assignments)
        expected = 0.5263195283063458

        self.assertEqual(result, expected)

    #energy calc - pass
    def test_flat_bottom(self):
        tolerance = 1
        x = 5 #distance
        result = self.energy.flat_bottom(x, tolerance)
        expected = 20 # x^2 - tolerance*x
        self.assertEqual(result, expected)

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
