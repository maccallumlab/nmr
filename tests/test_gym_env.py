import unittest
import numpy as np
import sys
from pathlib import Path

# Add parent directory to path to allow imports from nmr package
sys.path.insert(0, str(Path(__file__).parent.parent))

from nmr.nmr_gym.gym_env import GymEnv
from nmr.nmr_gym.data_structures import Connectivity, HSQCPeak, NOEPeak, Protein

class TestGymEnv(unittest.TestCase):

    def setUp(self):
        self.num_resid = 4
        # Create proper named tuples instead of plain lists
        coordinates = [
            Protein(0.35, 0.83, 0.89, 0.17, 0.044),
            Protein(0.69, 0.12, 0.92, 0.41, 0.34),
            Protein(0.05, 0.94, 0.51, 0.18, 0.61),
            Protein(0.45, 0.06, 0.70, 0.64, 0.23)
        ]
        obs_chemical_shifts = [
            HSQCPeak(0.17, 0.044),
            HSQCPeak(0.41, 0.34),
            HSQCPeak(0.18, 0.61),
            HSQCPeak(0.64, 0.23)
        ]
        noes = [
            NOEPeak(0.16, 0.040, 0.17),
            NOEPeak(0.40, 0.34, 0.63),
            NOEPeak(0.18, 0.61, 0.16),
            NOEPeak(0.64, 0.24, 0.39)
        ]
        connectivity = []  # Add connectivity if needed

        self.state = {
            "coordinates": coordinates,
            "obs_chemical_shifts": obs_chemical_shifts,
            "noes": noes,
            "connectivity": connectivity,
            "restraints": [[(0, 2)], [(1, 3)], [(0, 2)], [(1, 3)]],
            "assignments": {},
            "total_energy": 0
        }

        self.gym_env = GymEnv(self.num_resid)

    def test_step_energy_zeros(self):
        actions = [3, 1, 0, 2]

        expected_energy = [0, 0, 0, 0]
        expected_reward = [0, 0, 0, 0]

        observation = self.gym_env.custom_state(self.state)

        for i, action in enumerate(actions):
            observation, reward, terminated, total_energy = self.gym_env.step(action)

            self.assertEqual(observation['total_energy'], expected_energy[i])
            self.assertEqual(reward, expected_reward[i])
        
    def test_step_energy_goes_up(self):
        actions = [2, 3, 0, 1]
        
        # dist1 = np.linalg.norm((np.array(self.state['coords'][2]) - np.array(self.state['coords'][3])))
        # dist2 = np.linalg.norm((np.array(self.state['coords'][1]) - np.array(self.state['coords'][0])))

        # energy1 = dist1**2 - 0.5*dist1 # for restraint (0,2)
        # energy2 = dist2**2 - 0.5*dist2 # for restraint (1,3)
        # print(energy1*2 + energy2*2)
        # print(energy1*2 - (energy1*2 + energy2*2)) # final reward

        """
        Step1: No NOEs activated --> energy 0
        Step2: Two NOEs activated --> energy 0.47/NOE
        Step3: Two NOEs activated (no new additions) --> energy 0.47/NOE
        Step4: All NOEs activated (two more NOEs activated) --> energy 0.47/NOE prior + 0.22/NOE new

        """
        expected_energy = [0, 0.9558604159815726, 0.9558604159815726, 1.4092787203323272]
        expected_reward = [0, -0.9558604159815726, 0, -0.4534183043507546]

        observation = self.gym_env.custom_state(self.state)

        for i, action in enumerate(actions):
            observation, reward, terminated, total_energy = self.gym_env.step(action)

            self.assertEqual(observation['total_energy'], expected_energy[i])
            self.assertEqual(reward, expected_reward[i])

unittest.main(argv=[''], verbosity=2, exit=False)