import unittest
import numpy as np

from nmr_gym_env import GymEnv

class TestGymEnv(unittest.TestCase):

    def setUp(self):
        self.num_resid = 4
        self.state = {
            "coords": [[0.35, 0.83, 0.89], [0.69, 0.12, 0.92], [0.05, 0.94, 0.51], [0.45, 0.06, 0.70]],
            "actual_shifts": [[0.17, 0.044], [0.41, 0.34], [0.18, 0.61], [0.64, 0.23]],
            "noes": [[0.16, 0.040, 0.17], [0.40, 0.34, 0.63], [0.18, 0.61, 0.16], [0.64, 0.24, 0.39]],
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
        terminated = False
        
        for i, action in enumerate(actions):
            observation, reward, terminated = self.gym_env.step(action)

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
        terminated = False
        
        for i, action in enumerate(actions):
            observation, reward, terminated = self.gym_env.step(action)

            self.assertEqual(observation['total_energy'], expected_energy[i]) 
            self.assertEqual(reward, expected_reward[i])

unittest.main(argv=[''], verbosity=2, exit=False)