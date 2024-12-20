import gym
from gym import error, spaces
import numpy as np

from fake_data import generate_data
from energy import Energy, noe_combinations
from assignment_order import FracAct

class GymEnv(gym.Env):

    def __init__(self, num_resid):
        self.num_resid = num_resid
        self.assignments = {}
        self.assign_step = 0
        self.intermediate_energy = 0
        self.reward = 0

        self.action_space = spaces.Discrete(self.num_resid)

        self.observation_space = spaces.Dict({
            "coords": spaces.Box(0, 1, shape=(self.num_resid, 3), dtype=np.float32),
            "actual_shifts": spaces.Box(0, 1, shape=(self.num_resid, 2), dtype=np.float32),
            "predicted_shifts": spaces.Box(0, 1, shape=(self.num_resid, 2), dtype=np.float32),
            "noes": spaces.Box(0, 1, shape=(self.num_resid, 3), dtype=np.float32)
        })

    def custom_state(self, state):
        self.state = state
        frac_act = FracAct(self.num_resid, self.state['restraints'])
        self.assign_order = frac_act.fractional_activation()
        print(self.assign_order)
        return self.state

    def reset(self):
        # generate fake data
        coords, actual_shifts, predicted_shifts, noes = generate_data(self.num_resid)
        # get restraints
        restraints = noe_combinations(noes, actual_shifts)

        # get assignment order 
        frac_act = FracAct(self.num_resid, restraints)
        self.assign_order = frac_act.fractional_activation()
        print(self.assign_order)

        self.assignments = {}
        # self.assign_step = 0
        # self.intermediate_energy = 0
        self.total_energy = 0
        # self.reward = 0

        self.state = {
            "coords": coords,
            "actual_shifts": actual_shifts,
            "predicted_shifts": predicted_shifts,
            "noes": noes,
            "restraints": restraints,
            "assignments": self.assignments,
            "total_energy": self.total_energy
        }

        return self.state

    def step(self, action):
        
        energy = Energy(self.state['coords'], self.state['actual_shifts'], self.state['noes'])

        # add action to assignments
        self.assignments[(self.assign_order[self.assign_step])] = action
        print(self.assignments)

        # calc energy and store temporarily
        temp_intermediate_energy = energy.get_energy(self.state['restraints'], self.assignments)
        # print(f'this is the temp energy {temp_intermediate_energy}')

        # calc reward based on previous energy and temporary energy (+ve means energy went down, -ve means energy went up)
        self.reward = (self.intermediate_energy - temp_intermediate_energy) #if self.intermediate_energy != 0 else 0
        # print(f'this is the reward {self.reward}')

        # store energy as intermediate
        self.intermediate_energy = temp_intermediate_energy

        # assign intermediate energy as running total
        self.state['total_energy'] = self.intermediate_energy
        print(f"Running energy = {self.state['total_energy']}, Current reward = {self.reward}")

        self.assign_step += 1
        terminated = True if len(self.assignments) == self.num_resid else False
    
        if terminated:
            print(f"Final assignments (shift:atom) = {self.assignments}\nFinal energy evaluation = {self.state['total_energy']}")
            return self.state, self.reward, terminated
        else:
            return self.state, self.reward, terminated
    
        
