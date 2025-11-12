import gymnasium as gym
from gymnasium import error, spaces
import numpy as np
import pickle
import math

from .energy import Energy
from .assignment_order import FractionalActivation
from scripts.visualize_data import Visualization

import cProfile
import pstats
import time



class GymEnv(gym.Env):

    def __init__(self, num_resid):
        self.state = None
        self.num_resid = num_resid

        self.assignments: Dict[int, int] = None
        self.assign_step: int = None
        self.shift_to_assign: int = None
        self.restraints = None
        self.intermediate_energy: float = None
        self.dist_grid = []

        self.energy: Energy = None
        self.activation: FractionalActivation = None
        self.visualize: Visualization = None

        self.action_space = spaces.Discrete(self.num_resid)

        self.observation_space = spaces.Dict({
            "coordinates": spaces.Box(0, 1, shape=(self.num_resid, 5), dtype=np.float32),
            "obs_chemical_shifts": spaces.Box(0, 1, shape=(self.num_resid, 2), dtype=np.float32),
            "noes": spaces.Box(0, 1, shape=(self.num_resid, 3), dtype=np.float32),
            "assignments": spaces.Box(0, self.num_resid, shape=(self.num_resid, 3), dtype=np.float32),
            "assign_order": spaces.Box(0, self.num_resid, shape=(self.num_resid, 1), dtype=np.float32),
            "shift_to_assign": spaces.Discrete(self.num_resid),
            "total_energy": spaces.Box(0, math.inf, shape=(1, 1), dtype=np.float32),
            "reward": spaces.Box(0, math.inf, shape=(1, 1), dtype=np.float32)
        })

    def step(self, action):
        assert action < self.num_resid and action not in self.state['assignments'].values()

        # Add action to assignments
        self.assignments[(self.assign_order[self.assign_step])] = action
        # print(self.assignments)

        # Calculate energy and store temporarily
        temp_intermediate_energy = self.energy.get_total_energy(self.restraints, self.assignments, self.dist_grid)

        ### TEST WHEN REMOVING ENERGY FUNCTION ###
        # temp_intermediate_energy = 0
        ##########################################

        # Calculate reward based on previous energy and temporary energy
        # Positive reward when energy goes down (good), negative when energy goes up (bad)
        self.state['reward'] = self.intermediate_energy - temp_intermediate_energy

        # Store energy as intermediate
        self.intermediate_energy = temp_intermediate_energy

        # Assign intermediate energy as running total
        self.state['total_energy'] = self.intermediate_energy
        # print(f"Running energy = {self.state['total_energy']}, Current reward = {self.state['reward']}")

        self.assign_step += 1
        terminated = True if len(self.assignments) == self.num_resid else False
    
        if terminated:
            # print(f"Final assignments (shift:atom) = {self.assignments}\nFinal energy evaluation = {self.state['total_energy']}")
            self.state['shift_to_assign'] = None
            return self.state, self.state['reward'], terminated, self.state['total_energy']
        
        self.state['shift_to_assign'] = self.assign_order[self.assign_step]
        if not terminated:
            # print(f'Shift to be assigned: {self.state["shift_to_assign"]}')
            return self.state, self.state['reward'], terminated, self.state['total_energy']
    
    def custom_state(self, state):
        self.state = state

        # Get restraints
        self.energy = Energy(self.state['coordinates'], self.state['obs_chemical_shifts'], self.state['noes'])
        self.restraints = self.energy.noe_combinations()

        self.dist_grid = self.energy.calc_pdist()

        self.assignments = {}
        self.assign_step = 0
        self.intermediate_energy = 0.0

        # Get assignment order 
        self.activation = FractionalActivation(self.num_resid, self.restraints)
        self.assign_order = self.activation.fractional_activation() 

        ### TEST WHEN REMOVING FRACTIONAL ACTIVATION ###
        # test_range = np.arange(0, self.num_resid)
        # self.assign_order = test_range
        ################################################

        self.state["assign_order"] = self.assign_order
        self.shift_to_assign = self.assign_order[self.assign_step]
        
        self.state['shift_to_assign'] = self.shift_to_assign
        self.state['assignments'] = self.assignments

        return self.state

    def reset(self, coordinates, obs_chemical_shifts, noes, connectivity, seed=None, options=None): #pickled=False, pickle_data=True, example=True, random_key=False, seed=None, options=None):
        super().reset(seed=seed)

        # Get restraints
        self.energy = Energy(coordinates, obs_chemical_shifts, noes)
        self.restraints = self.energy.noe_combinations()

        self.dist_grid = self.energy.calc_pdist()

        self.assignments = {}
        self.assign_step = 0
        self.intermediate_energy = 0.0

        # Get assignment order 
        self.activation = FractionalActivation(self.num_resid, self.restraints)
        self.assign_order = self.activation.fractional_activation() 

        ### TEST WHEN REMOVING FRACTIONAL ACTIVATION ###
        # test_range = np.arange(0, self.num_resid)
        # self.assign_order = test_range
        ################################################

        self.shift_to_assign = self.assign_order[self.assign_step]
        # print(self.assign_order)
        # print(f'Shift to be assigned: {self.shift_to_assign}')

        self.state = {
            "coordinates": coordinates,
            "obs_chemical_shifts": obs_chemical_shifts,
            "noes": noes,
            "connectivity": connectivity,
            "assignments": self.assignments,
            "assign_order": self.assign_order,
            "shift_to_assign": self.shift_to_assign,
            "total_energy": 0.0,
            "reward": 0.0
        }

        return self.state

    def render(self):
        self.visualize = Visualization(self.state['coordinates'], self.state['obs_chemical_shifts'], self.state['connectivity'], self.restraints)
        self.visualize.plot_shifts(named_tuple_used=True)
        # self.visualize.plot_step(self.state['shift_to_assign'], self.assign_order)

    
        
