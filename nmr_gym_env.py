import gym
from gym import error, spaces
import numpy as np
import pickle
import math
import glob as glob

from fake_data import generate_data
from energy import Energy, noe_combinations
from assignment_order import FracAct

from visualize_data import plot_shifts

class GymEnv(gym.Env):

    def __init__(self, num_resid):
        self.state = None
        self.num_resid = num_resid

        self.assignments = {}
        self.assign_step = 0
        self.assign_shift = 0
        self.restraints = 0
        self.total_energy = 0.0
        self.intermediate_energy = 0.0
        self.reward = 0.0

        self.action_space = spaces.Discrete(self.num_resid)

        self.observation_space = spaces.Dict({
            "coords": spaces.Box(0, 1, shape=(self.num_resid, 5), dtype=np.float32),
            "actual_shifts": spaces.Box(0, 1, shape=(self.num_resid, 2), dtype=np.float32),
            "noes": spaces.Box(0, 1, shape=(self.num_resid, 3), dtype=np.float32),
            "assignments": spaces.Box(0, self.num_resid, shape=(self.num_resid, 3), dtype=np.float32),
            "assign_order": spaces.Box(0, self.num_resid, shape=(self.num_resid, 1), dtype=np.float32),
            "assign_shift": spaces.Discrete(self.num_resid),
            "total_energy": spaces.Box(0, math.inf, shape=(1, 1), dtype=np.float32),
            "reward": spaces.Box(0, math.inf, shape=(1, 1), dtype=np.float32)
        })

    def custom_state(self, state):
        self.state = state
        self.restraints = noe_combinations(self.state['noes'], self.state['actual_shifts'], tolerance_h=0.02, tolerance_n=0.02)

        self.assignments = {}
        self.assign_step = 0
        self.assign_shift = 0
        self.total_energy = 0.0
        self.intermediate_energy = 0.0
        self.reward = 0.0

        # get assignment order 
        # frac_act = FracAct(self.num_resid, self.restraints)
        # self.assign_order = frac_act.fractional_activation()
        self.assign_shift = self.state['assign_order'][self.assign_step]
        # print(self.assign_order)
        # print(f'Shift to be assigned: {self.assign_shift}')

        # self.state['assign_order'] = self.assign_order
        self.state['assign_shift'] = self.assign_shift
        self.state['assignments'] = self.assignments

        return self.state

    def reset(self, pickled=False, pickle_data=True, example=True, random_key=False):

        name = f"./*r{self.num_resid}.pkl" if example else "./current_run.pkl"

        if self.intermediate_energy > 0 or pickled:
            pickle_file = glob.glob(name)
            with open(pickle_file[0], 'rb') as f:
                coords = pickle.load(f)
                actual_shifts = pickle.load(f)
                noes = pickle.load(f)
                connectivity = pickle.load(f)
        
        elif pickle_data:
            generate_data(self.num_resid, pickle_data=pickle_data, example=example, random_key=random_key)
            pickle_file = glob.glob(name)
            with open(pickle_file[0], 'rb') as f:
                coords = pickle.load(f)
                actual_shifts = pickle.load(f)
                noes = pickle.load(f)
                connectivity = pickle.load(f)

        else:
            coords, actual_shifts, noes, connectivity = generate_data(self.num_resid, pickle_data=pickle_data, example=example, random_key=random_key)

        # get restraints
        """
        float(float(0.001*(self.num_resid)+0.01)) shift tolerance needs to increase with protein size somehow or else it can miss the correct answer,
        but this will increase the complexity of the problem
        """
        self.restraints = noe_combinations(noes, actual_shifts, tolerance_h=0.02, tolerance_n=0.02) 
        # get plot of shifts and restraints
        plot_shifts(actual_shifts, self.restraints, coords, connectivity, named_tuple_used=True)

        self.assignments = {}
        self.assign_step = 0
        self.assign_shift = 0
        self.total_energy = 0.0
        self.intermediate_energy = 0.0
        self.reward = 0.0

        # get assignment order 
        frac_act = FracAct(self.num_resid, self.restraints)
        self.assign_order = frac_act.fractional_activation()
        self.assign_shift = self.assign_order[self.assign_step]
        # print(self.assign_order)
        # print(f'Shift to be assigned: {self.assign_shift}')

        self.state = {
            "coords": coords,
            "actual_shifts": actual_shifts,
            "noes": noes,
            "connectivity": connectivity,
            "assignments": self.assignments,
            "assign_order": self.assign_order,
            "assign_shift": self.assign_shift,
            "total_energy": self.total_energy,
            "reward": self.reward
        }

        return self.state

    def step(self, action):
        # print(action)
        assert action < self.num_resid and action not in self.state['assignments'].values()

        energy = Energy(self.state['coords'], self.state['actual_shifts'], self.state['noes'])

        # add action to assignments
        self.assignments[(self.assign_order[self.assign_step])] = action
        # print(self.assignments)

        # calc energy and store temporarily
        temp_intermediate_energy = energy.get_total_energy(self.restraints, self.assignments, tolerance=float(1/np.cbrt(self.num_resid)))
        # print(f'this is the temp energy {temp_intermediate_energy}')

        # calc reward based on previous energy and temporary energy 
        self.state['reward'] = (self.intermediate_energy - temp_intermediate_energy)
        # before correction, +ve means energy went down - we DO NOT want this, -ve means energy went up
        assert self.state['reward'] <= 0
        self.state['reward'] = abs(self.state['reward'])
        # print(f'this is the reward {self.reward}')

        # store energy as intermediate
        self.intermediate_energy = temp_intermediate_energy

        # assign intermediate energy as running total
        self.state['total_energy'] = self.intermediate_energy
        # print(f"Running energy = {self.state['total_energy']}, Current reward = {self.state['reward']}")

        self.assign_step += 1
        # self.assign_shift = self.assign_order[self.assign_step]
        terminated = True if len(self.assignments) == self.num_resid else False
    
        if terminated:
            # print(f"Final assignments (shift:atom) = {self.assignments}\nFinal energy evaluation = {self.state['total_energy']}")
            return self.state, self.state['reward'], terminated, self.state['total_energy']
        
        self.state['assign_shift'] = self.assign_order[self.assign_step]
        if not terminated:
            # print(f'Shift to be assigned: {self.state["assign_shift"]}')
            return self.state, self.state['reward'], terminated, self.state['total_energy']
    
        
