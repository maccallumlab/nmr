import gymnasium as gym
import argparse
import pickle
import copy
import numpy as np

from nmr_gym_env import GymEnv
from fake_data import FakeDataGenerator
from fake_histories import FakeHistoryGenerator
from visualize_data import Visualization

from nmr_text_adventure import get_data

import time



if __name__ == '__main__':
    start = time.process_time()
    """
    Runs through protein examples of one size following the 1:1 shift to atom assignment.
    Once example is assigned, generates history and repeats assignment process.
    All histories are then saved to disk.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('num_resid', type=int, help='Max protein size')
    parser.add_argument('history_length', type=int, help='Number of synthetic examples to generate')
    args = parser.parse_args()

    num_resid = args.num_resid
    history_length = args.history_length

    gym_env = GymEnv(num_resid)

    fakedata = FakeDataGenerator(num_resid)
    fakehistory = FakeHistoryGenerator(num_resid)

    ######################################################

    print("\nAvailable options to work with:\n1. Generate and save an example\n2. Grab a previous saved example\n3. Run a new example (not saved)")

    selection = int(input("Option Selection: "))

    if selection == 1:
        # 1. Need to generate and save an example? Saved in all instances as fakedata_r{num_resid}.pkl
        coordinates, obs_chemical_shifts, noes, connectivity = get_data(num_resid, pickled=False, pickle_data=True, example=True, random_key=True)
    if selection == 2:
        # 2. Grab one of the previous examples? Make sure example exists with desired resid number
        coordinates, obs_chemical_shifts, noes, connectivity = get_data(num_resid, pickled=True, pickle_data=False, random_key=True)
    if selection == 3:
        # 3. Run a new example? Saved in all instances as current_run.pkl
        coordinates, obs_chemical_shifts, noes, connectivity = get_data(num_resid, pickled=False, pickle_data=True, example=False, random_key=True)

    ######################################################

    print(f'\nCumulative Time:')
    print(f'{time.process_time() - start}s --> Data Generated')

    observation = gym_env.reset(coordinates, obs_chemical_shifts, noes, connectivity)

    print(f'{time.process_time() - start}s --> Observation Reset')

    histories = []

    for i in range(history_length+1):

        # If solving for fake histories off the original
        if i > 0:
            observation = gym_env.custom_state(fake_observation)
            print(f'{time.process_time() - start}s --> Custom Reset ')

        if i <= 0:
            original_observation = copy.deepcopy(observation)
        
        # Initial observation (nothing assigned)
        histories.append(copy.deepcopy(observation))

        terminated = False
        while not terminated:
            for action in observation['assign_order']:
                observation, reward, terminated, total_energy = gym_env.step(action)
                if not terminated:
                    histories.append(copy.deepcopy(observation))

        # Generate new history
        if i != history_length + 1:
            fake_observation = fakehistory.generate_history(original_observation)
        
        # Final observation (everything assigned)
        histories.append(copy.deepcopy(observation))

        print(f"{time.process_time() - start} --> {i} Histories Complete")

    with open(f'fake_histories_r{num_resid}_{history_length}.pkl', 'wb') as f:
        pickle.dump(histories, f)
    
    print(f'{time.process_time() - start}s --> {history_length} Histories Saved\n')