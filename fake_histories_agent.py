import gym
import argparse
import pickle
import copy

from nmr_gym_env import GymEnv
from fake_histories import generate_history
from visualize_data import plot_shifts

if __name__ == '__main__':
    """
    Runs through protein examples of one size following the 1:1 shift to atom assignment.
    Once example is assigned, generates history and repeats assignment process.
    All histories are then saved to disk.
    """
    parser = argparse.ArgumentParser()
    # parser.add_argument('min_protein', type=int, help='Minimum protein size')
    parser.add_argument('max_protein', type=int, help='Max protein size')
    parser.add_argument('history_length', type=int, help='Number of synthetic examples to generate')
    args = parser.parse_args()

    # min_protein = args.min_protein
    max_protein = args.max_protein
    history_length = args.history_length

    gym_env = GymEnv(max_protein)
    observation = gym_env.reset(pickled=False, pickle_data=False)

    histories = []

    for i in range(history_length):
        if i > 0:
            observation = gym_env.custom_state(fake_observation)

        histories.append(copy.deepcopy(observation))

        terminated = False
        while not terminated:
            for action in observation['assign_order']:
                observation, reward, terminated, total_energy = gym_env.step(action)
                histories.append(copy.deepcopy(observation))

        # don't know if this is needed - issues with permanence of observation
        if i <= 0:
            original_observation = copy.deepcopy(observation)

        fake_observation = generate_history(original_observation, i, history_length=history_length)

    # print(histories)
    with open('fake_histories.pkl', 'wb') as f:
        pickle.dump(histories, f)
    
    print(f'{history_length} fake histories saved')