import gym
import argparse
import pickle

from nmr_gym_env import GymEnv
from fake_histories import generate_history
from visualize_data import plot_shifts

if __name__ == '__main__':
    """
    Runs through protein examples of various sizes (min to max range) following the 1:1 shift to atom assignment.
    Once example is assigned, generates histories and answers of original and moves on to next protein size.
    All histories and answers are then saved to disk.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('min_protein', type=int, help='Minimum protein size')
    parser.add_argument('max_protein', type=int, help='Max protein size')
    parser.add_argument('history_length', type=int, help='Number of synthetic examples to generate')
    args = parser.parse_args()

    min_protein = args.min_protein
    max_protein = args.max_protein
    history_length = args.history_length

    histories = []
    answers = []

    for protein_length in range(min_protein, max_protein, 10):
        gym_env = GymEnv(protein_length)
        observation = gym_env.reset(pickled=False, pickle_data=False)

        terminated = False
        while not terminated:
            for action in observation['assign_order']:
                observation, reward, terminated, total_energy = gym_env.step(action)

        histories, answer = generate_history(observation, history_length=history_length)
        histories.append(histories)
        answers.extend(answer)

    with open('fake_histories.pkl', 'wb') as f:
        pickle.dump(histories, f)
        pickle.dump(answers, f)    