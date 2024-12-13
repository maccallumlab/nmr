import gym
import argparse

from nmr_gym_env import GymEnv

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('num_resid', type=int, help='Number of residues')
    args = parser.parse_args()

    num_resid = args.num_resid
    gym_env = GymEnv(num_resid)
    episodes = 1

    for episode in range(episodes):
        observation = gym_env.reset()
        terminated = False
        
        while not terminated:
            action = gym_env.action_space.sample()
            # action = int(input())
            # need to prevent sample from grabbing the same atom more than once
            # but also need the loop to terminate (without last bit it won't go into else for last iteration)
            if action in observation['assignments'].values() and len(observation['assignments']) != num_resid:
                continue
            else:
                observation, reward, terminated = gym_env.step(action)
        
        print(f"Final assignments (shift:atom) = {observation['assignments']}\nFinal energy evaluation = {observation['total_energy']}")