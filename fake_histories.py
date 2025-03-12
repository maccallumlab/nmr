import numpy as np
from fake_data import *
from energy import noe_combinations

def perturb_data(coords):
    """
    Adds noise to original fake data and generates new 'actual' shifts based on predicted shifts.
    """
    protein = add_noise(np.array(coords)[:, :3], scale=0.1)
    actual_fake_shifts = add_noise(np.array(coords)[:, 3:], scale=0.1)

    return protein, actual_fake_shifts

def weighted_error(num_resid, scale=1):
    """
    Generates weights for number of errors added based on exponential distribution.
    Chooses error number to add based on weights.
    """
    # Weight each point
    weights = np.exp(-np.arange(num_resid)/scale)
    # Normalized
    weights = weights/np.sum(weights)
    # Number of errors based on weight
    error_num = np.random.choice(num_resid, 1, p=weights)

    # Can't make a singular error - needs to be replaced by another choice
    return error_num if error_num > 1 else error_num*2

def add_errors(answer):
    """
    If errors are present chooses that number of random shifts and move answer one over from true.
    """
    error_num = weighted_error(len(answer), scale=1)

    if error_num == 0:
        return answer
    else:
        selections = np.random.choice(list(answer.values()), error_num, replace=False)
        mutations = np.roll(selections, 1)
        for i, j in enumerate(selections):
            answer[j] = mutations[i]

    return answer

def generate_history(original, history_length=5):
    """
    Loops over chosen history length, adding noise to original data, reorganizes, and edits answer key if errors are introduced.
    """
    answer = {}
    history = []

    for i in range(history_length):
        protein, actual_shifts = perturb_data(original['coords'])
        noes, predicted_shifts = distance_noe(protein, actual_shifts, cutoff=0.5)
        connectivity = connectivity_data(protein, cutoff=0.37)

        coords, actual_shifts, noes, connectivity = order_data(protein, actual_shifts, predicted_shifts, noes, connectivity)

        restraints = noe_combinations(noes, actual_shifts)

        history.append(coords)
        history.extend((actual_shifts, noes, restraints, connectivity))

        answer[i] = dict(original['assignments'])
        answer[i] = add_errors(answer[i])

    return history, answer

        