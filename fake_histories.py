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
        # print(selections)
        mutations = np.roll(selections, 1)
        # print(mutations)
        for i, j in enumerate(selections):
            # print(answer, answer[j], mutations[i], error_num)
            answer[j] = mutations[i]
            
    return answer

def generate_history(original, i, history_length=5):
    """
    Loops over chosen history length, adding noise to original data, reorganizes, and edits answer key if errors are introduced.
    """

    protein, actual_shifts = perturb_data(original['coords'])
    noes, predicted_shifts = distance_noe(protein, actual_shifts, cutoff=float(1/np.cbrt(len(actual_shifts))))
    
    connectivity = connectivity_data(protein, cutoff=float(0.8/np.cbrt(len(actual_shifts))))

    coords, actual_shifts, noes, connectivity = order_data(protein, actual_shifts, predicted_shifts, noes, connectivity)

    # answer = original['assignments']
    # answer, error_num = add_errors(answer)
    
    # issue with permanence - need to recreate full assignment for now
    answer = {i:i for i in original['assign_order']}
    answer = add_errors(answer)

    state = {
    "coords": coords,
    "actual_shifts": actual_shifts,
    "noes": noes,
    "connectivity": connectivity,
    "assignments": {},
    "assign_order": list(answer.values()),
    "assign_shift": 0,
    "total_energy": 0.0,
    "reward": 0.0
    }

    return state