import numpy as np
from fake_data import *
from energy import noe_combinations

def perturb_data(coords):
    """
    Adds noise to original fake data and generates new 'actual' shifts based on predicted shifts.
    """
    protein = add_noise(np.array(coords)[:, :3], scale=0.1)
    predicted_shifts = add_noise(np.array(coords)[:, 3:], scale=0.1)
    actual_fake_shifts = add_noise(np.array(coords)[:, 3:], scale=0.1)

    return protein, predicted_shifts, actual_fake_shifts

def recalculate_noe(protein, shifts, shift_order, cutoff):
    """
    Grabs close coordinates and 'associated' HSQC shift peaks to create new NOES.
    Adds gaussian noise to all points at the end.
    """
    # Can use this to randomize if it was used in the original
    # shifts = [shifts[i] for i in shift_order.values()]

    noes = []

    for i, atom1 in enumerate(protein):
        for j, atom2 in enumerate(protein):
            if i != j:
                dist = calc_dist(atom1, atom2)
                if dist < cutoff:
                    # print(dist, shifts[i], shifts[j])
                    noe = list(shifts[i][:]) # H1, N1
                    noe.append(shifts[j][0]) # H2
                    noes.append(noe)

    noisy_noe = add_noise(np.array(noes), scale=0.01)

    return noisy_noe

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

def add_errors(answers):
    """
    If errors are present chooses that number of random shifts and moves answers one over from true.
    """
    error_num = weighted_error(len(answers), scale=1)

    if error_num == 0:
        return answers
    else:
        selections = np.random.choice(list(answers.values()), error_num, replace=False)
        mutations = np.roll(selections, 1)
        for i, j in enumerate(selections):
            answers[j] = mutations[i]

    return answers

def generate_history(original, history_len=5):
    """
    Loops over chosen history length, adding noise to original data, reorganizes, and edits answer key if errors are introduced.
    """
    answers = {}
    history = []

    for i in range(history_len):
        protein, predicted_shifts, actual_shifts = perturb_data(original['coords'])
        noes = recalculate_noe(protein, actual_shifts, original['assignments'], cutoff=0.5)
        connectivity = connectivity_data(protein, cutoff=0.37)

        # Lists of namedtuples (one object per residue)
        coords = [Protein(x=resid[0], y=resid[1], z=resid[2], H1=shift[0], N15=shift[1]) for resid, shift in zip(protein, predicted_shifts)]
        actual_shifts = [HSQCPeak(H1=shift[0], N15=shift[1]) for shift in actual_shifts]
        noes = [NOEPeak(H1=shift[0], N15=shift[1], H2=shift[2]) for shift in noes]
        connectivity = [Connectivity(atom1=connect[0], atom2=connect[1], distance=connect[2]) for connect in connectivity]

        restraints = noe_combinations(noes, actual_shifts)

        history.append(coords)
        history.append(actual_shifts)
        history.append(noes)
        history.append(restraints)
        history.append(connectivity)

        answers[i] = dict(original['assignments'])
        answers[i] = add_errors(answers[i])

    return history, answers

        