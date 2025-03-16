import torch
import pickle
import numpy as np
from nmr_transformer import *

def extract_data(pickle_file='fake_histories.pkl'):
    with open(pickle_file, 'rb') as f:
        histories = pickle.load(f)
        answers = pickle.load(f)
    return histories, answers

def preprocess_data(histories, answers, batch_size):
    """
    Loops over chosen histories and answers, splits up components, and adds shifts if indices were used.
    *should be reorganized for clarity after fake histories is fixed*
    """
    predicted_shifts = []
    obs_chemical_shifts = []
    noes = []
    close_dist = []
    assignments = []
    peak_to_assign = []

    for history, answer in zip(histories[:batch_size], answers[:batch_size]):

        predicted_shifts_temp = ([(i.H1, i.N15) for i in history[0]])
        predicted_shifts.append(predicted_shifts_temp)
        
        obs_chemical_shifts_temp = ([(i.H1, i.N15) for i in history[1]])
        obs_chemical_shifts.append(obs_chemical_shifts_temp)

        noes.append(history[2])

        # these have parentheses issues from list comprehension
        close_dist.append([(predicted_shifts_temp[i.atom1], predicted_shifts_temp[i.atom2]) for i in history[4]])
        assignments.append([(obs_chemical_shifts_temp[i], predicted_shifts_temp[j]) for i,j in list(answer[0].items())]) #shift:atom assignment, therefore obs_shift:predicted_shift

        # peak_to_assign += [] # need to fix the fake histories for this
    
    return predicted_shifts, obs_chemical_shifts, noes, close_dist, assignments, peak_to_assign

def organize_nmr_inputs(histories, answers, batch_size):
    """
    Sets up NMR inputs as list of named tuples.
    """
    predicted_shifts, actual_shifts, noes, close_dist, assignments, assign_peak = preprocess_data(histories, answers, batch_size=batch_size)
    # print(actual_shifts)
    # print(assignments)
    nmr_inputs = [
        NMRInput(
            obs_chemical_shifts=torch.tensor(actual_shifts[i]),
            pred_chemical_shifts=torch.tensor(predicted_shifts[i]),
            obs_noes=torch.tensor(noes[i]),
            close_distances=torch.tensor(close_dist[i]),
            assigned_peaks=torch.tensor(assignments[i]),
            peak_to_assign=0) # need to fix
            for i in range(batch_size)
    ]

    print(nmr_inputs)

if __name__ == "__main__":
    # need to loop over here
    histories, answers = extract_data()
    organize_nmr_inputs(histories, answers, batch_size=2)

    