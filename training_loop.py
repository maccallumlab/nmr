import torch
import pickle
import numpy as np
from nmr_transformer import *

def extract_data(pickle_file='fake_histories.pkl'):
    with open(pickle_file, 'rb') as f:
        histories = pickle.load(f)
    return histories

def preprocess_data(histories, batch_size):
    """
    Loops over chosen histories and answers, splits up components, and adds shifts if indices were used.
    """
    predicted_shifts = []
    obs_chemical_shifts = []
    noes = []
    close_dist = []
    assignments = []
    peak_to_assign = []

    # batch size needs to be reworked in here (not randomized - selects from start up until 'batch_size' index)
    for history in histories[:batch_size]:
        # Predicted shifts
        predicted_shifts_temp = ([(i.H1, i.N15) for i in history['coords']])
        predicted_shifts.append(predicted_shifts_temp)
        # Observed shifts
        obs_chemical_shifts_temp = ([(i.H1, i.N15) for i in history['actual_shifts']])
        obs_chemical_shifts.append(obs_chemical_shifts_temp)
        # NOEs
        noes.append(history['noes'])
        # Close distances (duplicated for reverse pair order)
        close_dist.append([(predicted_shifts_temp[i.atom1], predicted_shifts_temp[i.atom2]) for i in history['connectivity']])
        close_dist.append([(predicted_shifts_temp[i.atom2], predicted_shifts_temp[i.atom1]) for i in history['connectivity']])
        # Assignments
        assignments.append([(obs_chemical_shifts_temp[i], predicted_shifts_temp[j]) for i,j in list(history['assignments'].items())]) #shift:atom assignment, therefore obs_shift:predicted_shift
        # Peak to assign
        peak_to_assign.append(obs_chemical_shifts_temp[history['assign_shift']])
    
    return predicted_shifts, obs_chemical_shifts, noes, close_dist, assignments, peak_to_assign

def organize_nmr_inputs(histories, batch_size):
    """
    Sets up NMR inputs as list of named tuples.
    """
    predicted_shifts, actual_shifts, noes, close_dist, assignments, assign_peak = preprocess_data(histories, batch_size=batch_size)
    
    nmr_inputs = [
        NMRInput(
            obs_chemical_shifts=torch.tensor(actual_shifts[i]),
            pred_chemical_shifts=torch.tensor(predicted_shifts[i]),
            obs_noes=torch.tensor(noes[i]),
            close_distances=torch.tensor(close_dist[i]),
            assigned_peaks=torch.tensor(assignments[i]),
            peak_to_assign=assign_peak[i]) # int or tensor for H,N peak?
            for i in range(batch_size)
    ]
    print(nmr_inputs)
    return nmr_inputs

if __name__ == "__main__":
    # need to loop over here
    histories = extract_data()
    nmr_inputs = organize_nmr_inputs(histories, batch_size=3)

    