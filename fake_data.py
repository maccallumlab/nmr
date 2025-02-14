import numpy as np
from itertools import product
from typing import NamedTuple
import random
import argparse
import pickle

class HSQCPeak(NamedTuple):
    H1: float
    N15: float

class NOEPeak(NamedTuple):
    H1: float
    N15: float
    H2: float

class Protein(NamedTuple):
    x: float
    y: float
    z: float
    H1: float
    N15: float

class Connectivity(NamedTuple):
    atom1: float
    atom2: float
    distance: float

# Sampled points from unit square/cube
def sample_unit(n, num_sides, min=0, max=1):
    return np.random.uniform(min, max, size=(n, num_sides))

# Noise to sampled points
def add_noise(point, scale=0.1, min=0, max=1):
    """
    Random selection from normal (gaussian) distribution of 'scale' width from 0 (center).
    'size' makes sure that it's the same shape as the point we are adding noises to.
    ex. point = [0,1,2], noise = [1,1,1], noisy_point = [1,2,3]
    """
    noise = np.random.normal(0, scale, size=point.shape)
    noisy_point = point + noise

    # fold over points outside of boundaries 
    for point in noisy_point:
        for i, value in enumerate(point):
            if max < value:
                point[i] = (max-(value-max))
            if value < min:
                point[i] = (min-value)

    return noisy_point

def calc_dist(p1, p2):
    return np.linalg.norm((p2-p1))

def distance_noe(protein, shifts, cutoff):
    """
    Grabs close coordinates and 'associated' HSQC shift peaks (randomized index) to create NOES.
    Predicted shifts assigned as the new randomized shift order - this order is associated with the coordinate order (ie. the answer key).
    Adds gaussian noise to all points at the end.

    **Can end up with no NOEs depending on the cutoff**
    """
    randomized_shifts = np.random.permutation(shifts)

    noes = []

    for i, atom1 in enumerate(protein):
        for j, atom2 in enumerate(protein):
            if i != j:
                dist = calc_dist(atom1, atom2)
                if dist < cutoff:
                    # print(dist, shifts[i], shifts[j])
                    # noe = list(shifts[i][:]) # H1, N1
                    # noe.append(shifts[j][0]) # H2
                    noe = list(randomized_shifts[i][:])
                    noe.append(randomized_shifts[j][0])
                    noes.append(noe)

    noisy_noe = add_noise(np.array(noes), scale=0.01)
    predicted_shifts = add_noise(np.array(randomized_shifts), scale=0.1)

    return noisy_noe, predicted_shifts

def close_contacts(protein, cutoff):
    contacts = []

    for i, atom1 in enumerate(protein):
        for j, atom2 in enumerate(protein):
            if i != j:
                dist = calc_dist(atom1, atom2)
                if dist < cutoff:
                    contacts.append((i,j))

    return contacts

def connectivity_data(protein, cutoff):
    connectivity = []
    
    for i, atom1 in enumerate(protein):
      for j, atom2 in enumerate(protein):
        if i != j:
            dist = calc_dist(atom1, atom2) #dist = np.linalg.norm(atom1 - atom2)
            if dist < cutoff:
                connectivity.append((atom1, atom2, dist))

    return connectivity

def generate_data(num_resid, pickle_data=True, example=True):
    """
    Generates all fake data and orders it in lists of namedtuples.
    """
    # 3D structure [x,y,z]
    protein = sample_unit(num_resid, num_sides=3)
    # "Actual" shifts [H1,N1]
    actual_shifts = sample_unit(num_resid, num_sides=2)
    # NOES [H1,N1,H2] and predicted shifts [H1,N1]
    noes, predicted_shifts = distance_noe(protein, actual_shifts, cutoff=0.5)
    # connectivity [atom1,atom2,dist]
    connectivity = connectivity_data(protein, cutoff=0.5)

    # Lists of namedtuples (one object per residue)
    coords = [Protein(x=resid[0], y=resid[1], z=resid[2], H1=shift[0], N15=shift[1]) for resid, shift in zip(protein, predicted_shifts)]
    actual_shifts = [HSQCPeak(H1=shift[0], N15=shift[1]) for shift in actual_shifts]
    noes = [NOEPeak(H1=shift[0], N15=shift[1], H2=shift[2]) for shift in noes]
    connectivity = [Connectivity(atom1=connect[0], atom2=connect[1], distance=connect[2]) for connect in connectivity]

    if pickle_data:
        name = f'fakedata_r{num_resid}.pkl' if example else f'current_run.pkl'

        with open(name, 'wb') as f:
            pickle.dump(coords, f)
            pickle.dump(actual_shifts, f)
            pickle.dump(noes, f)
            pickle.dump(connectivity, f)

    else:
        return coords, actual_shifts, noes, connectivity

# if __name__ == '__main__':
# parser = argparse.ArgumentParser()
# parser.add_argument('num_resid', type=int, help='Number of residues')
# args = parser.parse_args()

# num_resid = args.num_resid
# generate_data(num_resid)

#     # These should be grouped to export into an environment
#     coords, actual_shifts, predicted_shifts, noes = generate_data(num_resid)
    # print(f'coords {(np.array(coords)).tolist()}')#\n shifts{(np.array(actual_shifts)).tolist()}\n noes{(np.array(noes)).tolist()}')
    # # print(actual_shifts)
    # # print(noes)

    # energy_obj = Energy(coords, actual_shifts, noes)
    # test = energy_obj.setup_noe_restraints()
    # # print(test)
    # # print(coords)
        
    # test_energy = energy_obj.get_total_energy(test, {0:2, 1:1, 2:0})
    # print(test_energy)