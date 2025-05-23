import numpy as np
from itertools import product
from typing import NamedTuple
import random
import argparse
import pickle
import glob as glob


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


class FakeDataGenerator():

    def __init__(self, num_resid):
        self.num_resid: int = num_resid
        self.nshift_min = 100
        self.nshift_max = 135
        self.hshift_min = 6
        self.hshift_max = 10
        self.cutoff = 0.5

    def sample_unit(self, num_points, num_sides, min_len=0, max_len=1):
        """ 
        Samples points from unit square or cube.
        """
        return np.random.uniform(low=min_len, high=max_len, size=(num_points, num_sides))

    def scale_unit(self, coordinates):
        """
        Scaling box size to reflect a globular protein.
        PDB:1CRC (~100 resid, globular, ~3.2 nm diameter)

        Rg = RN**v
        Scaling factor (v) of 0.4, instead of 0.3 (cube-root)
        R of 0.2 nm used in paper for fit
        DOI: 10.1142/S021972002050050X
        """
        radius_gyration = 0.2*(self.num_resid**0.4)

        com = np.mean(coordinates, axis=0) # should be ~0.5
        distances_sampled = np.linalg.norm(coordinates - com, axis=1) # distances around origin
        radius_sampled = np.sqrt(np.mean((distances_sampled**2))) # radius of gyration based on these distances
        
        # Expected is radius_gyration, current is radius_sampled --> scale to fit
        # Will overlap occur?
        scale = radius_gyration / radius_sampled 
        coordinates_scaled = com + ((coordinates - com) * scale)

        return coordinates_scaled

    def create_hsqc(self):
        """
        Sample points within NMR window for H1 and N15. Stack values to create HSQC peaks.
        """
        # H shifts
        h_shifts = self.sample_unit(self.num_resid, num_sides=1, min_len=self.hshift_min, max_len=self.hshift_max)
        # N shifts
        n_shifts = self.sample_unit(self.num_resid, num_sides=1, min_len=self.nshift_min, max_len=self.nshift_max)

        return np.hstack((h_shifts, n_shifts))

    def add_noise(self, points, scale=0.1):
        """
        Random selection from normal (gaussian) distribution of 'scale' width from 0 (center).
        'size' makes sure that it's the same shape as the point we are adding noises to.
        ex. point = [0,1,2], noise = [1,1,1], noisy_point = [1,2,3]
        """
        noise = np.random.normal(loc=0, scale=scale, size=points.shape)
        noisy_point = points + noise

        # fold over points outside of boundaries 
        # for point in noisy_point:
        #     for i, axis in enumerate(point):
        #         if max_len < axis:
        #             point[i] = (max_len - (axis - max_len))
        #         if axis < min_len:
        #             point[i] = (min_len - axis)
        #         else:
        #             pass

        return noisy_point

    def calculate_dist(self, p1, p2):
        """
        Euclidian distance between two points.
        """
        return np.linalg.norm((p2 - p1))

    def create_noes(self, coordinates, shifts, random_key=False):
        """
        Grabs close coordinates and 'associated' HSQC shift peaks (randomized index or 1:1 correlation) to create noes.
        Predicted shifts assigned as shift order - this order is associated with the coordinate order if randomized (ie. the answer key).
        Adds gaussian noise to all points at the end.

        **Can end up with no noes depending on the cutoff**
        """
        if random_key:
            shifts = np.random.permutation(shifts)

        noe_list = []
        for i, atom1 in enumerate(coordinates):
            for j, atom2 in enumerate(coordinates):
                if i != j:
                    dist = self.calculate_dist(atom1, atom2)
                    # dist = dist_grid[i][j]
                    if dist < self.cutoff:
                        #print(dist, shifts[i], shifts[j])
                        noe = list(shifts[i][:]) # H1, N15
                        noe.append(shifts[j][0]) # H2
                        noe_list.append(noe)

        noisy_noes = self.add_noise(np.array(noe_list), scale=0.01)
        pred_chemical_shifts = self.add_noise(np.array(shifts), scale=0.1)

        return noisy_noes, pred_chemical_shifts

    def calculate_connectivity(self, coordinates):
        """
        Calculates close contacts based on protein coordinates.
        """
        contacts = []

        for i, atom1 in enumerate(coordinates):
            for j, atom2 in enumerate(coordinates):
                if i != j:
                    dist = self.calculate_dist(atom1, atom2)
                    # dist = dist_grid[i][j]
                    if dist < self.cutoff:
                        contacts.append((i, j, dist))

        return contacts

    # def pdist_test(self, coordinates):
    #     pairwise_dists = pdist(coordinates)
    #     square_dists = squareform(pairwise_dists)
    #     return square_dists

    def generate_data_arrays(self, random_key):
        """
        Generates all qualities of the system as individual arrays.
        """
        # 3D structure [x,y,z]
        pred_coordinates = self.sample_unit(self.num_resid, num_sides=3)
        pred_coordinates = self.scale_unit(pred_coordinates)
    
        # "Actual" shifts [H1,N15]
        obs_chemical_shifts = self.create_hsqc()
        
        # Distance grid
        # dist_grid = self.pdist_test(pred_coordinates)

        # NOEs [H1,N15,H2] and predicted shifts [H1,N15]
        noes, pred_chemical_shifts = self.create_noes(pred_coordinates, obs_chemical_shifts, random_key=random_key)

        # Connectivity [atom1,atom2,dist]
        connectivity = self.calculate_connectivity(pred_coordinates)

        return pred_coordinates, obs_chemical_shifts, pred_chemical_shifts, noes, connectivity

    def order_data(self, pred_coordinates, obs_chemical_shifts, pred_chemical_shifts, noes, connectivity):
        """
        Compiles data in a list of named tuples with one object per residue.
        """
        pred_coordinates = [Protein(x=resid[0], y=resid[1], z=resid[2], H1=shift[0], N15=shift[1]) for resid, shift in zip(pred_coordinates, pred_chemical_shifts)]
        # pred_chemical_shifts = [HSQCPeak(H1=shift[0], N15=shift[1]) for shift in pred_chemical_shifts]
        obs_chemical_shifts = [HSQCPeak(H1=shift[0], N15=shift[1]) for shift in obs_chemical_shifts]
        noes = [NOEPeak(H1=shift[0], N15=shift[1], H2=shift[2]) for shift in noes]
        connectivity = [Connectivity(atom1=contact[0], atom2=contact[1], distance=contact[2]) for contact in connectivity]

        return pred_coordinates, obs_chemical_shifts, noes, connectivity

    def dump_pickle(self, pred_coordinates, obs_chemical_shifts, noes, connectivity, example=True):
        """
        Saves data to disk. Run is given a proper name if used as a saved example. 
        """
        name = f'fakedata_r{self.num_resid}.pkl' if example else f'current_run.pkl'
        
        with open(name, 'wb') as f:
            pickle.dump(pred_coordinates, f)
            pickle.dump(obs_chemical_shifts, f)
            pickle.dump(noes, f)
            pickle.dump(connectivity, f)
    
    def load_pickle(self, example):

        name = f"./*r{self.num_resid}.pkl" if example else "./current_run.pkl"

        pickle_file = glob.glob(name)
        with open(pickle_file[0], 'rb') as f:
            coordinates = pickle.load(f)
            obs_chemical_shifts = pickle.load(f)
            noes = pickle.load(f)
            connectivity = pickle.load(f)
        
        return coordinates, obs_chemical_shifts, noes, connectivity

    def generate_data(self, pickle_data=True, example=True, random_key=False):
        """
        Generates all fake data, orders it in lists of namedtuples, and pickles if required.
        """
        pred_coordinates, obs_chemical_shifts, pred_chemical_shifts, noes, connectivity = self.generate_data_arrays(random_key=random_key)
        pred_coordinates, obs_chemical_shifts, noes, connectivity = self.order_data(pred_coordinates, obs_chemical_shifts, pred_chemical_shifts, noes, connectivity)

        if pickle_data:
            self.dump_pickle(pred_coordinates, obs_chemical_shifts, noes, connectivity, example=example)

        else:
            return pred_coordinates, obs_chemical_shifts, noes, connectivity



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('num_resid', type=int, help='Number of residues')
    args = parser.parse_args()

    num_resid = args.num_resid

    fakedata = FakeDataGenerator(num_resid)
    pred_coordinates, obs_chemical_shifts, noes, connectivity = fakedata.generate_data(pickle_data=False, example=False)
    