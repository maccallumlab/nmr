import numpy as np
from itertools import product, combinations
from typing import NamedTuple
import random
import argparse

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
		for value in point:
			if max < value:
				value = (max-(value-max))
			elif value < min:
				value = (min-value)

	return noisy_point

def calc_dist(p1, p2):
	return np.linalg.norm((p2-p1))

def distance_noe(protein, shifts, cutoff):
	"""
	Grabs close coordinates and 'associated' HSQC shift peaks (by same index) to create NOES.
	Adds gaussian noise to all points.

	**Can end up with no NOEs depending on the cutoff**
	"""
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

def generate_data(num_resid):
	"""
	Generates all fake data and orders it in lists of namedtuples.
	"""
	# 3D structure [x,y,z]
	protein = sample_unit(num_resid, num_sides=3)
	# "Actual" shifts [H1,N1]
	actual_shifts = sample_unit(num_resid, num_sides=2)
	# Predicted shifts [H1,N1]
	predicted_shifts = add_noise(actual_shifts)
	# NOES [H1,N1,H2]
	noes = distance_noe(protein, actual_shifts, cutoff=0.5) # likely need to change dist cutoff and only accounting for actual_shifts right now

	# Lists of namedtuples (one object per residue)
	coords = [Protein(x=resid[0], y=resid[1], z=resid[2]) for resid in protein]
	actual_shifts = [HSQCPeak(H1=shift[0], N15=shift[1]) for shift in actual_shifts]
	predicted_shifts = [HSQCPeak(H1=shift[0], N15=shift[1]) for shift in predicted_shifts]
	noes = [NOEPeak(H1=shift[0], N15=shift[1], H2=shift[2]) for shift in noes]

	# print(coords)
	# print(actual_shifts)
	# print(noe)

	return coords, actual_shifts, predicted_shifts, noes

# match H1 shifts
def matchH(hshift, hsqc, tolerance_h):
	"""
	Loops through the HSQC list and matches the second proton shift of the NOE to a HSQC shift based on H cutoff.
	Returns the index of said shift.
	"""
	h_index = []
	for i, shift in enumerate(hsqc):
		if abs(hshift - shift.H1) <= tolerance_h:
			h_index.append(i)

	return h_index

# match H1 and N15 shifts
def matchNH(hshift, nshift, hsqc, tolerance_h, tolerance_n):
	"""
	Loops through the HSQC list and matches the first proton and nitrogen shift of the NOE to a HSQC shift based on H and N cutoffs.
	Returns the index of said shift.
	"""
	nh_index = []
	for i, shift in enumerate(hsqc):
		if abs(hshift - shift.H1) <= tolerance_h and abs(nshift - shift.N15) <= tolerance_n:
			nh_index.append(i)

	return nh_index

def noe_combinations(noes, actual_shifts, tolerance_h=0.2, tolerance_n=0.2):
	"""
	Loops through the NOES to match the H1, N1 and H2 shifts to two HSQC shifts.
	Lists the HSQC shifts separately (H1/N1 as contact1 and H2 as contact2) for each NOE.
	Returns a lists of tuples of all possible shift index combinations for each NOE.
	ex. [[(1,2),(1,3)],[(3,0)]]
		1 and 2 are a pair of potential shift indices for one NOE, 1 and 3 are another for the same NOE.
	"""
	contact1 = []
	contact2 = []
	for i, peak in enumerate(noes):
		# matching H1 and N15 in NOE to shifts in HSQC
		nh_index = matchNH(peak.H1, peak.N15, actual_shifts, tolerance_h, tolerance_n)
		# matching H2 in NOE to H1 shifts in HSQC
		h_index = matchH(peak.H2, actual_shifts, tolerance_h)

		contact1.append(nh_index)
		contact2.append(h_index)

	# print(contact1)
	# print(contact2)
	
	contacts = []
	for i, contact in enumerate(contact1):
		prod = product(contact, contact2[i])
		# filter out same shift pairs and inverted duplicates (ex. (0,0) and [(0,1),(1,0)])
		pairs = set(tuple(sorted(combo)) for combo in prod if combo[0] != combo[1])
		contacts.append(list(pairs))

	return contacts

if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('num_resid', type=int, help='Number of residues')
	args = parser.parse_args()

	num_resid = args.num_resid
	# These should be grouped to export into an environment
	coords, actual_shifts, predicted_shifts, noes = generate_data(num_resid)
	combinations = noe_combinations(noes, actual_shifts)
	print(combinations)
