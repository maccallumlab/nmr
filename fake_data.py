import numpy as np
from itertools import product
from typing import NamedTuple
import random
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('num_resid', type=int, help='Number of residues')
args = parser.parse_args()

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
def add_noise(point, scale=0.1):
  """
  Random selection from normal (gaussian) distribution of 'scale' width from 0 (center).
  'size' makes sure that it's the same shape as the point we are adding noises to.
  ex. point = [0,1,2], noise = [1,1,1], noisy_point = [1,2,3]
  """
  noise = np.random.normal(0, scale, size=point.shape)
  noisy_point = point + noise
  return np.clip(noisy_point, 0, 1)

def calc_dist(p1, p2):
  return np.sqrt((p2[0] - p1[0])**2 + (p2[1] - p1[1])**2 + (p2[2] - p1[2])**2)

def distance_noe(protein, shifts, cutoff):
  """
  Grabs close coordinates and 'associated' HSQC shift peaks (by same index) to create NOES.
  Adds gaussian noise to all points.

  **Can end up with no NOEs depending on the cutoff**
  """
  noe = []

  for i, atom1 in enumerate(protein):
    for j, atom2 in enumerate(protein):
      if i != j:
        dist = calc_dist(atom1, atom2)
        if dist < cutoff:
          # print(dist, shifts[i], shifts[j])
          val = len(noe) # Append both shifts to correct noe index
          noe.append(list(shifts[i][:])) # H1, N1
          noe[val].append(shifts[j][0]) # H2

  noisy_noe = add_noise(np.array(noe), scale=0.01)

  return noisy_noe

def generate_data(num_resid):
  """
  Generates all fake data and orders it in lists of namedtuples.
  """

  # 3D structure [x,y,z]
  protein = sample_unit(num_resid,3)

  # "Actual" shifts [H1,N1]
  actual_shifts = sample_unit(num_resid,2)

  # Predicted shifts [H1,N1]
  predicted_shifts = add_noise(actual_shifts)

  # NOES [H1,N1,H2]
  noe = distance_noe(protein, actual_shifts, 0.5) # likely need to change dist cutoff and only accounting for actual_shifts right now

  # Probably not the best way to do this
  # Lists of namedtuples (one object per residue)
  coords = [Protein(x=resid[0], y=resid[1], z=resid[2]) for resid in protein]
  actual_shifts = [HSQCPeak(H1=shift[0], N15=shift[1]) for shift in actual_shifts]
  predicted_shifts = [HSQCPeak(H1=shift[0], N15=shift[1]) for shift in predicted_shifts]
  noe = [NOEPeak(H1=shift[0], N15=shift[1], H2=shift[2]) for shift in noe]

  # Fake_data = namedtuple('FakeData', ['protein', 'actual_shifts', 'predicted_shifts', 'noe'])
  # fake_data = Fake_data(protein=protein, actual_shifts=actual_shifts, predicted_shifts=predicted_shifts, noe=noe)

  # print(coords)
  # print(actual_shifts)
  # print(noe)

  return coords, actual_shifts, predicted_shifts, noe

# match H1 shifts
def matchH(hshift, hsqc, tolerance_h):
  """
  Loops through the HSQC list and matches the second proton shift of the NOE to a HSQC shift based on H cutoff.
  Returns the index of said shift.
  """
  h_idx = []
  for i, shift in enumerate(hsqc):
    if abs(hshift - shift.H1) <= tolerance_h:
      h_idx.append(i)
  return h_idx

# match H1 and N15 shifts
def matchNH(hshift, nshift, hsqc, tolerance_h, tolerance_n):
  """
  Loops through the HSQC list and matches the first proton and nitrogen shift of the NOE to a HSQC shift based on H and N cutoffs.
  Returns the index of said shift.
  """
  nh_idx = []
  for i, shift in enumerate(hsqc):
    if abs(hshift - shift.H1) <= tolerance_h and abs(nshift - shift.N15) <= tolerance_n:
      nh_idx.append(i)
  return nh_idx

def noe_combinations(noe, actual_shifts, tolerance_h=0.2, tolerance_n=0.2):
  """
  Loops through the NOES to match the H1, N1 and H2 shifts to two HSQC shifts.
  Lists the HSQC shifts separately (H1/N1 as contact1 and H2 as contact2) for each NOE.
  Returns a lists of tuples of all possible shift index combinations for each NOE.
  ex. [[(1,2),(1,3)],[(3,0)]]
      1 and 2 are a pair of potential shift indices for one NOE, 1 and 3 are another for the same NOE.

  Does not consider:
  - Same shift assignments (nh_index and h_index could be the same for one NOE).
    Filter these out initially, but itertools product can still combine one shift as a possible pair.

  - Shift pair repetitions (two shifts are included in both the nh_index and h_index).
    Gives a double count ex. contact1/2 = [[3,0]], product = [[(0,3),(3,0)]]
  """
  contact1 = []
  contact2 = []

  for i, peak in enumerate(noe):
    # matching H1 and N15 in NOE to shifts in HSQC
    nh_idx = matchNH(peak.H1, peak.N15, actual_shifts, tolerance_h, tolerance_n)
    # matching H2 in NOE to H1 shifts in HSQC
    h_idx = matchH(peak.H2, actual_shifts, tolerance_h)

    if nh_idx != h_idx: # Filter out same shift assignments for one NOE
      contact1.append(nh_idx)
      contact2.append(h_idx)

  contacts = [[] for i in range(len(contact1))] # List to match number of NOES
  for i in range(len(contact1)):
    contacts[i] += list(product(contact1[i], contact2[i]))

  return contacts

num_resid = args.num_resid
# These should be grouped to export into an environment
coords, actual_shifts, predicted_shifts, noe = generate_data(num_resid)
combinations = noe_combinations(noe, actual_shifts)
print(combinations)
