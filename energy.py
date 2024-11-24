import numpy as np
from itertools import product
import math

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

class Energy():

	def __init__(self, coords, actual_shifts, noes):
		self.coords = coords
		self.actual_shifts = actual_shifts
		self.noes = noes

	def setup_noe_restraints(self):
		combos = noe_combinations(self.noes, self.actual_shifts)
		return combos

	def flat_bottom(self, x, tolerance):
		""" 
		Defines the flat bottom equation with a flat region from zero to the distance tolerance.
		"""
		assert x > 0

		# ends of flat bottom region
		xmax = tolerance
		xmin = 0

		# if x <= tolerance return energy of zero
		# if x > tolerance return energy relating to quadratic
		y = np.piecewise(x,
					[(xmin <= x) & (x <= xmax), x > xmax],
					[lambda x: 0,
					lambda x: x**2 - xmax*x])    
		return y
	
	def get_energy(self, restraints, assignments):
		"""
		Loops over the NOE restraints and each possible shift option.
		If these shift options have not been assigned the NOE is deactivated and passed, if not the NOE restraint stays activated.
		Activation means the coordinate indices are grabbed from the assignment dictionary and used in the flat_bottom function.
		The lowest energy restraint is then summed across all activated NOEs.
		"""

		total_energy = 0
		for restraint in restraints:
			restraint_energy = math.inf
			activated = True
			for i, j in restraint:
				if i not in assignments.values() or j not in assignments.values():
					activated = False
					break

				elif activated:
					k, l = assignments.get(i), assignments.get(j)

					# distance to energy evaluation
					dist = np.linalg.norm((np.array(self.coords[k]) - np.array(self.coords[l])))
					energy_value = self.flat_bottom(dist, tolerance=0.5)

					# keep the smallest energy (distance)
					restraint_energy = energy_value if energy_value < restraint_energy else restraint_energy
			
			if restraint_energy != math.inf:
				total_energy += restraint_energy

		return total_energy
