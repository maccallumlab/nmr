import numpy as np

from fake_data import FakeDataGenerator, HSQCPeak, Protein
from energy import Energy
from assignment_order import FractionalActivation



class FakeHistoryGenerator():

    def __init__(self, num_resid):
        self.num_resid = num_resid
        self.fakedata = FakeDataGenerator(num_resid)
        self.cutoff = 0.5

    def perturb_data(self, coordinates):
        """
        Adds noise to original fake data and generates new 'actual' shifts based on predicted shifts.
        """
        coords = self.fakedata.add_noise(np.array(coordinates)[:, :3], scale=0.1)
        synthetic_shifts = self.fakedata.add_noise(np.array(coordinates)[:, 3:], scale=0.1)

        return coords, synthetic_shifts

    def weighted_error(self, scale=1):
        """
        Generates weights for number of errors added based on exponential distribution.
        Chooses error number to add based on weights.
        """
        # Weight each point
        weights = np.exp(-np.arange(self.num_resid)/scale)
        # Normalized
        weights = weights/np.sum(weights)
        # Number of errors based on weight
        error_num = np.random.choice(self.num_resid, 1, p=weights)

        # Can't make a singular error - needs to be replaced by another choice
        return error_num if error_num > 1 else error_num*2

    def add_errors(self, answer):
        """
        If errors are present chooses that number of random shifts and move answer one over from true.
        """
        error_num = self.weighted_error(scale=1)

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

    def generate_history(self, original):
        """
        Adds noise to original data, reorganizes, and edits answer key if errors are introduced.
        """

        coordinates, obs_chemical_shifts = self.perturb_data(original['coordinates'])

        noes, pred_chemical_shifts = self.fakedata.create_noes(coordinates, obs_chemical_shifts)

        connectivity = self.fakedata.calculate_connectivity(coordinates)

        coordinates, obs_chemical_shifts, noes, connectivity = self.fakedata.order_data(coordinates, obs_chemical_shifts, pred_chemical_shifts, noes, connectivity)

        # Errors not included currently
        # answer = original['assignments']
        # answer = add_errors(answer)
        
        
        state = {
        "coordinates": coordinates,
        "obs_chemical_shifts": obs_chemical_shifts,
        "noes": noes,
        "connectivity": connectivity,
        "assignments": {},
        "assign_order": [],
        "shift_to_assign": 0,
        "total_energy": 0.0,
        "reward": 0.0
        }
        
        return state
