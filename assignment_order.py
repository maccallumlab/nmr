import numpy as np
from itertools import chain



class FractionalActivation():

    def __init__(self, num_resid, restraints):
        self.num_resid = num_resid
        self.restraints = restraints

    def activation_loop(self, target, noe, assign_order):
        """
        Checks if target is in NOE, if true, checks if it has already been assigned as higher priority.
        """
        if target in noe and target not in assign_order:
                return 1/(len(noe))
        else:
            return 0
    
    def fractional_activation(self):
        """
        Sets up residue assignment order based on fractional prevelence across NOEs.

        """
        assign_order = [] 

        # Get all possible values from NOEs
        noes_set = set(chain.from_iterable(chain.from_iterable(self.restraints)))

        for i in range(len(noes_set)):
            running_sums = np.zeros(self.num_resid)
            for noe in self.restraints:
                # values from current NOE, excluding assigned values
                noe_set = (set(chain.from_iterable(noe))).difference(set(assign_order))
                running_sum = ([self.activation_loop(j, noe_set, assign_order) for j in range(self.num_resid)])
                # print(running_sum)
                running_sums += running_sum
            assign_order.append(np.argmax(running_sums))
            # print(i, assign_order)

        missing_shifts = list(sorted(set(range(self.num_resid)).difference(set(assign_order)))) # need to preserve sorted index order for manual check
        assign_order.extend(missing_shifts)

        return assign_order

    # def activation_loop(self, target, noe, assign_order):
    #     s = set()
    #     for combo in noe:
    #         if combo[0] not in assign_order: # check if values have already been assigned as higher priority
    #             s.add(combo[0])
    #         if combo[1] not in assign_order:
    #             s.add(combo[1])

    #     if target in s:
    #         return 1/(len(s))
    #     else:
    #         return 0

    # def fractional_activation(self):

    #     assign_order = []
    #     missing_shifts = []

    #     while len(assign_order) < self.num_resid:
    #         temp2 = []
    #         for i in range(self.num_resid):
    #             temp = 0
    #             for noe in self.restraints:
    #                 # print(i)
    #             # print([(self.activation_loop(j, noe, assign_order)) for j, noe in zip(self.restraints)])
    #                 temp += self.activation_loop(i, noe, assign_order) # sum up the probabilities for 'i' (the given shift target)
    #                 # print(assign_order, temp)
    #                 # print(i, temp2)

    #             # identifies any missing shifts
    #             if temp == 0 and len(assign_order) == 0:
    #                 missing_shifts.append(i)

    #             # appends sum to list (one sum per shift)
    #             temp2.append(temp)
    #             # print(temp2)
    #         # makes sure extra zeros are not added during final iterations (if shifts are missing)
    #         if any(temp2) == True:
    #             # for x in temp2:
    #             # proper order based on max value
    #             assign_order.append(np.argmax(temp2))
    #                 # temp2[np.argmax(temp2)] = 0
    #                 # print(temp2)
    #         else:
    #             # add missing shifts to end of list
    #             assign_order = assign_order + missing_shifts

    #     return assign_order




