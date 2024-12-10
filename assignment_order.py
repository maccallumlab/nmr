import numpy as np

class FracAct():

    def __init__(self, num_resid, noes):
        self.num_resid = num_resid # value would need to be the number of shifts/atoms
        self.noes = noes

    def activation_loop(self, target, noe, assign_order):
        s = set()
        for combo in noe:
            if combo[0] not in assign_order: # check if values have already been assigned as higher priority
                s.add(combo[0])
            if combo[1] not in assign_order:
                s.add(combo[1])

        if target in s:
            return 1/(len(s))
        else:
            return 0

    # def fractional_activation(self):
    #     assign_order = []
    #     for j in range(self.num_resid): # I don't know why I need this loop here it just makes it work
    #         temp2 = []
    #         for i in range(self.num_resid):
    #             temp =  0
    #             for noe in self.noes:
    #                 temp += self.activation_loop(i, noe, assign_order) # sum up the probabilities for 'i' (the given shift target)
    #             temp2.append(temp)

    #         assign_order.append(np.argmax(temp2)) # append the max value's index (takes first occurence if equivalent)

    #     return assign_order
    def fractional_activation(self):
        assign_order = []
        missing_shifts = []

        while len(assign_order) < self.num_resid:
            temp2 = []
            for i in range(self.num_resid):
                temp = 0
                for noe in self.noes:
                    temp += self.activation_loop(i, noe, assign_order) # sum up the probabilities for 'i' (the given shift target)
                
                # identifies any missing shifts
                if temp == 0 and len(assign_order) == 0:
                    missing_shifts.append(i)
                
                # appends sum to list (one sum per shift)
                temp2.append(temp)

            # makes sure extra zeros are not added during final iterations (if shifts are missing)
            if any(temp2) == True:
                # proper order based on max value
                assign_order.append(np.argmax(temp2))
            else:
                # add missing shifts to end of list
                assign_order = assign_order + missing_shifts

        return assign_order
    
    # intermediate check for first set of probabilites assigned - not used as final calculation
    def fractional_activation_summation(self):

        assign_order = []
        for j in range(self.num_resid):
            temp2 = []
            for i in range(self.num_resid):
                temp =  0
                for noe in self.noes:
                    temp += activation_loop(i, noe, assign_order)
                temp2.append(temp)

            return temp2




