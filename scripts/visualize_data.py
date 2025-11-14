from typing import NamedTuple
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from itertools import chain
import math



class Visualization():

    def __init__(self, coordinates, obs_chemical_shifts, connectivity, restraints):
        self.coordinates = coordinates
        self.obs_chemical_shifts = obs_chemical_shifts
        self.connectivity = connectivity
        self.restraints = restraints

    def get_data(self, named_tuple_used=True):
        if named_tuple_used:
            # Actual
            actual_x = [shift.H1 for shift in self.obs_chemical_shifts]
            actual_y = [shift.N15 for shift in self.obs_chemical_shifts]
            # Predicted
            predict_x = [shift.H1 for shift in self.coordinates]
            predict_y = [shift.N15 for shift in self.coordinates]
            # Connectivity
            contacts = [(contact.atom1, contact.atom2) for contact in self.connectivity]

        else:
            # Actual
            actual_x = [shift[0] for shift in self.obs_chemical_shifts]
            actual_y = [shift[1] for shift in self.obs_chemical_shifts]
            # Predicted
            predict_x = [shift[3] for shift in self.coordinates]
            predict_y = [shift[4] for shift in self.coordinates]
        
        return actual_x, actual_y, predict_x, predict_y, contacts

    def plot_shifts(self, named_tuple_used=True):

        actual_x, actual_y, predict_x, predict_y, contacts = self.get_data(named_tuple_used=named_tuple_used)

        plt.scatter(predict_x, predict_y, color='blue', label='Predicted shifts')

        plt.scatter(actual_x, actual_y, color='red', label='Measured shifts')

        # Label measured shifts by index and predicted shifts by associated coordinate
        for i in range(len(self.obs_chemical_shifts)):
            plt.annotate(i, (actual_x[i]+0.01, actual_y[i]+0.01))
            plt.annotate(i, (predict_x[i]+0.01, predict_y[i]+0.01))

        # Draw restraints
        for restraint in self.restraints:
            for j,k in restraint:
                plt.plot((actual_x[j], actual_x[k]), (actual_y[j], actual_y[k]), color='red')
        
        # Draw close contacts
        for i,j in contacts:
            plt.plot((predict_x[i], predict_x[j]), (predict_y[i], predict_y[j]), color='blue', ls='dotted', alpha=0.5)

        plt.xlim([6, 10])
        plt.ylim([100, 135])

        plt.xlabel('H1')
        plt.ylabel('N15')
        plt.legend()

        plt.savefig("shifts_plot.png", dpi = 150)
        plt.close()

    def plot_step(self, shift_to_assign, assign_order):

        plt.subplot(1, 2, 1)

        self.plot_shifts()

        plt.subplot(1, 2, 2)

        actual_x, actual_y, predict_x, predict_y, contacts = self.get_data(named_tuple_used=True)

        plt.scatter(actual_x[shift_to_assign], actual_y[shift_to_assign], color='green', label='Shift to Assign')
        plt.annotate(shift_to_assign, (actual_x[shift_to_assign]+0.01, actual_y[shift_to_assign]+0.01))

        # to be plotted as restraints
        test_values = [restraint for restraint in chain.from_iterable(self.restraints) if shift_to_assign in restraint]
        test_values = set(chain.from_iterable(test_values)).difference({shift_to_assign})

        if len(test_values) > 0:
            for j in test_values:
                plt.scatter(actual_x[j], actual_y[j], color='red')
                plt.plot((actual_x[j], actual_x[shift_to_assign]), (actual_y[j], actual_y[shift_to_assign]), color='red')
                plt.annotate(j, (actual_x[j]+0.01, actual_y[j]+0.01))

        for i in range(len(assign_order)):
            if abs(predict_x[i] - actual_x[shift_to_assign]) <= 0.15:
                plt.scatter(predict_x[i], predict_y[i], color='blue')
                [plt.plot((predict_x[i], predict_x[j]), (predict_y[i], predict_y[j]), color='blue', ls='dotted', alpha=0.5) for i,j in contacts if i == shift_to_assign or j == shift_to_assign]
                plt.annotate(i, (predict_x[i]+0.02, predict_y[i]+0.02))

            if abs(predict_y[i] - actual_y[shift_to_assign]) <= 0.15:
                plt.scatter(predict_x[i], predict_y[i], color='blue')
                [plt.plot((predict_x[i], predict_x[j]), (predict_y[i], predict_y[j]), color='blue', ls='dotted', alpha=0.5) for i,j in contacts if i == shift_to_assign or j == shift_to_assign]
                plt.annotate(i, (predict_x[i]+0.02, predict_y[i]+0.02))

        plt.xlim([6, 10])
        plt.ylim([100, 135])

        plt.xlabel('H1')
        plt.ylabel('N15')
        plt.legend()

        plt.savefig("shifts_plot.png", dpi = 150)
        plt.close()

    # def new_plot(coordinates):

    #     fig = plt.figure(figsize=(5,5), layout='tight')
    #     ax = fig.add_subplot(111, projection='3d')
    #     x = []
    #     y = []
    #     z = []
    #     for i, coord in enumerate(coordinates):
    #             # x.append(coord.x)
    #             # y.append(coord.y)
    #             # z.append(coord.z)
    #             x = coord.x
    #             y = coord.y
    #             z = coord.z
            
    #             ax.scatter(x,y,z, s=40, label=f'shift {i}')

    #     ax.legend()

    #     plt.show()
    #     #plt.savefig("coord_plot.png", dpi = 150)

    # def close_contacts(coordinates, cutoff=0.37):
    #     """
    #     Calculates close contacts based on coordinates.
    #     """
    #     contacts = []

    #     for i, atom1 in enumerate(coordinates):
    #         for j, atom2 in enumerate(coordinates):
    #             if i != j:
    #                 dist = np.linalg.norm((np.array(atom1[:3]) - np.array(atom2[:3])))
    #                 if dist < cutoff:
    #                     contacts.append((i,j))
    #     return contacts