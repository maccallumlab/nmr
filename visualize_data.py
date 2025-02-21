from typing import NamedTuple
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D

def get_data(actual_shifts, coords, connectivity, named_tuple_used=True):
    if named_tuple_used:
        # Actual
        actual_x = [shift.H1 for shift in actual_shifts]
        actual_y = [shift.N15 for shift in actual_shifts]
        # Predicted
        predict_x = [shift.H1 for shift in coords]
        predict_y = [shift.N15 for shift in coords]
        # Connectivity
        contacts = [(contact.atom1, contact.atom2) for contact in connectivity]

    else:
        # Actual
        actual_x = [shift[0] for shift in actual_shifts]
        actual_y = [shift[1] for shift in actual_shifts]
        # Predicted
        predict_x = [shift[3] for shift in coords]
        predict_y = [shift[4] for shift in coords]
    
    return actual_x, actual_y, predict_x, predict_y, contacts

# def new_plot(coords):

#     fig = plt.figure(figsize=(5,5), layout='tight')
#     ax = fig.add_subplot(111, projection='3d')
#     x = []
#     y = []
#     z = []
#     for i, coord in enumerate(coords):
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

# def close_contacts(coords, cutoff=0.37):
#     """
#     Calculates close contacts based on coordinates.
#     """
#     contacts = []

#     for i, atom1 in enumerate(coords):
#         for j, atom2 in enumerate(coords):
#             if i != j:
#                 dist = np.linalg.norm((np.array(atom1[:3]) - np.array(atom2[:3])))
#                 if dist < cutoff:
#                     contacts.append((i,j))
#     return contacts

def plot_shifts(actual_shifts, restraints, coords, connectivity, named_tuple_used=True):

    actual_x, actual_y, predict_x, predict_y, contacts = get_data(actual_shifts, coords, connectivity, named_tuple_used=named_tuple_used)

    plt.scatter(predict_x, predict_y, color='blue', label='Predicted shifts')

    plt.scatter(actual_x, actual_y, color='red', label='Measured shifts')

    # Label measured shifts by index and predicted shifts by associated coordinate
    for i in range(len(actual_shifts)):
        plt.annotate(i, (actual_x[i]+0.01, actual_y[i]+0.01))
        plt.annotate(i, (predict_x[i]+0.01, predict_y[i]+0.01))

    # Draw restraints
    for restraint in restraints:
        for j,k in restraint:
            plt.plot((actual_x[j], actual_x[k]), (actual_y[j], actual_y[k]), color='red')
    
    # Draw close contacts
    for i,j in contacts:
        plt.plot((predict_x[i], predict_x[j]), (predict_y[i], predict_y[j]), color='blue', ls='dotted', alpha=0.5)

    plt.xlabel('H1')
    plt.ylabel('N15')
    plt.legend()

    plt.savefig("shifts_plot.png", dpi = 150)
    plt.close()