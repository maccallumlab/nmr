from typing import NamedTuple
import numpy as np
import argparse
import sys
from pathlib import Path

# Add parent directory to path to allow imports from nmr package
sys.path.insert(0, str(Path(__file__).parent.parent))

from nmr.nmr_gym.fake_data import FakeDataGenerator



def write_to_file(num_resid, coordinates, noes, protein=False):
    if protein:
        name = f"protein_r{num_resid}_noe{len(noes)}.pdb"
    else:
        name = f"coordinates_r{num_resid}_noe{len(noes)}.pdb"

    with open(name, "w") as f:
        for (index, coord) in zip(range(num_resid), coordinates):
            
            x = f"{coord[0]*10:.3f}"
            y = f"{coord[1]*10:.3f}"
            z = f"{coord[2]*10:.3f}"

            f.write(f"ATOM{index:>7}  N   A A{index:>4}     {x:>7} {y:>7} {z:>7}  1.00  0.00           N\n")


def parse_pdb(PDB):
    coords = []
    with open(f"{PDB}.pdb", "r") as f:
        for line in f:
            if line.startswith("TER"):
                break
            if line.startswith("ATOM"):
                 parts = line.split()
                 if parts[2] == "CA":
                    x = float(parts[6])/10
                    y = float(parts[7])/10
                    z = float(parts[8])/10
                    coords.append((x,y,z))

    return np.array(coords)



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('num_resid', type=int, help='Number of residues')
    parser.add_argument('pdb', type=int, help='PDB being used (any value) or not (zero)')
    args = parser.parse_args()

    num_resid = args.num_resid
    pdb = args.pdb

    fake_data = FakeDataGenerator(num_resid)

    coordinates, obs_chemical_shifts, pred_chemical_shifts, noes, connectivity = fake_data.generate_data_arrays(random_key=False)
    
    protein = False

    if pdb:
        protein = True
        coordinates = parse_pdb("")
        noes, pred_chemical_shifts = fake_data.create_noes(coordinates, obs_chemical_shifts)

    print(len(noes))
    write_to_file(num_resid, coordinates, noes, protein=protein)
