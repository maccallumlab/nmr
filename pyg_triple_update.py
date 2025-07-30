import torch
import torch_geometric
from torch_geometric.data import Data

from torch_geometric.loader import DataLoader
from torch_geometric.nn import MessagePassing
from torch_geometric.data import HeteroData
import torch_geometric.transforms as T
import torch.nn as nn

# Set up data (random to test)
data = HeteroData()

# Spatial and non-spatial
data['NOE'].x = torch.tensor([[1, 2, 3], [4, 5, 6]], dtype=torch.float32)
data['NOE'].f = torch.zeros((2, 1), dtype=torch.float32)
data['RES'].x = torch.tensor([[7, 8, 9, 1, 2], [10, 11, 12, 7, 3]], dtype=torch.float32)
data['RES'].f = torch.zeros((2, 1), dtype=torch.float32)
data['SHIFT'].x = torch.tensor([[4, 5], [1, 2]], dtype=torch.float32)
data['SHIFT'].f = torch.zeros((2, 1), dtype=torch.float32)

# Triple
data['TRIPLE'].x = torch.tensor(())

# Edges
data['NOE', 'NOE_extract', 'TRIPLE'].edge_index = torch.tensor([[0, 1], [0, 1]])
data['RES', 'NH1_extract', 'TRIPLE'].edge_index = torch.tensor([[0, 1], [0, 1]])
data['RES', 'NH2_extract', 'TRIPLE'].edge_index = torch.tensor([[1, 0], [0, 1]])
data['SHIFT', 'NH1_extract', 'TRIPLE'].edge_index = torch.tensor([[0, 1], [0, 1]])
data['SHIFT', 'NH2_extract', 'TRIPLE'].edge_index = torch.tensor([[1, 0], [0, 1]])

data['TRIPLE', 'update', 'TRIPLE'].edge_index = torch.tensor([[0, 1], [0, 1]]) # self loop

data['TRIPLE', 'NOE_add', 'NOE'].edge_index = torch.tensor([[0, 1], [0, 1]])
data['TRIPLE', 'NH1_add', 'RES'].edge_index = torch.tensor([[0, 1], [0, 1]])
data['TRIPLE', 'NH2_add', 'RES'].edge_index = torch.tensor([[0, 1], [1, 0]])
data['TRIPLE', 'res1_add', 'RES'].edge_index = torch.tensor([[0, 1], [0, 1]])
data['TRIPLE', 'res2_add', 'RES'].edge_index = torch.tensor([[0, 1], [1, 0]])
data['TRIPLE', 'NH1_add', 'SHIFT'].edge_index = torch.tensor([[0, 1], [0, 1]])
data['TRIPLE', 'NH2_add', 'SHIFT'].edge_index = torch.tensor([[0, 1], [1, 0]])

class TripleIn():
    def __init__(self, data):
        self.data = data

    def grab_node(self, node_type, edge_type):
        """
        Source node to triple via indexing.
        """
        # Spatial features
        xi = self.data[node_type].x[self.data[edge_type].edge_index[0]]
        # Non-spatial features
        fi = self.data[node_type].f[self.data[edge_type].edge_index[0]]

        return xi, fi
    
    def construct_triple(self, edge_type1, edge_type2, edge_type3):
        """
        Constructs triple based on type:
            - residue, residue, NOE
            - residue, shift, NOE
            - shift, residue, NOE
            - shift, shift, NOE
        """
        x1, f1 = self.grab_node(node_type=edge_type1[0], edge_type=edge_type1)
        x2, f2 = self.grab_node(node_type=edge_type2[0], edge_type=edge_type2)
        x3, f3 = self.grab_node(node_type=edge_type3[0], edge_type=edge_type3)

        return x1, x2, x3, f1, f2, f3


class TripleUpdateResResNoe(MessagePassing):
    """
    Message passing class for triple self updates (only set up for residue, residue, NOE triple type).
    Residue, residue, NOE triple type is the only one that deviates in calculations (coordinate calculation used), 
    other three would have same base set up with some differences in indexing.
    Pairwise calculations would use the same functions but would have a different message passing scheme.
    
    Options --> if/else statements for triples because of similarity, different class for pairwise?
    """
    def __init__(self):
        super().__init__(aggr='add')
        
        # Input index organization - slices to keep proper tensor dimension
        self.NOE_N1 = slice(0,1)
        self.NOE_H1 = slice(1,2)
        self.NOE_H2 = slice(2,3)

        self.RES_XYZ = slice(0,3)
        self.RES_N = slice(3,4)
        self.RES_H = slice(4,5)

        self.SHIFT_N = slice(0,1)
        self.SHIFT_H = slice(1,2)

        self.hidden = 64

        self.mlp1 = nn.Sequential(
                    nn.Linear(9, self.hidden),
                    nn.ReLU(),
                    nn.Linear(self.hidden, 16))

    def calc_noe_difference(self, x1, x2, noe, N1, H1, H2):
        """
        Calculates shift difference between NOE and residue/measured shifts (direct/indirect only reverse option is available by index).
        """
        diff_N = (noe[:, self.NOE_N1] - x1[:, N1]) # N [n, 1]
        diff_H1 = (noe[:, self.NOE_H1] - x1[:, H1]) # H' [n, 1]
        diff_H2 = (noe[:, self.NOE_H2] - x2[:, H2]) # H" [n, 1]
        return diff_N, diff_H1, diff_H2

    def calc_shift_difference(self, x1, x2, N1, H1):
        """
        Calculates shift difference between residue/measured shifts.
        """
        diff_N = (x1[:, N1] - x2[:, N1]) # N [n, 1]
        diff_H = (x1[:, H1] - x2[:, H1]) # H [n, 1]
        return diff_N, diff_H

    def calc_res_distance(self, x1, x2):
        """
        Calculates relative distance between residues and this value squared for equivariant calculations. 
        """
        rel_dist = ((x1[:, self.RES_XYZ]) - (x2[:, self.RES_XYZ])) # [n, 3]
        dist2 = torch.norm(rel_dist, dim=-1, keepdim=True)**2 # [n, 1]
        return rel_dist, dist2
        
    def forward(self, x1, x2, x3, f1, f2, f3, edge_index):
        out = self.propagate(edge_index, x1=x1, x2=x2, x3=x3, f1=f1, f2=f2, f3=f3)
        return out

    def message(self, x1_j, x2_j, x3_j, f1_j, f2_j, f3_j):
        # Residue distances
        rel_dist, dist2 = self.calc_res_distance(x1_j, x2_j)

        # Differences relative to NOE shifts (N, H', H")
        diff1, diff2, diff3 = self.calc_noe_difference(x1_j, x2_j, x3_j, self.RES_N, self.RES_H, self.RES_H)

        # Shift differences
        diff4, diff5 = self.calc_shift_difference(x1_j, x2_j, self.RES_N, self.RES_H)

        # Input for MLP 
        # (N, H', H", N, H, dist2, features)
        mlp_in = (torch.cat((diff1, diff2, diff3, diff4, diff5, dist2, f1_j, f2_j, f3_j), dim=-1)) # [n, 9]

        # Output from MLP
        # Out should include values for each 'change' wanting to make
        # (N, H', H", N1, H1, N2, H2, dist1, dist2, features)
        mlp_out = self.mlp1(mlp_in) # [n, 16]

        # NOE deltas
        delta1x = (diff1 * mlp_out[:, 0:1]) # N [n, 1]
        delta2x = (diff2 * mlp_out[:, 1:2]) # H' [n, 1]
        delta3x = (diff3 * mlp_out[:, 2:3]) # H" [n, 1]

        # SHIFT deltas residue
        delta4x = (diff4 * mlp_out[:, 3:4]) # N1 [n, 1]
        delta5x = (diff5 * mlp_out[:, 4:5]) # H1 [n, 1]
        delta6x = (diff4 * mlp_out[:, 5:6]) # N2 [n, 1]
        delta7x = (diff5 * mlp_out[:, 6:7]) # H2 [n, 1]

        # DISTANCE deltas
        delta12x = (rel_dist * (mlp_out[:, 7:10])) # [n, 3]
        delta13x = (rel_dist * (mlp_out[:, 10:13])) # [n, 3]

        # FEATURE deltas
        delta1f = mlp_out[:, 13:14] # [n, 1]
        delta2f = mlp_out[:, 14:15] # [n, 1]
        delta3f = mlp_out[:, 15:] # [n, 1]

        return torch.cat((delta1x, delta2x, delta3x, delta4x, delta5x, delta6x, delta7x, delta12x, delta13x, delta1f, delta2f, delta3f), dim=-1) # [n, 16]

    def update(self, aggr_out):
        # res involved in multiple different triples --> need to add across these (in different message passing class)
        # output shapes: noe[n, 3], shift1[n, 2], shift2[n, 2], dist1[n, 3], dist2[n, 3], f1[n, 1], f2[n, 1], f3[n, 1]
        delta_noe, delta_shift1, delta_shift2, delta_dist1, delta_dist2, delta_f1, delta_f2, delta_f3 = aggr_out[:, 0:3], aggr_out[:, 3:5], aggr_out[:, 5:7], aggr_out[:, 7:10], aggr_out[:, 10:13], aggr_out[:, 13:14], aggr_out[:, 14:15], aggr_out[:, 15:]
        return delta_shift1, delta_shift2, delta_noe, delta_dist1, delta_dist2, delta_f1, delta_f2, delta_f3



class TripleMessagePass(MessagePassing):
    """
    Standard message passing class for outgoing triple messages.
    """
    def __init__(self):
        super().__init__(aggr='add')

    def forward(self, x_source, x_target, edge_index):
        # SIZE (n, m) (source, target)
        return self.propagate(edge_index=edge_index, x=(x_source, x_target), size=(x_source.size(0), x_target.size(0)))

    def message(self, x_j):
        return x_j

    def update(self, aggr_out, x):
        # non-spatial features can also be included in same call but requires more work if they're not being updated (don't want to do residue feature updates 2x - only call once on shifts or coordinates)
        # x_val, f_val = aggr_out[:, :len(x[1][1])], aggr_out[:, len(x[1][1]):]
        # outf = f_val + f[1]
        # outx = x_val + x[1]

        out = aggr_out + x[1]
        return out

    

class TripleOut():
    def __init__(self, data):
        self.data = data
        self.triple_messgage = TripleMessagePass()

    def update_data(self, x_source, f_source, edge_type, x_index, update_f=True):
        """
        Calls message passing and directly updates the heterodata object based on target.
        """
        source, edge, target = edge_type
        edge_index = self.data[edge_type].edge_index

        # Message passing/update for spatial features
        self.data[target].x[:, x_index] = self.triple_messgage(x_source, self.data[target].x[:, x_index], edge_index)
        
        # Message passing/update for non-spatial features if needed (don't want a duplicate update for residue features)
        if update_f:
            self.data[target].f = self.triple_messgage(f_source, self.data[target].f, edge_index)

        return self.data



class ProteinGNN():
    """
    Test class that:
        - construct the triple based on edge types (only res, res, NOE as example)
        - self updates
        - sends message back to original nodes (note number of reverse edges depends on triple type)

    Can make separate classes to construct each triple type and do following updates or generalize this as a base class?
    """
    def __init__(self, data):
        self.data = data
        self.triple_in = TripleIn(data)
        self.triple_self = TripleUpdateResResNoe() # update would need to match data coming in
        self.triple_out = TripleOut(data)

        # Original data object indexing
        self.NOE = slice(0,3)
        self.RES_XYZ = slice(0,3)
        self.RES_NH = slice(3,5)
        self.SHIFT_NH = slice(0,2)
    
    def forward(self):
        # Triple in
        x1, x2, x3, f1, f2, f3 = self.triple_in.construct_triple(('RES', 'NH1_extract', 'TRIPLE'), ('RES', 'NH2_extract', 'TRIPLE'), ('NOE', 'NOE_extract', 'TRIPLE'))
        
        # Self update
        delta_shift1, delta_shift2, delta_noe, delta_dist1, delta_dist2, deltaf1, deltaf2, deltaf3 = self.triple_self(x1, x2, x3, f1, f2, f3, self.data['TRIPLE', 'update', 'TRIPLE'].edge_index)
        
        # Triple out (typically 3 reverse edges - 5 if coordinates are present)
        self.data = self.triple_out.update_data(delta_shift1, deltaf1, ('TRIPLE', 'NH1_add', 'RES'), self.RES_NH)
        self.data = self.triple_out.update_data(delta_shift2, deltaf2, ('TRIPLE', 'NH2_add', 'RES'), self.RES_NH)
        self.data = self.triple_out.update_data(delta_noe, deltaf3, ('TRIPLE', 'NOE_add', 'NOE'), self.NOE)
        self.data = self.triple_out.update_data(delta_dist1, deltaf1, ('TRIPLE', 'res1_add', 'RES'), self.RES_XYZ, update_f=False)
        self.data = self.triple_out.update_data(delta_dist2, deltaf2, ('TRIPLE', 'res2_add', 'RES'), self.RES_XYZ, update_f=False)

        return self.data



if __name__ == "__main__":
    print(f"original RES: {data['RES']}")
    print(f"original NOE: {data['NOE']}")
    testgnn = ProteinGNN(data)
    data = testgnn.forward()
    print(f"updated RES: {data['RES']}")
    print(f"updated NOE: {data['NOE']}")

