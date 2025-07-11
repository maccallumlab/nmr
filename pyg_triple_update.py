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
data['NOE', 'NOE_extract', 'TRIPLE'].edge_index = torch.tensor([[0], [0]])
data['RES', 'NH1_extract', 'TRIPLE'].edge_index = torch.tensor([[0], [0]])
data['RES', 'NH2_extract', 'TRIPLE'].edge_index = torch.tensor([[1], [0]])
data['SHIFT', 'NH1_extract', 'TRIPLE'].edge_index = torch.tensor([[0], [0]])
data['SHIFT', 'NH2_extract', 'TRIPLE'].edge_index = torch.tensor([[1], [0]])
data['TRIPLE', 'update', 'TRIPLE'].edge_index = torch.tensor([[0], [0]]) # self loop

# Generate reverse for message passing
data = T.ToUndirected()(data)

def triple_in(data, node, edge):
    """
    Source node to triple via indexing.
    """
    # Spatial features
    xi = data[node].x[data[edge].edge_index[0]]
    # Non-spatial features
    fi = data[node].f[data[edge].edge_index[0]]

    return xi, fi


class TripleUpdateResResNoe(MessagePassing):
    def __init__(self):
        super().__init__(aggr='add')
        
        self.NOE_N1 = 0
        self.NOE_H1 = 1
        self.NOE_H2 = 2

        self.RES_X = 0
        self.RES_Y = 1
        self.RES_Z = 2
        self.RES_N = 3
        self.RES_H = 4

        self.SHIFT_N = 0
        self.SHIFT_H = 1

        self.hidden = 64

        self.mlp1 = nn.Sequential(
                    nn.Linear(9, self.hidden),
                    nn.ReLU(),
                    nn.Linear(self.hidden, 9))
        self.mlp2 = nn.Sequential(
                    nn.Linear(3, self.hidden),
                    nn.ReLU(),
                    nn.Linear(self.hidden, 3))

    def calc_noe_difference(self, x1, x2, noe, N1, H1, H2):
        diff_N = (noe[:, self.NOE_N1] - x1[:, N1]).unsqueeze(1) # N
        diff_H1 = (noe[:, self.NOE_H1] - x1[:, H1]).unsqueeze(1) # H'
        diff_H2 = (noe[:, self.NOE_H2] - x2[:, H2]).unsqueeze(1) # H"
        return diff_N, diff_H1, diff_H2
    
    def calc_res_distance(self, x1, x2):
        rel_positions = ((x1[:, self.RES_X:self.RES_Z+1]) - (x2[:, self.RES_X:self.RES_Z+1])).squeeze(dim=-1)
        distances = torch.norm(rel_positions, dim=-1, keepdim=True)**2
        return rel_positions, distances
        
    def forward(self, x1, x2, x3, f1, f2, f3, edge_index):
        out = self.propagate(edge_index, x1=x1, x2=x2, x3=x3, f1=f1, f2=f2, f3=f3)
        return out

    def message(self, x1_i, x1_j, x2_j, x3_j, f1_j, f2_j, f3_j):
        # Residue distances
        rel_positions, distances = self.calc_res_distance(x1_j, x2_j)

        # Differences relative to NOE shifts (N, H', H")
        # SHIFT 1
        diff1, diff2, diff3 = self.calc_noe_difference(x1_j, x1_j, x3_j, self.RES_N, self.RES_H, self.RES_H)
        # SHIFT 2
        diff4, diff5, diff6 = self.calc_noe_difference(x2_j, x2_j, x3_j, self.RES_N, self.RES_H, self.RES_H)

        # Inputs for MLP 
        # (N, H', H", features)
        mlp_input1 = (torch.cat((diff1, diff2, diff3, diff4, diff5, diff6, f1_j, f2_j, f3_j), dim=-1))
        # (distances, features)
        mlp_input2 = (torch.cat((distances, f1_j, f2_j), dim=-1))

        # MLP
        mlp_out1 = self.mlp1(mlp_input1)
        mlp_out2 = self.mlp2(mlp_input2)

        # NOE deltas
        delta1x = (diff1 * mlp_out1[:, 0].unsqueeze(1)) + (diff4 * mlp_out1[:, 3].unsqueeze(1)) # N
        delta2x = (diff2 * mlp_out1[:, 1].unsqueeze(1)) + (diff5 * mlp_out1[:, 4].unsqueeze(1)) # H'
        delta3x = (diff3 * mlp_out1[:, 2].unsqueeze(1)) + (diff6 * mlp_out1[:, 5].unsqueeze(1)) # H"
        deltaf3 = mlp_out1[:, 8].unsqueeze(1) # features

        # SHIFT deltas
        # SHIFT1
        delta4x = -diff1 * mlp_out1[:, 0].unsqueeze(1) # N1
        delta5x = (-diff2 * mlp_out1[:, 1].unsqueeze(1)) + (-diff3 * mlp_out1[:, 2].unsqueeze(1)) # H1
        deltaf1 = mlp_out1[:, 6]
        # SHIFT2
        delta6x = -diff4 * mlp_out1[:, 3].unsqueeze(1) # N2
        delta7x = (-diff5 * mlp_out1[:, 4].unsqueeze(1)) + (-diff6 * mlp_out1[:, 5].unsqueeze(1)) # H2
        deltaf2 = mlp_out1[:, 7]

        # DISTANCE deltas?
        rel_positions = rel_positions
        delta8x = (rel_positions * (mlp_out2[:, 0]).unsqueeze(1))
        deltaf1 = (deltaf1 + mlp_out2[:, 1]).unsqueeze(1)
        deltaf2 = (deltaf2 + mlp_out2[:, 2]).unsqueeze(1)

        return (torch.cat((delta1x, delta2x, delta3x, delta4x, delta5x, delta6x, delta7x, delta8x, deltaf1, deltaf2, deltaf3), dim=-1))

    def update(self, aggr_out, x1, x2, x3, f1, f2, f3):
        update_NOE, update_x1, update_x2, update_f1, update_f2, update_f3 = aggr_out[:, 0:3], torch.cat((aggr_out[:, 7:10], aggr_out[:, 3:5]), dim=-1), torch.cat((aggr_out[:, 7:10]*-1, aggr_out[:, 5:7]), dim=-1), aggr_out[:, 10], aggr_out[:, 11], aggr_out[:, 12]
        return (x3 + update_NOE, 
        x1 + update_x1, 
        x2 + update_x2, 
        f1 + update_f1, 
        f2 + update_f2, 
        f3 + update_f3)

        # return aggr_out


if __name__ == "__main__":

    # Can grab these from data - would need some form of string input for each combination (node and edge)
    # data.node_types
    # data.edge_types
    node1, node2, node3 = 'RES', 'RES', 'NOE'
    edge1, edge2, edge3 = ('RES', 'NH1_extract', 'TRIPLE'), ('RES', 'NH2_extract', 'TRIPLE'), ('NOE', 'NOE_extract', 'TRIPLE')

    # Data --> Triple node
    x1, f1 = triple_in(data, node=node1, edge=edge1)
    x2, f2 = triple_in(data, node=node2, edge=edge2)
    x3, f3 = triple_in(data, node=node3, edge=edge3)

    gnn = TripleUpdateResResNoe()

    out = gnn(x1, x2, x3, f1, f2, f3, data['TRIPLE', 'update', 'TRIPLE'].edge_index)
    print(f'Update: {out}')
