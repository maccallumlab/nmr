import torch
import torch_geometric
from torch_geometric.data import Data

from torch_geometric.loader import DataLoader
from torch_geometric.nn import MessagePassing
from torch_geometric.data import HeteroData
import torch_geometric.transforms as T
import torch.nn as nn

from itertools import product

# Set up data
data = HeteroData()

# Spatial and non-spatial
data['NOE'].x = torch.tensor([[1, 2, 3]], dtype=torch.float32)
data['NOE'].f = torch.zeros((2, 1), dtype=torch.float32)
data['RES'].x = torch.tensor([[7, 8, 9, 1, 2], [10, 11, 12, 7, 3]], dtype=torch.float32)
data['RES'].f = torch.zeros((2, 1), dtype=torch.float32)
data['SHIFT'].x = torch.tensor([[4, 5], [1, 2]], dtype=torch.float32)
data['SHIFT'].f = torch.zeros((2, 1), dtype=torch.float32)

# Triples (nothing is ever stored in these - they are placeholders to be used in edge types)
data['TRIPLE'].x = torch.zeros((1, 1), dtype=torch.float32) # easier just to use separate triple for self updates
data['TRIPLE0'].x = torch.zeros((1, 1), dtype=torch.float32)
data['TRIPLE1'].x = torch.zeros((1, 1), dtype=torch.float32)
data['TRIPLE2'].x = torch.zeros((1, 1), dtype=torch.float32)
data['TRIPLE3'].x = torch.zeros((1, 1), dtype=torch.float32)

# # Edges
# data['NOE', 'NOE_extract', 'TRIPLE'].edge_index = torch.tensor([[0, 0], [0, 0]])
# data['RES', 'NH1_extract', 'TRIPLE'].edge_index = torch.tensor([[0, 1], [0, 0]])
# data['RES', 'NH2_extract', 'TRIPLE'].edge_index = torch.tensor([[1, 0], [0, 0]]) #[[1, 0], [0, 1]
# data['SHIFT', 'NH1_extract', 'TRIPLE'].edge_index = torch.tensor([[0, 1], [0, 0]])
# data['SHIFT', 'NH2_extract', 'TRIPLE'].edge_index = torch.tensor([[1, 0], [0, 0]])

# data['TRIPLE', 'update', 'TRIPLE'].edge_index = torch.tensor([[0], [0]]) # self loop

# data['TRIPLE', 'NOE_add', 'NOE'].edge_index = torch.tensor([[0, 0], [0, 0]])
# data['TRIPLE', 'NH1_add', 'RES'].edge_index = torch.tensor([[0, 0], [0, 1]])
# data['TRIPLE', 'NH2_add', 'RES'].edge_index = torch.tensor([[0, 0], [1, 0]])
# data['TRIPLE', 'res1_add', 'RES'].edge_index = torch.tensor([[0, 0], [0, 1]])
# data['TRIPLE', 'res2_add', 'RES'].edge_index = torch.tensor([[0, 0], [1, 0]])
# data['TRIPLE', 'NH1_add', 'SHIFT'].edge_index = torch.tensor([[0, 0], [0, 1]])
# data['TRIPLE', 'NH2_add', 'SHIFT'].edge_index = torch.tensor([[0, 0], [1, 0]])

class ConstructNodes():
    def __init__(self, data_history):
        self.data_history = data_history
    
    def construct_data_nodes(self, data):
        # will need to extract each component of data from data_history
        # data['NOE'].x
        # data['NOE'].f
        # data['SHIFT'].x
        # data['SHIFT'].f
        # data['RES'].x
        # data['RES'].f
        return data

    def construct_triple_nodes(self, data):
        # Triples (nothing is ever stored in these - they are placeholders to be used in edge types)
        data['TRIPLE'].x = torch.zeros((1, 1), dtype=torch.float32) # easier just to use separate triple for self updates
        data['TRIPLE0'].x = torch.zeros((1, 1), dtype=torch.float32)
        data['TRIPLE1'].x = torch.zeros((1, 1), dtype=torch.float32)
        data['TRIPLE2'].x = torch.zeros((1, 1), dtype=torch.float32)
        data['TRIPLE3'].x = torch.zeros((1, 1), dtype=torch.float32)
        return data
    
    def construct_data(self):
        data = HeteroData()
        self.construct_data_nodes(data)
        self.construct_triple_nodes(data)
        return data

class ConstructEdges():
    def __init__(self, data):
        self.data = data
        self.num_noe = len(data['NOE'].x)
        self.num_shift = len(data['SHIFT'].x)
        self.num_res = len(data['RES'].x)
    
    def get_triple_edges(self):
        """
        Grabs combinations from the three ranges.
        Orders these into source and target indices.
        """
        combo = list(product(range(self.num_noe), range(self.num_shift), range( self.num_res)))
        combo_tensor = torch.tensor(combo)
        # switches tensor dimension (columns set up for each edge)
        source_nodes = torch.transpose(combo_tensor, 0, 1)
        # repeats target edges for number of occurences in source (3 for the triple in this instance)
        target_nodes = torch.tensor(range(len(source_nodes[0]))).repeat(len(source_nodes), 1)
        return torch.stack([source_nodes, target_nodes], dim=0)
    
    def get_edges(self, triple_num, triple_type):
        source1, source2, source3 = triple_type

        edges_in = self.get_triple_edges()
        self.data['NOE', 'NOE_extract', f'TRIPLE{triple_num}'].edge_index = edges_in[:, 0]
        self.data[f'{source1}', 'NH1_extract', f'TRIPLE{triple_num}'].edge_index = edges_in[:, 1]
        self.data[f'{source2}', 'NH2_extract', f'TRIPLE{triple_num}'].edge_index = edges_in[:, 2]

        # reverse ordering for nodes going out
        edges_out = torch.stack([edges_in[1], edges_in[0]], dim=0)
        self.data[f'TRIPLE{triple_num}', 'NOE_add', 'NOE'].edge_index = edges_out[:, 0]
        self.data[f'TRIPLE{triple_num}', 'NH1_add', f'{source1}'].edge_index = edges_out[:, 1]
        self.data[f'TRIPLE{triple_num}', 'NH2_add', f'{source2}'].edge_index = edges_out[:, 2]

        # only residue nodes will have coordinate edges
        if source1 == 'RES':
            self.data[f'TRIPLE{triple_num}', 'res1_add', 'RES'].edge_index = edges_out[:, 1]
        if source2 == 'RES':
            self.data[f'TRIPLE{triple_num}', 'res2_add', 'RES'].edge_index = edges_out[:, 2]

        # self updates are the same across triple types - only need one
        if triple_num == 0:
            self.data['TRIPLE', 'update', 'TRIPLE'].edge_index = torch.stack([edges_in[1, 0], edges_in[1, 0]], dim=0) # self loop
        return self.data
    
    def generate_edge_indices(self):
        self.data = self.get_edges(0, ('RES', 'RES', 'NOE'))
        self.data = self.get_edges(1, ('RES', 'SHIFT', 'NOE'))
        self.data = self.get_edges(2, ('SHIFT', 'RES', 'NOE'))
        self.data = self.get_edges(3, ('SHIFT', 'SHIFT', 'NOE'))
        return self.data



class TripleIn():
    def __init__(self):
        # Input index organization - slices to keep proper tensor dimension
        self.RES_XYZ = slice(0,3)
        self.RES_NH = slice(3,5)

    def grab_node(self, data, node_type, edge_type):
        """
        Source node to triple via indexing.
        For residue data, extracts shift and coordinate information to process separately.
        """
        # Spatial features
        xi = data[node_type].x[data[edge_type].edge_index[0]]
        # Non-spatial features
        fi = data[node_type].f[data[edge_type].edge_index[0]]

        # Splits coordinate and shift features if combined
        xj = None
        if edge_type[0] == 'RES':
            xi, xj = xi[:, self.RES_NH], xi[:, self.RES_XYZ]
        return xi, xj, fi
    
    def construct_triple(self, data, edge_type1, edge_type2, edge_type3):
        """
        Constructs triple based on type.
        """
        # Measured shift or residue (shifts will return nonetype value)
        x1, x12, f1 = self.grab_node(data, node_type=edge_type1[0], edge_type=edge_type1)
        x2, x22, f2 = self.grab_node(data, node_type=edge_type2[0], edge_type=edge_type2)
        # NOE (does not require any value split)
        x3, _, f3 = self.grab_node(data, node_type=edge_type3[0], edge_type=edge_type3)
        return x1, x12, x2, x22, x3, f1, f2, f3



class CalculationManager():
    """
    Common calculations used across triple and pairwise comparisons.
    """
    def __init__(self):
        # Input index organization - slices to keep proper tensor dimension
        self.NOE_N1 = slice(0,1)
        self.NOE_H1 = slice(1,2)
        self.NOE_H2 = slice(2,3)
        self.RES_XYZ = slice(0,3)
        self.SHIFT_N = slice(0,1)
        self.SHIFT_H = slice(1,2)
    
    def calc_noe_difference(self, x1, x2, noe):
        """
        Calculates shift difference between NOE and residue/measured shifts (direct/indirect only reverse option is available by index).
        """
        diff_N = (noe[:, self.NOE_N1] - x1[:, self.SHIFT_N]) # N [n, 1]
        diff_H1 = (noe[:, self.NOE_H1] - x1[:, self.SHIFT_H]) # H' [n, 1]
        diff_H2 = (noe[:, self.NOE_H2] - x2[:, self.SHIFT_H]) # H" [n, 1]
        return diff_N, diff_H1, diff_H2

    def calc_shift_difference(self, x1, x2):
        """
        Calculates shift difference between residue/measured shifts.
        """
        diff_N = (x1[:, self.SHIFT_N] - x2[:, self.SHIFT_N]) # N [n, 1]
        diff_H = (x1[:, self.SHIFT_H] - x2[:, self.SHIFT_H]) # H [n, 1]
        return diff_N, diff_H

    def calc_res_distance(self, x1, x2):
        """
        Calculates relative distance between residues and this value squared for equivariant calculations. 
        """
        rel_dist = (x1 - x2) # [n, 3]
        dist2 = torch.norm(rel_dist, dim=-1, keepdim=True)**2 # [n, 1]
        return rel_dist, dist2    



class TripleUpdate(MessagePassing):
    """
    Message passing class for triple self updates.
    """
    def __init__(self):
        super().__init__(aggr='add')
        self.calc_manager = CalculationManager()

        self.hidden = 64

        self.mlp1 = nn.Sequential(
                    nn.Linear(9, self.hidden),
                    nn.ReLU(),
                    nn.Linear(self.hidden, 16))
        
        self.mlp2 = nn.Sequential(
                    nn.Linear(8, self.hidden),
                    nn.ReLU(),
                    nn.Linear(self.hidden, 10))
    
    def resresnoe_message(self, x1_j, x2_j, x3_j, f1_j, f2_j, f3_j, x12_j, x22_j):
        """
        Constructs message for (res, res, NOE) triple type.
        """
        # Residue distances
        rel_dist, dist2 = self.calc_manager.calc_res_distance(x12_j, x22_j)

        # Differences relative to NOE shifts (N, H', H")
        diff1, diff2, diff3 = self.calc_manager.calc_noe_difference(x1_j, x2_j, x3_j)

        # Shift differences
        diff4, diff5 = self.calc_manager.calc_shift_difference(x1_j, x2_j)
        
        # Input for MLP 
        # (N, H', H", N, H, dist2, features)
        mlp_in = (torch.cat((diff1, diff2, diff3, diff4, diff5, dist2, f1_j, f2_j, f3_j), dim=-1)) # [n, 9]

        # Output from MLP
        # Out should include values for each 'change' wanting to make
        # (N, H', H", N1, H1, N2, H2, dist1, dist2, features)
        mlp_out = self.mlp1(mlp_in) # [n, 16]

        # NOE deltas [n, 1]
        delta1x = (diff1 * mlp_out[:, 0:1]) # N
        delta2x = (diff2 * mlp_out[:, 1:2]) # H'
        delta3x = (diff3 * mlp_out[:, 2:3]) # H"

        # SHIFT deltas residue [n, 1]
        delta4x = (diff4 * mlp_out[:, 3:4]) # N1 
        delta5x = (diff5 * mlp_out[:, 4:5]) # H1 
        delta6x = (diff4 * mlp_out[:, 5:6]) # N2 
        delta7x = (diff5 * mlp_out[:, 6:7]) # H2

        # DISTANCE deltas [n, 3]
        delta12x = (rel_dist * (mlp_out[:, 7:10]))
        delta13x = (rel_dist * (mlp_out[:, 10:13]))

        # FEATURE deltas [n, 1]
        delta1f = mlp_out[:, 13:14]
        delta2f = mlp_out[:, 14:15] 
        delta3f = mlp_out[:, 15:]
        return torch.cat((delta1x, delta2x, delta3x, delta4x, delta5x, delta6x, delta7x, delta12x, delta13x, delta1f, delta2f, delta3f), dim=-1) # [n, 16]
    
    def shiftshiftnoe_message(self, x1_j, x2_j, x3_j, f1_j, f2_j, f3_j):
        """
        Constructs message for (shift, shift, NOE), (shift, res, NOE), (res, shift, NOE) triple types.
        """
        # Differences relative to NOE shifts (N, H', H")
        diff1, diff2, diff3 = self.calc_manager.calc_noe_difference(x1_j, x2_j, x3_j)

        # Shift differences
        diff4, diff5 = self.calc_manager.calc_shift_difference(x1_j, x2_j)
        
        # Input for MLP 
        # (N, H', H", N, H, dist2, features)
        mlp_in = (torch.cat((diff1, diff2, diff3, diff4, diff5, f1_j, f2_j, f3_j), dim=-1)) # [n, 8]

        # Output from MLP
        # Out should include values for each 'change' wanting to make
        # (N, H', H", N1, H1, N2, H2, features)
        mlp_out = self.mlp2(mlp_in) # [n, 10]

        # NOE deltas [n, 1]
        delta1x = (diff1 * mlp_out[:, 0:1]) # N
        delta2x = (diff2 * mlp_out[:, 1:2]) # H'
        delta3x = (diff3 * mlp_out[:, 2:3]) # H"

        # SHIFT deltas residue [n, 1]
        delta4x = (diff4 * mlp_out[:, 3:4]) # N1 
        delta5x = (diff5 * mlp_out[:, 4:5]) # H1 
        delta6x = (diff4 * mlp_out[:, 5:6]) # N2 
        delta7x = (diff5 * mlp_out[:, 6:7]) # H2

        # FEATURE deltas [n, 1]
        delta1f = mlp_out[:, 7:8]
        delta2f = mlp_out[:, 8:9] 
        delta3f = mlp_out[:, 9:]
        return torch.cat((delta1x, delta2x, delta3x, delta4x, delta5x, delta6x, delta7x, delta1f, delta2f, delta3f), dim=-1) # [n, 10]
    
    def forward(self, x1, x12, x2, x22, x3, f1, f2, f3, edge_index):
        out = self.propagate(edge_index, x1=x1, x2=x2, x3=x3, f1=f1, f2=f2, f3=f3, x12=x12, x22=x22) # not sure how to deal with size here
        return out

    def message(self, x1_j, x2_j, x3_j, f1_j, f2_j, f3_j, x12_j=None, x22_j=None):
        # residue based triple type will have two sets of coordinates (shifts will have nonetype)
        if x12_j != None and x22_j != None:
            return self.resresnoe_message(x1_j, x2_j, x3_j, f1_j, f2_j, f3_j, x12_j, x22_j)
        else:
            return self.shiftshiftnoe_message(x1_j, x2_j, x3_j, f1_j, f2_j, f3_j)

    def update(self, aggr_out, x12, x22):
        if x12 != None and x22 != None:
            # output shapes: noe[n, 3], shift1[n, 2], shift2[n, 2], dist1[n, 3], dist2[n, 3], f1[n, 1], f2[n, 1], f3[n, 1]
            delta_noe, delta_shift1, delta_shift2, delta_dist1, delta_dist2, delta_f1, delta_f2, delta_f3 = aggr_out[:, 0:3], aggr_out[:, 3:5], aggr_out[:, 5:7], aggr_out[:, 7:10], aggr_out[:, 10:13], aggr_out[:, 13:14], aggr_out[:, 14:15], aggr_out[:, 15:]
            return delta_shift1, delta_shift2, delta_noe, delta_dist1, delta_dist2, delta_f1, delta_f2, delta_f3
        else:
            # output shapes: noe[n, 3], shift1[n, 2], shift2[n, 2], f1[n, 1], f2[n, 1], f3[n, 1]
            delta_noe, delta_shift1, delta_shift2, delta_f1, delta_f2, delta_f3 = aggr_out[:, 0:3], aggr_out[:, 3:5], aggr_out[:, 5:7], aggr_out[:, 7:8], aggr_out[:, 8:9], aggr_out[:, 9:]
            return delta_shift1, delta_shift2, delta_noe, delta_f1, delta_f2, delta_f3



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
        # non-spatial features can also be included in same call but requires more work if they're not being updated 
        # (don't want to do residue feature updates 2x - only call once on shifts or coordinates)
        # x_val, f_val = aggr_out[:, :len(x[1][1])], aggr_out[:, len(x[1][1]):]
        # outf = f_val + f[1]
        # outx = x_val + x[1]
        out = aggr_out + x[1]
        return out

    

class TripleOut():
    def __init__(self):
        self.triple_messgage = TripleMessagePass()

    def update_data(self, data, x_source, f_source, edge_type, x_index, update_f=True):
        """
        Calls message passing and directly updates the heterodata object based on target.
        Set to update non-spatial features in all cases, but can be turned off case by case to prevent duplicate updates.
        """
        source, edge, target = edge_type
        edge_index = data[edge_type].edge_index

        # Message passing/update for spatial features
        data[target].x[:, x_index] = self.triple_messgage(x_source, data[target].x[:, x_index], edge_index)
        
        # Message passing/update for non-spatial features if needed (don't want a duplicate update for residue features)
        if update_f:
            data[target].f = self.triple_messgage(f_source, data[target].f, edge_index)
        return data



class ProteinGNN():
    def __init__(self):
        self.triple_in = TripleIn()
        self.triple_self = TripleUpdate()
        self.triple_out = TripleOut()

        self.NOE = slice(0,3)
        self.RES_XYZ = slice(0,3)
        self.RES_NH = slice(3,5)
        self.SHIFT_NH = slice(0,2)
        
    def update_noes(self, data, noe_delta, feature, i):
        data = self.triple_out.update_data(data, noe_delta, feature, (f'TRIPLE{i}', 'NOE_add', 'NOE'), self.NOE)
        return data

    def update_coordinates(self, data, dist_delta1, dist_delta2, feature1, feature2, i):
        data = self.triple_out.update_data(data, dist_delta1, feature1, (f'TRIPLE{i}', 'res1_add', 'RES'), self.RES_XYZ, update_f=False)
        data = self.triple_out.update_data(data, dist_delta2, feature2, (f'TRIPLE{i}', 'res2_add', 'RES'), self.RES_XYZ, update_f=False)
        return data

    def update_shifts(self, data, shift_delta1, shift_delta2, feature1, feature2, target1, target2, i):
        # selects target indexing for residue or shift node
        target_range1 = self.RES_NH if target1 == 'RES' else self.SHIFT_NH
        target_range2 = self.RES_NH if target2 == 'RES' else self.SHIFT_NH
        
        data = self.triple_out.update_data(data, shift_delta1, feature1, (f'TRIPLE{i}', 'NH1_add', target1), target_range1)
        data = self.triple_out.update_data(data, shift_delta2, feature2, (f'TRIPLE{i}', 'NH2_add', target2), target_range2)
        return data
        
    def do_updates(self, data, triple_type, i):
        # Triple in  
        shift_x1, coord_x1, shift_x2, coord_x2, noe_x, shift_f1, shift_f2, noe_f = self.triple_in.construct_triple(data, (triple_type[0], 'NH1_extract', f'TRIPLE{i}'), (triple_type[1], 'NH2_extract', f'TRIPLE{i}'), ('NOE', 'NOE_extract', f'TRIPLE{i}'))
        
        # Self update and output
        if triple_type == ('RES', 'RES', 'NOE'):
            delta_shift1, delta_shift2, delta_noe, delta_dist1, delta_dist2, deltaf1, deltaf2, deltaf3 = self.triple_self(shift_x1, coord_x1, shift_x2, coord_x2, noe_x, shift_f1, shift_f2, noe_f, data['TRIPLE', 'update', 'TRIPLE'].edge_index)
            data = self.update_noes(data, delta_noe, deltaf3, i)
            data = self.update_coordinates(data, delta_dist1, delta_dist2, deltaf1, deltaf2, i)
            data = self.update_shifts(data, delta_shift1, delta_shift2, deltaf1, deltaf2, triple_type[0], triple_type[1], i)

        if triple_type in (('SHIFT', 'RES', 'NOE'), ('RES', 'SHIFT', 'NOE'), ('SHIFT', 'SHIFT', 'NOE')):
            delta_shift1, delta_shift2, delta_noe, deltaf1, deltaf2, deltaf3 = self.triple_self(shift_x1, coord_x1, shift_x2, coord_x2, noe_x, shift_f1, shift_f2, noe_f, data['TRIPLE', 'update', 'TRIPLE'].edge_index)
            data = self.update_noes(data, delta_noe, deltaf3, i)
            data = self.update_shifts(data, delta_shift1, delta_shift2, deltaf1, deltaf2, triple_type[0], triple_type[1], i)
        return data

    def forward(self, data):
        data = self.do_updates(data, ('RES', 'RES', 'NOE'), 0)
        data = self.do_updates(data, ('RES', 'SHIFT', 'NOE'), 1)
        data = self.do_updates(data, ('SHIFT', 'RES', 'NOE'), 2)
        data = self.do_updates(data, ('SHIFT', 'SHIFT', 'NOE'), 3)
        return data


if __name__ == "__main__":
    edges = ConstructEdges(data)
    data = edges.generate_edge_indices()
    # testgnn = ProteinGNN()

    # data = testgnn.forward(data)
    # print(f"updated RES: {data['RES']}")
    # print(f"updated NOE: {data['NOE']}")
    # print(f"updated SHIFT: {data['SHIFT']}")

    graph_list = [data, data]
    data_loader = DataLoader(graph_list, batch_size=1, shuffle=False)
    
    unbatched_data_list = []
    testgnn = ProteinGNN()
    for batch in data_loader:
        test = testgnn.forward(batch)
        unbatched_data_list += test.to_data_list()

    # results = []
    # for subgraph in unbatched_data_list:
    #     results += torch.matmul(subgraph['RES'].x[:, 3:5], subgraph['SHIFT'].x.T)

    # print(results)