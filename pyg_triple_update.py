import torch
import torch_geometric
from torch_geometric.data import Data

from torch_geometric.loader import DataLoader
from torch_geometric.nn import MessagePassing
from torch_geometric.data import HeteroData
import torch.nn as nn
import numpy as np

import matplotlib.pyplot as plt
import matplotlib.animation

from itertools import product
import pickle

# # Set up data
# data = HeteroData()

# # Spatial and non-spatial
# data['NOE'].x = torch.tensor([[1, 2, 3], [4, 5, 6], [7, 8 , 9]], dtype=torch.float32)
# data['NOE'].f = torch.zeros((3, 2), dtype=torch.float32)
# data['RES'].x = torch.tensor([[7, 8, 9, 1, 2], [10, 11, 12, 7, 3]], dtype=torch.float32)
# data['RES'].f = torch.zeros((2, 2), dtype=torch.float32)
# data['SHIFT'].x = torch.tensor([[4, 5], [1, 2]], dtype=torch.float32)
# data['SHIFT'].f = torch.zeros((2, 2), dtype=torch.float32)

# # Triples (nothing is ever stored in these - they are placeholders to be used in edge types)
# data['TRIPLE0'].x = torch.zeros((1, 1), dtype=torch.float32)
# data['TRIPLE1'].x = torch.zeros((1, 1), dtype=torch.float32)
# data['TRIPLE2'].x = torch.zeros((1, 1), dtype=torch.float32)
# data['TRIPLE3'].x = torch.zeros((1, 1), dtype=torch.float32)

# data['VALUE_NOE'].x = torch.zeros(1, 1, dtype=torch.float32) # has to be num graphs batched?
# data['VALUE_SHIFT'].x = torch.zeros(1, 1, dtype=torch.float32)
# data['VALUE_RES'].x = torch.zeros(1, 1, dtype=torch.float32)



class ConstructNodes():
    """
    Builds graph nodes from input
    """
    def __init__(self, device):
        self.device = device

    def construct_data_nodes(self, data, histories):
        data['NOE'].x = torch.tensor(histories['noes'], dtype=torch.float32, device=self.device)
        data['NOE'].f = torch.zeros((len(histories['noes']), 2), dtype=torch.float32, device=self.device)
        data['SHIFT'].x = torch.tensor(histories['obs_chemical_shifts'], dtype=torch.float32, device=self.device)
        data['SHIFT'].f = torch.zeros((len(histories['obs_chemical_shifts']), 2), dtype=torch.float32, device=self.device)
        data['RES'].x = torch.tensor(histories['coordinates'], dtype=torch.float32, device=self.device)
        data['RES'].f = torch.zeros((len(histories['coordinates']), 2), dtype=torch.float32, device=self.device)
        return data
    
    def construct_node_features(self, data, histories):
        for i in range(len(data['SHIFT'].f)):
            # is the shift being assigned right now?
            if i == histories['shift_to_assign']:
                data['SHIFT'].f[i, 0] = 1
            # has the shift already been assigned?
            if i in histories['assignments'].keys():
                data['SHIFT'].f[i, 1] = 1

        for i in range(len(data['RES'].f)):
            # has the residue been assigned?
            if i in histories['assignments'].values():
                data['RES'].f[i] = 1
        return data

    def construct_triple_nodes(self, data):
        """
        Generates triple node types. Nothing is ever stored in these - they are simply placeholders to construct/use in edge types.
        """
        data['TRIPLE0'].x = torch.zeros((1, 1), dtype=torch.float32, device=self.device)
        data['TRIPLE1'].x = torch.zeros((1, 1), dtype=torch.float32, device=self.device)
        data['TRIPLE2'].x = torch.zeros((1, 1), dtype=torch.float32, device=self.device)
        data['TRIPLE3'].x = torch.zeros((1, 1), dtype=torch.float32, device=self.device)
        return data
    
    def construct_value_nodes(self, data):
        data['VALUE_NOE'].x = torch.zeros(1, 1, dtype=torch.float32, device=self.device) # has to be num graphs batched?
        data['VALUE_SHIFT'].x = torch.zeros(1, 1, dtype=torch.float32, device=self.device)
        data['VALUE_RES'].x = torch.zeros(1, 1, dtype=torch.float32, device=self.device)
        return data
    
    def construct_data(self, histories):
        data = HeteroData()
        data = self.construct_data_nodes(data, histories)
        data = self.construct_triple_nodes(data)
        data = self.construct_value_nodes(data)
        data = self.construct_node_features(data, histories) # need to sort out all features first across types (consistent or not?)
        return data



class ConstructEdges():
    def __init__(self, data, device):
        self.data = data
        self.num_noe = len(data['NOE'].x)
        self.num_shift = len(data['SHIFT'].x)
        self.num_res = len(data['RES'].x)
        self.device = device
    
    def get_triple_edges(self, source1, source2):
        """
        Grabs all index combinations from the three ranges (residue, shift, noe) and orders these into source and target indices for the edges.
        Combinations occur along columns into a triple node.
        """
        combo = list(product(range(self.num_noe), range(source1), range(source2)))
        combo_tensor = torch.tensor(combo, device=self.device)

        # Switches tensor dimension for source (columns set up for each edge [0,0,0] --> [0],[0],[0])
        source_nodes = torch.transpose(combo_tensor, 0, 1)
        # Repeats target edges for number of occurences in source (3 for the triple in this instance, to be assigned to each incoming node type)
        target_nodes = torch.tensor(range(len(source_nodes[0])), device=self.device).repeat(len(source_nodes), 1)

        # print(torch.stack([source_nodes, target_nodes], dim=0))
        return torch.stack([source_nodes, target_nodes], dim=0)
    
    def get_pairwise_edges(self):
        """
        Grabs index combinations between shifts and residues and orders these into source and target indices for the edges.
        """
        # shift = torch.arange(0, self.num_shift)
        shift = np.nonzero(self.data['SHIFT'].f[:, 0])[0] # need to specify if it's the shift being assigned
        resid = torch.arange(0, self.num_res, device=self.device)
        
        shift_repeats = shift.repeat_interleave(self.num_res)
        # resid_repeats = resid.repeat(self.num_shift)
        resid_repeats = resid
        return torch.stack((resid_repeats, shift_repeats), dim=0)
    
    def get_batch_edges(self):
        shift = torch.arange(0, self.num_shift, device=self.device)
        noe = torch.arange(0, self.num_noe, device=self.device)
        resid = torch.arange(0, self.num_res, device=self.device)

        shift_repeats = torch.zeros(self.num_shift, device=self.device).long()
        noe_repeats = torch.zeros(self.num_noe, device=self.device).long()
        resid_repeats = torch.zeros(self.num_res, device=self.device).long()

        self.data['SHIFT', 'SHIFT_extract', f'VALUE_SHIFT'].edge_index = torch.stack((shift, shift_repeats), dim=0)
        self.data['NOE', 'NOE_extract', f'VALUE_NOE'].edge_index = torch.stack((noe, noe_repeats), dim=0)
        self.data['RES', 'RES_extract', f'VALUE_RES'].edge_index = torch.stack((resid, resid_repeats), dim=0)
        
        return self.data

    def get_edges(self, triple_num, triple_type):
        """
        Constructs all edges for the graph.
        """
        source1, source2, source3 = triple_type

        # Need to make sure I'm using the right node range (shift and residue number could vary)
        num_node1 = self.num_res if source1 == 'RES' else self.num_shift
        num_node2 = self.num_res if source2 == 'RES' else self.num_shift

        # Triple in
        edges_in = self.get_triple_edges(num_node1, num_node2)
        self.data[f'{source3}', 'NOE_extract', f'TRIPLE{triple_num}'].edge_index = edges_in[:, 0]
        self.data[f'{source1}', 'NH1_extract', f'TRIPLE{triple_num}'].edge_index = edges_in[:, 1]
        self.data[f'{source2}', 'NH2_extract', f'TRIPLE{triple_num}'].edge_index = edges_in[:, 2]

        # Triple out (reverse in/out ordering)
        edges_out = torch.stack([edges_in[1], edges_in[0]], dim=0)
        self.data[f'TRIPLE{triple_num}', 'NOE_add', f'{source3}'].edge_index = edges_out[:, 0]
        self.data[f'TRIPLE{triple_num}', 'NH1_add', f'{source1}'].edge_index = edges_out[:, 1]
        self.data[f'TRIPLE{triple_num}', 'NH2_add', f'{source2}'].edge_index = edges_out[:, 2]

        # Only residue nodes will have coordinate edges
        if source1 == 'RES':
            self.data[f'TRIPLE{triple_num}', 'res1_add', 'RES'].edge_index = edges_out[:, 1]
        if source2 == 'RES':
            self.data[f'TRIPLE{triple_num}', 'res2_add', 'RES'].edge_index = edges_out[:, 2]

        # Edges for self loop 
        self.data[f'TRIPLE{triple_num}', 'update', f'TRIPLE{triple_num}'].edge_index = torch.stack([edges_in[1, 0], edges_in[1, 0]], dim=0)

        if triple_num == 0:
            self.data['RES', 'pair', 'SHIFT'].edge_index = self.get_pairwise_edges() # pairwise edges - no message passing
        return self.data
    
    def generate_edge_indices(self):
        self.data = self.get_edges(0, ('RES', 'RES', 'NOE'))
        self.data = self.get_edges(1, ('RES', 'SHIFT', 'NOE'))
        self.data = self.get_edges(2, ('SHIFT', 'RES', 'NOE'))
        self.data = self.get_edges(3, ('SHIFT', 'SHIFT', 'NOE'))
        self.data = self.get_batch_edges()
        # print(self.data['RES', 'pair', 'SHIFT'].edge_index)
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
    def __init__(self, device):
        super().__init__(aggr='add')
        self.calc_manager = CalculationManager()
        self.device = device

        self.hidden = 64

        self.mlp1 = nn.Sequential(
                    nn.Linear(12, self.hidden, device=self.device),
                    nn.ReLU(),
                    nn.Linear(self.hidden, 19, device=self.device),
                    nn.LayerNorm(19, device=self.device))
        
        self.mlp2 = nn.Sequential(
                    nn.Linear(11, self.hidden, device=self.device),
                    nn.ReLU(),
                    nn.Linear(self.hidden, 13, device=self.device),
                    nn.LayerNorm(13, device=self.device))
    
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
        delta1f = mlp_out[:, 13:15]
        delta2f = mlp_out[:, 15:17] 
        delta3f = mlp_out[:, 17:]
        return torch.cat((delta1x, delta2x, delta3x, delta4x, delta5x, delta6x, delta7x, delta12x, delta13x, delta1f, delta2f, delta3f), dim=-1) # [n, 16] [n, 19]
    
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
        delta1f = mlp_out[:, 7:9]
        delta2f = mlp_out[:, 9:11] 
        delta3f = mlp_out[:, 11:]
        return torch.cat((delta1x, delta2x, delta3x, delta4x, delta5x, delta6x, delta7x, delta1f, delta2f, delta3f), dim=-1) # [n, 10] [n, 13]
    
    def forward(self, x1, x12, x2, x22, x3, f1, f2, f3, edge_index):
        out = self.propagate(edge_index, x1=x1, x2=x2, x3=x3, f1=f1, f2=f2, f3=f3, x12=x12, x22=x22) # not sure how to deal with size here
        return out

    def message(self, x1_j, x2_j, x3_j, f1_j, f2_j, f3_j, x12_j=None, x22_j=None):
        # residue based triple type will have two sets of coordinates (x12_j and x22_j) where shifts will have nonetype
        if x12_j != None and x22_j != None:
            return self.resresnoe_message(x1_j, x2_j, x3_j, f1_j, f2_j, f3_j, x12_j, x22_j)
        else:
            return self.shiftshiftnoe_message(x1_j, x2_j, x3_j, f1_j, f2_j, f3_j)

    def update(self, aggr_out, x12, x22):
        # residue based triple type will have two sets of coordinates (x12_j and x22_j) where shifts will have nonetype
        if x12 != None and x22 != None:
            # output shapes: noe[n, 3], shift1[n, 2], shift2[n, 2], dist1[n, 3], dist2[n, 3], f1[n, 2], f2[n, 2], f3[n, 2]
            delta_noe, delta_shift1, delta_shift2, delta_dist1, delta_dist2, delta_f1, delta_f2, delta_f3 = aggr_out[:, 0:3], aggr_out[:, 3:5], aggr_out[:, 5:7], aggr_out[:, 7:10], aggr_out[:, 10:13], aggr_out[:, 13:15], aggr_out[:, 15:17], aggr_out[:, 17:]
            return delta_shift1, delta_shift2, delta_noe, delta_dist1, delta_dist2, delta_f1, delta_f2, delta_f3
        else:
            # output shapes: noe[n, 3], shift1[n, 2], shift2[n, 2], f1[n, 2], f2[n, 2], f3[n, 2]
            delta_noe, delta_shift1, delta_shift2, delta_f1, delta_f2, delta_f3 = aggr_out[:, 0:3], aggr_out[:, 3:5], aggr_out[:, 5:7], aggr_out[:, 7:9], aggr_out[:, 9:11], aggr_out[:, 11:]
            return delta_shift1, delta_shift2, delta_noe, delta_f1, delta_f2, delta_f3



class TripleMessagePass(MessagePassing):
    """
    Standard message passing class for outgoing triple messages.
    """
    def __init__(self, aggr):
        super().__init__(aggr=aggr)

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



class BatchMessagePass(MessagePassing):
    def __init__(self, aggr, device):
        super().__init__(aggr=aggr)
        self.device = device

        self.noe_reduce = (nn.Linear(3, 1, device=self.device))
        self.shift_reduce = (nn.Linear(2, 1, device=self.device))
        self.res_reduce = (nn.Linear(5, 1, device=self.device))
        
    def forward(self, x_source, x_target, edge_index):
        # SIZE (n, m) (source, target)
        return self.propagate(edge_index=edge_index, x=(x_source, x_target), size=(x_source.size(0), x_target.size(0)))

    def message(self, x_j):
        if len(x_j[0]) == 3:
            return self.noe_reduce(x_j)
        if len(x_j[0]) == 2:
            return self.shift_reduce(x_j)
        if len(x_j[0]) == 5:
            return self.res_reduce(x_j)

    def update(self, aggr_out, x):
        return aggr_out

    

class TripleOut():
    def __init__(self):
        self.triple_message = TripleMessagePass(aggr='add')

    def update_data(self, data, x_source, f_source, edge_type, x_index, update_f=True):
        """
        Calls message passing and directly updates the heterodata object based on target.
        Set to update non-spatial features in all cases, but can be turned off case by case to prevent duplicate updates.
        """
        source, edge, target = edge_type
        edge_index = data[edge_type].edge_index

        # Message passing/update for spatial features
        data[target].x[:, x_index] = self.triple_message(x_source, data[target].x[:, x_index], edge_index)
        
        # Message passing/update for non-spatial features if needed (don't want a duplicate update for residue features)
        if update_f:
            data[target].f = self.triple_message(f_source, data[target].f, edge_index)
        return data

    

class NMRLayer(nn.Module):
    def __init__(self, device):
        super(NMRLayer, self).__init__()
        self.triple_in = TripleIn()
        self.triple_self = TripleUpdate(device)
        self.triple_out = TripleOut()
        self.calc_manager = CalculationManager()

        self.NOE = slice(0,3)
        self.RES_XYZ = slice(0,3)
        self.RES_NH = slice(3,5)
        self.SHIFT_NH = slice(0,2)
        
    def update_noes(self, data, noe_delta, feature, i):
        # 'i' is for the triple type in all of these cases (TRIPLE0 --> RES RES NOE, etc.)
        # Should really simplify this to be one or the other from the very start (string --> number identification)
        data = self.triple_out.update_data(data, noe_delta, feature, (f'TRIPLE{i}', 'NOE_add', 'NOE'), self.NOE)
        return data

    def update_coordinates(self, data, dist_delta1, dist_delta2, feature1, feature2, i):
        data = self.triple_out.update_data(data, dist_delta1, feature1, (f'TRIPLE{i}', 'res1_add', 'RES'), self.RES_XYZ, update_f=False)
        data = self.triple_out.update_data(data, dist_delta2, feature2, (f'TRIPLE{i}', 'res2_add', 'RES'), self.RES_XYZ, update_f=False)
        return data

    def update_shifts(self, data, shift_delta1, shift_delta2, feature1, feature2, target1, target2, i):
        # Selects target indexing for residue or shift node
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
            delta_shift1, delta_shift2, delta_noe, delta_dist1, delta_dist2, deltaf1, deltaf2, deltaf3 = self.triple_self(shift_x1, coord_x1, shift_x2, coord_x2, noe_x, shift_f1, shift_f2, noe_f, data[f'TRIPLE{i}', 'update', f'TRIPLE{i}'].edge_index)
            data = self.update_noes(data, delta_noe, deltaf3, i)
            data = self.update_coordinates(data, delta_dist1, delta_dist2, deltaf1, deltaf2, i)
            data = self.update_shifts(data, delta_shift1, delta_shift2, deltaf1, deltaf2, triple_type[0], triple_type[1], i)

        if triple_type in (('SHIFT', 'RES', 'NOE'), ('RES', 'SHIFT', 'NOE'), ('SHIFT', 'SHIFT', 'NOE')):
            delta_shift1, delta_shift2, delta_noe, deltaf1, deltaf2, deltaf3 = self.triple_self(shift_x1, coord_x1, shift_x2, coord_x2, noe_x, shift_f1, shift_f2, noe_f, data[f'TRIPLE{i}', 'update', f'TRIPLE{i}'].edge_index)
            data = self.update_noes(data, delta_noe, deltaf3, i)
            data = self.update_shifts(data, delta_shift1, delta_shift2, deltaf1, deltaf2, triple_type[0], triple_type[1], i)
        return data

    def forward(self, data):
        data = self.do_updates(data, ('RES', 'RES', 'NOE'), 0)
        data = self.do_updates(data, ('RES', 'SHIFT', 'NOE'), 1)
        data = self.do_updates(data, ('SHIFT', 'RES', 'NOE'), 2)
        data = self.do_updates(data, ('SHIFT', 'SHIFT', 'NOE'), 3)
        return data



class ValueCalc():
    def __init__(self, device):
        self.device = device

        self.batch_message = BatchMessagePass(aggr='mean', device=self.device)
        self.hidden = 64

        self.testmlp = nn.Sequential(
            nn.Linear(3, self.hidden, device=self.device),
            nn.ReLU(),
            nn.Linear(self.hidden, 1, device=self.device)
        )

    def get_aggr(self, x_source, x_target, edge_index):
        aggr = self.batch_message(x_source, x_target, edge_index)
        return aggr
    
    def calc_value(self, data):
        noe = self.get_aggr(data['NOE'].x, data['VALUE_NOE'].x, data['NOE', 'NOE_extract', 'VALUE_NOE'].edge_index)
        shift = self.get_aggr(data['SHIFT'].x, data['VALUE_SHIFT'].x, data['SHIFT', 'SHIFT_extract', 'VALUE_SHIFT'].edge_index)
        resid = self.get_aggr(data['RES'].x, data['VALUE_RES'].x, data['RES', 'RES_extract', 'VALUE_RES'].edge_index)
        
        concat_aggr = torch.cat((noe, shift, resid), dim=-1)
        value = self.testmlp(concat_aggr)
        return value



class PolicyCalc():
    def __init__(self, device):
        self.RES_NH = slice(3,5)
        self.device = device

    def pairwise_distance(self, data, edge_type):
        node_type1, edge, node_type2 = edge_type

        # residue 
        resid_nodes = data[node_type1].x[data[edge_type].edge_index[0]][:, self.RES_NH]
        # shift
        shift_nodes = data[node_type2].x[data[edge_type].edge_index[1]]

        rel_dist = (shift_nodes - resid_nodes)
        pair_dist2 = -(torch.norm(rel_dist, dim=-1, keepdim=True)**2)
        return pair_dist2.squeeze() # reduce dimension [num_res, 1] --> [num_res]
    
    def calc_policy(self, data):
        data_unbatched = data.to_data_list()
        policy = [self.pairwise_distance(batch, ('RES', 'pair', 'SHIFT')).tolist() for batch in data_unbatched]
        return torch.tensor(policy, dtype=torch.float32, requires_grad=True, device=self.device).unsqueeze(1) # add dimension back, now transposed [1, num_res]



class TestLayer(nn.Module):
    def __init__(self, device):
        super(TestLayer, self).__init__()
        self.value = ValueCalc(device)
        self.policy = PolicyCalc(device)
        
        self.nmr = nn.Sequential(
                    NMRLayer(device),
                    )

    def forward(self, data):
        out_data = self.nmr(data)
        value = self.value.calc_value(out_data)
        policy = self.policy.calc_policy(out_data)
        return value, policy
        


if __name__ == "__main__":

    def extract_data(pickle_file="fake_histories_r4_0.pkl"):
        with open(pickle_file, "rb") as f:
            histories = pickle.load(f)
        return histories

    def construct_graph(history, device):
        nodes = ConstructNodes(device)
        data = nodes.construct_data(history)
        edges = ConstructEdges(data, device)
        data = edges.generate_edge_indices()
        return data

    def preprocess_data(histories, batch_size, device):
        graphs = []
        for history in histories:
            graphs.append(construct_graph(history, device))

        data_loader = DataLoader(graphs, batch_size=batch_size, shuffle=False)
        return data_loader
    
    data_loader = preprocess_data(extract_data(), batch_size=1, device='cuda')
    
    plt.ion()
    fig, ax = plt.subplots()

    epochs = 5
 
    testgnn = TestLayer(device='cuda')
    for epoch in range(epochs):
        batch_num = 0
        for batch in data_loader:

            batch_num += 1

            value, policy = testgnn.forward(batch)
            
            ax.cla()
            ax.scatter(batch['SHIFT'].x[:, 0].tolist(), batch['SHIFT'].x[:, 1].tolist(), color='red')
            ax.scatter(batch['RES'].x[:, 3].tolist(), batch['RES'].x[:, 4].tolist(), color='blue')
            ax.set_title(f"Epoch {epoch} Batch {batch_num}")

            plt.pause(0.01)

    plt.ioff()
    plt.show()