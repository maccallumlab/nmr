"""
Triple-based graph message passing components.

Contains classes for constructing, updating, and propagating messages
through triple nodes in the heterogeneous NMR graph.
"""

import torch
import torch.nn as nn
from torch_geometric.nn import MessagePassing


class CalculationManager:
    """
    Common calculations used across triple and pairwise comparisons.
    """

    def __init__(self):
        # Input index organization - slices to keep proper tensor dimension
        self.NOE_N1 = slice(0, 1)
        self.NOE_H1 = slice(1, 2)
        self.NOE_H2 = slice(2, 3)
        self.RES_XYZ = slice(0, 3)
        self.SHIFT_N = slice(0, 1)
        self.SHIFT_H = slice(1, 2)

    def calc_noe_difference(self, x1, x2, noe):
        """
        Calculates shift difference between NOE and residue/measured shifts (direct/indirect only reverse option is available by index).
        """
        diff_N = noe[:, self.NOE_N1] - x1[:, self.SHIFT_N]  # N [n, 1]
        diff_H1 = noe[:, self.NOE_H1] - x1[:, self.SHIFT_H]  # H' [n, 1]
        diff_H2 = noe[:, self.NOE_H2] - x2[:, self.SHIFT_H]  # H" [n, 1]
        return diff_N, diff_H1, diff_H2

    def calc_shift_difference(self, x1, x2):
        """
        Calculates shift difference between residue/measured shifts.
        """
        diff_N = x1[:, self.SHIFT_N] - x2[:, self.SHIFT_N]  # N [n, 1]
        diff_H = x1[:, self.SHIFT_H] - x2[:, self.SHIFT_H]  # H [n, 1]
        return diff_N, diff_H

    def calc_res_distance(self, x1, x2):
        """
        Calculates relative distance between residues and this value squared for equivariant calculations.
        """
        rel_dist = x1 - x2  # [n, 3]
        dist2 = torch.norm(rel_dist, dim=-1, keepdim=True) ** 2  # [n, 1]
        return rel_dist, dist2


class TripleIn(nn.Module):
    def __init__(self):
        super().__init__()
        # Input index organization - slices to keep proper tensor dimension
        self.RES_XYZ = slice(0, 3)
        self.RES_NH = slice(3, 5)

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
        if edge_type[0] == "RES":
            xi, xj = xi[:, self.RES_NH], xi[:, self.RES_XYZ]
        return xi, xj, fi

    def construct_triple(self, data, edge_type1, edge_type2, edge_type3):
        """
        Constructs triple based on type.
        """
        # Measured shift or residue (shifts will return nonetype value)
        x1, x12, f1 = self.grab_node(
            data, node_type=edge_type1[0], edge_type=edge_type1
        )
        x2, x22, f2 = self.grab_node(
            data, node_type=edge_type2[0], edge_type=edge_type2
        )
        # NOE (does not require any value split)
        x3, _, f3 = self.grab_node(data, node_type=edge_type3[0], edge_type=edge_type3)
        return x1, x12, x2, x22, x3, f1, f2, f3


class TripleUpdate(MessagePassing):
    """
    Message passing class for triple self updates.
    """

    def __init__(self, device):
        super().__init__(aggr="add")
        self.calc_manager = CalculationManager()
        self.device = device

        self.hidden = 64

        self.mlp1 = nn.Sequential(
            nn.Linear(12, self.hidden, device=self.device),
            nn.ReLU(),
            nn.Linear(self.hidden, 19, device=self.device),
            nn.LayerNorm(19, device=self.device),
        )

        self.mlp2 = nn.Sequential(
            nn.Linear(11, self.hidden, device=self.device),
            nn.ReLU(),
            nn.Linear(self.hidden, 13, device=self.device),
            nn.LayerNorm(13, device=self.device),
        )

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
        mlp_in = torch.cat(
            (diff1, diff2, diff3, diff4, diff5, dist2, f1_j, f2_j, f3_j), dim=-1
        )  # [n, 9]

        # Output from MLP
        # Out should include values for each 'change' wanting to make
        # (N, H', H", N1, H1, N2, H2, dist1, dist2, features)
        mlp_out = self.mlp1(mlp_in)  # [n, 16]

        # NOE deltas [n, 1]
        delta1x = diff1 * mlp_out[:, 0:1]  # N
        delta2x = diff2 * mlp_out[:, 1:2]  # H'
        delta3x = diff3 * mlp_out[:, 2:3]  # H"

        # SHIFT deltas residue [n, 1]
        delta4x = diff4 * mlp_out[:, 3:4]  # N1
        delta5x = diff5 * mlp_out[:, 4:5]  # H1
        delta6x = diff4 * mlp_out[:, 5:6]  # N2
        delta7x = diff5 * mlp_out[:, 6:7]  # H2

        # DISTANCE deltas [n, 3]
        delta12x = rel_dist * (mlp_out[:, 7:10])
        delta13x = rel_dist * (mlp_out[:, 10:13])

        # FEATURE deltas [n, 1]
        delta1f = mlp_out[:, 13:15]
        delta2f = mlp_out[:, 15:17]
        delta3f = mlp_out[:, 17:]
        return torch.cat(
            (
                delta1x,
                delta2x,
                delta3x,
                delta4x,
                delta5x,
                delta6x,
                delta7x,
                delta12x,
                delta13x,
                delta1f,
                delta2f,
                delta3f,
            ),
            dim=-1,
        )  # [n, 16] [n, 19]

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
        mlp_in = torch.cat(
            (diff1, diff2, diff3, diff4, diff5, f1_j, f2_j, f3_j), dim=-1
        )  # [n, 8]

        # Output from MLP
        # Out should include values for each 'change' wanting to make
        # (N, H', H", N1, H1, N2, H2, features)
        mlp_out = self.mlp2(mlp_in)  # [n, 10]

        # NOE deltas [n, 1]
        delta1x = diff1 * mlp_out[:, 0:1]  # N
        delta2x = diff2 * mlp_out[:, 1:2]  # H'
        delta3x = diff3 * mlp_out[:, 2:3]  # H"

        # SHIFT deltas residue [n, 1]
        delta4x = diff4 * mlp_out[:, 3:4]  # N1
        delta5x = diff5 * mlp_out[:, 4:5]  # H1
        delta6x = diff4 * mlp_out[:, 5:6]  # N2
        delta7x = diff5 * mlp_out[:, 6:7]  # H2

        # FEATURE deltas [n, 1]
        delta1f = mlp_out[:, 7:9]
        delta2f = mlp_out[:, 9:11]
        delta3f = mlp_out[:, 11:]
        return torch.cat(
            (
                delta1x,
                delta2x,
                delta3x,
                delta4x,
                delta5x,
                delta6x,
                delta7x,
                delta1f,
                delta2f,
                delta3f,
            ),
            dim=-1,
        )  # [n, 10] [n, 13]

    def forward(self, x1, x12, x2, x22, x3, f1, f2, f3, edge_index):
        out = self.propagate(
            edge_index, x1=x1, x2=x2, x3=x3, f1=f1, f2=f2, f3=f3, x12=x12, x22=x22
        )  # not sure how to deal with size here
        return out

    def message(self, x1_j, x2_j, x3_j, f1_j, f2_j, f3_j, x12_j=None, x22_j=None):
        # residue based triple type will have two sets of coordinates (x12_j and x22_j) where shifts will have nonetype
        if x12_j != None and x22_j != None:
            return self.resresnoe_message(
                x1_j, x2_j, x3_j, f1_j, f2_j, f3_j, x12_j, x22_j
            )
        else:
            return self.shiftshiftnoe_message(x1_j, x2_j, x3_j, f1_j, f2_j, f3_j)

    def update(self, aggr_out, x12, x22):
        # residue based triple type will have two sets of coordinates (x12_j and x22_j) where shifts will have nonetype
        if x12 != None and x22 != None:
            # output shapes: noe[n, 3], shift1[n, 2], shift2[n, 2], dist1[n, 3], dist2[n, 3], f1[n, 2], f2[n, 2], f3[n, 2]
            (
                delta_noe,
                delta_shift1,
                delta_shift2,
                delta_dist1,
                delta_dist2,
                delta_f1,
                delta_f2,
                delta_f3,
            ) = (
                aggr_out[:, 0:3],
                aggr_out[:, 3:5],
                aggr_out[:, 5:7],
                aggr_out[:, 7:10],
                aggr_out[:, 10:13],
                aggr_out[:, 13:15],
                aggr_out[:, 15:17],
                aggr_out[:, 17:],
            )
            return (
                delta_shift1,
                delta_shift2,
                delta_noe,
                delta_dist1,
                delta_dist2,
                delta_f1,
                delta_f2,
                delta_f3,
            )
        else:
            # output shapes: noe[n, 3], shift1[n, 2], shift2[n, 2], f1[n, 2], f2[n, 2], f3[n, 2]
            delta_noe, delta_shift1, delta_shift2, delta_f1, delta_f2, delta_f3 = (
                aggr_out[:, 0:3],
                aggr_out[:, 3:5],
                aggr_out[:, 5:7],
                aggr_out[:, 7:9],
                aggr_out[:, 9:11],
                aggr_out[:, 11:],
            )
            return delta_shift1, delta_shift2, delta_noe, delta_f1, delta_f2, delta_f3


class TripleMessagePass(MessagePassing):
    """
    Standard message passing class for outgoing triple messages.
    """

    def __init__(self, aggr):
        super().__init__(aggr=aggr)

    def forward(self, x_source, x_target, edge_index):
        # SIZE (n, m) (source, target)
        return self.propagate(
            edge_index=edge_index,
            x=(x_source, x_target),
            size=(x_source.size(0), x_target.size(0)),
        )

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


class TripleOut(nn.Module):
    def __init__(self):
        super().__init__()
        self.triple_message = TripleMessagePass(aggr="add")

    def update_data(self, data, x_source, f_source, edge_type, x_index, update_f=True):
        """
        Calls message passing and directly updates the heterodata object based on target.
        Set to update non-spatial features in all cases, but can be turned off case by case to prevent duplicate updates.
        """
        source, edge, target = edge_type
        edge_index = data[edge_type].edge_index

        # Message passing/update for spatial features
        data[target].x[:, x_index] = self.triple_message(
            x_source, data[target].x[:, x_index], edge_index
        )

        # Message passing/update for non-spatial features if needed (don't want a duplicate update for residue features)
        if update_f:
            data[target].f = self.triple_message(f_source, data[target].f, edge_index)
        return data
