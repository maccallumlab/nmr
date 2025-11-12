"""
Top-level NMR GNN network architecture.

Combines triple-based message passing with value and policy heads
to create the complete neural network for NMR assignment.
"""

import torch
import torch.nn as nn

from .triple import CalculationManager, TripleIn, TripleOut, TripleUpdate
from .heads import ValueCalc, PolicyCalc


class NMRLayer(nn.Module):
    """
    Single layer of NMR-specific message passing.

    Orchestrates the full triple update pipeline:
    1. Extract features from nodes into triples (TripleIn)
    2. Update triple representations via message passing (TripleUpdate)
    3. Propagate updated information back to nodes (TripleOut)

    Handles all four triple types:
    - TRIPLE0: (RES, RES, NOE)
    - TRIPLE1: (RES, SHIFT, NOE)
    - TRIPLE2: (SHIFT, RES, NOE)
    - TRIPLE3: (SHIFT, SHIFT, NOE)
    """

    def __init__(self, device):
        super(NMRLayer, self).__init__()
        self.triple_in = TripleIn()
        self.triple_self = TripleUpdate(device)
        self.triple_out = TripleOut()
        self.calc_manager = CalculationManager()

        self.NOE = slice(0, 3)
        self.RES_XYZ = slice(0, 3)
        self.RES_NH = slice(3, 5)
        self.SHIFT_NH = slice(0, 2)

    def update_noes(self, data, noe_delta, feature, i):
        # 'i' is for the triple type in all of these cases (TRIPLE0 --> RES RES NOE, etc.)
        # Should really simplify this to be one or the other from the very start (string --> number identification)
        data = self.triple_out.update_data(
            data, noe_delta, feature, (f"TRIPLE{i}", "NOE_add", "NOE"), self.NOE
        )
        return data

    def update_coordinates(self, data, dist_delta1, dist_delta2, feature1, feature2, i):
        data = self.triple_out.update_data(
            data,
            dist_delta1,
            feature1,
            (f"TRIPLE{i}", "res1_add", "RES"),
            self.RES_XYZ,
            update_f=False,
        )
        data = self.triple_out.update_data(
            data,
            dist_delta2,
            feature2,
            (f"TRIPLE{i}", "res2_add", "RES"),
            self.RES_XYZ,
            update_f=False,
        )
        return data

    def update_shifts(
        self, data, shift_delta1, shift_delta2, feature1, feature2, target1, target2, i
    ):
        # Selects target indexing for residue or shift node
        target_range1 = self.RES_NH if target1 == "RES" else self.SHIFT_NH
        target_range2 = self.RES_NH if target2 == "RES" else self.SHIFT_NH

        data = self.triple_out.update_data(
            data,
            shift_delta1,
            feature1,
            (f"TRIPLE{i}", "NH1_add", target1),
            target_range1,
        )
        data = self.triple_out.update_data(
            data,
            shift_delta2,
            feature2,
            (f"TRIPLE{i}", "NH2_add", target2),
            target_range2,
        )
        return data

    def do_updates(self, data, triple_type, i):
        # Triple in
        shift_x1, coord_x1, shift_x2, coord_x2, noe_x, shift_f1, shift_f2, noe_f = (
            self.triple_in.construct_triple(
                data,
                (triple_type[0], "NH1_extract", f"TRIPLE{i}"),
                (triple_type[1], "NH2_extract", f"TRIPLE{i}"),
                ("NOE", "NOE_extract", f"TRIPLE{i}"),
            )
        )

        # Self update and output
        if triple_type == ("RES", "RES", "NOE"):
            (
                delta_shift1,
                delta_shift2,
                delta_noe,
                delta_dist1,
                delta_dist2,
                deltaf1,
                deltaf2,
                deltaf3,
            ) = self.triple_self(
                shift_x1,
                coord_x1,
                shift_x2,
                coord_x2,
                noe_x,
                shift_f1,
                shift_f2,
                noe_f,
                data[f"TRIPLE{i}", "update", f"TRIPLE{i}"].edge_index,
            )
            data = self.update_noes(data, delta_noe, deltaf3, i)
            data = self.update_coordinates(
                data, delta_dist1, delta_dist2, deltaf1, deltaf2, i
            )
            data = self.update_shifts(
                data,
                delta_shift1,
                delta_shift2,
                deltaf1,
                deltaf2,
                triple_type[0],
                triple_type[1],
                i,
            )

        if triple_type in (
            ("SHIFT", "RES", "NOE"),
            ("RES", "SHIFT", "NOE"),
            ("SHIFT", "SHIFT", "NOE"),
        ):
            delta_shift1, delta_shift2, delta_noe, deltaf1, deltaf2, deltaf3 = (
                self.triple_self(
                    shift_x1,
                    coord_x1,
                    shift_x2,
                    coord_x2,
                    noe_x,
                    shift_f1,
                    shift_f2,
                    noe_f,
                    data[f"TRIPLE{i}", "update", f"TRIPLE{i}"].edge_index,
                )
            )
            data = self.update_noes(data, delta_noe, deltaf3, i)
            data = self.update_shifts(
                data,
                delta_shift1,
                delta_shift2,
                deltaf1,
                deltaf2,
                triple_type[0],
                triple_type[1],
                i,
            )
        return data

    def forward(self, data):
        data = self.do_updates(data, ("RES", "RES", "NOE"), 0)
        data = self.do_updates(data, ("RES", "SHIFT", "NOE"), 1)
        data = self.do_updates(data, ("SHIFT", "RES", "NOE"), 2)
        data = self.do_updates(data, ("SHIFT", "SHIFT", "NOE"), 3)
        return data


class NormalizeShifts(nn.Module):
    """
    Normalize chemical shifts to have a typical range between -1 and 1
    """

    def __init__(self, H_lower=6.0, H_upper=10.0, N_lower=100.0, N_upper=135.0):
        super().__init__()
        self.H_lower = H_lower
        self.H_upper = H_upper
        self.H_delta = H_upper - H_lower
        self.N_lower = N_lower
        self.N_upper = N_upper
        self.N_delta = N_upper - N_lower

    def forward(self, data):
        # split RES into coords and shifts
        res_x = data["RES"].x
        res_xyz = res_x[:, :3]
        res_H = res_x[:, 3].unsqueeze(-1)
        res_N = res_x[:, 4].unsqueeze(-1)
        # split SHIFTS into 1H and 15N
        shift_H = data["SHIFT"].x[:, 0].unsqueeze(-1)
        shift_N = data["SHIFT"].x[:, 1].unsqueeze(-1)
        # split NOEs into 1H and 15N
        noe_H1 = data["NOE"].x[:, 0].unsqueeze(-1)
        noe_N1 = data["NOE"].x[:, 1].unsqueeze(-1)
        noe_H2 = data["NOE"].x[:, 2].unsqueeze(-1)
        # transform all 1H
        res_H = self._transform_H(res_H)
        shift_H = self._transform_H(shift_H)
        noe_H1 = self._transform_H(noe_H1)
        noe_H2 = self._transform_H(noe_H2)
        # transofrm all 15N
        res_N = self._transform_N(res_N)
        shift_N = self._transform_N(shift_N)
        noe_N1 = self._transform_N(noe_N1)
        # reassemble tensors
        res_x = torch.cat([res_xyz, res_H, res_N], dim=-1)
        shift_x = torch.cat([shift_H, shift_N], dim=-1)
        noe_x = torch.cat([noe_H1, noe_N1, noe_H2], dim=-1)

        data["RES"].x = res_x
        data["SHIFT"].x = shift_x
        data["NOE"].x = noe_x
        return data

    def _transform_H(self, value):
        return 2 * (value - self.H_lower) / self.H_delta - 1

    def _transform_N(self, value):
        return 2 * (value - self.N_lower) / self.N_delta - 1


class NMRNet(nn.Module):
    """
    Complete NMR GNN model combining message passing with prediction heads.

    Stacks NMRLayer(s) for graph message passing, then uses ValueCalc and
    PolicyCalc heads to predict state value and action probabilities.
    """

    def __init__(self, device):
        super().__init__()
        self.value = ValueCalc(device)
        self.policy = PolicyCalc(device)

        self.normalize = NormalizeShifts()
        self.nmr = nn.Sequential(
            NMRLayer(device),
        )

    def forward(self, data):
        data = self.normalize(data)
        out_data = self.nmr(data)
        value = self.value.calc_value(out_data)
        policy = self.policy.calc_policy(out_data)
        return out_data, value, policy
