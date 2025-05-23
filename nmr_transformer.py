import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import NamedTuple, List, Optional, Tuple


class NMRInput(NamedTuple):
    # input array of observed chemical shifts
    # 2-dimensions: N_shift, H_shift
    # (n_peaks, 2)
    obs_chemical_shifts: torch.Tensor

    # input array of observed chemical shifts
    # 2-dimensions: N_shift, H_shift
    # (n_peaks, 2)
    pred_chemical_shifts: torch.Tensor

    # input array of observed noes
    # 3-dimensions: N_shift1, H_shift1, H_shift2
    # (n_noe, 3)
    obs_noes: Optional[torch.Tensor]

    # input array of close distances
    # 4-dimensions: N_shift1, H_shift1, H_shift2, H_shift2
    # note: each close distance should be listed twice, with the residue order swapped
    # (n_close, 4)
    close_distances: Optional[torch.Tensor]

    # input array of peaks assigned so far
    # 4-dimensions: N_obs_shift, H_obs_shift, N_pred_shift, H_pred_shift
    # (n_assigned, 4)
    # note: can set to None if no peaks have been assigned
    assigned_peaks: Optional[torch.Tensor]

    # which peak should be assigned?
    peak_to_assign: int


class NMRTransformer(nn.Module):
    """
    A transformer model for predicting the next peak to assign in an NMR spectrum.
    """

    def __init__(
        self,
        n_hidden: int = 128,
        n_heads: int = 4,
        n_layers: int = 4,
        dropout: float = 0.1,
        device=None,
    ):
        super(NMRTransformer, self).__init__()
        self.embedder = NMRInitialEmbedding(n_hidden, device=device)
        self.linearmapping = NMRInputMapping()
        encoder_layers = nn.TransformerEncoderLayer(
            d_model=n_hidden,
            dim_feedforward=n_hidden * n_heads,
            nhead=n_heads,
            dropout=dropout,
            device=device,
        )
        self.encoder = nn.TransformerEncoder(encoder_layers, num_layers=n_layers)
        self.layer_norm = nn.LayerNorm(n_hidden, device=device)###############

        self.policy_linear1 = nn.Linear(2 * n_hidden, 2 * n_hidden, device=device)
        self.policy_linear2 = nn.Linear(2 * n_hidden, 1, device=device)
        self.value_linear1 = nn.Linear(n_hidden, n_hidden, device=device)
        self.value_linear2 = nn.Linear(n_hidden, 1, device=device)

    def forward(self, nmr_inputs: List[NMRInput]) -> Tuple[torch.Tensor, torch.Tensor]:
        
        linear_values = self.linearmapping(nmr_inputs)
        embedded, mask = self.embedder(linear_values)
        
        # embedded, mask = self.embedder(nmr_inputs)  # (N, S, D), (N, S)
        embedded = embedded.transpose(
            0, 1
        )  # (S, N, D) for nn.Transformer compatibility

        # embedded_norm = self.layer_norm(embedded)

        x = self.encoder(
            embedded, src_key_padding_mask=mask, is_causal=False
        )  # (S, N, D)
        x = x.transpose(0, 1)  # (N, S, D)

        policies = []
        for i, nmr_input in enumerate(linear_values):
            n_res = len(nmr_input.pred_chemical_shifts)
            residue_embeddings = x[i, 1 : (n_res + 1), :]
            peak_to_assign = nmr_input.peak_to_assign
            peak_embedding = x[i, n_res + peak_to_assign + 1, :]
            policy = (torch.matmul(residue_embeddings, peak_embedding))
            policies.append(policy)

        embeddings = x[:, 0, :]  # Global tokens for each input in the batch
        values = self.value_linear2(F.relu(self.value_linear1(embeddings))).reshape(-1)

        return policies, values


class NMRInputMapping(nn.Module):
    def __init__(
        self,
        xmin_h: int = 6,
        xmax_h: int = 10,
        xmin_n: int = 100,
        xmax_n: int = 135
    ):
        super(NMRInputMapping, self).__init__()
        self.xmin_h = xmin_h
        self.xmax_h = xmax_h
        self.xmin_n = xmin_n
        self.xmax_n = xmax_n
    
    def linear_transform(self, x, xmin, xmax):
        value = 2*((x - xmin)/(xmax - xmin)) - 1
        return value

    def forward(self, nmr_inputs: List[NMRInput]): #-> Tuple[torch.Tensor, torch.Tensor]:

        linear_values = []
        for i, nmr_input in enumerate(nmr_inputs):
            pred_shift1 = self.linear_transform(nmr_input.pred_chemical_shifts[:,0], self.xmin_h, self.xmax_h)
            pred_shift2 = self.linear_transform(nmr_input.pred_chemical_shifts[:,1], self.xmin_n, self.xmax_n)
            pred_chemical_shifts = torch.stack((pred_shift1, pred_shift2), dim=-1)
            # print(pred_chemical_shifts.shape)

            obs_shift1 = self.linear_transform(nmr_input.obs_chemical_shifts[:,0], self.xmin_h, self.xmax_h)
            obs_shift2 = self.linear_transform(nmr_input.obs_chemical_shifts[:,1], self.xmin_n, self.xmax_n)
            obs_chemical_shifts = torch.stack((obs_shift1, obs_shift2), dim=-1)
            # print(obs_chemical_shifts.shape)

            if nmr_input.obs_noes is not None:
                noe1 = self.linear_transform(nmr_input.obs_noes[:,0], self.xmin_h, self.xmax_h)
                noe2 = self.linear_transform(nmr_input.obs_noes[:,1], self.xmin_n, self.xmax_n)
                noe3 = self.linear_transform(nmr_input.obs_noes[:,2], self.xmin_h, self.xmax_h)
                obs_noes = torch.stack((noe1, noe2, noe3), dim=-1)
            else:
                obs_noes = nmr_input.obs_noes
            # print(obs_noes.shape)

            if nmr_input.close_distances is not None:
                dist1 = self.linear_transform(nmr_input.close_distances[:,0], self.xmin_h, self.xmax_h)
                dist2 = self.linear_transform(nmr_input.close_distances[:,1], self.xmin_n, self.xmax_n)
                dist3 = self.linear_transform(nmr_input.close_distances[:,2], self.xmin_h, self.xmax_h)
                dist4 = self.linear_transform(nmr_input.close_distances[:,3], self.xmin_n, self.xmax_n)
                close_distances = torch.stack((dist1, dist2, dist3, dist4), dim=-1)
            else:
                close_distances = nmr_input.close_distances
            # print(close_distances.shape)

            if nmr_input.assigned_peaks is not None:
                peak1 = self.linear_transform(nmr_input.assigned_peaks[:,0], self.xmin_h, self.xmax_h)
                peak2 = self.linear_transform(nmr_input.assigned_peaks[:,1], self.xmin_n, self.xmax_n)
                peak3 = self.linear_transform(nmr_input.assigned_peaks[:,2], self.xmin_h, self.xmax_h)
                peak4 = self.linear_transform(nmr_input.assigned_peaks[:,3], self.xmin_n, self.xmax_n)
                assigned_peaks = torch.stack((peak1, peak2, peak3, peak4), dim=-1)
            else:
                assigned_peaks = nmr_input.assigned_peaks

            peak_to_assign = nmr_input.peak_to_assign
        
            linear_values.append(NMRInput(obs_chemical_shifts=obs_chemical_shifts,
            pred_chemical_shifts=pred_chemical_shifts,
            obs_noes=obs_noes,
            close_distances=close_distances,
            assigned_peaks=assigned_peaks,
            peak_to_assign=peak_to_assign))

        return linear_values



class NMRInitialEmbedding(nn.Module):
    def __init__(self, n_hidden: int = 128, device=None):
        super(NMRInitialEmbedding, self).__init__()
        self.device = device
        self.n_hidden = n_hidden
        self.linear_obs_shifts = nn.Linear(2, n_hidden, device=device)
        self.linear_pred_shifts = nn.Linear(2, n_hidden, device=device)
        self.linear_obs_noes = nn.Linear(3, n_hidden, device=device)
        self.linear_close_distances = nn.Linear(4, n_hidden, device=device)
        self.linear_assigned_peaks = nn.Linear(4, n_hidden, device=device)
        self.embeddings = nn.Embedding(6, n_hidden, device=device)

    def forward(self, nmr_inputs: List[NMRInput]):
        batch_embeddings = []
        lengths = []

        for nmr in nmr_inputs:
            parts = []

            global_token = self.embeddings(
                torch.tensor([5], device=nmr.pred_chemical_shifts.device)
            ).expand(1, -1)
            parts.append(global_token)

            x = self.linear_pred_shifts(nmr.pred_chemical_shifts)
            x += self.embeddings(torch.tensor([0], device=x.device))
            parts.append(x)

            x = self.linear_obs_shifts(nmr.obs_chemical_shifts)
            x += self.embeddings(torch.tensor([1], device=x.device))
            parts.append(x)

            if nmr.obs_noes is not None:
                x = self.linear_obs_noes(nmr.obs_noes)
                x += self.embeddings(torch.tensor([2], device=x.device))
                parts.append(x)

            if nmr.close_distances is not None:
                x = self.linear_close_distances(nmr.close_distances)
                x += self.embeddings(torch.tensor([3], device=x.device))
                parts.append(x)

            if nmr.assigned_peaks is not None:
                x = self.linear_assigned_peaks(nmr.assigned_peaks)
                x += self.embeddings(torch.tensor([4], device=x.device))
                parts.append(x)

            full_seq = torch.cat(parts, dim=0)
            batch_embeddings.append(full_seq)
            lengths.append(full_seq.size(0))

        max_len = max(lengths)
        padded = []
        padding_mask = []

        for emb in batch_embeddings:
            emb_len = emb.size(0)

            if emb_len < max_len:
                # Create a padding tensor to append
                pad_size = max_len - emb_len
                pad_tensor = torch.zeros(pad_size, self.n_hidden, device=emb.device)
                padded_emb = torch.cat([emb, pad_tensor], dim=0)

                # Create padding mask
                mask = torch.cat(
                    [
                        torch.zeros(emb_len, dtype=torch.bool, device=emb.device),
                        torch.ones(pad_size, dtype=torch.bool, device=emb.device),
                    ]
                )
            else:
                padded_emb = emb
                mask = torch.zeros(emb_len, dtype=torch.bool, device=emb.device)

            padded.append(padded_emb)
            padding_mask.append(mask)

        padded = torch.stack(padded)
        padding_mask = torch.stack(padding_mask)

        return padded, padding_mask
