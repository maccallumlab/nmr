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
        device = None
    ):
        super(NMRTransformer, self).__init__()
        self.embedder = NMRInitialEmbedding(n_hidden, device=device)
        encoder_layers = nn.TransformerEncoderLayer(
            d_model=n_hidden, nhead=n_heads, dropout=dropout, device=device
        )
        self.encoder = nn.TransformerEncoder(encoder_layers, num_layers=n_layers)
        self.policy_linear1 = nn.Linear(2 * n_hidden, 2 * n_hidden, device=device)
        self.policy_linear2 = nn.Linear(2 * n_hidden, 1, device=device)
        self.value_linear1 = nn.Linear(n_hidden, n_hidden, device=device)
        self.value_linear2 = nn.Linear(n_hidden, 1, device=device)

    def forward(self, nmr_inputs: List[NMRInput]) -> Tuple[torch.Tensor, torch.Tensor]:
        embedded, mask = self.embedder(nmr_inputs)  # (N, S, D), (N, S)
        embedded = embedded.transpose(
            0, 1
        )  # (S, N, D) for nn.Transformer compatibility

        x = self.encoder(embedded, src_key_padding_mask=mask, is_causal=False)  # (S, N, D)
        x = x.transpose(0, 1)  # (N, S, D)

        policies = []
        for i, nmr_input in enumerate(nmr_inputs):
            n_res = len(nmr_input.pred_chemical_shifts)
            residue_embeddings = x[i, 1 : (n_res + 1), :]
            peak_to_assign = nmr_input.peak_to_assign
            peak_embedding = x[i, n_res + peak_to_assign + 1, :].expand(n_res, -1)
            embeddings = torch.cat([residue_embeddings, peak_embedding], dim=1)
            policy = self.policy_linear2(
                F.relu(self.policy_linear1(embeddings))
            ).reshape(-1)
            policies.append(policy)

        embeddings = x[:, 0, :]  # Global tokens for each input in the batch
        values = self.value_linear2(F.relu(self.value_linear1(embeddings))).reshape(-1)

        return policies, values


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
