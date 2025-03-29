import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import NamedTuple, List, Optional


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


class NMRTransformer(torch.nn.Module):
    """
    A transformer model for predicting the next peak to assign in an NMR spectrum.
    """

    def __init__(
        self,
        n_hidden: int = 128,
        n_heads: int = 4,
        n_layers: int = 4,
        dropout: float = 0.1,
    ):
        super(NMRTransformer, self).__init__()
        self.embedder = NMRInitialEmbedding(n_hidden)
        encoder_modules = [
            TransformerEncoderLayer(d_model=n_hidden, nhead=n_heads, dropout=dropout)
            for _ in range(n_layers)
        ]
        self.encoder_layers = nn.Sequential(*encoder_modules)
        self.policy_linear1 = nn.Linear(2 * n_hidden, 2 * n_hidden)
        self.policy_linear2 = nn.Linear(2 * n_hidden, 1)
        self.value_linear1 = nn.Linear(n_hidden, n_hidden)
        self.value_linear2 = nn.Linear(n_hidden, 1)

    def forward(self, nmr_inputs: List[NMRInput]) -> torch.Tensor:
        embedded = self.embedder(nmr_inputs)
        x = self.encoder_layers(embedded)

        # compute the policy for each input in the batch
        policies = []
        for i in range(len(nmr_inputs)):
            n_res = len(nmr_inputs[i].pred_chemical_shifts)
            residue_embeddings = x[i][:n_res]
            peak_to_assign = nmr_inputs[i].peak_to_assign
            assert peak_to_assign >= 0 and peak_to_assign < len(
                nmr_inputs[i].obs_chemical_shifts
            )
            peak_embedding = x[i][n_res + peak_to_assign].expand(n_res, -1)
            embeddings = torch.cat([residue_embeddings, peak_embedding], dim=1)
            policy = self.policy_linear2(
                F.relu(self.policy_linear1(embeddings))
            ).reshape(-1)
            policies.append(policy)

        # compute the value for each input in the batch
        embeddings = []
        for i in range(len(nmr_inputs)):
            # the global token is always the last one
            embedding = x[i][-1]
            embeddings.append(embedding)
        embeddings = torch.stack(embeddings)
        values = self.value_linear2(F.relu(self.value_linear1(embeddings))).reshape(-1)

        return policies, values


class NMRInitialEmbedding(nn.Module):
    """
    Embeds the initial input for the NMR transformer.
    """

    def __init__(self, n_hidden: int = 128):
        super(NMRInitialEmbedding, self).__init__()
        self.linear_obs_shifts = nn.Linear(2, n_hidden)
        self.linear_pred_shifts = nn.Linear(2, n_hidden)
        self.linear_obs_noes = nn.Linear(3, n_hidden)
        self.linear_close_distances = nn.Linear(4, n_hidden)
        self.linear_assigned_peaks = nn.Linear(4, n_hidden)
        self.embeddings = nn.Embedding(6, n_hidden)

    def forward(self, nmr_inputs: List[NMRInput]) -> torch.Tensor:
        # embed each of the inputs
        embedded_obs_shifts = self._handle_input(
            [nmr.obs_chemical_shifts for nmr in nmr_inputs], self.linear_obs_shifts, 0
        )
        embedded_pred_shifts = self._handle_input(
            [nmr.pred_chemical_shifts for nmr in nmr_inputs], self.linear_pred_shifts, 1
        )
        embedded_obs_noes = self._handle_input(
            [nmr.obs_noes for nmr in nmr_inputs], self.linear_obs_noes, 2
        )
        embedded_close_distances = self._handle_input(
            [nmr.close_distances for nmr in nmr_inputs], self.linear_close_distances, 3
        )
        embedded_assigned_peaks = self._handle_input(
            [nmr.assigned_peaks for nmr in nmr_inputs], self.linear_assigned_peaks, 4
        )

        # collect the "global" token for each input
        embedded_global = []
        embed = self.embeddings(torch.tensor([5]))
        for _ in nmr_inputs:
            embedded_global.append(embed)

        # concatenate the two sets of embeddings
        concatendated = []
        variables_to_zip = [
            embedded_pred_shifts,
            embedded_obs_shifts,
            embedded_obs_noes,
            embedded_close_distances,
            embedded_assigned_peaks,
            embedded_global,
        ]
        for x1, x2, x3, x4, x5, x6 in zip(*variables_to_zip):
            concatendated.append(torch.cat([x1, x2, x3, x4, x5, x6], dim=0))

        return torch.nested.nested_tensor(concatendated, layout=torch.jagged)

    def _handle_input(self, xs, linear, embed_index):
        embedded = []
        for x in xs:
            if x is None:
                embedded.append(torch.Tensor())
            else:
                x = linear(x)
                x += self.embeddings(torch.tensor([embed_index]))
                embedded.append(x)
        return embedded


class TransformerEncoderLayer(nn.Module):
    """
    TransformerEncoderLayer is made up of self-attn and feedforward network
    with residual connections.
    """

    def __init__(
        self,
        d_model,
        nhead,
        dim_feedforward=2048,
        dropout=0.1,
        activation: nn.Module = torch.nn.functional.relu,
        layer_norm_eps=1e-5,
        norm_first=True,
        bias=True,
        device=None,
        dtype=None,
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.self_attn = MultiHeadAttention(
            d_model,
            d_model,
            d_model,
            d_model,
            nhead,
            dropout=dropout,
            bias=bias,
            **factory_kwargs,
        )
        self.linear1 = nn.Linear(d_model, dim_feedforward, bias=bias, **factory_kwargs)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model, bias=bias, **factory_kwargs)

        self.norm_first = norm_first
        self.norm1 = nn.LayerNorm(
            d_model, eps=layer_norm_eps, bias=bias, **factory_kwargs
        )
        self.norm2 = nn.LayerNorm(
            d_model, eps=layer_norm_eps, bias=bias, **factory_kwargs
        )

        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.activation = activation

    def _sa_block(self, x):
        x = self.self_attn(x, x, x)
        return self.dropout1(x)

    def _ff_block(self, x):
        x = self.linear2(self.dropout(self.activation(self.linear1(x))))
        return self.dropout2(x)

    def forward(self, src):
        """
        Arguments:
            src: (batch_size, seq_len, d_model)
        """
        x = src
        if self.norm_first:
            x = x + self._sa_block(self.norm1(x))
            x = x + self._ff_block(self.norm2(x))
        else:
            x = self.norm1(x + self._sa_block(x))
            x = self.norm2(x + self._ff_block(x))
        return x


class MultiHeadAttention(nn.Module):
    """
    Computes multi-head attention. Supports nested or padded tensors.

    Args:
        E_q (int): Size of embedding dim for query
        E_k (int): Size of embedding dim for key
        E_v (int): Size of embedding dim for value
        E_total (int): Total embedding dim of combined heads post input projection. Each head
            has dim E_total // nheads
        nheads (int): Number of heads
        dropout (float, optional): Dropout probability. Default: 0.0
        bias (bool, optional): Whether to add bias to input projection. Default: True
    """

    def __init__(
        self,
        E_q: int,
        E_k: int,
        E_v: int,
        E_total: int,
        nheads: int,
        dropout: float = 0.0,
        bias=True,
        device=None,
        dtype=None,
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.nheads = nheads
        self.dropout = dropout
        self.q_proj = nn.Linear(E_q, E_total, bias=bias, **factory_kwargs)
        self.k_proj = nn.Linear(E_k, E_total, bias=bias, **factory_kwargs)
        self.v_proj = nn.Linear(E_v, E_total, bias=bias, **factory_kwargs)
        E_out = E_q
        self.out_proj = nn.Linear(E_total, E_out, bias=bias, **factory_kwargs)
        assert E_total % nheads == 0, "Embedding dim is not divisible by nheads"
        self.E_head = E_total // nheads
        self.bias = bias

    def forward(
        self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor
    ) -> torch.Tensor:
        """
        Forward pass; runs the following process:
            1. Apply input projection
            2. Split heads and prepare for SDPA
            3. Run SDPA
            4. Apply output projection

        Args:
            query (torch.Tensor): query of shape (N, L_q, E_qk)
            key (torch.Tensor): key of shape (N, L_kv, E_qk)
            value (torch.Tensor): value of shape (N, L_kv, E_v)

        Returns:
            attn_output (torch.Tensor): output of shape (N, L_t, E_q)
        """
        # Step 1. Apply input projection
        query = self.q_proj(query)
        key = self.k_proj(key)
        value = self.v_proj(value)

        # Step 2. Split heads and prepare for SDPA
        # reshape query, key, value to separate by head
        # (N, L_t, E_total) -> (N, L_t, nheads, E_head) -> (N, nheads, L_t, E_head)
        query = query.unflatten(-1, [self.nheads, self.E_head]).transpose(1, 2)
        # (N, L_s, E_total) -> (N, L_s, nheads, E_head) -> (N, nheads, L_s, E_head)
        key = key.unflatten(-1, [self.nheads, self.E_head]).transpose(1, 2)
        # (N, L_s, E_total) -> (N, L_s, nheads, E_head) -> (N, nheads, L_s, E_head)
        value = value.unflatten(-1, [self.nheads, self.E_head]).transpose(1, 2)

        # Step 3. Run SDPA
        # (N, nheads, L_t, E_head)
        attn_output = F.scaled_dot_product_attention(
            query, key, value, dropout_p=self.dropout
        )
        # (N, nheads, L_t, E_head) -> (N, L_t, nheads, E_head) -> (N, L_t, E_total)
        attn_output = attn_output.transpose(1, 2).flatten(-2)

        # Step 4. Apply output projection
        # (N, L_t, E_total) -> (N, L_t, E_out)
        attn_output = self.out_proj(attn_output)

        return attn_output


if __name__ == "__main__":
    # stupid code to find simple errors
    nmr_inputs = [
        NMRInput(
            obs_chemical_shifts=torch.randn(2, 2),
            pred_chemical_shifts=torch.randn(2, 2),
            obs_noes=torch.randn(3, 3),
            close_distances=torch.randn(2, 4),
            assigned_peaks=torch.randn(2, 4),
            peak_to_assign=0,
        ),
        NMRInput(
            obs_chemical_shifts=torch.randn(5, 2),
            pred_chemical_shifts=torch.randn(5, 2),
            obs_noes=torch.randn(4, 3),
            close_distances=torch.randn(7, 4),
            assigned_peaks=None,
            peak_to_assign=0,
        ),
    ]
    trans = NMRTransformer()
    output = trans(nmr_inputs)
    print(output)
