"""Transformer Encoder Model.

Uses transformer encoder to get contextualized representations of all tokens and we take the CLS token
representation as the embedding.
"""

import torch
from omegaconf import DictConfig
from torch import nn
from x_transformers import Encoder, TransformerWrapper

from meds_torch.utils.module_class import Module


class TransformerEncoderModel(torch.nn.Module, Module):
    """Wrapper of Encoder Transformer for use in MEDS with triplet token embeddings."""

    def __init__(self, cfg: DictConfig):
        super().__init__()
        dropout = cfg.dropout
        self.model = TransformerWrapper(
            num_tokens=cfg.token_dim,  # placeholder as this is not used
            max_seq_len=cfg.max_seq_len,
            emb_dropout=dropout,
            use_abs_pos_emb=False,
            attn_layers=Encoder(
                dim=cfg.token_dim,
                depth=cfg.n_layers,
                heads=cfg.nheads,
                layer_dropout=dropout,  # stochastic depth - dropout entire layer
                attn_dropout=dropout,  # dropout post-attention
                ff_dropout=dropout,  # feedforward dropout
            ),
        )
        self.model.token_emb = nn.Identity()
        self.rep_token = torch.nn.Parameter(torch.randn(1, 1, cfg.token_dim))

    def forward(self, batch, mask):
        # Add representation token to the beginning of the sequence
        repeated_rep_token = self.rep_token.repeat(batch.shape[0], 1, 1)
        batch = torch.column_stack((repeated_rep_token, batch.transpose(1, 2)))
        mask = torch.cat((torch.ones((2, 1), dtype=torch.bool), mask), dim=1)
        # pass tokens and attention mask to the transformer
        output = self.model(batch, mask=mask)
        # extract the representation token's embedding
        output = output[:, 0, :]
        return output
