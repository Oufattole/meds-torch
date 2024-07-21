"""Transformer Encoder Model.

Uses transformer encoder to get contextualized representations of all tokens and we take the CLS token
representation as the embedding.
"""

import torch
from omegaconf import DictConfig
from torch import nn

from meds_torch.model.architectures.utils import get_last_token


class LstmModel(torch.nn.Module):
    """Wrapper of Encoder Transformer for use in MEDS with triplet token embeddings."""

    def __init__(self, cfg: DictConfig):
        super().__init__()
        dropout = cfg.model.params.dropout
        self.model = nn.LSTM(
            cfg.model.embedder.token_dim,
            cfg.model.embedder.token_dim,
            num_layers=cfg.model.params.n_layers,
            batch_first=True,
            dropout=dropout,
        )
        self.get_last_token = cfg.model.get_last_token

    def forward(self, batch, mask=None):
        # pass tokens and attention mask to the lstm
        output = self.model(batch.transpose(1, 2))[0]
        # extract the representation token's embedding
        if self.get_last_token:
            if mask is None:
                return output[:, -1, :]
            else:
                return get_last_token(output, mask)
        else:
            return output
