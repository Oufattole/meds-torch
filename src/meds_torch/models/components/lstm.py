"""Transformer Encoder Model.

Uses transformer encoder to get contextualized representations of all tokens and we take the CLS token
representation as the embedding.
"""

import torch
from omegaconf import DictConfig
from torch import nn

from meds_torch.input_encoder import INPUT_ENCODER_MASK_KEY, INPUT_ENCODER_TOKENS_KEY
from meds_torch.models import BACKBONE_EMBEDDINGS_KEY, BACKBONE_TOKENS_KEY
from meds_torch.models.components.utils import get_last_token
from meds_torch.utils.module_class import Module


class LstmModel(torch.nn.Module, Module):
    """Wrapper of Encoder Transformer for use in MEDS with triplet token embeddings."""

    def __init__(self, cfg: DictConfig):
        super().__init__()
        dropout = cfg.dropout
        self.model = nn.LSTM(
            cfg.token_dim,
            cfg.token_dim,
            num_layers=cfg.n_layers,
            batch_first=True,
            dropout=dropout,
        )
        # TODO (oufattole): add support for other positional encodings
        self.get_last_token = cfg.get_last_token

    def forward(self, batch):
        input_data, mask = batch[INPUT_ENCODER_TOKENS_KEY], batch[INPUT_ENCODER_MASK_KEY]
        # pass tokens and attention mask to the lstm
        output = self.model(input_data.transpose(1, 2))[0]
        # extract the representation token's embedding
        batch[BACKBONE_TOKENS_KEY] = output
        batch[BACKBONE_EMBEDDINGS_KEY] = get_last_token(output, mask)
        return batch
