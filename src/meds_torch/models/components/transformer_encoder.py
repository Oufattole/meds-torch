"""Transformer Encoder Model.

Uses transformer encoder to get contextualized representations of all tokens and we take the CLS token
representation as the embedding.
"""

import torch
from omegaconf import DictConfig
from torch import nn
from x_transformers import Encoder, TransformerWrapper

from meds_torch.input_encoder import INPUT_ENCODER_MASK_KEY, INPUT_ENCODER_TOKENS_KEY
from meds_torch.models import BACKBONE_EMBEDDINGS_KEY, BACKBONE_TOKENS_KEY
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
            use_abs_pos_emb=(cfg.pos_encoding == "absolute_sinusoidal"),
            attn_layers=Encoder(
                dim=cfg.token_dim,
                depth=cfg.n_layers,
                heads=cfg.nheads,
                layer_dropout=dropout,  # stochastic depth - dropout entire layer
                attn_dropout=dropout,  # dropout post-attention
                ff_dropout=dropout,  # feedforward dropout
            ),
        )
        if cfg.pos_encoding != "absolute_sinusoidal" and cfg.pos_encoding is not None:
            raise ValueError(f"Unknown positional encoding: {cfg.pos_encoding}")
        if not cfg.use_xtransformers_token_emb:
            self.model.token_emb = nn.Identity()
        self.rep_token = torch.nn.Parameter(torch.randn(1, 1, cfg.token_dim))

    def forward(self, batch):
        input_data, mask = batch[INPUT_ENCODER_TOKENS_KEY], batch[INPUT_ENCODER_MASK_KEY]
        # Add representation token to the beginning of the sequence
        repeated_rep_token = self.rep_token.repeat(input_data.shape[0], 1, 1)
        input_data = torch.column_stack((repeated_rep_token, input_data.transpose(1, 2)))
        mask = torch.cat((torch.ones((2, 1), dtype=torch.bool, device=mask.device), mask), dim=1)
        # pass tokens and attention mask to the transformer
        output = self.model(input_data, mask=mask)
        # extract the representation token's embedding
        rep = output[:, 0, :]
        batch[BACKBONE_TOKENS_KEY] = output
        batch[BACKBONE_EMBEDDINGS_KEY] = rep
        return batch
