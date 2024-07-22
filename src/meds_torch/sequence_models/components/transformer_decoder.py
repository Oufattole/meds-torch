# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the Llama 2 Community License Agreement.


import torch
from omegaconf import DictConfig
from torch import nn
from x_transformers import Decoder, TransformerWrapper

from meds_torch.sequence_models.components.utils import get_last_token
from meds_torch.utils.module_class import Module


class TransformerDecoderModel(torch.nn.Module, Module):
    """Wrapper of Decoder Transformer for use in MEDS with triplet token embeddings."""

    def __init__(self, cfg: DictConfig):
        super().__init__()
        dropout = cfg.dropout
        self.model = TransformerWrapper(
            num_tokens=cfg.token_dim,  # placeholder as this is not used
            max_seq_len=cfg.max_seq_len,
            emb_dropout=dropout,
            use_abs_pos_emb=False,
            attn_layers=Decoder(
                dim=cfg.token_dim,
                depth=cfg.n_layers,
                heads=cfg.nheads,
                layer_dropout=dropout,  # stochastic depth - dropout entire layer
                attn_dropout=dropout,  # dropout post-attention
                ff_dropout=dropout,  # feedforward dropout
            ),
        )
        self.model.token_emb = nn.Identity()
        self.get_last_token = cfg.get_last_token

        # Setup LM Heads
        self.value_head = nn.Linear(cfg.token_dim, 1, bias=False)
        self.code_head = nn.Linear(cfg.token_dim, cfg.vocab_size, bias=False)
        self.time_head = nn.Linear(cfg.token_dim, 1, bias=False)

    def forward(self, batch, mask):
        output = self.model(batch.transpose(1, 2), mask=mask)
        if self.get_last_token:
            batch["REP"] = get_last_token(output, mask)
        else:
            batch["TOKENS"] = output
        return batch
