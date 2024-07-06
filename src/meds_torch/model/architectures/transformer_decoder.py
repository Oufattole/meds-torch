# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the Llama 2 Community License Agreement.


import torch
from omegaconf import DictConfig
from torch import nn
from x_transformers import Decoder, TransformerWrapper


class TransformerDecoderModel(torch.nn.Module):
    """Wrapper of Decoder Transformer for use in MEDS with triplet token embeddings."""

    def __init__(self, cfg: DictConfig):
        super().__init__()
        dropout = cfg.model.params.dropout
        self.model = TransformerWrapper(
            num_tokens=cfg.model.embedder.token_dim,  # placeholder as this is not used
            max_seq_len=cfg.model.embedder.max_seq_len,
            emb_dropout=dropout,
            use_abs_pos_emb=False,
            attn_layers=Decoder(
                dim=cfg.model.embedder.token_dim,
                depth=cfg.model.params.n_layers,
                heads=cfg.model.params.nheads,
                layer_dropout=dropout,  # stochastic depth - dropout entire layer
                attn_dropout=dropout,  # dropout post-attention
                ff_dropout=dropout,  # feedforward dropout
            ),
        )
        self.model.token_emb = nn.Identity()
        self.get_last_token = cfg.model.get_last_token

        # Setup LM Heads
        self.value_head = nn.Linear(cfg.model.embedder.token_dim, 1, bias=False)
        self.code_head = nn.Linear(cfg.model.embedder.token_dim, cfg.model.embedder.vocab_size, bias=False)
        self.time_head = nn.Linear(cfg.model.embedder.token_dim, 1, bias=False)

    def forward(self, batch, mask):
        output = self.model(batch.transpose(1, 2), mask=mask)
        if self.get_last_token:
            mask_max = (~mask).max(dim=1)
            lengths = mask_max.indices
            lengths[~mask_max.values] = mask.shape[1]
            lengths -= 1

            # expand to match the shape of all_token_embeddings
            indices = lengths.view(-1, 1, 1).expand(-1, -1, output.shape[-1])
            last_token = torch.gather(output, dim=1, index=indices).squeeze(1)
            return last_token
        else:
            return output
