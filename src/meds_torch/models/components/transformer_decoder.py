import torch
from omegaconf import DictConfig
from torch import nn
from x_transformers import Decoder, TransformerWrapper

from meds_torch.input_encoder import INPUT_ENCODER_MASK_KEY, INPUT_ENCODER_TOKENS_KEY
from meds_torch.models import BACKBONE_EMBEDDINGS_KEY, BACKBONE_TOKENS_KEY
from meds_torch.models.components.utils import get_last_token
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
            use_abs_pos_emb=(cfg.pos_encoding == "absolute_sinusoidal"),
            attn_layers=Decoder(
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

        # Setup LM Heads
        self.value_head = nn.Linear(cfg.token_dim, 1, bias=False)
        self.code_head = nn.Linear(cfg.token_dim, cfg.vocab_size, bias=False)
        self.time_head = nn.Linear(cfg.token_dim, 1, bias=False)

    def forward(self, batch):
        input_data, mask = batch[INPUT_ENCODER_TOKENS_KEY], batch[INPUT_ENCODER_MASK_KEY]
        output = self.model(input_data.transpose(1, 2), mask=mask)
        batch[BACKBONE_TOKENS_KEY] = output
        batch[BACKBONE_EMBEDDINGS_KEY] = get_last_token(output, mask)
        return batch
