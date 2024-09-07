import torch
from omegaconf import DictConfig
from torch import nn

from meds_torch.input_encoder import INPUT_ENCODER_MASK_KEY, INPUT_ENCODER_TOKENS_KEY
from meds_torch.models import BACKBONE_EMBEDDINGS_KEY, BACKBONE_TOKENS_KEY
from meds_torch.models.components.utils import get_last_token
from meds_torch.utils.module_class import Module


class TransformerDecoderModel(torch.nn.Module, Module):
    """Wrapper of Decoder Transformer for use in MEDS with triplet token embeddings."""

    def __init__(self, cfg: DictConfig):
        super().__init__()
        self.cfg = cfg
        self.model = cfg.model

        if cfg.token_emb:
            self.model.token_emb = cfg.token_emb

        # Setup LM Heads
        self.value_head = nn.Linear(cfg.token_dim, 1, bias=False)
        self.code_head = nn.Linear(cfg.token_dim, cfg.vocab_size, bias=False)
        self.time_head = nn.Linear(cfg.token_dim, 1, bias=False)

    def forward(self, batch):
        input_data, mask = batch[INPUT_ENCODER_TOKENS_KEY], batch[INPUT_ENCODER_MASK_KEY]
        output, embeddings = self.model(input_data, mask=mask, return_logits_and_embeddings=True)
        if self.cfg.get_last_token:
            embeddings = get_last_token(embeddings, ~mask)
        batch[BACKBONE_TOKENS_KEY] = output
        batch[BACKBONE_EMBEDDINGS_KEY] = embeddings
        return batch
