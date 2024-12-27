from functools import wraps

import torch
from omegaconf import DictConfig

from meds_torch.input_encoder import INPUT_ENCODER_MASK_KEY, INPUT_ENCODER_TOKENS_KEY
from meds_torch.models import BACKBONE_EMBEDDINGS_KEY, BACKBONE_TOKENS_KEY
from meds_torch.models.components.utils import get_last_token
from meds_torch.utils.module_class import Module


def eval_decorator(fn):
    @wraps(fn)
    def inner(model, *args, **kwargs):
        was_training = model.training
        model.eval()
        out = fn(model, *args, **kwargs)
        model.train(was_training)
        return out

    return inner


class TransformerDecoderModel(torch.nn.Module, Module):
    """Wrapper of Decoder Transformer for use in MEDS with triplet token embeddings.

    This model handles both forward passes and generation with different budget types.

    Examples:
        >>> # Setup mock configuration and model components
        >>> import torch
        >>> from omegaconf import OmegaConf
        >>> from x_transformers import TransformerWrapper, Decoder
        >>> from hydra.utils import instantiate
        >>> from enum import Enum

        >>> # Mock configuration
        >>> B, S, L = 2, 5, 8  # batch_size, seq_len, dim
        >>> max_seq_len = 7
        >>> vocab_size = 4
        >>> cfg = instantiate({
        ...     'token_dim': L,
        ...     'vocab_size': vocab_size,
        ...     'max_seq_len': max_seq_len,
        ...     'get_last_token': True,
        ...     'temperature': 1.0,
        ...     'token_emb': None,
        ...     'max_tokens_budget': 10,
        ...     'model': {
        ...         '_target_': 'x_transformers.TransformerWrapper',
        ...         'num_tokens': vocab_size,
        ...         'max_seq_len': max_seq_len,
        ...         'use_abs_pos_emb': False,
        ...         'attn_layers': {
        ...             '_target_': 'x_transformers.Decoder',
        ...             'dim': L,
        ...             'depth': 2,
        ...             'heads': 2,
        ...             'rotary_pos_emb': True,
        ...         }
        ...     }
        ... })

        >>> # Initialize model
        >>> model = TransformerDecoderModel(cfg)
        >>> # Test basic forward pass
        >>> prompts = torch.randint(0, vocab_size, (B, S))
        >>> mask = torch.ones(B, S, dtype=torch.bool)
        >>> get_time = lambda x: torch.ones(B, dtype=torch.float)
        >>> get_value = lambda x: torch.ones(B, dtype=torch.float)
        >>> batch = {INPUT_ENCODER_TOKENS_KEY: torch.ones((B,4)),
        ...          INPUT_ENCODER_MASK_KEY: torch.ones((B,4), dtype=torch.bool)}
        >>> output = model.forward(batch)
        >>> output[BACKBONE_TOKENS_KEY].shape
        torch.Size([2, 4, 4])
        >>> output[BACKBONE_EMBEDDINGS_KEY].shape
        torch.Size([2, 8])
    """

    def __init__(self, cfg: DictConfig):
        super().__init__()
        self.cfg = cfg
        self.model = cfg.model

        if cfg.token_emb:
            self.model.token_emb = cfg.token_emb

    def forward(self, batch):
        input_data, mask = batch[INPUT_ENCODER_TOKENS_KEY], batch[INPUT_ENCODER_MASK_KEY]
        output, embeddings = self.model(input_data, mask=mask, return_logits_and_embeddings=True)
        if self.cfg.get_last_token:
            embeddings = get_last_token(embeddings, ~mask)
        batch[BACKBONE_TOKENS_KEY] = output
        batch[BACKBONE_EMBEDDINGS_KEY] = embeddings
        return batch
