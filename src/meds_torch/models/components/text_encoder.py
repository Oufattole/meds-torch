"""Text Encoder Model.

Uses transformer encoder to get contextualized representations of all tokens and we take the CLS token
representation as the embedding.
"""

import torch
from omegaconf import DictConfig

from transformers import BertModel, BertConfig
from meds_torch.utils.module_class import Module


class TextEncoderModel(torch.nn.Module, Module):
    """Wrapper of Text Encoder Transformer."""

    def __init__(self, cfg: DictConfig):
        super().__init__()
        print(cfg)
        config = BertConfig(
            vocab_size=cfg.num_tokens,
            hidden_size=cfg.logits_dim,
            num_hidden_layers=cfg.notes_num_layers,
            num_attention_heads=cfg.notes_num_heads,
            intermediate_size=3072,
            hidden_act="gelu",
            hidden_dropout_prob=cfg.notes_dropout,
            attention_probs_dropout_prob=cfg.notes_dropout,
            max_position_embeddings=cfg.notes_max_seq_len,
            type_vocab_size=2,
            initializer_range=0.02,
            layer_norm_eps=1e-12,
            position_embedding_type="absolute",
            use_cache=True,
            classifier_dropout=None,
            output_attentions=False,
        )
        self.model = BertModel(config)

    def forward(self, text_inputs):
        return self.model(**text_inputs).last_hidden_state[:, 0]
