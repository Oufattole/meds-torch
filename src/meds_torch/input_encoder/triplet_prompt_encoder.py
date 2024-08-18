import dataclasses
import enum

import torch
from omegaconf import DictConfig
from torch import nn

from meds_torch.input_encoder import INPUT_ENCODER_MASK_KEY, INPUT_ENCODER_TOKENS_KEY
from meds_torch.utils.module_class import Module


@dataclasses.dataclass
class ModelOutput:
    rep: torch.Tensor
    hidden_states: torch.Tensor = None


class Triplet(enum.Enum):
    DATE = "date"
    VALUE = "value"
    VARIABLE = "variable"


def sequence_mask(lengths, maxlen, dtype=torch.bool):
    row_vector = torch.arange(0, maxlen, 1, device=lengths.device)
    matrix = torch.unsqueeze(lengths, dim=-1)
    mask = row_vector < matrix

    mask.type(dtype)
    return mask


class CVE(nn.Module):
    """Continuous Value Encoder (CVE) module.

    Assumes input is a single continuous value, and encodes it as an `output_dim` size embedding vector.
    """

    def __init__(self, cfg):
        super().__init__()
        self.layer = nn.Linear(1, cfg.token_dim)

    def forward(self, x):
        return self.layer(x)


class TripletPromptEncoder(nn.Module, Module):
    """Container module with an encoder, a recurrent or transformer module, and a decoder.

    Copied from: https://github.com/pytorch/examples/blob/main/word_language_model/model.py
    """

    def __init__(self, cfg: DictConfig):
        super().__init__()
        self.cfg = cfg
        # Define Triplet Embedders
        # TODO add to config
        self.code_embedder = torch.nn.Embedding(cfg.vocab_size + 2, embedding_dim=cfg.token_dim)
        self.numerical_value_embedder = CVE(cfg)

    def embed_func(self, embedder, x):
        out = embedder.forward(x[None, :].transpose(2, 0)).permute(1, 2, 0)
        return out

    def get_embedding(self, batch):
        code = batch["code"]
        numerical_value = batch["numeric_value"]
        numerical_value_mask = batch["numerical_value_mask"]

        # Embed codes
        code_emb = self.code_embedder.forward(code).permute(0, 2, 1)
        # Embed numerical values and mask nan values
        val_emb = self.embed_func(
            self.numerical_value_embedder, numerical_value
        ) * numerical_value_mask.unsqueeze(dim=1)

        # Sum the (time, code, value) triplets and
        embedding = code_emb + val_emb

        assert embedding.isfinite().all(), "Embedding is not finite"
        return embedding

    def forward(self, batch):
        embedding = self.get_embedding(batch)
        batch[INPUT_ENCODER_MASK_KEY] = batch["mask"]
        batch[INPUT_ENCODER_TOKENS_KEY] = embedding
        return batch
