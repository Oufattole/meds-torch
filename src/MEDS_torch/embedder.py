import dataclasses
import enum

import torch
from omegaconf import DictConfig
from torch import nn


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

    Assumes input is a single continuous value, and encodes it
    as an `output_dim` size embedding vector.
    """

    def __init__(self, cfg):
        super().__init__()
        self.layer = nn.Linear(1, cfg.embedder.token_dim)

    def forward(self, x):
        return self.layer(x)


class ObservationEmbedder(nn.Module):
    """Container module with an encoder, a recurrent or transformer module, and a decoder.

    Copied from: https://github.com/pytorch/examples/blob/main/word_language_model/model.py
    """

    def __init__(self, cfg: DictConfig):
        super().__init__()
        self.cfg = cfg
        # Define Triplet Embedders
        self.date_embedder = CVE(cfg)
        code_vocab_size = 1  # TODO: FIX THIS
        self.code_embedder = torch.nn.Embedding(code_vocab_size, embedding_dim=cfg.embedder.token_dim)
        self.numerical_value_embedder = CVE(cfg)

    def embed_func(self, embedder, x):
        out = embedder.forward(x[None, :].T).permute(1, 2, 0)
        return out

    def get_embedding(self, batch):
        event_mask = batch["event_mask"]
        dynamic_values_mask = batch["dynamic_values_mask"]
        time_delta_days = batch["time_delta_days"]
        dynamic_indices = batch["dynamic_indices"]
        dynamic_values = batch["dynamic_values"]
        static_indices = batch["static_indices"]
        static_values = batch["static_values"]

        time_emb = self.embed_func(self.date_embedder, timestamp)
        code_emb = self.code_embedder(code).transpose(1, 2)
        val_emb = self.embed_func(self.numerical_value_embedder, numerical_value).squeeze(dim=1)

        embedding = time_emb + code_emb + val_emb
        embedding *= mask

        assert embedding.isfinite().all(), "Embedding is not finite"

        return embedding

    def embed(self, batch):
        embedding = self.get_embedding(batch)
        mask = batch["mask"]
        return embedding, mask
