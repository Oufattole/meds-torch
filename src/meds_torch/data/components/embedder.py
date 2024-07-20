import dataclasses
import enum

import polars as pl
import torch
from omegaconf import DictConfig, OmegaConf
from torch import nn

OmegaConf.register_new_resolver(
    "get_vocab_size",
    lambda code_metadata_fp: pl.scan_parquet(code_metadata_fp)
    .select("code/vocab_index")
    .max()
    .collect()
    .item()
    + 1,
)


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
        self.layer = nn.Linear(1, cfg.model.embedder.token_dim)

    def forward(self, x):
        return self.layer(x)


class TripletEmbedder(nn.Module):
    """Container module with an encoder, a recurrent or transformer module, and a decoder.

    Copied from: https://github.com/pytorch/examples/blob/main/word_language_model/model.py
    """

    def __init__(self, cfg: DictConfig):
        super().__init__()
        self.cfg = cfg
        # Define Triplet Embedders
        self.date_embedder = CVE(cfg)
        self.code_embedder = torch.nn.Embedding(
            cfg.model.embedder.vocab_size, embedding_dim=cfg.model.embedder.token_dim
        )
        self.numerical_value_embedder = CVE(cfg)

    def embed_func(self, embedder, x):
        out = embedder.forward(x[None, :].transpose(2, 0)).permute(1, 2, 0)
        return out

    def get_embedding(self, batch):
        static_mask = batch["static_mask"]
        code = batch["code"]
        numerical_value = batch["numerical_value"]
        time_delta_days = batch["time_delta_days"]
        numerical_value_mask = batch["numerical_value_mask"]

        # Embed times and mask static value times
        time_emb = self.embed_func(self.date_embedder, time_delta_days) * ~static_mask.unsqueeze(dim=1)
        # Embed codes
        code_emb = self.code_embedder.forward(code).permute(0, 2, 1)
        # Embed numerical values and mask nan values
        val_emb = self.embed_func(
            self.numerical_value_embedder, numerical_value
        ) * numerical_value_mask.unsqueeze(dim=1)

        # Sum the (time, code, value) triplets and
        embedding = time_emb + code_emb + val_emb

        assert embedding.isfinite().all(), "Embedding is not finite"
        return embedding

    def embed(self, batch):
        embedding = self.get_embedding(batch)
        mask = batch["mask"]
        return embedding, mask
