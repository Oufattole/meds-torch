import dataclasses
import enum

import polars as pl
import torch
from omegaconf import DictConfig
from torch import nn
from transformers import AutoModel, AutoTokenizer

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


class TextCodeEmbedder(nn.Module, Module):
    def __init__(self, cfg: DictConfig):
        super().__init__()
        self.cfg = cfg
        self.code_to_tokens_map = self.build_code_to_tokens_map()
        self.code_embedder = AutoModel.from_pretrained(self.cfg.code_embedder)

        # TODO: add caching for common embeddings

    def build_code_to_tokens_map(self):
        code_metadata = pl.scan_parquet(self.cfg.code_metadata_fp).select(["code/vocab_index", "description"])
        tokenizer = AutoTokenizer.from_pretrained(self.cfg.code_tokenizer)
        tokenized_code_metadata = tokenizer(
            code_metadata.select(["description"]).collect().fill_null("").to_series().to_list(),
            **self.cfg.tokenizer_config,
        )
        codes = code_metadata.select(["code/vocab_index"]).collect().get_column("code/vocab_index").to_list()

        keys = tokenized_code_metadata.keys()
        code_to_tokens_map = {
            code: {key: tokenized_code_metadata[key][i] for key in keys} for i, code in enumerate(codes)
        }
        del code_metadata, tokenized_code_metadata, codes, tokenizer  # , attention_mask, tokens
        return code_to_tokens_map

    def forward(self, codes, mask):
        # mask codes
        masked_codes = codes[mask]

        unique_codes = masked_codes.unique()

        available_keys = list(self.code_to_tokens_map[unique_codes[0].item()].keys())
        embedder_inputs = {
            key: torch.stack([self.code_to_tokens_map[code.item()][key] for code in unique_codes]).to(
                codes.device
            )
            for key in available_keys
        }
        code_embeddings = self.code_embedder(**embedder_inputs)
        keys = code_embeddings.keys()
        code_to_embeddings = {
            code.item(): {key: code_embeddings[key][i] for key in keys} for i, code in enumerate(unique_codes)
        }

        mask_embedding = torch.zeros_like(
            code_to_embeddings[unique_codes[0].item()]["last_hidden_state"][:, 0]
        )

        embeddings = [
            torch.stack(
                [
                    code_to_embeddings[code.item()]["last_hidden_state"][:, 0]
                    if mask_row[i]
                    else mask_embedding
                    for i, code in enumerate(row)
                ]
            )
            for row, mask_row in zip(codes, mask)
        ]
        embeddings = torch.stack(embeddings).to(codes.device)
        return embeddings


class TextCodeEncoder(nn.Module, Module):
    """Container module with an encoder, a recurrent or transformer module, and a decoder.

    Copied from: https://github.com/pytorch/examples/blob/main/word_language_model/model.py
    """

    def __init__(self, cfg: DictConfig):
        super().__init__()
        self.cfg = cfg
        # Define Triplet Embedders
        self.date_embedder = CVE(cfg)
        self.code_embedder = TextCodeEmbedder(cfg)
        self.numeric_value_embedder = CVE(cfg)

    def embed_func(self, embedder, x):
        out = embedder.forward(x[None, :].transpose(2, 0)).permute(1, 2, 0)
        return out

    def get_embedding(self, batch):
        static_mask = batch["static_mask"]
        code = batch["code"]
        code_mask = batch["mask"]
        numeric_value = batch["numeric_value"]
        time_delta_days = batch["time_delta_days"]
        numeric_value_mask = batch["numeric_value_mask"]
        # Embed times and mask static value times
        time_emb = self.embed_func(self.date_embedder, time_delta_days) * ~static_mask.unsqueeze(dim=1)
        # Embed codes
        code_emb = self.code_embedder.forward(code, code_mask)
        code_emb = code_emb.permute(0, 2, 1)

        # Embed numerical values and mask nan values
        val_emb = self.embed_func(self.numeric_value_embedder, numeric_value) * numeric_value_mask.unsqueeze(
            dim=1
        )

        # Sum the (time, code, value) triplets and
        embedding = time_emb + code_emb + val_emb

        assert embedding.isfinite().all(), "Embedding is not finite"
        if embedding.shape[-1] > self.cfg.max_seq_len:
            raise ValueError(
                f"Triplet embedding length {embedding.shape[-1]} "
                "is greater than max_seq_len {self.cfg.max_seq_len}"
            )
        return embedding.transpose(1, 2)

    def forward(self, batch):
        embedding = self.get_embedding(batch)
        batch[INPUT_ENCODER_MASK_KEY] = batch["mask"]
        batch[INPUT_ENCODER_TOKENS_KEY] = embedding
        return batch
