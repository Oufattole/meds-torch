import dataclasses

import torch
from omegaconf import DictConfig
from torch import nn
from transformers import AutoModel

from meds_torch.input_encoder import INPUT_ENCODER_MASK_KEY, INPUT_ENCODER_TOKENS_KEY
from meds_torch.input_encoder.triplet_encoder import CVE
from meds_torch.utils.module_class import Module


@dataclasses.dataclass
class ModelOutput:
    rep: torch.Tensor
    hidden_states: torch.Tensor = None


class AutoEmbedder(nn.Module):
    def __init__(self, cfg: DictConfig):
        super().__init__()
        self.cfg = cfg
        self.code_embedder = AutoModel.from_pretrained(self.cfg.code_embedder.pretrained_model)

    def forward(self, x, mask):
        batch_size, sequence_length, feature_dimension = x.shape
        x_reshaped = x.view(batch_size * sequence_length, feature_dimension)
        mask_reshaped = mask.view(batch_size * sequence_length, feature_dimension)
        outputs = self.code_embedder(input_ids=x_reshaped, attention_mask=mask_reshaped)
        # TODO(@oufattole) generalize to other hf models
        pooler_output = outputs["pooler_output"]
        pooler_output = pooler_output.view(batch_size, sequence_length, -1)

        return pooler_output


class TextCodeEncoder(nn.Module, Module):
    """TODO(teya): Add docstring."""

    def __init__(self, cfg: DictConfig):
        super().__init__()
        self.cfg = cfg
        self.date_embedder = CVE(cfg)
        self.code_embedder = AutoEmbedder(cfg)
        self.numerical_value_embedder = CVE(cfg)

    def embed_func(self, embedder, x):
        out = embedder.forward(x[None, :].transpose(2, 0)).permute(1, 2, 0)
        return out

    def get_embedding(self, batch):
        static_mask = batch["static_mask"]
        code_tokens = batch["code_tokens"]
        code_mask = batch["code_mask"]
        numerical_value = batch["numeric_value"]
        time_delta_days = batch["time_delta_days"]
        numerical_value_mask = batch["numerical_value_mask"]

        # Embed times and mask static value times
        time_emb = self.embed_func(self.date_embedder, time_delta_days) * ~static_mask.unsqueeze(dim=1)
        code_emb = self.code_embedder.forward(code_tokens, code_mask).permute(0, 2, 1)
        val_emb = self.embed_func(
            self.numerical_value_embedder, numerical_value
        ) * numerical_value_mask.unsqueeze(dim=1)

        # Sum the (time, code, value) triplets and
        embedding = time_emb + code_emb + val_emb

        assert embedding.isfinite().all(), "Embedding is not finite"
        return embedding

    def embed(self, batch):
        embedding = self.get_embedding(batch)
        batch[INPUT_ENCODER_MASK_KEY] = batch["mask"]
        batch[INPUT_ENCODER_TOKENS_KEY] = embedding
        return batch


class TextObservationEncoder(nn.Module, Module):
    """TODO(teya): Add docstring."""

    def __init__(self, cfg: DictConfig):
        super().__init__()
        self.cfg = cfg
        self.code_embedder = AutoEmbedder(cfg)

    def get_embedding(self, batch):
        observation_tokens = batch["observation_tokens"]
        observation_mask = batch["observation_mask"]

        embedding = self.code_embedder.forward(observation_tokens, observation_mask).permute(0, 2, 1)

        assert embedding.isfinite().all(), "Embedding is not finite"
        return embedding

    def embed(self, batch):
        embedding = self.get_embedding(batch)
        batch[INPUT_ENCODER_MASK_KEY] = batch["mask"]
        batch[INPUT_ENCODER_TOKENS_KEY] = embedding
        return batch
