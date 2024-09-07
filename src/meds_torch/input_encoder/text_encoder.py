"""Converts EHR history to text and generates sequence of embedded tokens.

This module implements various encoder classes for embedding medical data, including times, code, and numeric
values. It allows the use of pre-trained text models.

The module includes classes for encoding triplets of (time, code, value), as well as text observations. These
encoders can be used as part of larger systems for tasks such as patient representation learning, medical
prediction, or healthcare analytics.

Classes:     AutoEmbedder: A wrapper for HuggingFace's AutoModel for embedding meds data.     TextCodeEncoder:
An encoder for (time, code, value) triplets in medical data.     TextObservationEncoder: An encoder for text
observations in medical data.
"""
import torch
from omegaconf import DictConfig
from torch import nn

from meds_torch.input_encoder import INPUT_ENCODER_MASK_KEY, INPUT_ENCODER_TOKENS_KEY
from meds_torch.input_encoder.triplet_encoder import CVE
from meds_torch.utils.module_class import Module


class AutoEmbedder(nn.Module, Module):
    """A wrapper class for HuggingFace's AutoModel to embed code sequences.

    This class initializes a pre-trained model specified in the configuration and provides a forward method to
    embed input sequences.

    Attributes:     cfg (DictConfig): Configuration object containing model parameters.     code_embedder
    (AutoModel): The pre-trained model for code embedding.
    """

    def __init__(self, code_embedder):
        super().__init__()
        self.code_embedder = code_embedder

    def forward(self, x, mask):
        """Embed the input sequences using the pre-trained model.

        Args:     x (torch.Tensor): Input tensor of shape (batch_size, sequence_length, feature_dimension).
        mask (torch.Tensor): Attention mask tensor of the same shape as x.

        Returns:     torch.Tensor: Embedded representation of shape (batch_size, sequence_length,
        embedding_dim).
        """
        batch_size, sequence_length, feature_dimension = x.shape
        x_reshaped = x.view(batch_size * sequence_length, feature_dimension)
        mask_reshaped = mask.view(batch_size * sequence_length, feature_dimension)

        outputs = self.code_embedder(x_reshaped, mask=mask_reshaped.to(bool))
        pooler_output = outputs.view(batch_size, sequence_length, -1)

        return pooler_output


class TextCodeEncoder(nn.Module, Module):
    """An encoder for processing triplets of (time, code, value) in medical data.

    This class combines embeddings for dates, codes, and numerical values to create a unified representation
    of medical events.

    Attributes:     cfg (DictConfig): Configuration object containing model parameters.     date_embedder
    (CVE): Encoder for date information.     code_embedder (AutoEmbedder): Encoder for medical codes.
    numeric_value_embedder (CVE): Encoder for numerical values.
    """

    def __init__(self, cfg: DictConfig):
        super().__init__()
        self.cfg = cfg
        self.date_embedder = CVE(cfg)
        self.code_embedder = cfg.auto_embedder
        self.code_head = nn.Linear(cfg.text_token_dim, cfg.token_dim)
        self.numeric_value_embedder = CVE(cfg)

    def embed_func(self, embedder, x):
        out = embedder.forward(x[None, :].transpose(2, 0)).permute(1, 2, 0)
        return out

    def get_embedding(self, batch):
        """Generate embeddings for the input batch of medical data.

        Args:     batch (dict): A dictionary containing input tensors with keys ["static_mask", "code_tokens",
        "code_mask", "numeric_value", "time_delta_days", "numeric_value_mask"].

        Returns:     torch.Tensor: Combined embedding tensor for the batch.
        """
        static_mask = batch["static_mask"]
        code_tokens = batch["code_tokens"]
        code_mask = batch["code_mask"]
        numeric_value = batch["numeric_value"]
        time_delta_days = batch["time_delta_days"]
        numeric_value_mask = batch["numeric_value_mask"]

        # Embed times and mask static value times
        time_emb = self.embed_func(self.date_embedder, time_delta_days) * ~static_mask.unsqueeze(dim=1)
        for param in self.code_embedder.code_embedder.parameters():
            param.requires_grad = False
        text_code_embedding = self.code_embedder.forward(code_tokens, code_mask)
        code_emb = torch.nan_to_num(self.code_head(text_code_embedding).permute(0, 2, 1))
        val_emb = self.embed_func(self.numeric_value_embedder, numeric_value) * numeric_value_mask.unsqueeze(
            dim=1
        )

        # Sum the (time, code, value) triplets and
        embedding = time_emb + code_emb + val_emb

        assert embedding.isfinite().all(), "Embedding is not finite"
        return embedding

    def forward(self, batch):
        """Embed the input batch and update it with the generated embeddings.

        Args:     batch (dict): A dictionary containing various input tensors.

        Returns:     dict: Updated batch dictionary with added embedding information.
        """
        embedding = self.get_embedding(batch)
        batch[INPUT_ENCODER_MASK_KEY] = batch["mask"]
        batch[INPUT_ENCODER_TOKENS_KEY] = embedding.transpose(1, 2)
        return batch
