import polars as pl
import torch
from torch import nn

from meds_torch.input_encoder import INPUT_ENCODER_MASK_KEY, INPUT_ENCODER_TOKENS_KEY
from meds_torch.utils.module_class import Module


def get_dummy_batch_and_cfg(num_samples: int = 3):
    class DummyConfig:
        vocab_size = 7
        token_dim = 3

    cfg = DummyConfig()
    histogram_values = [
        [0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
        [0.0, 1.0, 1.0, 0.0, 1.0, 1.0, 0.0],
        [0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
        [1.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0],
        [1.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
        [0.0, 1.0, 1.0, 0.0, 1.0, 1.0, 0.0],
        [0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
        [0.0, 1.0, 0.0, 1.0, 1.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
    ]
    histogram_tensor = torch.stack([torch.tensor(histogram_values)] * num_samples)
    # torch.zeros((num_samples, 21, 7))

    data_dict = {
        "code": torch.tensor([[4, 5, 1, 2, 4, 5, 3, 0, 4, 5, 1, 2, 4, 5, 1, 3, 4, 5, 2, 3, 4]] * num_samples),
        "mask": torch.ones((num_samples, 21), dtype=torch.int64),
        "histogram": histogram_tensor,
        "subject_id": torch.arange(num_samples),
    }
    return data_dict, cfg


import torch
import torch.nn as nn


class HistogramEicEncoder(nn.Module, Module):
    """
    Embeds integer codes and combines them with a weighted average of category embeddings
    using a gating mechanism.

    Instead of using an MLP for the histogram embedding, this version computes a
    weighted average of category embeddings. For example, a histogram [1, 0, 3] is normalized
    to [0.25, 0, 0.75] and then used as:
        weighted_embedding = 0.25 * cat0_embed + 0 * cat1_embed + 0.75 * cat2_embed

    Example usage:
        >>> batch, cfg = get_dummy_batch_and_cfg()
        >>> encoder = HistogramEicEncoder(cfg)
        >>> output = encoder(batch)
        >>> output[INPUT_ENCODER_TOKENS_KEY].shape
        torch.Size([batch_size, seq_length, token_dim])
        >>> output[INPUT_ENCODER_MASK_KEY].shape
        torch.Size([batch_size, seq_length])
    """

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg

        # Embedding for discrete codes
        self.code_embedder = nn.Embedding(cfg.vocab_size, cfg.token_dim)

        # Embedding for histogram categories. This will be used to compute a weighted average.
        self.category_embedding = nn.Embedding(cfg.subvocab_size, cfg.token_dim)
        # Get NTP token index from metadata
        self.ntp_token = pl.read_parquet(cfg.metadata_fp).filter(pl.col("code").eq("[NTP]"))[
            "code/vocab_index"
        ][0]

    def forward(self, batch):
        # Expecting keys "code", "histogram", and "mask" in the batch.
        batch[INPUT_ENCODER_MASK_KEY] = batch["mask"]

        # Code embeddings: shape (batch_size, seq_length, token_dim)
        embedded_codes = self.code_embedder(batch["code"])

        # Convert the histogram to float if it isn't already.
        histogram = batch["histogram"].float()
        # Normalize each histogram by its sum to get probabilities.
        histogram_sum = histogram.sum(dim=-1, keepdim=True)
        normalized_histogram = histogram / (
            histogram_sum + 1e-8
        )  # add small epsilon to avoid division by zero

        # Compute the weighted average of the category embeddings.
        # If histogram has shape (B, S, subvocab_size) and the embedding matrix has shape (subvocab_size, token_dim),
        # then the resulting weighted average has shape (B, S, token_dim)
        embedded_histograms = torch.matmul(normalized_histogram, self.category_embedding.weight)
        # Fuse the code embeddings with the gated histogram embeddings.
        ntp_mask = (batch["code"] == self.ntp_token).unsqueeze(-1)
        fused_embeddings = embedded_codes + ntp_mask * embedded_histograms

        batch[INPUT_ENCODER_TOKENS_KEY] = fused_embeddings
        return batch

    def process_sample(self, codes, histograms):
        """
        Process a single sample (or independent tensors) with the same logic.
        """
        embedded_codes = self.code_embedder(codes)
        histograms = histograms.float()
        histogram_sum = histograms.sum(dim=-1, keepdim=True)
        normalized_histogram = histograms / (histogram_sum + 1e-8)
        embedded_histograms = torch.matmul(normalized_histogram, self.category_embedding.weight)
        ntp_mask = (codes == self.ntp_token).unsqueeze(-1)
        fused_embeddings = embedded_codes + ntp_mask * embedded_histograms
        return fused_embeddings
