import torch
from torch import nn

from meds_torch.input_encoder import INPUT_ENCODER_MASK_KEY, INPUT_ENCODER_TOKENS_KEY
from meds_torch.input_encoder.histogram_eic_encoder import BaseHistogramEicEncoder


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


# Set the histogram values for both batches
import torch.nn as nn


class HistogramEicEncoder(BaseHistogramEicEncoder):
    """
    Embeds integer codes and combines them with histogram embeddings using
    an MLP for histogram embedding and a gating mechanism.

    Example:
        >>> batch, cfg = get_dummy_batch_and_cfg()
        >>> encoder = HistogramEicEncoder(cfg)
        >>> output = encoder(batch)
        >>> output[INPUT_ENCODER_TOKENS_KEY].shape
        torch.Size([batch_size, seq_length, token_dim])
        >>> output[INPUT_ENCODER_MASK_KEY].shape
        torch.Size([batch_size, seq_length])
    """

    def __init__(self, cfg):
        super().__init__(cfg)
        self.cfg = cfg

        # Embedding for discrete codes
        self.code_embedder = nn.Embedding(cfg.vocab_size, cfg.token_dim)

        # MLP embedder for the histogram (two-layer MLP with non-linearity)
        self.histogram_embedder = nn.Sequential(
            nn.Linear(
                cfg.subvocab_size,
                cfg.token_dim * 2,
            ),
            nn.ReLU(),
            nn.Linear(cfg.token_dim * 2, cfg.token_dim),
        )

    def _forward(self, batch):
        # Assume batch contains "code", "histogram", and "mask" keys.
        batch[INPUT_ENCODER_MASK_KEY] = batch["mask"]

        # Code embeddings: shape (batch_size, seq_length, token_dim)
        embedded_codes = self.code_embedder(batch["code"])

        # Normalize histograms. This normalization assumes that dividing by cfg.max_count
        # brings the histogram values to a similar scale as the learned embeddings.
        normalized_histogram = batch["histogram"] / self.cfg.max_count

        # Histogram embeddings via a non-linear MLP: shape (batch_size, seq_length, token_dim)
        embedded_histograms = self.histogram_embedder(normalized_histogram)

        # Fuse the signals: histogram information is scaled by the gate then added to code embeddings.
        ntp_mask = (batch["code"] == self.ntp_token).unsqueeze(-1)
        fused_embeddings = torch.where(ntp_mask, embedded_codes + embedded_histograms, embedded_codes)

        batch[INPUT_ENCODER_TOKENS_KEY] = fused_embeddings
        return batch

    def _process_sample(self, codes, histograms):
        # Process a single sample (or independent tensors) with the same logic.
        embedded_codes = self.code_embedder(codes)
        normalized_histogram = histograms / self.cfg.max_count
        embedded_histograms = self.histogram_embedder(normalized_histogram)
        ntp_mask = (codes == self.ntp_token).unsqueeze(-1)
        fused_embeddings = torch.where(ntp_mask, embedded_codes + embedded_histograms, embedded_codes)
        return fused_embeddings
