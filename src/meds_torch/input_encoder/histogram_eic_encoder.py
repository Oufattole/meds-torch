import polars as pl
import torch
from torch import nn
from torchvision.ops import MLP

from meds_torch.input_encoder import INPUT_ENCODER_MASK_KEY, INPUT_ENCODER_TOKENS_KEY
from meds_torch.utils import RankedLogger
from meds_torch.utils.module_class import Module

log = RankedLogger(__name__, rank_zero_only=True)


import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import nn


class VQPrompt(nn.Module):
    def __init__(self, D, K, commit_coef=0.25):
        super().__init__()
        self.codebook = nn.Embedding(K, D)
        nn.init.uniform_(self.codebook.weight, -1.0, 1.0)
        self.commit_coef = commit_coef
        # these will be set during forward
        self.codebook_loss = None
        self.commit_loss = None

    def forward(self, x):
        # x: (B, L, D)
        B, L, D = x.size()
        flat = x.view(-1, D)  # (B*L, D)

        # 1) Find nearest codebook entry
        # compute distances
        flat_sq = (flat**2).sum(-1, keepdim=True)  # (B*L, 1)
        cb = self.codebook.weight  # (K, D)
        cb_sq = (cb**2).sum(-1)  # (K,)
        dot = flat @ cb.t()  # (B*L, K)
        dists = flat_sq + cb_sq.unsqueeze(0) - 2 * dot  # (B*L, K)

        # select
        idx = dists.argmin(dim=1)  # (B*L,)
        quant = self.codebook(idx)  # (B*L, D)

        # 2) Compute VQ losses
        # codebook should move toward the encoder output
        self.codebook_loss = F.mse_loss(quant, flat.detach())
        # encoder should commit to the codebook (with a smaller weight)
        self.commit_loss = F.mse_loss(quant.detach(), flat) * self.commit_coef

        # 3) Straight-through output: update encoder via the flat term,
        #    but forward the quantized vector so the LM sees discrete codes.
        out_flat = flat + (quant - flat).detach()  # grad→flat only
        return out_flat.view(B, L, D)


class BaseHistogramEicEncoder(nn.Module, Module):
    def __init__(self, cfg):
        super().__init__()
        self.ntp_token = pl.read_parquet(cfg.metadata_fp).filter(pl.col("code").eq("[NTP]"))[
            "code/vocab_index"
        ][0]
        self.h_token = pl.read_parquet(cfg.metadata_fp).filter(pl.col("code").eq("[H]"))["code/vocab_index"][
            0
        ]

        if cfg.do_prompt_tune:
            # self.prompt_mlp = nn.Linear(cfg.token_dim, cfg.token_dim)
            if cfg.encode_for_prompt_tuning:
                input_dim = cfg.encoder_dims[-1]
            else:
                input_dim = cfg.token_dim
            # self.prompt_mlp = nn.Linear(input_dim, cfg.token_dim)
            self.prompt_mlp = MLP(
                in_channels=input_dim,
                hidden_channels=[cfg.token_dim, cfg.token_dim, cfg.token_dim, cfg.token_dim],
                norm_layer=nn.LayerNorm,
                activation_layer=nn.GELU,
                dropout=0.1,
                bias=True,
            )
            if cfg.do_vector_quantize_prompts:
                self.prompt_quantizer = VQPrompt(K=64, D=cfg.token_dim)
            else:
                self.prompt_quantizer = nn.Identity()

    def forward(self, batch, histogram_embedding=None):
        """
        We will roll the embeddings forward one token
        """
        # Expecting keys "code", "histogram", and "mask" in the batch.
        batch[INPUT_ENCODER_MASK_KEY] = batch["mask"]

        # Code embeddings: shape (batch_size, seq_length, token_dim)
        embedded_codes = self.code_embedder(batch["code"])
        ntp_mask = (batch["code"] == self.ntp_token).unsqueeze(-1)
        if histogram_embedding is not None:
            if not hasattr(self, "prompt_mlp"):
                raise ValueError("prompt_mlp must be defined if histogram_embedding is provided")
            # roll embeddings forward since we want to use the embedding of the h token on the following ntp token
            prompts = self.prompt_mlp(histogram_embedding)
            prompts = prompts.roll(shifts=1, dims=1)
            prompts = self.prompt_quantizer(prompts)
            # torch.tensor([1, 2, 3, 4, 5]).roll(shifts=1, dims=0), this will be tensor([5, 1, 2, 3, 4]), perfect
            batch[INPUT_ENCODER_TOKENS_KEY] = torch.where(ntp_mask, prompts, embedded_codes)
            return batch
        else:
            return self._forward(batch)

    def process_sample(self, codes, histograms, histogram_embedding=None):
        """
        Process a single sample (or independent tensors) with the same logic.

        If histogram_embedding is passed, prompt tuning is applied

        When the code is an NTP token, this means it was just sampled from the embedding of the h_token.
        So we can just use the histogram_embedding then the ntp_mask is true.
        """
        embedded_codes = self.code_embedder(codes)
        ntp_mask = (codes == self.ntp_token).unsqueeze(-1)
        if histogram_embedding is not None:
            if not hasattr(self, "prompt_mlp"):
                raise ValueError("prompt_mlp must be defined if histogram_embedding is provided")
            prompts = self.prompt_mlp(histogram_embedding)
            prompts = self.prompt_quantizer(prompts)
            return torch.where(ntp_mask, prompts, embedded_codes)
        else:
            return self._process_sample(codes, histograms)


class HistogramEicEncoder(BaseHistogramEicEncoder):
    def __init__(self, cfg):
        super().__init__(cfg)
        self.cfg = cfg

        # Embedding for discrete codes
        self.code_embedder = nn.Embedding(cfg.vocab_size, cfg.token_dim)

        # Embedding for histogram categories. This will be used to compute a weighted average.
        self.category_embedding = nn.Embedding(cfg.subvocab_size, cfg.token_dim)
        # Get NTP token index from metadata

    def _forward(self, batch):
        # Expecting keys "code", "histogram", and "mask" in the batch.
        batch[INPUT_ENCODER_MASK_KEY] = batch["mask"]

        # Code embeddings: shape (batch_size, seq_length, token_dim)
        embedded_codes = self.code_embedder(batch["code"])
        ntp_mask = (batch["code"] == self.ntp_token).unsqueeze(-1)
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
        fused_embeddings = torch.where(ntp_mask, embedded_codes + embedded_histograms, embedded_codes)

        batch[INPUT_ENCODER_TOKENS_KEY] = fused_embeddings
        return batch

    def _process_sample(self, codes, histograms):
        """
        Process a single sample (or independent tensors) with the same logic.

        If histogram_embedding is passed, prompt tuning is applied
        """
        embedded_codes = self.code_embedder(codes)
        ntp_mask = (codes == self.ntp_token).unsqueeze(-1)
        histograms = histograms.float()
        histogram_sum = histograms.sum(dim=-1, keepdim=True)
        normalized_histogram = histograms / (histogram_sum + 1e-8)
        with torch.autocast("cuda", torch.float32):
            embedded_histograms = torch.matmul(
                normalized_histogram.to(torch.float32), self.category_embedding.weight.to(torch.float32)
            )
        fused_embeddings = torch.where(ntp_mask, embedded_codes + embedded_histograms, embedded_codes)
        return fused_embeddings.to(embedded_codes.dtype)
