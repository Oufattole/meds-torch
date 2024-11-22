import dataclasses
import enum

import torch
from torch import nn
from omegaconf import DictConfig

from meds_torch.models.components.cve import CVE
from meds_torch.models.components.ecg_encoder import ECGEncoder
from meds_torch.input_encoder import INPUT_ENCODER_MASK_KEY, INPUT_ENCODER_TOKENS_KEY
from meds_torch.utils.module_class import Module

class ECGTripletEncoder(nn.Module, Module):
    """
    ECG / EHR Triplet Encoder Module.

    This module encodes ECG and EHR triplet data.

    The EHR triplets are comprised of three fields:
      - time_delta_days (relative time differences)
      - code
      - numeric_value (i.e. measurements)

    The time_delta_days and numeric_values are embedded with Continuous Value
    Embedding (CVE), while the code uses an embedding vocabulary. These
    embeddings are then summed up.

    If cfg.early_fusion is True, then the ECGs are embedded and added to the
    spot in the EHR time series at which they were recorded. Otherwise, the
    ECG and EHR embeddings are returned separately in a dictionary.

    Args:
        cfg (DictConfig): Configuration object containing the following parameters:
            - token_dim (int): Dimensionality of the embeddings.
            - vocab_size (int): Size of the vocabulary for categorical codes.
            - max_seq_len (int): Maximum allowed sequence length.
            - early_fusion (bool): Whether to fuse the ECG and EHR or not.

    Inputs:
        batch (dict): A dictionary containing the following keys:
            - "code": Tensor of hospital codes (shape: [batch_size, seq_len]).
            - "numeric_value": Tensor of measurements (shape: [batch_size, seq_len]).
            - "time_delta_days": Tensor of relative times (shape: [batch_size, seq_len]).
            - "mask": Boolean tensor indicating valid positions (shape: [batch_size, seq_len]).
            - "static_mask": Boolean tensor marking static features (shape: [batch_size, seq_len]).
            - "numeric_value_mask": Boolean tensor marking valid numeric values (shape: [batch_size, seq_len]).
            - "modality": Tensor of ECGs (shape: [num_ecgs, 12, 5000]).
            - "modality_batch_idx": Tensor of positions in batch each ECG corresponds to (shape: [num_ecgs]).
            - "modality_sequence_idx": Tensor of positions in time series each ECG corresponds to (shape: [num_ecgs]).

    Outputs:
        batch (dict): Updated dictionary containing:
            - INPUT_ENCODER_MASK_KEY: The input mask for valid positions.
            - INPUT_ENCODER_TOKENS_KEY: If cfg.early_fusion == True, this is the fused embeddings,
                                        shape (batch_size, seq_len, token_dim).
                                        If cfg.early_fusion == False, this is a dictionary including
                                        the separate EHR and ECG embeddings.
    
    Example 1 -- early_fusion==True:
        >>> import torch
        >>> from omegaconf import OmegaConf
        >>> cfg = OmegaConf.create({"token_dim": 128, "vocab_size": 100, "max_seq_len": 512, "early_fusion": True, "num_patches": 50, "patch_size": 100, "num_leads": 12})
        >>> encoder = ECGTripletEncoder(cfg)
        >>> # Create a dummy batch
        >>> #   - Batch size: 2
        >>> #   - Sequence length: 10
        >>> #   - Number of ECGs: 3
        >>> batch = {
        ...     "code": torch.randint(0, 100, (2, 10)),
        ...     "numeric_value": torch.randn(2, 10),
        ...     "time_delta_days": torch.randn(2, 10),
        ...     "mask": torch.ones(2, 10, dtype=torch.bool),
        ...     "static_mask": torch.zeros(2, 10, dtype=torch.bool),
        ...     "numeric_value_mask": torch.ones(2, 10, dtype=torch.bool),
        ...     "modality": torch.randn(3, 12, 5000),
        ...     "modality_batch_idx": [0, 1, 1],
        ...     "modality_sequence_idx": [4, 7, 9],
        ... }
        >>> output = encoder(batch)
        >>> # Check the updated batch
        >>> output[INPUT_ENCODER_MASK_KEY].shape
        torch.Size([2, 10])
        >>> output[INPUT_ENCODER_TOKENS_KEY].shape
        torch.Size([2, 10, 128])
    
    Example 2 -- early_fusion==False:
        >>> import torch
        >>> from omegaconf import OmegaConf
        >>> cfg = OmegaConf.create({"token_dim": 128, "vocab_size": 100, "max_seq_len": 512, "early_fusion": False, "num_patches": 50, "patch_size": 100, "num_leads": 12})
        >>> encoder = ECGTripletEncoder(cfg)
        >>> # Create a dummy batch
        >>> #   - Batch size: 2
        >>> #   - Sequence length: 10
        >>> #   - Number of ECGs: 3
        >>> batch = {
        ...     "code": torch.randint(0, 100, (2, 10)),
        ...     "numeric_value": torch.randn(2, 10),
        ...     "time_delta_days": torch.randn(2, 10),
        ...     "mask": torch.ones(2, 10, dtype=torch.bool),
        ...     "static_mask": torch.zeros(2, 10, dtype=torch.bool),
        ...     "numeric_value_mask": torch.ones(2, 10, dtype=torch.bool),
        ...     "modality": torch.randn(3, 12, 5000),
        ...     "modality_batch_idx": [0, 1, 1],
        ...     "modality_sequence_idx": [4, 7, 9],
        ... }
        >>> output = encoder(batch)
        >>> # Check the updated batch
        >>> output[INPUT_ENCODER_MASK_KEY].shape
        torch.Size([2, 10])
        >>> output[INPUT_ENCODER_TOKENS_KEY]["EHR_embedding"].shape
        torch.Size([2, 10, 128])
        >>> output[INPUT_ENCODER_TOKENS_KEY]["ECG_embedding"].shape
        torch.Size([2, 128])
    """

    def __init__(self, cfg: DictConfig):
        super().__init__()
        self.cfg = cfg
        self.date_embedder = CVE(cfg)
        self.code_embedder = torch.nn.Embedding(cfg.vocab_size, embedding_dim=cfg.token_dim)
        self.numeric_value_embedder = CVE(cfg)
        self.ecg_embedder = ECGEncoder(cfg)

    def embed_func(self, embedder, x):
        out = embedder.forward(x[None, :].transpose(2, 0)).permute(1, 2, 0)
        return out

    def encode(self, batch):
        # Embed times then mask static value times.
        time_emb = (
            self.embed_func(self.date_embedder, batch["time_delta_days"])
            * ~(batch["static_mask"]).unsqueeze(dim=1)
        )

        # Embed codes.
        code_emb = self.code_embedder.forward(batch["code"]).permute(0, 2, 1)

        # Embed numerical values and mask NaN values.
        val_emb = (
            self.embed_func(self.numeric_value_embedder, batch["numeric_value"])
            * batch["numeric_value_mask"].unsqueeze(dim=1)
        )

        # Embed the ECGs.
        # Shape: (E, token_dim).
        ecg_emb = self.ecg_embedder.forward(batch["modality"])

        # Sum the (time, code, value) triplets.
        # Shape is (B, token_dim, seq_len), transposed to (B, seq_len, token_dim).
        ehr_emb = time_emb + code_emb + val_emb
        ehr_emb = ehr_emb.transpose(1, 2)

        assert ehr_emb.isfinite().all(), "EHR embedding is not finite"
        if ehr_emb.shape[-1] > self.cfg.max_seq_len:
            raise ValueError(
                f"Triplet EHR embedding length {ehr_emb.shape[-1]} "
                "is greater than max_seq_len {self.cfg.max_seq_len}"
            )

        if not self.cfg.early_fusion:
            # Squash ECG embeddings from (E, token_dim) to (B, token_dim) by
            # mean pooling ECGs that come from the same sample within the batch.
            #
            # Initialize tensors to accumulate embeddings and counts
            B = ehr_emb.shape[0]
            token_dim = ecg_emb.shape[1]
            ecg_emb_sums = torch.zeros(B, token_dim, device=ecg_emb.device)
            counts = torch.zeros(B, 1, device=ecg_emb.device)

            modality_batch_idx = torch.tensor(batch["modality_batch_idx"],
                                        dtype=torch.long, device=ecg_emb.device)

            # Accumulate ECG embeddings per batch index and count the number of
            # ECGs per batch index.
            ecg_emb_sums.index_add_(0, modality_batch_idx, ecg_emb)
            counts.index_add_(0, modality_batch_idx, torch.ones(len(modality_batch_idx), 1, device=ecg_emb.device))

            # Avoid division by zero by setting counts to 1 where counts are zero
            counts = counts + (counts == 0).float()

            # Compute the mean ECG embedding per batch
            ecg_mean_emb = ecg_emb_sums / counts

            return {"EHR_embedding": ehr_emb, "ECG_embedding": ecg_mean_emb}
        
        # Add ECG embeddings to EHR embedding via early fusion.
        #
        # batch["modality_batch_idx"] and batch["modality_sequence_idx"] are
        # size (E), the number of ECGs. These tell you which batch and which
        # element in the time series to add each ECG into.
        fused_emb = ehr_emb
        fused_emb[batch["modality_batch_idx"], batch["modality_sequence_idx"]] += ecg_emb

        assert fused_emb.isfinite().all(), "Fused embedding is not finite"
        return fused_emb

    def forward(self, batch):
        batch[INPUT_ENCODER_MASK_KEY] = batch["mask"]
        batch[INPUT_ENCODER_TOKENS_KEY] = self.encode(batch)
        return batch

if __name__ == "__main__":
    import doctest
    doctest.testmod(verbose=True)
