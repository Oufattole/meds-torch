from collections.abc import Callable
from contextlib import nullcontext

import numpy as np
import polars as pl
import torch
import torch.nn.functional as F
import torch.utils
from clinical_zeroshot_labeler.labeler import SequenceLabeler, WindowStatus
from clinical_zeroshot_labeler.model import BaseGenerativeModel, RateColumn, slice_cache
from loguru import logger
from mixins import TimeableMixin
from omegaconf import DictConfig
from rich.progress import (
    BarColumn,
    Progress,
    TaskProgressColumn,
    TextColumn,
    TimeRemainingColumn,
)
from torch import nn
from torchmetrics import Metric, MetricCollection
from torchmetrics.classification import MulticlassAccuracy, MulticlassAUROC
from torchvision.ops import MLP
from x_transformers import Decoder, TransformerWrapper
from x_transformers.autoregressive_wrapper import (
    FILTER_LOGITS_FN,
    align_right,
    exists,
    identity,
    join,
)

from meds_torch.input_encoder import INPUT_ENCODER_MASK_KEY, INPUT_ENCODER_TOKENS_KEY
from meds_torch.models import (
    BACKBONE_EMBEDDINGS_KEY,
    BACKBONE_TOKENS_KEY,
    GENERATE_PREFIX,
    MODEL_BATCH_LOSS_KEY,
    MODEL_EMBEDDINGS_KEY,
    MODEL_LOGITS_SEQUENCE_KEY,
    MODEL_LOSS_KEY,
    MODEL_PRED_PROBA_KEY,
    MODEL_PRED_STATUS_KEY,
    MODEL_PREFIX,
    MODEL_TOKENS_KEY,
)
from meds_torch.models.base_model import BaseModule
from meds_torch.models.components.utils import TrajectoryBatch, get_time_days_delta
from meds_torch.utils import TIME_DELTA_TOKEN

MODEL_LOSS_KEYS = ["MODEL//code_loss", "MODEL//vae_loss", "MODEL//vae_rec_loss", "MODEL//vae_kl_loss"]


def eval_decorator(fn):
    def inner(self, *args, **kwargs):
        was_training = self.model.model.training
        self.model.model.eval()
        out = fn(self, *args, **kwargs)
        self.model.model.train(was_training)
        return out

    return inner


class DummyTrajectoryLabeler:
    def __init__(self, B):
        self.counter = 0
        self.status = [
            torch.tensor([WindowStatus.UNDETERMINED.value] * B),
            torch.tensor([WindowStatus.ACTIVE.value] * B),
            torch.tensor([WindowStatus.SATISFIED.value] * B),
        ]
        self.labels = torch.zeros((B,), dtype=torch.bool)

    def process_step(self, tokens, times, values):
        status = self.status[min(self.counter, len(self.status) - 1)]
        self.counter += 1
        return status

    def is_finished(self):
        return self.counter >= len(self.status)

    def get_labels(self):
        return self.labels


def create_dummy_sequence_labeler(batch_size: int = 2):
    """Create a dummy sequence labeler with a simple ACES task configuration.

    Args:
        batch_size: Number of sequences to process in parallel

    Returns:
        Tuple containing:
            - Dummy labeler instance
            - Metadata DataFrame
            - Sample input batch
            - ACES task configuration string

    Examples:
        >>> labeler, metadata_df, batch, task_config = create_dummy_sequence_labeler()
        >>> import torch
        >>> assert isinstance(batch['code'], torch.Tensor)
        >>> assert batch['code'].shape == (2, 8)  # batch_size=2, seq_len=8
        >>> assert 'mask' in batch
        >>> # Test indices are within vocab range
        >>> max_idx = batch['code'].max()
        >>> assert max_idx < len(metadata_df)
    """
    from datetime import datetime

    import polars as pl
    import torch
    from clinical_zeroshot_labeler.labeler import SequenceLabeler

    # Define simple ACES task configuration
    task_config = """
    predicates:
        hospital_discharge:
            code: {regex: "HOSPITAL_DISCHARGE//.*"}
        lab:
            code: {regex: "LAB//.*"}
        high_lab:
            code: {regex: "LAB//.*"}
            value_min: 2.0
            value_min_inclusive: True

    trigger: hospital_discharge

    windows:
        input:
            start: NULL
            end: trigger
            start_inclusive: True
            end_inclusive: True
            index_timestamp: end
        target:
            start: input.end
            end: start + 365d
            start_inclusive: False
            end_inclusive: True
            has:
                lab: (1, None)
            label: high_lab
    """

    # Create metadata DataFrame with test codes
    metadata_df = pl.DataFrame(
        {
            "code": [
                "PAD",
                "HOSPITAL_DISCHARGE//MEDICAL",
                "LAB//_Q_1",
                "LAB//_Q_2",
                "LAB//_Q_3",
                "TIME//DELTA//TOKEN//_Q_1",
                "TIME//DELTA//TOKEN//_Q_2",
                "TIME//DELTA//TOKEN//_Q_3",
                "[H]",
                "[NTP]",
            ],
            "code/vocab_index": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
            "values/min": [None, None, 2.0, 0.0, 1.0, 0, 1, 2, None, None],
            "values/max": [None, None, 3.0, 1.0, 2.0, 1, 2, 3, None, None],
            "values/sum": [None, None, 0.5, 1.5, 2.5, 0.5, 1.5, 2.5, None, None],
            "values/n_occurrences": [None, None, 1, 1, 1, 1, 1, 1, None, None],
            "values/quantiles": [
                {"values/quantile/0.5": None},
                {"values/quantile/0.5": None},
                {"values/quantile/0.5": 1},
                {"values/quantile/0.5": 1},
                {"values/quantile/0.5": 1},
                {"values/quantile/0.5": 1},
                {"values/quantile/0.5": 1},
                {"values/quantile/0.5": 1},
                {"values/quantile/0.5": None},
                {"values/quantile/0.5": None},
            ],
        }
    )

    # Create sample input batch
    # Sequence 1: Hospital discharge -> High lab value -> Time token
    # Sequence 2: Hospital discharge -> Normal lab value -> Time token
    # Note: All indices should be < len(metadata_df)
    from meds_torch.data.components.histogram_pytorch_dataset import (
        compute_count_histogram,
        insert_h_o_tokens,
    )

    h_token = 8
    o_token = 9
    codes = np.array([[1, 2, 4], [1, 3, 4]])
    token_bin_size = 2
    vocab_size = metadata_df.shape[0]

    patient_1_codes = codes[0]
    p1_inserted_codes = insert_h_o_tokens(patient_1_codes, token_bin_size, h_token, o_token)
    p1_histogram = compute_count_histogram(p1_inserted_codes, vocab_size, o_token)

    patient_2_codes = codes[1]
    p2_inserted_codes = insert_h_o_tokens(patient_2_codes, token_bin_size, h_token, o_token)
    p2_histogram = compute_count_histogram(p2_inserted_codes, vocab_size, o_token)

    batch_codes = torch.tensor([p1_inserted_codes, p2_inserted_codes], dtype=torch.long)
    batch_histograms = torch.tensor([p1_histogram, p2_histogram], dtype=torch.float32)
    batch_masks = torch.ones(2, batch_codes.shape[1], dtype=torch.bool)

    batch = {
        "code": batch_codes,  # Using vocab indices
        "histogram": batch_histograms,
        "mask": batch_masks,
        "subject_id": torch.tensor([1, 2]),
        "prediction_time": [datetime(2020, 1, 1), datetime(2020, 1, 1)],
        "end_time": [datetime(2020, 1, 1), datetime(2020, 1, 1)],
    }

    # Initialize sequence labeler
    from functools import partial

    labeler = partial(SequenceLabeler.from_yaml_str, yaml_str=task_config, early_stop=True)

    return labeler, metadata_df, batch, task_config


def create_model_config(metadata_df_path: str):
    """Create a model configuration for testing.

    Args:
        metadata_df_path: Path to metadata DataFrame parquet file

    Returns:
        Instantiated model configuration

    Examples:
        >>> import tempfile, polars as pl
        >>> with tempfile.NamedTemporaryFile(suffix='.parquet') as temp_file:
        ...     df = pl.DataFrame({"code": ["A", "[H]", "[NTP]"], "code/vocab_index": [0, 1, 2]})
        ...     df.write_parquet(temp_file.name)
        ...     cfg = create_model_config(temp_file.name)
        >>> assert cfg.vocab_size == 3, cfg.vocab_size  # Original size + pad token
    """
    from hydra.utils import instantiate

    metadata_df = pl.read_parquet(metadata_df_path)

    vocab_size = metadata_df.height
    token_dim = 5

    cfg = {
        "augmented_code_metadata_fp": metadata_df_path,
        "backbone": {
            "_target_": "meds_torch.models.histogram_forecasting.DummyModel",
            "token_dim": token_dim,
            "vocab_size": vocab_size,
        },
        "beta": 1e-3,
        "token_insertion_strategy": "token_count",
        "token_bin_size": 2,
        "vocab_size": vocab_size,  # Add 1 for pad token
        "generate_id": None,
        "store_generated_trajectory": True,
        "max_seq_len": 10,
        "temperature": 1.0,
        "eos_tokens": None,
        "optimizer": {
            "_target_": "meds_torch.models.histogram_forecasting.DummyOptimizer",
            "_partial_": True,
        },
        "scheduler": {
            "_target_": "meds_torch.models.histogram_forecasting.DummyScheduler",
            "_partial_": True,
        },
        "input_encoder": {
            "_target_": "meds_torch.models.histogram_forecasting.DummyEncoder",
            "token_dim": token_dim,
            "vocab_size": vocab_size,
        },
        "code_head": {
            "_target_": "meds_torch.models.histogram_forecasting.DummyCodeHead",
            "vocab_size": vocab_size,
        },
        "compile": False,
        "top_k_acc": [1],
        "next_token_auc": False,
        "max_tokens_budget": 10,
        "return_tokens": False,
        "return_logits": False,
        "return_labeler": False,
        "prune_terminated": False,
        "histogram_batch_mul": 2,
        "n_bits": 8,
        "scale": 1.0,
        "token_dim": 5,
    }
    return instantiate(cfg)


class DummyModel:
    """Dummy model that generates two fixed sequences."""

    cfg = dict(token_emb=torch.nn.Identity())

    def __init__(self, token_dim, vocab_size):
        self.token_dim = token_dim
        self.vocab_size = vocab_size
        self.model = TransformerWrapper(
            num_tokens=vocab_size,
            max_seq_len=10,
            attn_layers=Decoder(dim=token_dim, depth=1, heads=2, rotary_pos_emb=True),
            use_abs_pos_emb=False,
        )
        self.model.token_emb = torch.nn.Identity()

    def __call__(self, batch, do_get_last_token=False):
        B, S = batch["code"].shape
        histogram = torch.ones(B, S, self.vocab_size)
        histogram[:, :, :1] = 0
        return {
            BACKBONE_TOKENS_KEY: torch.ones(B, S, self.token_dim),
            BACKBONE_EMBEDDINGS_KEY: torch.rand(B, S, self.token_dim),
            "histogram": histogram,
        }

    def generate(self, prompts, **kwargs):
        # Always generate two fixed sequences
        generated = torch.tensor(
            [
                [5, 2, 2, 7, 7],  # Sequence 1: low labs only
                [5, 4, 4, 5, 5],  # Sequence 2: high labs only
            ]
        )
        out_lengths = torch.tensor([5, 5])
        labels = dict()
        if kwargs.get("trajectory_labeler") is not None:
            labels = dict(
                labels=torch.tensor([1.0, 0.0]),  # Sequence 1 positive, Sequence 2 negative
                status=torch.ones(2) * WindowStatus.SATISFIED.value,
            )
        return generated, out_lengths, labels


class DummyCodeHead:
    def __init__(self, vocab_size):
        self.vocab_size = vocab_size  # Match metadata size

    def __call__(self, x):
        B, S, _ = x.shape
        logits = torch.zeros(B, S, self.vocab_size)
        logits[..., 1:] = 1.0  # All tokens except PAD equally likely
        return logits


class DummyEncoder:
    def __init__(self, vocab_size, token_dim):
        self.token_encoder = torch.nn.Embedding(vocab_size, token_dim)
        self.histogram_encoder = torch.nn.Linear(vocab_size, token_dim)

    def __call__(self, batch):
        batch[INPUT_ENCODER_TOKENS_KEY] = self.token_encoder(batch["code"])
        batch[INPUT_ENCODER_MASK_KEY] = torch.ones_like(batch["mask"]).bool()
        return batch

    def process_sample(self, codes, histograms):
        return self.token_encoder(codes) + self.histogram_encoder(histograms)


class DummyOptimizer:
    def __init__(self, params):
        self.params = list(params)

    def step(self):
        pass

    def zero_grad(self):
        pass


class DummyScheduler:
    def step(self):
        pass

    def get_last_lr(self):
        return [0.001]


CODE_LOGITS = "EIC_MODEL//CODE_LOGITS"


def topk(x: torch.Tensor, k: torch.Tensor) -> torch.Tensor:
    """
    Creates a mask where values are 0 for the top-k elements in each batch and 1 elsewhere.

    Args:
        x : tensor of shape [B, L] containing values to find top-k elements
        k : tensor of shape [B, 1] containing the number of top elements to find for each batch

    Returns:
        final_mask: tensor of shape [B,L] where final_mask[b,i] = 0 if x[b,i] is in
                   the k[b] biggest values of x[b,:], else final_mask[b,i] = 1
    """
    B, L = x.shape  # batchsize, list size

    # Get indices sorted in descending order
    _, indices_des = torch.sort(x, dim=-1, descending=True)

    # Create range mask [1, L] and repeat it B times
    mask = torch.arange(L, device=x.device).unsqueeze(0).expand(B, -1)
    k_expanded = k.expand(-1, L)
    mask = mask < k_expanded

    # Create one-hot encoding and apply mask
    one_hot = torch.nn.functional.one_hot(indices_des, num_classes=L).float()
    one_hot = one_hot * mask.unsqueeze(-1)

    # Sum along the appropriate dimension to get final mask
    final_mask = one_hot.sum(dim=1)

    # Flip the mask (0 for top-k, 1 for others)
    return final_mask


class HistogramNormalizer(torch.nn.Module):
    def __init__(self, h_token, o_token, vocab_size, num_bits=16, scale=1):
        super().__init__()
        self.h_token = h_token
        self.o_token = o_token
        self.vocab_size = vocab_size
        self.num_bits = num_bits
        self.scale = scale
        # Create powers of 2 as a buffer to avoid recomputing
        self.register_buffer("powers", torch.pow(2, torch.arange(num_bits - 1, -1, -1).float()))

    def get_normalized_size(self):
        return self.vocab_size * self.num_bits

    def count_to_binary(self, count):
        """Convert number(s) to binary representation using PyTorch operations."""
        return ((count.unsqueeze(-1) // self.powers) % 2).to(torch.int)

    def binary_to_count(self, binary):
        return (binary * self.powers).sum(dim=-1, keepdim=True)

    def transform(self, x):
        # Zero out special tokens
        x[:, self.h_token] = 0
        x[:, self.o_token] = 0

        # Get counts and normalize histogram
        b, _ = x.shape

        # Convert counts to binary representation
        data = self.count_to_binary(x.reshape(-1)).reshape(b, -1).to(torch.float32)

        # Shift from [0,1] to [-self.scale,self.scale] range
        return (data * 2 - 1) * self.scale

    def reverse_transform(self, x):
        # Shift from [-self.scale,self.scale] to [0,1]  range
        x = (x / self.scale + 1) / 2

        # Clip to [0,1] range
        x = x.clip(min=0, max=1).round().int()
        # Split into histogram and binary count
        b, _ = x.shape

        # Convert from batch_size x (vocab_size * num_bits) to (batch_size x vocab_size) x num_bits
        x = x.reshape(b * self.vocab_size, -1)
        # Convert binary back to count, and reshape
        x = self.binary_to_count(x)
        # Convert from (batch_size x vocab_size) x 1 to batch_size x vocab_size
        x = x.reshape(b, -1)

        # Restore special tokens
        x[:, self.h_token] = 1
        x[:, self.o_token] = 1

        return x, x.sum(dim=-1)


class VAEEncoder(nn.Module):
    def __init__(
        self,
        input_dim=3,
        latent_dim=16,
        hidden_dims=[64, 32],
        dropout=0.0,
    ):
        super().__init__()
        # Create MLP encoder using torchvision
        self.encoder = MLP(
            in_channels=input_dim,
            hidden_channels=hidden_dims,
            norm_layer=nn.LayerNorm,
            activation_layer=nn.SiLU,
            dropout=dropout,
            inplace=None,  # Explicit None for clarity
        )
        self.fc_mu = nn.Linear(hidden_dims[-1], latent_dim)
        self.fc_var = nn.Linear(hidden_dims[-1], latent_dim)

    def forward(self, x):
        result = self.encoder(x)
        mu = self.fc_mu(result)
        log_var = self.fc_var(result)
        return mu, log_var


def scaled_sigmoid(x):
    return 2 * torch.sigmoid(x) - 1  # Scales [0,1] to [-1,1]


class VAEDecoder(nn.Module):
    def __init__(
        self,
        output_dim=3,
        latent_dim=16,
        hidden_dims=[32, 64],
        dropout=0.0,
    ):
        super().__init__()
        self.decoder = MLP(
            in_channels=latent_dim,
            hidden_channels=hidden_dims + [output_dim],
            norm_layer=nn.LayerNorm,
            activation_layer=nn.SiLU,
            dropout=dropout,
            inplace=None,  # Explicit None for clarity
        )

    def forward(self, z):
        return self.decoder(z)
        return scaled_sigmoid(self.decoder(z))


class DiagonalGaussianDistribution:
    def __init__(self, parameters, deterministic=False):
        self.parameters = parameters
        self.mean, self.logvar = torch.chunk(parameters, 2, dim=1)
        self.logvar = torch.clamp(self.logvar, -30.0, 20.0)
        self.deterministic = deterministic
        self.std = torch.exp(0.5 * self.logvar)
        self.var = torch.exp(self.logvar)
        if self.deterministic:
            self.var = self.std = torch.zeros_like(self.mean).to(device=self.parameters.device)

    def sample(self):
        x = self.mean + self.std * torch.randn(self.mean.shape).to(device=self.parameters.device)
        return x

    def kl(self, other=None):
        if self.deterministic:
            return torch.Tensor([0.0])
        else:
            if other is None:
                return 0.5 * torch.sum(
                    torch.pow(self.mean, 2) + self.var - 1.0 - self.logvar,
                    dim=[1],
                )
            else:
                return 0.5 * torch.sum(
                    torch.pow(self.mean - other.mean, 2) / other.var
                    + self.var / other.var
                    - 1.0
                    - self.logvar
                    + other.logvar,
                    dim=[1],
                )

    def nll(self, sample, dims=[1]):
        if self.deterministic:
            return torch.Tensor([0.0])
        logtwopi = np.log(2.0 * np.pi)
        return 0.5 * torch.sum(
            logtwopi + self.logvar + torch.pow(sample - self.mean, 2) / self.var,
            dim=dims,
        )

    def mode(self):
        return self.mean


class AutoencoderKL(nn.Module):
    def __init__(
        self,
        embed_dim,
        input_dim,
        output_dim=None,
        encoder_hidden_dims=[64, 32],
        decoder_hidden_dims=[32, 64],
        use_variational=True,
        beta=1.0,
        warmup_steps=None,
        annealing_steps=None,
    ):
        super().__init__()
        assert use_variational
        if output_dim is None:
            output_dim = input_dim
        self.use_variational = use_variational
        self._encoder = VAEEncoder(input_dim=input_dim, latent_dim=embed_dim, hidden_dims=encoder_hidden_dims)
        self._decoder = VAEDecoder(
            output_dim=output_dim, latent_dim=embed_dim, hidden_dims=decoder_hidden_dims
        )
        self.embed_dim = embed_dim
        self.beta = beta
        self.step = 0
        self.warmup_steps = warmup_steps if warmup_steps else 0
        self.annealing_steps = annealing_steps if annealing_steps else 0
        self.annealing_steps += self.warmup_steps

    def encode(self, x):
        mu, log_var = self._encoder(x)
        moments = torch.cat((mu, log_var), 1)
        posterior = DiagonalGaussianDistribution(moments)
        return posterior

    def decode(self, z):
        dec = self._decoder(z)
        return dec

    def forward(self, inputs, outputs=None, disable=False):
        return self.training_step(inputs, outputs, disable)

    def training_step(self, inputs, outputs=None, disable=False):
        """
        Training step for the VAE model.

        Args:
            inputs: Input tensor
            disable: Flag to disable parts of the loss computation
            optimizer_idx: Index of the optimizer (not used in this implementation)

        Returns:
            dict: Dictionary containing loss components and reconstructed output
        """
        posterior = self.encode(inputs)

        if disable or (self.warmup_steps and self.step < self.warmup_steps):
            # Deterministic warmup: directly use mean
            z = posterior.mean
        else:
            # Sample from posterior
            z = posterior.sample()

        if outputs is None:
            outputs = inputs

        # Decode the latent representation
        dec = self.decode(z)

        # Compute reconstruction loss (mean squared error)
        rec_loss = torch.nn.functional.mse_loss(dec, outputs, reduction="mean")

        # Compute KL divergence loss if using variational mode
        kl_loss = torch.zeros_like(rec_loss)
        if self.use_variational and not disable:
            kl_loss = posterior.kl().mean()

        # Total loss is reconstruction loss + KL divergence
        if self.warmup_steps and self.step < self.warmup_steps:
            beta = 0.0
        elif self.annealing_steps:
            beta = self.beta * min(1.0, (self.step / self.annealing_steps))
        else:
            beta = self.beta
        loss = rec_loss + beta * kl_loss

        self.step += 1

        return {
            "vae_loss": loss,
            "vae_rec_loss": rec_loss,
            "vae_kl_loss": kl_loss,
            "vae_reconstruction": dec,
        }


# Function to pad a single array
def pad_array(arr, max_len):
    pad_width = ((0, 0), (0, max_len - arr.shape[1]))
    if arr.dtype == bool:
        return np.pad(arr, pad_width, mode="constant", constant_values=False)
    else:
        return np.pad(arr, pad_width, mode="constant", constant_values=0)


def three_d_align_right(t, lens, pad_id=0):
    """Aligns the second dimension of a 3D tensor to the maximum length in the first dimension.

    Args:
        t: 3D tensor to align
        lens: 1D tensor of lengths to align
        pad_id: value to pad with

    Returns:
        Aligned 3D tensor

    Examples:
        >>> import torch
        >>> import torch.nn.functional as F
        >>> prompts = torch.tensor([
        ...     [[1, 2, 3],
        ...      [4, 5, 6],
        ...      [7, 8, 9],
        ...      [10, 11, 12]]
        ... ]) # [1,4,3]
        >>> prompt_lens = torch.tensor([2])
        >>> aligned = three_d_align_right(prompts, prompt_lens)
        >>> print(aligned.shape)
        torch.Size([1, 4, 3])
        >>> print(aligned[0]) # Print first batch
        tensor([[0, 0, 0],
                [0, 0, 0],
                [1, 2, 3],
                [4, 5, 6]])
    """
    batch, seq_len, _, device, _ = *t.shape, t.device, t.dtype

    assert lens.ndim == 1 and lens.shape[0] == batch
    assert lens.amax() <= seq_len

    pad_lens = seq_len - lens
    max_pad_len = pad_lens.amax()

    batch_arange = torch.arange(batch, device=device, dtype=torch.long)[..., None]
    prompt_len_arange = torch.arange(seq_len, device=device, dtype=torch.long)

    # Pad along sequence dimension (dim=1) while preserving the hidden dimension
    t = F.pad(t, (0, 0, max_pad_len, 0), value=pad_id)  # Changed padding dimensions
    offset = max_pad_len - pad_lens

    # Add extra dimension to maintain the hidden_dim
    aligned = t[batch_arange, prompt_len_arange + offset[..., None], :]

    return aligned


class NextTokenPredictionMetric(Metric):
    """
    A metric class for calculating AUC and top-n accuracy for next token prediction in language models.

    This metric computes the Area Under the Receiver Operating Characteristic Curve (AUROC) and
    top-n accuracy for each position in the sequence, considering only the next token prediction.

    Attributes:
        vocab_size (int): The size of the vocabulary.
        top_n (tuple): The values of n for which to calculate top-n accuracy.
        auroc (MulticlassAUROC): The AUROC metric for multiclass classification.
        top_n_accuracy (dict): A dictionary of MulticlassAccuracy metrics for each n in top_n.
    """

    def __init__(self, vocab_size: int, top_k_acc: list[int], next_token_auc: bool, dist_sync_on_step=False):
        """
        Initialize the NextTokenPredictionMetric.

        Args:
            vocab_size (int): The size of the vocabulary.
            top_n (tuple): The values of n for which to calculate top-n accuracy. Default is (1, 5, 10).
            dist_sync_on_step (bool): Synchronize metric state across processes at each step. Default is
                False.
        """
        super().__init__(dist_sync_on_step=dist_sync_on_step)
        self.vocab_size = vocab_size

        self.top_k_acc = top_k_acc
        metrics = {
            f"top_{k}_accuracy": MulticlassAccuracy(num_classes=vocab_size, top_k=k) for k in top_k_acc
        }
        if next_token_auc:
            metrics["auroc"] = MulticlassAUROC(num_classes=vocab_size, average="macro", thresholds=100)
        self.next_token_metrics = MetricCollection(metrics)

    def update(self, logits: torch.Tensor, targets: torch.Tensor, mask: torch.Tensor):
        """
        Update the metric state with batch statistics.

        Args:
            logits (torch.Tensor): Predicted logits from the model, shape (batch_size, seq_length,
                vocab_size).
            targets (torch.Tensor): Ground truth labels, shape (batch_size, seq_length).
            mask (torch.Tensor): Mask to ignore padded elements, shape (batch_size,
                seq_length).

        The method shifts the targets to align with the next token prediction and updates AUROC and top-n
            accuracy.
        """

        # Shift targets to align with next token prediction
        shifted_targets = targets[:, 1:]
        shifted_mask = mask[:, :-1]

        # Reshape tensors for metric update
        flat_logits = logits[:, :-1][shifted_mask].view(-1, self.vocab_size)
        flat_targets = shifted_targets[shifted_mask].view(-1)

        # Update metrics
        self.next_token_metrics.update(flat_logits, flat_targets)

    def compute(self):
        """
        Compute the AUROC and top-n accuracy based on accumulated statistics.

        Returns:
            dict: A dictionary containing the computed AUROC and top-n accuracy for each n in top_n.
        """
        results = self.next_token_metrics.compute()
        return results


class HistogramForecastingModule(BaseModule, TimeableMixin, BaseGenerativeModel):
    """EIC token based GPT Forecasting Model.

    This model has three main capabilities:
    1. Autoregressive training (learning to predict next tokens)
    2. Data generation (creating synthetic medical event sequences)
    3. Zero-shot prediction (using generated sequences for prediction)

    Args:
        cfg (DictConfig): Configuration object containing:
            - vocab_size: Size of the vocabulary
            - max_seq_len: Maximum sequence length
            - zero_shot_labeler: Optional function for zero-shot prediction
            - augmented_code_metadata_fp: Path to augmented code metadata file (with H and NTP tokens)

    Examples:
        >>> import tempfile
        >>> from clinical_zeroshot_labeler.labeler import WindowStatus
        >>> # Create test setup using helper function
        >>> trajectory_labeler, metadata_df, batch, _ = create_dummy_sequence_labeler()

        >>> # Write metadata to temporary file and create config
        >>> temp_file = tempfile.NamedTemporaryFile(suffix='.parquet')
        >>> metadata_df.write_parquet(temp_file.name)
        >>> cfg = create_model_config(temp_file.name)

        >>> # Test workflow 1: Autoregressive training
        >>> model = HistogramForecastingModule(cfg)
        >>> loss = model.training_step(batch)
        >>> assert loss.isfinite().all()

        >>> # Test workflow 2: Data generation without labeling
        >>> cfg.generate_id = 1
        >>> model = HistogramForecastingModule(cfg)
        >>> output = model.forward(batch)
        >>> assert GENERATE_PREFIX + '1' in output
        >>> generated_df = output[GENERATE_PREFIX + '1']
        >>> # Check generated data structure
        >>> assert 'time' in generated_df.columns
        >>> assert 'code' in generated_df.columns
        >>> assert 'numeric_value' in generated_df.columns
        >>> assert 'subject_id' in generated_df.columns
        >>> assert 'prediction_time' in generated_df.columns
        >>> # Verify time token generation (code/vocab_index 4 in metadata)
        >>> generated_df.shape[0] > 0
        True

        >>> # Test workflow 3: Generation with zero-shot labeling
        >>> cfg.generate_id = 1
        >>> model = HistogramForecastingModule(cfg)
        >>> model.trajectory_labeler = trajectory_labeler
        >>> output = model.forward(batch)
        >>> # Check labeling output
        >>> assert MODEL_PRED_PROBA_KEY in output
        >>> assert MODEL_PRED_STATUS_KEY in output
        >>> assert output[MODEL_PRED_PROBA_KEY].shape == (2,)  # Binary prediction per sequence
        >>> assert output[MODEL_PRED_STATUS_KEY].shape == (2,)  # Status per sequence
        >>> # Verify status progression works
        >>> status_vals = output[MODEL_PRED_STATUS_KEY]
    """

    def __init__(self, cfg: DictConfig):
        super().__init__(cfg)
        self.code_head = self.cfg.code_head

        num_future_codes = self.cfg.get("num_future_codes", None)
        if num_future_codes is not None:
            logger.info(f"Using {num_future_codes} future codes for forecasting")
        self.train_next_token_metric = NextTokenPredictionMetric(
            self.cfg.vocab_size, self.cfg.top_k_acc, self.cfg.next_token_auc
        )
        self.val_next_token_metric = NextTokenPredictionMetric(
            self.cfg.vocab_size, self.cfg.top_k_acc, self.cfg.next_token_auc
        )
        self.test_next_token_metric = NextTokenPredictionMetric(
            self.cfg.vocab_size, self.cfg.top_k_acc, self.cfg.next_token_auc
        )

        self.metadata_df = pl.read_parquet(self.cfg.augmented_code_metadata_fp)
        self.trajectory_labeler = self.cfg.get("trajectory_labeler", None)
        self.initialize_weights()

        self.h_token = self.metadata_df.filter(pl.col("code") == "[H]")["code/vocab_index"][-1]
        self.ntp_token = self.metadata_df.filter(pl.col("code") == "[NTP]")["code/vocab_index"][-1]

        self.histogram_normalizer = HistogramNormalizer(
            self.h_token, self.ntp_token, self.cfg.vocab_size, self.cfg.n_bits, scale=self.cfg.scale
        )
        histogram_dim = self.histogram_normalizer.get_normalized_size()

        encoder_hidden_dims = [
            self.cfg.token_dim,
            self.cfg.token_dim,
            self.cfg.token_dim,
            self.cfg.token_dim // 2,
        ]
        decoder_hidden_dims = [self.cfg.token_dim, histogram_dim, histogram_dim, histogram_dim]
        self.autoencoder = AutoencoderKL(
            embed_dim=self.cfg.token_dim,
            input_dim=self.cfg.token_dim,
            output_dim=histogram_dim,
            encoder_hidden_dims=encoder_hidden_dims,
            decoder_hidden_dims=decoder_hidden_dims,
            use_variational=True,
            beta=self.cfg.beta,
            warmup_steps=0,
            annealing_steps=0,
        )

    @TimeableMixin.TimeAs
    def get_loss(self, batch):
        return self.get_loss_no_filter(batch)
        code_logits = batch[CODE_LOGITS]
        assert not torch.isnan(code_logits).any(), "code_logits is NaN"

        code_logits = batch[CODE_LOGITS]
        mask = batch["mask"]
        code_target = batch["code"]
        histogram_mask = batch["histogram"] > 0

        # Shift sequences
        shifted_code_target = code_target[:, 1:]  # Remove first token
        shifted_mask = mask[:, 1:]  # Remove first position from mask

        # Apply histogram mask to logits
        filtered_code_logits = code_logits.clone()
        filtered_code_logits[~histogram_mask] = -float("inf")
        # Remove last prediction and transpose for cross_entropy

        filtered_code_logits = filtered_code_logits[:, :-1]

        # Only compute loss on masked positions
        masked_logits = filtered_code_logits[shifted_mask, :]  # Get logits at masked positions
        masked_targets = shifted_code_target[shifted_mask]  # Get targets at masked positions

        # Calculate loss with both masks
        code_loss = F.cross_entropy(
            masked_logits,
            masked_targets,
            ignore_index=0,  # Assuming 0 is your padding index
            reduction="mean",
        )

        assert not torch.isnan(code_loss).any(), "code_loss is NaN"

        return code_loss

    def get_loss_no_filter(self, batch):
        code_logits = batch[CODE_LOGITS]
        assert not torch.isnan(code_logits).any(), "code_logits is NaN"

        code_logits = batch[CODE_LOGITS]
        mask = batch["mask"]
        code_target = batch["code"]

        # Shift sequences
        shifted_code_target = code_target[:, 1:]  # Remove first token
        shifted_mask = mask[:, 1:]  # Remove first position from mask

        # Only compute loss on masked positions
        masked_logits = code_logits[:, :-1][shifted_mask, :]  # Get logits at masked positions
        masked_targets = shifted_code_target[shifted_mask]  # Get targets at masked positions

        # Calculate loss with both masks
        code_loss = F.cross_entropy(
            masked_logits,
            masked_targets,
            ignore_index=0,  # Assuming 0 is your padding index
            reduction="mean",
        )

        assert not torch.isnan(code_loss).any(), "code_loss is NaN"

        return code_loss

    def initialize_weights(self):
        # initialize nn.Linear and nn.LayerNorm
        pass
        # self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, torch.nn.Linear):
            # we use xavier_uniform following official JAX ViT:
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, torch.nn.Linear) and m.bias is not None:
                torch.nn.init.constant_(m.bias, 0)
        elif isinstance(m, torch.nn.LayerNorm):
            if m.bias is not None:
                torch.nn.init.constant_(m.bias, 0)
            if m.weight is not None:
                torch.nn.init.constant_(m.weight, 1.0)

    @TimeableMixin.TimeAs
    def get_forecast_logits(self, model_output):
        if isinstance(model_output, torch.Tensor):
            all_token_embeddings = model_output
        else:
            all_token_embeddings = model_output[BACKBONE_TOKENS_KEY]
        code_logits = self.code_head(all_token_embeddings)
        # histogram_mask = model_output["histogram"] > 0
        # code_logits[~histogram_mask] = -float("inf")

        return {
            CODE_LOGITS: code_logits,
        }

    @TimeableMixin.TimeAs
    def get_histogram_loss(self, prompts, histogram, embeddings, mask):
        # All inputs except the last we can evaluate
        prompts = prompts[:, :-1]  # ignore last h token
        embeddings = embeddings[:, :-1]  # ignore last h token

        # Ground truth histogram is shifted by one, as we are predicting the next histogram
        histogram = histogram[:, 1:]
        mask = mask[:, 1:]  # last unmasked token has invalid histogram so mask it

        h_mask = (prompts == self.h_token) & mask
        patch_embeddings = embeddings[h_mask]
        num_histogram_samples = h_mask.sum()

        # Setup target -- histogram + counts
        target = histogram[h_mask, :]

        with torch.no_grad():
            normalized_gt_histogram = self.histogram_normalizer.transform(target)
        # Forward pass and loss computation
        loss_dict = self.autoencoder.forward(
            inputs=patch_embeddings.reshape(num_histogram_samples, -1).repeat(
                self.cfg.histogram_batch_mul, 1
            ),
            outputs=normalized_gt_histogram.detach().repeat(self.cfg.histogram_batch_mul, 1),
        )
        loss = loss_dict["vae_loss"]
        loss_dict = {"MODEL//" + k: v for k, v in loss_dict.items() if k != "vae_reconstruction"}
        assert not torch.isnan(loss).any(), "histogram loss is NaN"
        return loss, loss_dict

    @TimeableMixin.TimeAs
    def forward(self, batch, keep_code_logits=False):
        batch = self.input_encoder(batch)
        model_output = self.model(batch, do_get_last_token=False)
        embeddings = model_output[BACKBONE_EMBEDDINGS_KEY]

        if self.cfg.return_tokens:
            batch[MODEL_TOKENS_KEY] = model_output[BACKBONE_TOKENS_KEY]
        batch[MODEL_EMBEDDINGS_KEY] = model_output[BACKBONE_EMBEDDINGS_KEY]
        forecast = self.get_forecast_logits(model_output)
        if self.cfg.return_logits:
            batch[MODEL_LOGITS_SEQUENCE_KEY] = forecast[CODE_LOGITS]
        batch[CODE_LOGITS] = forecast[CODE_LOGITS]

        code_loss = self.get_loss(batch)
        histogram_loss, loss_dict = self.get_histogram_loss(
            batch["code"], batch["histogram"], embeddings, batch["mask"]
        )
        model_loss = code_loss + histogram_loss

        loss_dict["MODEL//code_loss"] = code_loss
        batch.update(loss_dict)

        batch[MODEL_LOSS_KEY] = model_loss
        batch[MODEL_BATCH_LOSS_KEY] = model_loss
        batch = self._generate(batch)

        if not keep_code_logits:
            del batch[CODE_LOGITS]
        return batch

    def _log(self, batch, split):
        on_step = split == "train"
        for loss_key in MODEL_LOSS_KEYS + [MODEL_LOSS_KEY]:
            loss_name = "/" + loss_key.split("/")[-1].lower()
            self.log(
                split + loss_name,
                batch[loss_key],
                on_step=on_step,
                on_epoch=True,
                prog_bar=True,
                logger=True,
            )
        if split == "train":
            self.train_next_token_metric.update(batch[CODE_LOGITS], batch["code"], batch["mask"])
        elif split == "val":
            self.val_next_token_metric.update(batch[CODE_LOGITS], batch["code"], batch["mask"])
        elif split == "test":
            self.test_next_token_metric.update(batch[CODE_LOGITS], batch["code"], batch["mask"])
        else:
            raise ValueError(f"Invalid split: {split}")

    def _generate(self, batch):
        if self.cfg.generate_id is not None:
            return self.generate_batch(batch)
        else:
            return batch

    def training_step(self, batch):
        batch = self(batch, True)
        assert not torch.isnan(batch[MODEL_BATCH_LOSS_KEY]), "Loss is NaN"
        self._log(batch, "train")
        del batch[CODE_LOGITS]
        return batch[MODEL_BATCH_LOSS_KEY]

    def on_train_epoch_end(self):
        next_token_results = self.train_next_token_metric.compute()
        for metric_name, value in next_token_results.items():
            self.log(f"test/NEXT_TOKEN/{metric_name.upper()}", value, on_epoch=True)
        self.train_next_token_metric.reset()

    def validation_step(self, batch):
        batch = self(batch, True)
        assert not torch.isnan(batch[MODEL_BATCH_LOSS_KEY]), "Loss is NaN"
        self._log(batch, "val")
        del batch[CODE_LOGITS]
        return batch[MODEL_BATCH_LOSS_KEY]

    def on_validation_epoch_end(self):
        next_token_results = self.val_next_token_metric.compute()
        for metric_name, value in next_token_results.items():
            self.log(f"test/NEXT_TOKEN/{metric_name.upper()}", value, on_epoch=True)
        self.val_next_token_metric.reset()

    def test_step(self, batch):
        batch = self(batch, True)
        assert not torch.isnan(batch[MODEL_BATCH_LOSS_KEY]), "Loss is NaN"
        self._log(batch, "test")
        del batch[CODE_LOGITS]
        loss = batch[MODEL_BATCH_LOSS_KEY]
        return loss

    def on_test_epoch_end(self):
        next_token_results = self.test_next_token_metric.compute()
        for metric_name, value in next_token_results.items():
            self.log(f"test/NEXT_TOKEN/{metric_name.upper()}", value, on_epoch=True)
        self.test_next_token_metric.reset()

    @classmethod
    def get_metadata_means(cls, metadata_df):
        if "values/sum" not in metadata_df or "values/n_occurrences" not in metadata_df:
            raise ValueError("Missing 'values/sum' and/or 'values/n_occurrences' columns in metadata_df")
        metadata_df = metadata_df.with_columns(
            (pl.col("values/sum") / pl.col("values/n_occurrences")).alias("values/mean")
        )
        return metadata_df

    @classmethod
    def get_code_to_time_map(cls, metadata_df) -> dict:
        """Convert the metadata DataFrame to a dictionary mapping code to time.

        Args:
            metadata_df: Polars DataFrame containing code metadata
                (includes 'code' and 'code/vocab_index' columns)

        Returns:
            dict: Mapping code to time in years

        Example:
        >>> metadata_df = pl.DataFrame({
        ...     "code": ["A", "B", "C", "TIME//DELTA//TOKEN//_Q_17"],
        ...     "code/vocab_index": [0, 1, 2, 3],
        ...     "values/sum": [None, None, None, 1],
        ...     "values/n_occurrences": [None, None, None, 1],
        ... })
        >>> HistogramForecastingModule.get_code_to_time_map(metadata_df)
        tensor([0., 0., 0., 1., 0.])
        """
        metadata_df = cls.get_metadata_means(metadata_df)
        # Assuming we know the vocab size
        num_vocab = metadata_df["code/vocab_index"].max()
        code_to_time_map = torch.zeros(num_vocab + 2)  # +2 since indices start at 1 and EOS token is added

        # Set values using the indices
        time_mask = pl.col("code").str.starts_with(TIME_DELTA_TOKEN)
        vocab_indices = metadata_df.filter(time_mask)["code/vocab_index"]
        time_values = metadata_df.filter(time_mask)["values/mean"]

        code_to_time_map[vocab_indices.to_list()] = time_values.to_torch().to(code_to_time_map.dtype)
        return code_to_time_map

    @classmethod
    def get_code_to_numeric_value_map(cls, metadata_df, get_raw_values=True) -> dict:
        """Convert the metadata DataFrame to a dictionary mapping code to numeric value.

        Args:
            metadata_df: Polars DataFrame containing code metadata
                (includes 'code' and 'code/vocab_index' columns)

        Returns:
            dict: Mapping code to time in years

        Example:
        >>> metadata_df = pl.DataFrame({
        ...     "code": ["A", "A//_Q_1", "A//_Q_2", "A//_Q_3", "A//_Q_4", "B"],
        ...     "code/vocab_index": [0, 1, 2, 3, 4, 5],
        ...     'values/min': [0, 0, 0, 0, 0, None],
        ...     'values/max': [4, 4, 4, 4, 4, None],
        ...     'values/sum': [None, .5, 1.5, 2.5, 3.5, None],
        ...     'values/n_occurrences': [None, 1, 1, 1, 1, None],
        ...     "values/quantiles": [
        ...         {'values/quantile/0.25': 1, 'values/quantile/0.5': 2, 'values/quantile/0.75': 3},
        ...         {'values/quantile/0.25': 1, 'values/quantile/0.5': 2, 'values/quantile/0.75': 3},
        ...         {'values/quantile/0.25': 1, 'values/quantile/0.5': 2, 'values/quantile/0.75': 3},
        ...         {'values/quantile/0.25': 1, 'values/quantile/0.5': 2, 'values/quantile/0.75': 3},
        ...         {'values/quantile/0.25': 1, 'values/quantile/0.5': 2, 'values/quantile/0.75': 3},
        ...         {'values/quantile/0.25': None, 'values/quantile/0.5': None,
        ...          'values/quantile/0.75': None},
        ...     ],
        ... })
        >>> HistogramForecastingModule.get_code_to_numeric_value_map(
        ...     metadata_df, get_raw_values=True).tolist()
        [nan, 0.5, 1.5, 2.5, 3.5, nan, nan]
        >>> HistogramForecastingModule.get_code_to_numeric_value_map(
        ...     metadata_df, get_raw_values=False).tolist()
        [nan, 0.125, 0.375, 0.625, 0.875, nan, nan]
        """
        # First, verify the input DataFrame is sorted by vocab_index
        assert metadata_df["code/vocab_index"].is_sorted()

        # Get the maximum vocab index to determine tensor size
        max_vocab_idx = metadata_df["code/vocab_index"].max()

        # Create a tensor filled with NaN values
        result = torch.full((max_vocab_idx + 1,), float("nan"))
        # TODO(Oufattole) remove this and enforce that metadata_df must include the values/min
        ordered_quantiles = [field.name for field in metadata_df.schema["values/quantiles"].fields]
        percentiles = [0, *[float(q.split("/")[-1]) for q in ordered_quantiles], 1]
        if "values/min" not in metadata_df.columns or "values/max" not in metadata_df.columns:
            raise ValueError("Missing values/min and/or values/max values in metadata_df")
        metadata_df = cls.get_metadata_means(metadata_df)

        # Process each row in the DataFrame
        for row in metadata_df.iter_rows(named=True):
            vocab_idx = row["code/vocab_index"]
            code = row["code"]
            if row["values/quantiles"] is None:  # Handle case with single None quantile
                raw_quantiles = [None]
            else:
                raw_quantiles = [row["values/quantiles"][each] for each in ordered_quantiles]
            min_value = row["values/min"]
            max_value = row["values/max"]
            raw_quantiles = [min_value, *raw_quantiles, max_value]
            mean_value = row["values/mean"]

            # Check if this is a quarterly code (contains "//_Q_")
            if code and "//_Q_" in code and not code.startswith("TIME//DELTA//TOKEN"):
                # Extract the number of quantiles the value is greater than, 0 for Q_1, 1 for Q_2, etc.
                rank = int(code.split("//_Q_")[1]) - 1
                # We estimate the numeric value is the average of the bordering quantiles it is between
                if get_raw_values:
                    result[vocab_idx] = torch.tensor(mean_value, dtype=result.dtype)
                else:
                    result[vocab_idx] = sum([percentiles[rank], percentiles[rank + 1]]) / 2

            # For non-quarterly codes, leave as NaN
            # This handles both the base code (e.g., "A") and any other non-quarterly codes
        return torch.cat([result, torch.Tensor([np.nan])])  # postpend a zero in case EOS token is postpended

    @classmethod
    def to_trajectory_batch(
        cls,
        code,
        mask,
        metadata_df,
        prediction_time_offset_years: torch.Tensor,
        code_to_time_map: torch.Tensor = None,
        code_to_numeric_value_map: torch.Tensor = None,
    ) -> TrajectoryBatch:
        """Convert the model output to MEDS format.

        Args:
            code (torch.Tensor): Tensor of shape (batch_size, sequence_length) containing event codes
            mask (torch.Tensor): Tensor of shape (batch_size, sequence_length) indicates valid
                measurements/codes
            metadata_df: Polars DataFrame containing code metadata (includes 'code' column)
            prediction_time_offset_days: Tensor of shape (batch_size,) containing the time difference in days
                between each input sequence's end time and its target prediction time. Used to calculate
                absolute timestamps since the TrajectoryBatch stores times relative to the prediction time.

        Returns:
            pl.DataFrame: MEDS format DataFrame with columns:
                - time_index: Time in years starting from 0
                - code: The medical code
                - value: Always 1.0 (presence indicator)
                - sample_id: ID of the generated sample

        Time will start from 0, and is measured in years.

        Example:
        >>> from datetime import datetime
        >>> metadata_df = pl.DataFrame({
        ...     "code": ["A", "A//_Q_1", "A//_Q_2", "A//_Q_3", "A//_Q_4", "TIME//DELTA//TOKEN//_Q_17"],
        ...     "code/vocab_index": [0, 1, 2, 3, 4, 5],
        ...     'values/min': [0, 0, 0, 0, 0, None],
        ...     'values/max': [4, 4, 4, 4, 4, None],
        ...     'values/sum': [None, .5, 1.5, 2.5, 3.5, 1],
        ...     'values/n_occurrences': [None, 1, 1, 1, 1, 1],
        ...     "values/quantiles": [
        ...         {'values/quantile/0.25': 1, 'values/quantile/0.5': 2, 'values/quantile/0.75': 3},
        ...         {'values/quantile/0.25': 1, 'values/quantile/0.5': 2, 'values/quantile/0.75': 3},
        ...         {'values/quantile/0.25': 1, 'values/quantile/0.5': 2, 'values/quantile/0.75': 3},
        ...         {'values/quantile/0.25': 1, 'values/quantile/0.5': 2, 'values/quantile/0.75': 3},
        ...         {'values/quantile/0.25': 1, 'values/quantile/0.5': 2, 'values/quantile/0.75': 3},
        ...         {'values/quantile/0.25': None, 'values/quantile/0.5': None,
        ...          'values/quantile/0.75': None},
        ...     ],
        ... })
        >>> code = torch.tensor([[0, 2, 5, 5], [2, 3, 4, 5], [5, 5, 0, 1]])
        >>> mask = torch.tensor([[1, 1, 1, 1], [1, 1, 1, 0], [1, 1, 1, 0]])
        >>> prediction_time_offset_years = torch.tensor([0.0, 1.0, 2.0])
        >>> from pprint import pprint, pformat
        >>> subject_ids = [1,2,3]
        >>> prediction_times = [1,2,3]
        >>> HistogramForecastingModule.to_trajectory_batch(code, mask, metadata_df,
        ...     prediction_time_offset_years).to_meds(subject_ids, prediction_times).columns
        ['subject_id', 'prediction_time', 'time', 'code', 'code/vocab_index', 'numeric_value']
        """
        if not code_to_time_map:
            code_to_time_map = cls.get_code_to_time_map(metadata_df)
        if not code_to_numeric_value_map:
            code_to_numeric_value_map = cls.get_code_to_numeric_value_map(metadata_df)
        # Initialize lists to store the DataFrame rows
        time = torch.cumsum(code_to_time_map[code], dim=1)
        numeric_value = code_to_numeric_value_map[code]
        numeric_value_mask = ~numeric_value.isnan()
        time += prediction_time_offset_years.unsqueeze(1)
        return TrajectoryBatch(time, code, mask, numeric_value, numeric_value_mask, metadata_df)

    def update_generation_state(
        self,
        tokens: torch.Tensor,
        cumulative_time: torch.Tensor,
        trajectory_labeler: SequenceLabeler | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, bool]:
        """Updates trajectory_labeler state, and returns state information.

        Examples:
            >>> import tempfile
            >>> from clinical_zeroshot_labeler.labeler import WindowStatus
            >>> # Create test setup using helper function
            >>> _, metadata_df, _, _ = create_dummy_sequence_labeler()

            >>> # Write metadata to temporary file and create config
            >>> temp_file = tempfile.NamedTemporaryFile(suffix='.parquet')
            >>> metadata_df.write_parquet(temp_file.name)
            >>> cfg = create_model_config(temp_file.name)

            >>> model = HistogramForecastingModule(cfg)
            >>> model._init_time_and_value_quantiles()
            >>> B = 2  # batch_size
            >>> device = 'cpu'

            >>> # Setup basic test case
            >>> cumulative = torch.tensor([0.0, 0.0], device=device)
            >>> tokens = torch.randint(0, 5, (B,3), device=device)

            >>> # Test trajectory labeler progression
            >>> labeler = DummyTrajectoryLabeler(B)
            >>> time, status, is_finished, ended = model.update_generation_state(
            ...     tokens=tokens,
            ...     cumulative_time=cumulative,
            ...     trajectory_labeler=labeler,
            ... )
            >>> assert time.shape == (B,)
            >>> assert status.shape == (B,)
            >>> assert not is_finished
            >>> assert not ended.any()

            # >>> # Test second step shows active status
            # >>> time, status, is_finished, ended = model.update_generation_state(
            # ...     tokens=tokens,
            # ...     cumulative_time=time,
            # ...     trajectory_labeler=labeler,
            # ... )
            # >>> assert (status == WindowStatus.ACTIVE.value).all()
            # >>> assert not is_finished
            # >>> assert not ended.any()

            # >>> # Test third step shows satisfied status and finished
            # >>> time, status, is_finished, ended = model.update_generation_state(
            # ...     tokens=tokens,
            # ...     cumulative_time=time,
            # ...     trajectory_labeler=labeler,
            # ... )
            # >>> assert (status == WindowStatus.SATISFIED.value).all()
            # >>> assert is_finished
            # >>> assert ended.all()

            # >>> # Test without trajectory labeler
            # >>> time, status, is_finished, ended = model.update_generation_state(
            # ...     tokens=tokens,
            # ...     cumulative_time=time,
            # ...     trajectory_labeler=None,
            # ... )
            # >>> assert time.shape == (B,)
            # >>> assert status is None
            # >>> assert not is_finished
            # >>> assert not ended.any()
        """
        current_sample = tokens[:, -1].cpu()
        pred_time = self.time_quantile_map[current_sample.flatten()]
        cumulative_time = cumulative_time.cpu() + pred_time.squeeze(-1)
        current_value = self.value_quantile_map[current_sample.flatten()]
        if trajectory_labeler is not None:
            status = trajectory_labeler.process_step(current_sample, cumulative_time, current_value)
            is_finished = trajectory_labeler.is_finished()
            ended_sequences = torch.logical_or(
                status == WindowStatus.SATISFIED.value, status == WindowStatus.IMPOSSIBLE.value
            )
        else:
            status = None
            is_finished = False
            ended_sequences = torch.zeros((current_sample.shape[0]), dtype=torch.bool)
        return cumulative_time, status, is_finished, ended_sequences

    def _init_time_and_value_quantiles(self):
        if not hasattr(self, "time_quantile_map"):
            self.time_quantile_map = self.get_code_to_time_map(self.metadata_df)
        if not hasattr(self, "value_quantile_map"):
            self.value_quantile_map = self.get_code_to_numeric_value_map(self.metadata_df)

    @torch.inference_mode()
    @eval_decorator
    @TimeableMixin.TimeAs
    def generate_batch(
        self,
        input_batch,
        **kwargs,
    ):
        """Generate evaluation metrics for the model."""
        if self.cfg.max_tokens_budget is None and self.trajectory_labeler is None:
            raise ValueError(
                "At least one of model.backbone.max_tokens_budget or model.trajectory_labeler must be "
                "set in the configuration."
            )
        if not self.cfg.backbone.cfg.get("token_emb", None):
            raise NotImplementedError(
                "Manual token embeddings should be used as we jointly embed histograms and codes"
            )
        else:
            prompts, mask = input_batch[INPUT_ENCODER_TOKENS_KEY], input_batch[INPUT_ENCODER_MASK_KEY]

        self._init_time_and_value_quantiles()

        if "prediction_time" not in input_batch or "end_time" not in input_batch:
            raise ValueError(
                "Prediction time and end time must be provided for zero-shot labeling. "
                "Enable the flags do_include_prediction_time and do_include_end_time."
            )
        prediction_time_offset_years = (
            -get_time_days_delta(input_batch["prediction_time"], input_batch["end_time"], prompts.device)
            / 365.25
        )
        if (prediction_time_offset_years > 0).any():
            raise ValueError("time_offset_years must be less than or equal to 0")

        if self.cfg.generate_id is not None:
            trajectory_labeler = (
                self.trajectory_labeler(batch_size=prompts.shape[0], metadata_df=self.metadata_df)
                if self.trajectory_labeler is not None
                else None
            )
            out, out_lengths, metadata = self.generate(
                prompts=prompts,
                mask=mask,
                trajectory_labeler=trajectory_labeler,
                time_offset_years=prediction_time_offset_years,
                temperature=self.cfg.temperature,
                eos_tokens=self.cfg.eos_tokens,
                log_progress=self.cfg.get("log_progress", False),
                prune_terminated=self.cfg.prune_terminated,
                histogram=input_batch["histogram"],
                code=input_batch["code"],
                **kwargs,
            )
            out_mask = torch.arange(out.size(1))[None, :].cpu() < out_lengths[:, None].cpu()

            # Store generated data
            null_data = torch.zeros_like(out).cpu()
            # Convert codes to time deltas
            time_deltas = self.time_quantile_map.to(out.device)[out]
            generated_data = {
                "code": out.cpu(),
                "mask": out_mask,
                "numeric_value": null_data,
                "numeric_value_mask": null_data,
                "static_mask": null_data,
                "time_delta_years": time_deltas.cpu(),
                "subject_id": input_batch["subject_id"].cpu(),
                "prediction_time": input_batch["prediction_time"],
                "end_time": input_batch["end_time"],
            }
            trajectory_batch = self.to_trajectory_batch(
                generated_data["code"],
                generated_data["mask"],
                self.metadata_df,
                prediction_time_offset_years.cpu(),
            )
            if self.cfg.store_generated_trajectory:
                input_batch[GENERATE_PREFIX + str(self.cfg.generate_id)] = trajectory_batch.to_meds(
                    generated_data["prediction_time"], generated_data["subject_id"]
                )
            logger.info(f"Completed generation for sample {self.cfg.generate_id}")

            if metadata:
                labels, status = metadata["labels"], metadata["status"]
                input_batch[MODEL_PREFIX + "STATUS"] = status
                unknown = status != WindowStatus.SATISFIED.value
                # Handle unknown values by setting their probability to 0.5
                if unknown.any().item() > 0:
                    logger.warning(f"Found {unknown.sum().item()} unknown zero-shot predictions")
                    labels[unknown] = 0.5
                input_batch[MODEL_PRED_PROBA_KEY] = labels
                input_batch[MODEL_PRED_STATUS_KEY] = status
                input_batch["MODEL//OFFSET"] = prediction_time_offset_years
                logger.info(f"Completed zero-shot labeling for sample {self.cfg.generate_id}")
            if trajectory_labeler is not None and self.cfg.return_labeler:
                input_batch["labeler"] = trajectory_labeler
        return input_batch

    @torch.inference_mode()
    @eval_decorator
    def generate(
        self,
        prompts: torch.Tensor,
        mask: torch.Tensor | None,
        trajectory_labeler: SequenceLabeler | None = None,
        time_offset_years: torch.Tensor | None = None,
        eos_tokens: list[int] | None = None,
        temperature: float = 1.0,
        filter_logits_fn: str | Callable = identity,
        filter_kwargs: dict = dict(),
        cache_kv: bool = True,
        pad_value: int = 0,
        log_progress: bool = False,
        prune_terminated: bool = False,
        histogram: torch.Tensor | None = None,
        code: torch.Tensor | None = None,
        **kwargs,
    ) -> tuple[torch.Tensor, torch.Tensor, dict | None]:
        """Generate token sequences with model-specific processing.

        This implements the core generation loop while delegating token
        processing to subclass implementations.

        Args:
            prompts: Input token sequences [batch_size, seq_len]
            mask: Optional attention mask for prompts
            trajectory_labeler: Optional labeler for monitoring conditions
            time_offset_years: Optional time offsets per sequence
            eos_tokens: Optional tokens that should end generation
            temperature: Sampling temperature
            filter_logits_fn: Optional logits filtering function
            filter_kwargs: Additional args for filtering
            cache_kv: Whether to use KV caching
            pad_value: Value to use for padding
            **kwargs: Additional arguments passed to model

        Returns:
            Tuple containing:
                - Generated token sequences
                - Sequence lengths
                - Optional metadata dict

        Examples:
            # >>> # Basic generation test
            # >>> model = TestModel()
            # >>> prompts = torch.randint(0, 5, (2, 3))  # batch_size=2, seq_len=3
            # >>> mask = torch.ones((2, 3), dtype=torch.bool)

            # >>> tokens, lengths, meta = model.generate(prompts, mask, temperature=1.0)
            # >>> assert tokens.shape[1] <= model.cfg.max_tokens_budget  # Respects budget
            # >>> assert lengths.shape == (2,)  # Batch size preserved

            # >>> # Test with trajectory labeler
            # >>> labeler = DummyLabeler()
            # >>> tokens, lengths, meta = model.generate(
            # ...     prompts,
            # ...     mask,
            # ...     trajectory_labeler=labeler,
            # ...     temperature=1.0
            # ... )
            # >>> assert meta is not None  # Metadata returned with labeler
            # >>> assert "labels" in meta
            # >>> assert "status" in meta

            # >>> # Test with EOS token
            # >>> tokens, lengths, meta = model.generate(
            # ...     prompts,
            # ...     mask,
            # ...     eos_tokens=[4],  # Use token 4 as EOS
            # ...     temperature=0.0  # Greedy sampling
            # ... )
            # >>> assert tokens.shape[1] <= model.cfg.max_tokens_budget

            # >>> # Test with time offset
            # >>> time_offset = torch.tensor([1.0, 2.0])
            # >>> tokens, lengths, meta = model.generate(
            # ...     prompts,
            # ...     mask,
            # ...     time_offset_years=time_offset,
            # ...     temperature=1.0
            # ... )
            # >>> assert tokens.shape[1] <= model.cfg.max_tokens_budget
        """
        assert code is not None and histogram is not None, "code and histogram must be provided"
        transformer_decoder = self.model.model
        max_seq_len = transformer_decoder.max_seq_len

        # prompts, ps = pack([prompts], "* n")
        b, t, token_dim = prompts.shape

        # Handle filter logits fn given as string
        if isinstance(filter_logits_fn, str):
            assert filter_logits_fn in FILTER_LOGITS_FN, f"only {join(FILTER_LOGITS_FN.keys())} are available"
            filter_logits_fn = FILTER_LOGITS_FN[filter_logits_fn]

        # Align prompts
        prompt_lens = mask.sum(dim=-1).view(-1)
        self._check_valid_mask(mask, prompt_lens)
        prompts = three_d_align_right(prompts, prompt_lens, pad_id=pad_value)
        code = align_right(code, prompt_lens, pad_id=pad_value)
        histogram = three_d_align_right(histogram, prompt_lens, pad_id=pad_value)

        seq_start_pos = t - prompt_lens

        if exists(eos_tokens):
            eos_tokens = torch.tensor(eos_tokens)

        # Initialize state
        out = prompts
        cache = None
        cumulative_time = (
            time_offset_years if time_offset_years is not None else torch.zeros(b, device=prompts.device)
        )
        ended_sequences = torch.zeros(b, dtype=torch.bool)
        is_finished = False
        num_generated_tokens = 0
        out_lengths = torch.zeros(b, dtype=torch.int32)
        metadata = None
        status = None

        log_progress = False
        progress = (
            Progress(
                TextColumn("[progress.description]{task.description} {task.completed}"),
                BarColumn(),
                TaskProgressColumn(),
                RateColumn(),  # Shows speed of updates
                TimeRemainingColumn(),
                transient=True,
            )
            if log_progress
            else nullcontext()
        )
        assert histogram is not None
        prev_histogram = histogram[:, -1]
        with progress:
            if log_progress:  # pragma: no cover
                tokens_task = progress.add_task(
                    "[cyan]Tokens Generated...",  # Static description
                    total=self.cfg.max_tokens_budget if self.cfg.max_tokens_budget is not None else None,
                )
                sequences_task = progress.add_task(
                    f"[green]Trajectories ({b} total)...",  # Static description with total
                    total=b,
                )

            while not is_finished:
                # Track sliding window for full batch
                x_full, cache_full, current_start_pos_full = self._track_sliding_window_generation(
                    out, max_seq_len, cache_kv, cache, transformer_decoder, seq_start_pos
                )
                if prune_terminated:
                    # Get indices relative to original batch
                    orig_indices = (~ended_sequences).nonzero().squeeze(-1)
                    orig_indices = orig_indices.flatten()  # Handle single active sequence

                    # Track which indices are active in our currently sliced tensors
                    active_indices = torch.arange(len(orig_indices), device=orig_indices.device)

                    # Select active sequences and their cache
                    x = x_full[orig_indices]  # Use orig_indices for first slice from full batch
                    current_start_pos = (
                        current_start_pos_full[orig_indices] if current_start_pos_full is not None else None
                    )
                    # Use active_indices for cache since it's already sliced
                    cache = slice_cache(cache_full, active_indices) if cache_full is not None else None
                else:
                    active_indices = torch.arange(b, device=prompts.device)
                    orig_indices = active_indices
                    cache = cache_full
                    current_start_pos = current_start_pos_full
                    x = x_full

                # Get next token predictions for active sequences only
                (logits, embeddings), new_cache = transformer_decoder(
                    x,
                    return_logits_and_embeddings=True,
                    return_intermediates=True,
                    cache=cache,
                    seq_start_pos=current_start_pos,
                    **kwargs,
                )

                # Map logits back to full batch size, used when pruning terminated trajectories
                logits = logits[:, -1]
                full_logits = torch.zeros((b, logits.shape[1]), device=logits.device, dtype=logits.dtype)
                full_logits[orig_indices[active_indices]] = logits
                logits = full_logits
                # mask logits given the prev_histogram
                logits[~(prev_histogram > 0)] = -float("inf")

                embeddings = embeddings[:, -1, :]
                full_embeddings = torch.zeros(
                    (b, embeddings.shape[1]), device=embeddings.device, dtype=embeddings.dtype
                )
                full_embeddings[orig_indices[active_indices]] = embeddings
                embeddings = full_embeddings

                next_histogram_posterior = self.autoencoder.encode(embeddings)
                next_ae_histogram_binary = self.autoencoder.decode(next_histogram_posterior.sample())
                next_ae_histogram, count = self.histogram_normalizer.reverse_transform(
                    next_ae_histogram_binary
                )

                # Update cache with pruned version
                if cache_kv and transformer_decoder.can_cache_kv:
                    cache = new_cache
                # Sample next tokens
                if temperature == 0.0:  # greedy sampling
                    raise ValueError("Not supported at the moment.")
                    # sample = logits.argmax(dim=-1, keepdim=True)
                else:
                    filtered_logits = filter_logits_fn(logits, **filter_kwargs)
                    # Apply the same masking logic for temperature sampling
                    mask = torch.ones_like(filtered_logits, dtype=torch.bool)
                    mask[..., self.h_token] = False
                    mask[..., self.ntp_token] = False

                    logits_finite = filtered_logits.isfinite()
                    has_h_token = logits_finite[:, self.h_token]
                    has_o_token = logits_finite[:, self.ntp_token]
                    num_finite_logits = logits_finite.sum(dim=-1)
                    is_histogram_only_h_o_tokens = num_finite_logits <= 2

                    mask[is_histogram_only_h_o_tokens.squeeze(-1) & has_h_token, self.h_token] = True
                    can_sample_o = is_histogram_only_h_o_tokens & ~has_h_token & has_o_token
                    mask[can_sample_o.squeeze(-1), self.ntp_token] = True

                    filtered_logits = filtered_logits.masked_fill(~mask, float("-inf"))
                    probs = F.softmax(filtered_logits / temperature, dim=-1)
                    sample = torch.multinomial(probs, 1)

                one_hot_sample = torch.zeros_like(prev_histogram).scatter_(1, sample, 1)
                next_decrement_histogram = prev_histogram - one_hot_sample

                h_token_histogram_mask = (
                    (code[:, -1] == self.h_token).reshape(-1, 1).repeat(1, next_ae_histogram.shape[1])
                )
                next_histogram = torch.where(
                    h_token_histogram_mask, next_ae_histogram, next_decrement_histogram
                )
                if (next_histogram == 0).all():
                    raise ValueError("All histogram counts are zero somehow, this should not happen.")

                prev_histogram = next_histogram

                # Update generation state
                num_generated_tokens += 1
                if log_progress:  # pragma: no cover
                    progress.update(
                        tokens_task,
                        advance=1,
                    )

                    completed_sequences = ended_sequences.int().sum().item()
                    progress.update(
                        sequences_task,
                        completed=completed_sequences,
                    )

                # Append new tokens
                code = torch.cat((code, sample), dim=-1)
                histogram = torch.cat((histogram, next_histogram.unsqueeze(1)), dim=1)
                next_sample_embedding = self.input_encoder.process_sample(sample, next_histogram.unsqueeze(1))
                out = torch.cat((out, next_sample_embedding), dim=1)

                # Update cumulative time and check status
                cumulative_time, status, is_finished, new_ended_sequences = self.update_generation_state(
                    code,
                    cumulative_time,
                    trajectory_labeler,
                )

                # Update sequence end flags
                new_ended_sequences |= ended_sequences
                if exists(eos_tokens):
                    new_ended_sequences |= sample.flatten().cpu() == eos_tokens
                out_lengths[new_ended_sequences != ended_sequences] = num_generated_tokens
                ended_sequences = new_ended_sequences
                # Check max token budget condition
                if (
                    self.cfg.max_tokens_budget is not None
                    and num_generated_tokens >= self.cfg.max_tokens_budget
                ):
                    is_finished = True
                    out_lengths[~ended_sequences] = num_generated_tokens

                if ended_sequences.all():
                    is_finished = True

        # Get final metadata if using labeler
        if status is not None:
            metadata = dict(labels=trajectory_labeler.get_labels(), status=status)

        # Process final sequences
        code = code[:, t:]

        return code, out_lengths, metadata

    def get_sample(self, batch, temperature=1.0):
        """Get next token logits and histogram prediction for a single step.

        Args:
            batch: Dictionary containing:
                - code: Token indices [batch_size, seq_len]
                - histogram: Token histograms [batch_size, seq_len, vocab_size]
                - mask: Attention mask [batch_size, seq_len]

        Returns:
            tuple: (next_token_logits, next_histogram)
                - next_token_logits: Logits for next token prediction [batch_size, vocab_size]
                - next_histogram: Predicted histogram [batch_size, vocab_size]
        """
        # Encode input
        batch = self.input_encoder(batch)
        tokens = batch[INPUT_ENCODER_TOKENS_KEY]

        # Get model predictions
        transformer_decoder = self.model.model
        (logits, embeddings), _ = transformer_decoder(
            tokens, return_logits_and_embeddings=True, return_intermediates=True
        )

        # Get last token predictions
        next_token_logits = logits[:, -1]
        last_embeddings = embeddings[:, -1]

        # Get histogram prediction using diffusion
        latent_next_histogram_posterior = self.autoencoder.encode(last_embeddings)
        next_histogram, counts = self.histogram_normalizer.reverse_transform(
            self.autoencoder.decode(latent_next_histogram_posterior.sample())
        )

        # Mask token logits based on histogram
        prev_histogram = batch["histogram"][:, -1]
        next_token_logits[~(prev_histogram > 0)] = -float("inf")

        return next_token_logits, next_histogram, latent_next_histogram_posterior, counts, last_embeddings
