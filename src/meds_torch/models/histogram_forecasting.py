from collections.abc import Callable, Sequence
from contextlib import nullcontext

import faiss
import matplotlib.pyplot as plt
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
from torchmetrics import Metric
from torchmetrics.utilities.plot import _AX_TYPE, _PLOT_OUT_TYPE
from torchvision.ops import MLP
from transformers import GPTNeoXForCausalLM
from x_transformers import Decoder, TransformerWrapper
from x_transformers.autoregressive_wrapper import (
    FILTER_LOGITS_FN,
    align_right,
    exists,
    identity,
    join,
)

from meds_torch.data.components.histogram_pytorch_dataset import SubvocabMapper
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
from meds_torch.models.eic_forecasting import NextTokenPredictionMetric
from meds_torch.utils import TIME_DELTA_TOKEN

MODEL_LOSS_KEYS = [
    "MODEL//code_loss",
    "MODEL//vae_loss",
    "MODEL//vae_rec_loss",
    "MODEL//vae_aux_loss",
    "MODEL//vae_true_lp",
    "MODEL//vae_sample_lp",
    "MODEL//vae_kl_loss",
    "MODEL//vae_clip_loss",
    "MODEL//vae_isolated_rec_loss",
    "MODEL//diffusion_loss",
]


class HistogramMetric(Metric):
    """
    Accumulates histograms (true, mean, sample) and, when computed, produces three plots and
    returns the summed Mean Absolute Error (MAE) across all categories (using the mean predictions).

    The plot() method follows the torchmetrics v1.0.0 plotting API.
    """

    # Optional attributes for the internal _plot method (not used here because we need custom plots)
    plot_lower_bound: float | None = None
    plot_upper_bound: float | None = None

    def __init__(self, dist_sync_on_step: bool = False):
        super().__init__(dist_sync_on_step=dist_sync_on_step)
        self.add_state("true_hist", default=[], dist_reduce_fx=None)
        self.add_state("mean_hist", default=[], dist_reduce_fx=None)
        self.add_state("sample_hist", default=[], dist_reduce_fx=None)

    def sync(self, *args, **kwargs):
        # Disable distributed synchronization on CPU to avoid the error.
        return

    def update(self, true_hist: torch.Tensor, mean_hist: torch.Tensor, sample_hist: torch.Tensor):
        """
        Update the metric state with a batch of histograms.
        Each input is expected to be a tensor of shape (batch_size, num_categories).
        """
        self.true_hist.append(true_hist.detach().cpu())
        self.mean_hist.append(mean_hist.detach().cpu())
        self.sample_hist.append(sample_hist.detach().cpu())

    def _aggregate(self):
        """
        Aggregates accumulated states and computes all per-category quantities needed for plotting and metrics.
        Returns a dictionary containing:
         - x: category indices
         - mae_mean: per-category MAE for mean histogram predictions
         - mae_sample: per-category MAE for sample histogram predictions
         - avg_true: per-category average count from the true histograms
         - corr_mean: per-category Pearson correlation (true vs. mean predictions)
         - corr_sample: per-category Pearson correlation (true vs. sample predictions)
         - true_mean, true_std: per-category mean and std for the true histograms
         - mean_mean, mean_std: per-category mean and std for the mean histograms
         - sample_mean, sample_std: per-category mean and std for the sample histograms
         - mae_sum: summed MAE over all categories (using mean histogram predictions)
         - num_categories: total number of categories
        """
        true_hist = torch.cat(self.true_hist, dim=0)  # shape: [N, C]
        mean_hist = torch.cat(self.mean_hist, dim=0)
        sample_hist = torch.cat(self.sample_hist, dim=0)
        # print("$$$$$$$$$$$$$$$$$$$$$$$$$$$$HISTOGRAM_DEBUG$$$$$$$$$$$$$$$$$$$$$$$$$$$$")
        # print(true_hist[0])
        # print(mean_hist[0])
        # print(sample_hist[0])

        num_categories = true_hist.shape[1]
        x = np.arange(num_categories)

        # Plot 1 data: MAE per category & average true count
        mae_mean = torch.mean(torch.abs(mean_hist - true_hist), dim=0).numpy()
        mae_sample = torch.mean(torch.abs(sample_hist - true_hist), dim=0).numpy()
        avg_true = torch.mean(true_hist, dim=0).numpy()
        # print(avg_true)
        # print(mae_mean)
        # print(mae_sample)

        # Plot 2 data: Pearson correlation per category
        true_np = true_hist.numpy()
        mean_np = mean_hist.numpy()
        sample_np = sample_hist.numpy()
        corr_mean = []
        corr_sample = []
        for i in range(num_categories):
            if np.std(true_np[:, i]) > 0 and np.std(mean_np[:, i]) > 0:
                corr_mean.append(np.corrcoef(mean_np[:, i], true_np[:, i])[0, 1])
            else:
                corr_mean.append(np.nan)
            if np.std(true_np[:, i]) > 0 and np.std(sample_np[:, i]) > 0:
                corr_sample.append(np.corrcoef(sample_np[:, i], true_np[:, i])[0, 1])
            else:
                corr_sample.append(np.nan)
        corr_mean = np.array(corr_mean)
        corr_sample = np.array(corr_sample)

        # Plot 3 data: Means and stds per category for each histogram type
        true_mean_val = torch.mean(true_hist, dim=0).numpy()
        true_std_val = torch.std(true_hist, dim=0).numpy()
        mean_mean_val = torch.mean(mean_hist, dim=0).numpy()
        mean_std_val = torch.std(mean_hist, dim=0).numpy()
        sample_mean_val = torch.mean(sample_hist, dim=0).numpy()
        sample_std_val = torch.std(sample_hist, dim=0).numpy()

        # Summed MAE across categories for mean histogram predictions
        mae_sum = float(mae_mean.sum())
        mae_sample_sum = float(mae_sample.sum())
        # print(mae_sum)
        # print(mae_sample_sum)

        return {
            "x": x,
            "mae_mean": mae_mean,
            "mae_sample": mae_sample,
            "avg_true": avg_true,
            "corr_mean": corr_mean,
            "corr_sample": corr_sample,
            "true_mean": true_mean_val,
            "true_std": true_std_val,
            "mean_mean": mean_mean_val,
            "mean_std": mean_std_val,
            "sample_mean": sample_mean_val,
            "sample_std": sample_std_val,
            "mae_sum": mae_sum,
            "mae_sample_sum": mae_sample_sum,
            "num_categories": num_categories,
        }

    def compute(self) -> float:
        """
        Computes and returns the summed MAE across all categories (using mean histogram predictions).
        Also clears the internal states if needed.
        """
        plot_data = self._aggregate()
        return plot_data["mae_sum"], plot_data["mae_sample_sum"]

    def plot(
        self, val: torch.Tensor | Sequence[torch.Tensor] | None = None, ax: _AX_TYPE | None = None
    ) -> _PLOT_OUT_TYPE:
        """
        Creates three plots:
         1. MAE per category vs. average true count (with twin y-axis).
         2. Pearson correlation per category for mean vs. sample histogram predictions.
         3. Grouped bar plot of the per-category mean and std (error bars) for true, mean, and sample histograms.

         Returns:
            A tuple (fig, axs) where axs is an array of Axes objects.
        """
        plot_data = self._aggregate()
        x = plot_data["x"]
        num_categories = plot_data["num_categories"]
        bar_width = 0.35  # width for bar plots

        # Create a figure with three subplots (stacked vertically)
        fig, axs = plt.subplots(3, 1, figsize=(12, 18))

        # -------------------------------
        # Plot 1: MAE vs Average True Count
        # -------------------------------
        ax1 = axs[0]
        ax1.bar(x - bar_width / 2, plot_data["mae_mean"], bar_width, label="Mean Histogram MAE")
        ax1.bar(x + bar_width / 2, plot_data["mae_sample"], bar_width, label="Sample Histogram MAE")
        ax1.set_xlabel("Category")
        ax1.set_ylabel("Mean Absolute Error")
        ax1.set_title("MAE per Category vs. Average True Count")
        ax1.set_xticks(x)
        ax1.set_xticklabels([f"Cat {i+1}" for i in x])

        # Twin axis for average true count
        ax1_twin = ax1.twinx()
        ax1_twin.plot(
            x, plot_data["avg_true"], color="black", marker="o", linewidth=2, label="Average True Count"
        )
        ax1_twin.set_ylabel("Average True Count")
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines1_twin, labels1_twin = ax1_twin.get_legend_handles_labels()
        ax1_twin.legend(lines1 + lines1_twin, labels1 + labels1_twin, loc="upper left")

        # -------------------------------
        # Plot 2: Pearson Correlation per Category
        # -------------------------------
        ax2 = axs[1]
        ax2.bar(x - bar_width / 2, plot_data["corr_mean"], bar_width, label="Mean Histogram Corr")
        ax2.bar(x + bar_width / 2, plot_data["corr_sample"], bar_width, label="Sample Histogram Corr")
        ax2.set_xlabel("Category")
        ax2.set_ylabel("Pearson Correlation Coefficient")
        ax2.set_title("Correlation per Category: Mean vs. Sample Histogram Predictions")
        ax2.set_xticks(x)
        ax2.set_xticklabels([f"Cat {i+1}" for i in x])
        ax2.legend()

        # -------------------------------
        # Plot 3: Grouped Bar Plot (Mean and Std for Each Histogram)
        # -------------------------------
        ax3 = axs[2]
        ax3.bar(
            x - bar_width,
            plot_data["true_mean"],
            bar_width,
            yerr=plot_data["true_std"],
            capsize=5,
            label="True Histogram",
        )
        ax3.bar(
            x,
            plot_data["mean_mean"],
            bar_width,
            yerr=plot_data["mean_std"],
            capsize=5,
            label="Mean Histogram",
        )
        ax3.bar(
            x + bar_width,
            plot_data["sample_mean"],
            bar_width,
            yerr=plot_data["sample_std"],
            capsize=5,
            label="Sample Histogram",
        )
        ax3.set_xlabel("Category")
        ax3.set_ylabel("Count")
        ax3.set_title("Histogram Mean and Standard Deviation per Category")
        ax3.set_xticks(x)
        ax3.set_xticklabels([f"Cat {i+1}" for i in x])
        ax3.legend()

        fig.tight_layout()
        return fig, axs


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
            "code/subvocab_index": [0, 0, 1, 1, 1, 2, 2, 2, 3, 4],
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
    def __init__(
        self,
        h_token,
        o_token,
        vocab_size,
        histogram_head_loss,
        num_bits=16,
        max_count=4,
        scale=1,
        use_diffusion=False,
        normalize=True,
    ):
        super().__init__()
        self.h_token = h_token
        self.o_token = o_token
        self.vocab_size = vocab_size
        self.histogram_head_loss = histogram_head_loss
        self.num_bits = num_bits
        self.max_count = max_count
        self.scale = scale
        self.use_diffusion = use_diffusion
        self.normalize = normalize
        # Create powers of 2 as a buffer to avoid recomputing
        self.register_buffer("powers", torch.pow(2, torch.arange(num_bits - 1, -1, -1).float()))

    def get_normalized_size(self):
        if self.histogram_head_loss.startswith("softmax"):
            return self.vocab_size + self.num_bits
        elif self.histogram_head_loss.startswith("multinomial"):
            return self.vocab_size * self.max_count  # TODO consider other sizes
        elif self.histogram_head_loss.startswith("cont_softmax"):
            return self.vocab_size
        return self.vocab_size * self.num_bits

    def count_to_binary(self, count):
        """Convert number(s) to binary representation using PyTorch operations."""
        return ((count.unsqueeze(-1) // self.powers) % 2).to(torch.int)

    def binary_to_count(self, binary):
        return (binary * self.powers).sum(dim=-1, keepdim=True)

    def multinomial_to_count(self, multinomial):
        """
        Recovers the count from a multinomial (one-hot) vector.
        """
        # multinomial is expected to be (…, max_count+1). We take the index of the maximum value.
        return multinomial.argmax(dim=-1, keepdim=True)

    def multinomial_transform(self, x):
        """
        Converts a batch of histograms (shape: [batch, vocab_size]) into
        a normalized representation using a multinomial expansion.
        """
        # Zero out special tokens so they are not affected by the transform.
        x[:, self.h_token] = 0
        x[:, self.o_token] = 0

        return x

    def multinomial_reverse_transform(self, x):
        """
        Converts the normalized representation back into a histogram.
        Returns:
            counts: a tensor of shape [batch, vocab_size] with the recovered counts.
            total: a tensor of shape [batch] with the sum of counts per histogram.
        """
        if isinstance(x, HistogramSample):
            x = x.tokens
        if len(x.shape) == 2:
            if self.normalize:
                counts = (
                    (x / x.sum(dim=-1, keepdim=True).clip(min=1) * (self.max_count - 1))
                    .round()
                    .to(torch.int64)
                )

            x[:, self.h_token] = 1
            x[:, self.o_token] = 1

            return x, x.sum(dim=-1)
        # Recover each count by taking the argmax over the multinomial dimension.
        x = x.reshape(x.shape[0], self.vocab_size, -1)
        counts = self.multinomial_to_count(x).squeeze(-1)
        backup = counts.clone()
        if self.normalize:
            counts = (
                (counts / backup.sum(dim=-1, keepdim=True).clip(min=1) * (self.max_count - 1))
                .round()
                .to(torch.int64)
            )
        # # TODO: REMOVE

        # Restore special tokens to count 1.
        counts[:, self.h_token] = 1
        counts[:, self.o_token] = 1

        return counts, counts.sum(dim=-1)

    def transform(self, x):
        if self.histogram_head_loss.startswith("multinomial"):
            return self.multinomial_transform(x)
        # Zero out special tokens
        x[:, self.h_token] = 0
        x[:, self.o_token] = 0

        # Get counts and normalize histogram
        b, _ = x.shape
        if self.histogram_head_loss.startswith("softmax"):
            count = x.sum(dim=-1) + 2  # Add H and NTP token
            data = torch.hstack([x / count.unsqueeze(-1), self.count_to_binary(count)])
        elif self.histogram_head_loss.startswith("cont_softmax"):
            count = x.sum(dim=-1) + 2  # Add H and NTP token
            data = x / count.unsqueeze(-1)
        else:
            # Convert counts to binary representation
            data = self.count_to_binary(x.reshape(-1)).reshape(b, -1).to(torch.float32)

        if self.use_diffusion:
            # Shift from [0,1] to [-self.scale,self.scale] range
            data = (data - 0.5) * 2 * self.scale

        return data

    def reverse_transform(self, x):
        if self.histogram_head_loss.startswith("multinomial"):
            return self.multinomial_reverse_transform(x)
        # Shift from [-self.scale,self.scale] to [0,1]  range
        if self.histogram_head_loss.startswith("softmax"):
            x, count = (
                torch.softmax(x[:, : -self.num_bits], dim=-1),
                torch.sigmoid(x[:, -self.num_bits :]).round(),
            )
            count = self.binary_to_count(count).reshape(x.shape[0], -1)
            x = (x * count).round()
        elif self.histogram_head_loss.startswith("cont_softmax"):
            x = torch.softmax(x, dim=-1)
            count = 64
            x = (x * count).round()
        else:
            if self.use_diffusion:
                # Shift from [-self.scale,self.scale] to [0,1]  range
                x = (x + self.scale) / (2 * self.scale)
            x = torch.sigmoid(x)

            # Clip to [0,1] range
            x = x.round().int()
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


class LinearVAEEncoder(nn.Module):
    def __init__(
        self,
        input_dim=3,
        latent_dim=16,
        hidden_dims=[64, 32],
        dropout=0.0,
    ):
        super().__init__()
        # Create MLP encoder using torchvision
        self.fc_mu = nn.Linear(input_dim // 2, latent_dim)
        self.fc_var = nn.Linear(input_dim // 2, latent_dim)

    def forward(self, x):
        result = x
        mu = self.fc_mu(result[:, : result.shape[1] // 2])
        log_var = self.fc_var(result[:, result.shape[1] // 2 :])
        return mu, log_var


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
        if self.deterministic:
            return self.mean
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


import random

import torch.nn as nn


class MultinomialAutoregressiveDecoder(nn.Module):
    def __init__(
        self, latent_dim, hidden_dim, num_categories, max_count, num_layers=3, teacher_forcing_ratio=0.5
    ):
        """
        Args:
            latent_dim (int): Dimensionality of the latent vector.
            hidden_dim (int): Hidden state size of the LSTM.
            num_categories (int): Number of histogram categories (bins) to decode.
            max_count (int): Maximum count value for each category.
            num_layers (int): Number of LSTM layers.
            teacher_forcing_ratio (float): Ratio for teacher forcing during training.
        """
        super().__init__()
        self.num_categories = num_categories
        self.teacher_forcing_ratio = teacher_forcing_ratio

        # Map latent vector to an initial hidden state.
        self.latent_to_hidden = nn.Linear(latent_dim, hidden_dim)
        # Embedding for input tokens. We reserve index 0 for a start token.
        self.embedding = nn.Embedding(max_count + 1, hidden_dim)
        self.lstm = nn.LSTM(hidden_dim, hidden_dim, num_layers, batch_first=True)
        # Output projection: for each time step, output logits over possible count values [0, max_count]
        self.out_proj = nn.Linear(hidden_dim, max_count + 1)

    def forward(self, z, target_seq=None):
        """
        Args:
            z (Tensor): Latent vectors of shape (batch, latent_dim).
            target_seq (Tensor or None): Ground truth histogram count indices for each category with shape (batch, num_categories)
                                         for teacher forcing during training.
        Returns:
            Tensor: Logits for each category with shape (batch, num_categories, max_count+1).
                    (Each time step outputs a probability distribution over possible count values.)
        """
        batch_size = z.size(0)
        # Initialize hidden and cell states from the latent vector.
        num_layers = self.lstm.num_layers  # e.g., 3
        hidden = self.latent_to_hidden(z).unsqueeze(0).repeat(num_layers, 1, 1)
        cell = torch.zeros_like(hidden)

        # Start token: assume index 0 is reserved for <start>.
        input_token = torch.zeros(batch_size, dtype=torch.long, device=z.device)
        outputs = []

        for t in range(self.num_categories):
            # Embed the current input token.
            input_embed = self.embedding(input_token).unsqueeze(1)  # shape: (batch, 1, hidden_dim)
            # Run one step of the LSTM.
            output, (hidden, cell) = self.lstm(input_embed, (hidden, cell))
            # Project LSTM output to logits over possible count values.
            logits = self.out_proj(output.squeeze(1))  # shape: (batch, max_count+1)
            outputs.append(logits)

            # Determine the next input token.
            if self.training and target_seq is not None and random.random() < self.teacher_forcing_ratio:
                # Use the ground truth token (teacher forcing)
                input_token = target_seq[:, t]
            else:
                # Use the model's prediction.
                input_token = logits.argmax(dim=-1)

        # Stack outputs along the time dimension.
        outputs = torch.stack(outputs, dim=1)  # shape: (batch, num_categories, max_count+1)
        return outputs


import torch
import torch.nn as nn
import torch.nn.functional as F
import xformers.ops


def tokens_to_inf_logits(tokens: torch.LongTensor, vocab_size: int) -> torch.FloatTensor:
    """
    Parameters
    ----------
    tokens      : (…, seq_len)  integer token IDs
    vocab_size  : size of the vocabulary

    Returns
    -------
    logits      : (…, seq_len, vocab_size)  where
                  logits[..., t, token_id[t]] = 0
                  logits[..., t, other]      = -inf
    """
    # create an output tensor full of -inf
    logits = torch.full((*tokens.shape, vocab_size), float("-inf"), device=tokens.device, dtype=torch.float32)

    # scatter 0.0 at the correct class for every position
    logits.scatter_(-1, tokens.unsqueeze(-1), 0.0)
    return logits


# ------------------------------------------------------------
# 1.  Transformer layer with xformers attention + KV caching
# ------------------------------------------------------------
class CachedXformersTransformerLayer(nn.Module):
    def __init__(self, hidden_dim: int, nhead: int, dropout: float = 0.1):
        super().__init__()
        self.hidden_dim, self.nhead, self.dropout = hidden_dim, nhead, dropout

        self.q_proj = nn.Linear(hidden_dim, hidden_dim)
        self.k_proj = nn.Linear(hidden_dim, hidden_dim)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)
        self.ln1 = nn.LayerNorm(hidden_dim)

        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.ReLU(),
            nn.Linear(hidden_dim * 4, hidden_dim),
            nn.Dropout(dropout),
        )
        self.ln2 = nn.LayerNorm(hidden_dim)

    # ---- helper -------------------------------------------------
    def _reshape(self, t: torch.Tensor, batch: int) -> torch.Tensor:
        """
        (seq_len, batch, hidden) -> (batch*nhead, seq_len, head_dim)
        """
        seq_len, _, _ = t.size()
        head_dim = self.hidden_dim // self.nhead
        t = t.view(seq_len, batch, self.nhead, head_dim)
        t = t.permute(1, 2, 0, 3).contiguous()
        return t.view(batch * self.nhead, seq_len, head_dim)

    # ---- forward ------------------------------------------------
    def forward(self, x: torch.Tensor, cache: dict | None = None):
        """
        x : (x_seq_len, batch, hidden_dim)
        cache : {'k': (cached_len, batch, hidden), 'v': ...}  or None
        """
        residual = x
        x = self.ln1(x)

        Q = self.q_proj(x)
        new_K = self.k_proj(x)
        new_V = self.v_proj(x)

        if cache is None:
            K, V = new_K, new_V
        else:
            K = torch.cat([cache["k"], new_K], dim=0)
            V = torch.cat([cache["v"], new_V], dim=0)
        new_cache = {"k": K, "v": V}

        x_seq_len, batch, _ = x.size()
        head_dim = self.hidden_dim // self.nhead

        Qh = self._reshape(Q, batch)  # (B*H, x_seq_len, head_dim)
        Kh = self._reshape(K, batch)  # (B*H, total_len, head_dim)
        Vh = self._reshape(V, batch)  # (B*H, total_len, head_dim)

        if cache is None:
            attn_bias = xformers.ops.LowerTriangularMask()  # full sequence
        else:
            attn_bias = None  # no future tokens exist, so no mask needed

        attn = xformers.ops.memory_efficient_attention(
            Qh,
            Kh,
            Vh,
            p=self.dropout,
            attn_bias=attn_bias,
        )  # (B*H, x_seq_len, head_dim)

        attn = attn.view(batch, self.nhead, x_seq_len, head_dim)
        attn = attn.permute(2, 0, 1, 3).contiguous().view(x_seq_len, batch, self.hidden_dim)
        x = residual + self.out_proj(attn)

        # Feed‑forward
        x = x + self.ffn(self.ln2(x))
        return x, new_cache


import math
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch.distributions import Categorical


@dataclass
class HistogramSample(nn.Module):
    logits: torch.Tensor
    tokens: torch.Tensor
    entropy: torch.Tensor | None
    likelihood: torch.Tensor | None


# ------------------------------------------------------------
# 2.  Autoregressive decoder that passes latent as 1st token
# ------------------------------------------------------------
class XformersAutoregressiveDecoder(nn.Module):
    def __init__(
        self,
        latent_dim: int,
        hidden_dim: int,
        num_categories: int,
        max_count: int,
        num_layers: int = 3,
    ):
        super().__init__()
        self.num_categories, self.max_count = num_categories, max_count

        self.token_emb = nn.Embedding(max_count + 1, hidden_dim)
        self.pos_emb = nn.Embedding(num_categories + 1, hidden_dim)  # pos0 = latent
        self.latent_proj = nn.Linear(latent_dim, hidden_dim)
        nhead = 12 if hidden_dim >= 144 else 8

        self.layers = nn.ModuleList(
            [CachedXformersTransformerLayer(hidden_dim, nhead=nhead) for _ in range(num_layers)]
        )
        self.out_proj = nn.Linear(hidden_dim, max_count + 1)

    # ------------------------------------------------------------------
    #  Teacher forcing  (single pass, shifted RIGHT by one position)
    # ------------------------------------------------------------------
    def teacher_forced_forward(self, z: torch.Tensor, target_seq: torch.Tensor):
        """
        z           : (B, latent_dim)
        target_seq  : (B, N)   (ground‑truth tokens to predict)
        returns logits : (B, N, max_count+1)
        """
        B, N = target_seq.shape
        device = z.device

        # latent token (position 0)
        latent_tok = self.latent_proj(z).unsqueeze(1)
        latent_tok = latent_tok + self.pos_emb(torch.zeros(B, 1, dtype=torch.long, device=device))

        # SHIFT the teacher‑forcing tokens right by one:
        #   input tokens = latent + target_seq[:, :-1]
        in_tokens = target_seq[:, :-1]  # (B, N-1)   (empty if N==1)
        if N == 1:
            in_embeds = torch.zeros(B, 0, latent_tok.size(-1), device=device)
        else:
            pos = torch.arange(1, N, device=device).unsqueeze(0).expand(B, -1)
            in_embeds = self.token_emb(in_tokens) + self.pos_emb(pos)

        x = torch.cat([latent_tok, in_embeds], dim=1)  # (B, N)   length N  (0..N-1)
        x = x.transpose(0, 1)  # (seq_len=N, B, hidden)

        # transformer stack (no caching needed)
        for layer in self.layers:
            x, _ = layer(x, cache=None)

        logits = self.out_proj(x)  # predictions for positions 1..N
        logits = logits.transpose(0, 1)  # (B, N, vocab)
        return self.get_histogram_sample(logits, tokens=logits.argmax(dim=-1))

    def get_histogram_sample(self, logits, tokens, multiple_logits=False, get_metadata=False):
        if get_metadata:
            entropy = Categorical(logits=logits).entropy()
            likelihood = Categorical(logits=logits).log_prob(tokens)
            if multiple_logits:
                entropy = entropy.mean(1)
                likelihood = torch.logsumexp(likelihood, dim=1) - math.log(likelihood.shape[1])
        else:
            entropy = None
            likelihood = None
        return HistogramSample(logits, tokens, entropy, likelihood)

    # ------------------------------------------------------------------
    #  Autoregressive inference  (produce BEFORE inserting new token)
    # ------------------------------------------------------------------
    def inference_forward(
        self,
        z: torch.Tensor,
        target_seq=None,
        return_tokens=False,
        use_teacher_forcing=False,
        temperature=1.0,
        get_metadata: bool = False,
    ):
        """
        If `use_teacher_forcing` and `target_seq` provided, runs forced decoding;
        otherwise autoregressively samples / argmaxes.
        """
        B, device = z.size(0), z.device
        caches = [None] * len(self.layers)

        # ----- start with latent token -----
        x = self.latent_proj(z).unsqueeze(1)
        x = x + self.pos_emb(torch.zeros(B, 1, dtype=torch.long, device=device))
        x = x.transpose(0, 1)  # (1, B, hidden)
        for i, layer in enumerate(self.layers):
            x, caches[i] = layer(x, cache=None)

        outputs = []
        tokens = []
        for t in range(self.num_categories):  # want N logits
            # 1) produce logits from CURRENT context (before adding next token)
            logits_t = self.out_proj(x[-1])  # (B, vocab)
            outputs.append(logits_t)

            # 2) decide next token
            if use_teacher_forcing and target_seq is not None:
                next_tok = target_seq[:, t].unsqueeze(1)  # teacher
            else:
                # TODO try different sampling strategies
                next_tok = torch.multinomial((logits_t / temperature).softmax(dim=-1), num_samples=1)
                # next_tok = logits_t.argmax(dim=-1, keepdim=True)  # greedy (or sample)
            tokens.append(next_tok)
            # 3) embed & append
            pos_id = torch.full((B, 1), t + 1, dtype=torch.long, device=device)  # position t+1
            next_embed = self.token_emb(next_tok) + self.pos_emb(pos_id)
            x_new = next_embed.transpose(0, 1)  # (1, B, hidden)

            # 4) run only the NEW token through each layer using caches
            new_caches = []
            for i, layer in enumerate(self.layers):
                x_new, new_cache = layer(x_new, cache=caches[i])
                new_caches.append(new_cache)
            caches = new_caches
            x = torch.cat([x, x_new], dim=0)  # extend sequence context
        logits = torch.stack(outputs, dim=1)  # (B, N, vocab)

        best_tokens = torch.stack(tokens, dim=1).squeeze(-1)  # (B, N)

        return self.get_histogram_sample(logits, best_tokens, get_metadata=get_metadata)

    @torch.no_grad()
    def inference_best_of_n(
        self,
        z: torch.Tensor,  # (B, latent_dim)
        *,
        n_samples: int = 16,
        temperature: float = 1.0,
        return_tokens: bool = False,
        get_metadata: bool = False,
    ):
        """
        Draw `n_samples` sequences *in parallel* and keep the one with the
        highest total log‑probability for each item in the batch.

        Returns
        -------
        token_logits : (B, num_categories, vocab)
        best_tokens  : (B, num_categories)          (only if return_tokens=True)
        """
        B, device = z.size(0), z.device
        V = self.max_count + 1
        N = self.num_categories

        # ------------------------------------------------------------
        # 1.  Tile the latent so we have (B*n_samples, latent_dim)
        # ------------------------------------------------------------
        z_rep = z.unsqueeze(1).expand(-1, n_samples, -1)  # (B, n, D)
        z_rep = z_rep.reshape(B * n_samples, -1)  # (B*n, D)

        # ------------------------------------------------------------
        # 2.  Initialise transformer with the latent token
        # ------------------------------------------------------------
        caches = [None] * len(self.layers)
        x = self.latent_proj(z_rep).unsqueeze(1)  # (B*n, 1, H)
        x = x + self.pos_emb(torch.zeros(B * n_samples, 1, dtype=torch.long, device=device))
        x = x.transpose(0, 1)  # (1, B*n, H)
        for i, layer in enumerate(self.layers):
            x, caches[i] = layer(x, cache=None)

        # storage
        seq_logp = torch.zeros(B * n_samples, device=device)
        seq_tokens = torch.empty(B * n_samples, N, dtype=torch.long, device=device)
        seq_logits = torch.empty(B * n_samples, N, V, dtype=z.dtype, device=device)

        # ------------------------------------------------------------
        # 3.  Autoregressive loop (still parallel over B*n trajectories)
        # ------------------------------------------------------------
        for t in range(N):
            logits_t = self.out_proj(x[-1]) / temperature  # (B*n, V)
            probs_t = logits_t.softmax(-1)

            next_tok = torch.multinomial(probs_t, 1)  # (B*n, 1)

            seq_logp += probs_t.gather(1, next_tok).log().squeeze(1)
            seq_tokens[:, t] = next_tok.squeeze(1)
            seq_logits[:, t] = logits_t

            # advance
            pos_id = torch.full((B * n_samples, 1), t + 1, dtype=torch.long, device=device)
            next_embed = self.token_emb(next_tok) + self.pos_emb(pos_id)
            x_new = next_embed.transpose(0, 1)  # (1, B*n, H)

            new_caches = []
            for i, layer in enumerate(self.layers):
                x_new, new_cache = layer(x_new, cache=caches[i])
                new_caches.append(new_cache)
            caches = new_caches
            x = torch.cat([x, x_new], 0)

        # ------------------------------------------------------------
        # 4.  Pick the best of the n_samples for every original batch item
        # ------------------------------------------------------------
        seq_logp = seq_logp.view(B, n_samples)  # (B,n)
        seq_tokens = seq_tokens.view(B, n_samples, N)  # (B,n,N)
        seq_logits = seq_logits.view(B, n_samples, N, V)  # (B,n,N,V)

        best_idx = seq_logp.argmax(dim=1)  # (B,)
        batch_idx = torch.arange(B, device=device)

        best_tokens = seq_tokens[batch_idx, best_idx]  # (B,N)
        best_logits = seq_logits[batch_idx, best_idx]  # (B,N,V)

        # (optional) convert tokens back to the “inference logits” format you use
        return self.get_histogram_sample(best_logits, best_tokens, get_metadata=get_metadata)

    @torch.no_grad()
    def inference_avg_of_n(
        self,
        z: torch.Tensor,  # (B, latent_dim)
        *,
        n_samples: int = 16,
        temperature: float = 1.0,
        weight_by_prob: bool = False,  # True → weight samples by exp(log‑prob)
        return_tokens: bool = False,
        get_metadata: bool = False,
    ):
        """
        Draw `n_samples` complete sequences **in parallel** and return the
        per‑time‑step average of the logits (and optionally the averaged tokens).

        Parameters
        ----------
        z              : (B, latent_dim)
        n_samples      : how many independent trajectories to draw
        temperature    : soft‑max temperature during sampling
        weight_by_prob : if True, average logits with weights ∝ exp(total log‑prob)
        return_tokens  : additionally return the averaged token IDs (float)

        Returns
        -------
        token_logits : (B, num_categories, vocab)
        avg_tokens   : (B, num_categories)  (only if return_tokens=True)
        """
        B, device = z.size(0), z.device
        V = self.max_count + 1
        N = self.num_categories

        # ------------------------------------------------------------
        # 1.  Tile latent  →  (B*n, latent_dim)
        # ------------------------------------------------------------
        z_rep = z.unsqueeze(1).expand(-1, n_samples, -1).reshape(B * n_samples, -1)

        # ------------------------------------------------------------
        # 2.  Run the autoregressive decoder for all B*n trajectories
        # ------------------------------------------------------------
        caches = [None] * len(self.layers)
        x = self.latent_proj(z_rep).unsqueeze(1)  # (B*n,1,H)
        x = x + self.pos_emb(torch.zeros(B * n_samples, 1, dtype=torch.long, device=device))
        x = x.transpose(0, 1)  # (1,B*n,H)
        for i, layer in enumerate(self.layers):
            x, caches[i] = layer(x, cache=None)

        seq_logits = torch.empty(B * n_samples, N, V, dtype=z.dtype, device=device)
        seq_tokens = torch.empty(B * n_samples, N, dtype=torch.long, device=device)
        seq_logp = torch.zeros(B * n_samples, device=device)

        for t in range(N):
            logits_t = self.out_proj(x[-1]) / temperature  # (B*n,V)
            probs_t = logits_t.softmax(-1)
            next_tok = torch.multinomial(probs_t, 1)  # (B*n,1)

            seq_logits[:, t] = logits_t
            seq_tokens[:, t] = next_tok.squeeze(1)
            seq_logp += probs_t.gather(1, next_tok).log().squeeze(1)

            # advance
            pos_id = torch.full((B * n_samples, 1), t + 1, dtype=torch.long, device=device)
            next_embed = self.token_emb(next_tok) + self.pos_emb(pos_id)
            x_new = next_embed.transpose(0, 1)

            new_caches = []
            for i, layer in enumerate(self.layers):
                x_new, new_cache = layer(x_new, cache=caches[i])
                new_caches.append(new_cache)
            caches = new_caches
            x = torch.cat([x, x_new], 0)

        # ------------------------------------------------------------
        # 3.  Reshape  (B*n, …)  →  (B, n_samples, …)
        # ------------------------------------------------------------
        seq_logits = seq_logits.view(B, n_samples, N, V)  # (B,n,N,V)
        seq_tokens = seq_tokens.view(B, n_samples, N)  # (B,n,N)
        seq_logp = seq_logp.view(B, n_samples)  # (B,n)

        # ------------------------------------------------------------
        # 4.  Average across the n_samples dimension
        # ------------------------------------------------------------
        if weight_by_prob:
            raise NotImplementedError(
                "weight_by_prob not supported anymore, if you want to use it, note that get_histogram_sample might not work with it"
            )
            w = (seq_logp - seq_logp.max(1, keepdim=True).values).exp()  # stability
            w = w / w.sum(1, keepdim=True)  # (B,n)
            w = w.unsqueeze(-1).unsqueeze(-1)  # (B,n,1,1)
            token_logits = (seq_logits * w).sum(1)  # (B,N,V)
            avg_tokens = (seq_tokens.float() * w.squeeze(-1)).sum(1)  # (B,N)
        else:
            token_logits = seq_logits  # (B,N,V)
            avg_tokens = seq_tokens  # (B,N)

        return self.get_histogram_sample(
            token_logits.mean(dim=1),
            seq_tokens.float().mean(dim=1).round().long(),
            multiple_logits=False,
            get_metadata=get_metadata,
        )

    @torch.no_grad()
    def sequence_log_likelihood(
        self, z: torch.Tensor, tokens: torch.Tensor, temperature: float = 1.0
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        z       : (B, latent_dim)       latent vectors
        tokens  : (B, num_categories)   ground‑truth token IDs

        Returns
        -------
        logp    : (B,)  total log‑likelihood (sum over sequence positions)
        """
        # 1) get per‑position logits from the teacher‑forced pass
        sample = self.teacher_forced_forward(z, tokens)

        logits = sample.logits / temperature  # (B, N, vocab)

        # 2) convert to log‑probs
        log_probs = F.log_softmax(logits, dim=-1)  # (B, N, vocab)

        # 3) pick the log‑prob assigned to the true token at every position
        token_logp = log_probs.gather(-1, tokens.unsqueeze(-1))  # (B, N, 1)
        token_logp = token_logp.squeeze(-1)  # (B, N)

        # 4) sum over the sequence to get the total log‑likelihood
        return token_logp.sum(dim=-1)

    # ------------------------------------------------------------------
    def forward(
        self,
        z,
        target_seq=None,
        log=False,
        return_tokens=False,
        inference_method="forward",
        temperature=1.0,
        get_metadata=False,
    ):
        if target_seq is not None:
            if self.training:
                output = self.teacher_forced_forward(z, target_seq)
                assert isinstance(output, HistogramSample)
            else:
                output = self.inference_forward(z, target_seq, use_teacher_forcing=True)
                assert isinstance(output, HistogramSample)
        else:
            if inference_method == "best_of_n":
                output = self.inference_best_of_n(
                    z, return_tokens=return_tokens, temperature=temperature, get_metadata=get_metadata
                )
                assert isinstance(output, HistogramSample)
            elif inference_method == "beam_search":
                output = self.inference_beam_search(z, return_tokens=return_tokens, temperature=temperature)
                assert isinstance(output, HistogramSample)
            elif inference_method == "forward":
                output = self.inference_forward(
                    z, return_tokens=return_tokens, temperature=temperature, get_metadata=get_metadata
                )
                assert isinstance(output, HistogramSample)
            elif inference_method == "avg_of_n":
                output = self.inference_avg_of_n(
                    z, return_tokens=return_tokens, temperature=temperature, get_metadata=get_metadata
                )
                assert isinstance(output, HistogramSample)
            else:
                raise ValueError(f"Unknown inference method: {inference_method}")
        return output


# ------------------------------------------------------------
#  three auxiliary‑loss helpers
# ------------------------------------------------------------
def distance_kernel_loss(logits, target, sigma=2.0, laplace=True):
    """
    logits : (B,N,V)   raw decoder logits
    target : (B,N)     integer counts
    returns scalar loss  (mean over B,N)
    """
    B, N, V = logits.shape
    device = logits.device

    # build distance‑based target distribution
    k = torch.arange(V, device=device).view(1, 1, V)  # (1,1,V)
    tgt = target.unsqueeze(-1)  # (B,N,1)
    dist = (k - tgt).abs().float()
    if laplace:
        log_w = -dist / sigma
    else:  # Gaussian
        log_w = -(dist**2) / (2 * sigma**2)
    smooth = log_w.exp()
    smooth = smooth / smooth.sum(-1, keepdim=True)  # (B,N,V)

    log_probs = F.log_softmax(logits, -1)
    loss = -(smooth * log_probs).sum(-1).mean()
    return loss


def expectation_mse_loss(logits, target):
    """
    MSE between model expected count  E[k]  and target
    """
    probs = logits.softmax(-1)  # (B,N,V)
    V = logits.size(-1)
    k = torch.arange(V, device=logits.device).view(1, 1, V)
    exp = (probs * k).sum(-1)  # (B,N)
    return F.mse_loss(exp, target.float())


def crps_loss(logits, target):
    """
    CRPS / discrete Earth‑Mover distance between predicted CDF and target CDF
    """
    probs = logits.softmax(-1)  # (B,N,V)
    cdf_pred = probs.cumsum(-1)  # (B,N,V)
    V = logits.size(-1)
    one_hot = F.one_hot(target.long(), V).float()
    cdf_true = one_hot.cumsum(-1)
    return (cdf_pred - cdf_true).abs().mean()


class AutoencoderKL_Autoregressive(nn.Module):
    def __init__(
        self,
        embed_dim,
        input_dim,
        num_categories,
        encoder_hidden_dims,
        beta,
        max_count,
        aux_loss: str | None = None,  # 'distance', 'expectation', 'crps', or None
        aux_lambda: float = 0.0,  # weight of the auxiliary term
        **kwargs,
    ):
        super().__init__()
        # Initialize the encoder as before.
        self._encoder = nn.Linear(input_dim, embed_dim)
        self.embed_dim = embed_dim

        # Choose the autoregressive decoder.
        self.auto_reg_decoder = XformersAutoregressiveDecoder(
            latent_dim=embed_dim,
            hidden_dim=embed_dim,
            num_categories=num_categories,
            max_count=max_count,
            num_layers=3,
        )

        self.use_variational = True
        self.beta = beta
        self.aux_loss = aux_loss
        self.aux_lambda = aux_lambda

    def encode(self, x):
        mu = self._encoder(x)
        log_var = torch.full_like(mu, 1e-6)
        moments = torch.cat((mu, log_var), dim=1)
        posterior = DiagonalGaussianDistribution(moments, deterministic=True)
        return posterior

    def decode(self, z, outputs=None, **kwargs):
        # Autoregressively decode histogram counts from latent z.
        if outputs is not None:
            outputs = outputs.long()
        dec = self.auto_reg_decoder(z, target_seq=outputs, **kwargs)
        return dec

    def forward(
        self,
        inputs,
        outputs=None,
        disable=True,
    ):
        """
        aux_loss   : choose which auxiliary loss to add (or None)
        aux_lambda : weight applied to that auxiliary term
        """
        posterior = self.encode(inputs)
        z = posterior.mean if disable else posterior.sample()

        histogram_sample = self.decode(z, outputs=outputs)  # (B,N,V)
        dec = histogram_sample.logits

        # --- primary reconstruction loss (token‑level CE) -------------

        rec_loss = F.cross_entropy(
            dec.transpose(1, 2),  # (B,V,N)
            outputs.long(),
            reduction="mean",
        )

        # --- optional auxiliary loss ----------------------------------
        aux_val = torch.tensor(0.0, device=dec.device)

        if self.aux_loss == "distance":
            aux_val = distance_kernel_loss(dec, outputs)
        elif self.aux_loss == "expectation":
            aux_val = expectation_mse_loss(dec, outputs)
        elif self.aux_loss == "crps":
            aux_val = crps_loss(dec, outputs)

        loss = rec_loss + self.aux_lambda * aux_val

        if not self.training:
            with torch.no_grad():
                histogram_sample = self.auto_reg_decoder.inference_forward(z)
                sample = histogram_sample.tokens
                sample_lp = self.auto_reg_decoder.sequence_log_likelihood(z, sample).mean()
        else:
            sample_lp = 0
        with torch.no_grad():
            true_lp = self.auto_reg_decoder.sequence_log_likelihood(z, outputs.long()).mean()

        return {
            "vae_loss": loss,
            "vae_rec_loss": rec_loss,
            "vae_aux_loss": self.aux_lambda * aux_val,
            "vae_kl_loss": torch.tensor(0.0, device=dec.device),
            "vae_reconstruction": dec,
            "vae_true_lp": true_lp,
            "vae_sample_lp": sample_lp,
        }


import torch
import torch.nn as nn
import torch.nn.functional as F

# (Assume DiagonalGaussianDistribution is defined elsewhere.)
# For example:
# class DiagonalGaussianDistribution:
#     def __init__(self, moments, deterministic=False):
#         self._mean, self._log_var = torch.chunk(moments, 2, dim=1)
#         self.deterministic = deterministic
#     @property
#     def mean(self):
#         return self._mean
#     def sample(self):
#         if self.deterministic:
#             return self._mean
#         std = torch.exp(0.5 * self._log_var)
#         return self._mean + std * torch.randn_like(std)


class TransformerMaskedImputationDecoder(nn.Module):
    def __init__(
        self,
        latent_dim,
        hidden_dim,
        num_categories,
        max_count,
        num_layers=3,
        mask_prob=0.8,  # If float, fraction of count tokens to mask; if None, randomize per forward.
        num_iters=1,  # Number of iterative refinement iterations during inference.
    ):
        """
        In this version the input sequence starts with a latent token (derived from z)
        at index 0 and tokens 1...num_categories correspond to the count categories.
        This allows the transformer encoder to attend to the latent representation
        when predicting each count.

        Args:
            latent_dim (int): Dimension of the latent code.
            hidden_dim (int): Dimension of the transformer embeddings.
            num_categories (int): Number of count tokens to predict.
            max_count (int): Maximum count value (vocabulary: 0...max_count).
            num_layers (int): Number of transformer encoder layers.
            mask_prob (float or None): If a float, the fraction of count tokens to mask during training.
                                       If set to None, a random masking probability is used for each forward pass.
            num_iters (int): Number of iterative refinement iterations during inference.
        """
        super().__init__()
        self.num_categories = num_categories
        # Total tokens now includes the latent token.
        self.total_tokens = num_categories + 1
        self.max_count = max_count
        self.mask_prob = mask_prob
        self.num_iters = num_iters
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim

        # Token embedding for valid count tokens (indices 0 ... max_count).
        self.token_embedding = nn.Embedding(max_count + 1, hidden_dim)
        # Learned mask embedding for positions that are masked.
        self.mask_embedding = nn.Parameter(torch.randn(hidden_dim))
        # Positional embeddings for each position in the sequence (total_tokens positions).
        self.pos_embedding = nn.Embedding(self.total_tokens, hidden_dim)
        # Standard transformer encoder layers.
        encoder_layer = nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=8)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        # Project the latent vector z to an embedding used as the first token.
        self.latent_to_context = nn.Linear(latent_dim, hidden_dim)
        # Output projection: from hidden_dim to logits over count tokens (0...max_count).
        self.out_proj = nn.Linear(hidden_dim, max_count + 1)

    def teacher_forced_forward(self, z, target_seq, log=False):
        """
        Training-time forward pass using masked imputation.

        The full input sequence has length num_categories+1. The 0th token is derived directly
        from z; tokens 1..N are obtained from the target sequence but are randomly masked.
        If self.mask_prob is None, a random fraction for masking is sampled for this forward.

        Args:
            z (Tensor): Latent vectors of shape (batch, latent_dim)
            target_seq (Tensor): Ground-truth tokens for counts, shape (batch, num_categories)
            log (bool): If True, print debug information.

        Returns:
            cat_logits (Tensor): Logits for count tokens, shape (batch, num_categories, max_count+1).
            mask (BoolTensor): Mask indicator for count tokens, shape (batch, num_categories),
                               where True indicates that the token was masked.
        """
        batch_size = z.size(0)
        device = z.device

        # Compute the latent token (position 0) from z.
        latent_token_embed = self.latent_to_context(z)  # (batch, hidden_dim)

        # Determine which tokens to mask for positions 1..num_categories.
        # If mask_prob is None, sample a random masking probability for this forward pass.
        if self.mask_prob is None:
            random_mask_prob = torch.rand(1, device=device).item()
            mask = torch.rand(target_seq.shape, device=device) < random_mask_prob
        else:
            mask = torch.rand(target_seq.shape, device=device) < self.mask_prob

        # Look up embeddings for target tokens.
        token_embeds = self.token_embedding(target_seq)  # (batch, num_categories, hidden_dim)
        # Expand the learned mask embedding.
        mask_embed_expanded = self.mask_embedding.unsqueeze(0).unsqueeze(0).expand_as(token_embeds)
        # For each count token: if masked, use the mask embedding; otherwise, use the token embedding.
        cat_embeds = torch.where(
            mask.unsqueeze(-1), mask_embed_expanded, token_embeds
        )  # (batch, num_categories, hidden_dim)

        # Construct the full sequence embeddings:
        # Position 0 is the latent token; positions 1..end are the (possibly masked) count tokens.
        full_seq_embeds = torch.cat(
            [latent_token_embed.unsqueeze(1), cat_embeds], dim=1
        )  # (batch, total_tokens, hidden_dim)

        # Add positional embeddings (positions 0 to total_tokens-1).
        positions = torch.arange(self.total_tokens, device=device).unsqueeze(0).expand(batch_size, -1)
        pos_embeds = self.pos_embedding(positions)
        input_embeds = full_seq_embeds + pos_embeds

        # Pass the sequence through the transformer encoder.
        x = input_embeds.transpose(0, 1)  # (total_tokens, batch, hidden_dim)
        encoded = self.transformer_encoder(x)  # (total_tokens, batch, hidden_dim)
        logits = self.out_proj(encoded)  # (total_tokens, batch, max_count+1)
        logits_seq = logits.transpose(0, 1)  # (batch, total_tokens, max_count+1)
        # We only care about the outputs corresponding to the count tokens (positions 1...end).
        cat_logits = logits_seq[:, 1:, :]  # (batch, num_categories, max_count+1)

        if log:
            print("Teacher forced logits sample:", cat_logits[0].argmax(dim=-1))

        return self.get_histogram_sample(cat_logits, tokens=cat_logits.argmax(dim=-1))

    def inference_forward(self, z, target_seq=None, log=False, temperature=0.3, confidence_threshold=0.95):
        """
        Inference-time forward pass using iterative refinement with selective re-masking.

        The input sequence is initialized with:
        - Position 0: the latent token computed from z.
        - Positions 1...end: initially masked.
        For each iteration, the model:
        1. Computes logits for all count tokens.
        2. Applies temperature scaling and computes a probability distribution.
        3. Samples tokens only for positions currently masked.
        4. Measures the confidence (probability) of the newly sampled tokens.
        5. For masked positions, updates the token only if the confidence exceeds the threshold;
            otherwise, these positions remain masked.
        Positions already fixed (from previous iterations) remain unchanged.

        Args:
            z (Tensor): Latent vectors of shape (batch, latent_dim)
            target_seq (Tensor or None): Not used in this implementation.
            use_teacher_forcing (bool): Not used in this implementation.
            log (bool): If True, prints debug information per iteration.
            temperature (float): Temperature parameter for scaling the logits when sampling.
            confidence_threshold (float): Confidence threshold below which tokens are re-masked.

        Returns:
            cat_logits (Tensor): Final output logits for count tokens,
                                shape (batch, num_categories, max_count+1).
        """
        batch_size = z.size(0)
        device = z.device

        # Compute the latent token embedding for position 0.
        latent_token_embed = self.latent_to_context(z)  # (batch, hidden_dim)
        # Initialize count tokens (positions 1...end) as masked (denoted by -1).
        cat_tokens = torch.full((batch_size, self.num_categories), -1, dtype=torch.long, device=device)

        for it in range(1):
            # Prepare embeddings:
            # For positions that remain masked (== -1), use the learned mask embedding;
            # For positions that are fixed, use the token embedding.
            cat_embeds = torch.where(
                cat_tokens.unsqueeze(-1) == -1,
                self.mask_embedding.unsqueeze(0)
                .unsqueeze(0)
                .expand(batch_size, self.num_categories, self.hidden_dim),
                self.token_embedding(torch.clamp(cat_tokens, min=0)),
            )
            # Construct the full sequence: latent token (position 0) plus count tokens.
            full_seq_embeds = torch.cat(
                [latent_token_embed.unsqueeze(1), cat_embeds], dim=1
            )  # (batch, total_tokens, hidden_dim)
            # Add positional embeddings.
            positions = torch.arange(self.total_tokens, device=device).unsqueeze(0).expand(batch_size, -1)
            pos_embeds = self.pos_embedding(positions)
            input_embeds = full_seq_embeds + pos_embeds

            # Transformer forward pass.
            x = input_embeds.transpose(0, 1)  # (total_tokens, batch, hidden_dim)
            encoded = self.transformer_encoder(x)
            logits = self.out_proj(encoded)
            logits_seq = logits.transpose(0, 1)  # (batch, total_tokens, max_count+1)
            # Extract logits for count tokens (positions 1...end).
            cat_logits = logits_seq[:, 1:, :]  # (batch, num_categories, max_count+1)

            # Temperature scaling and compute probabilities.
            probs = F.softmax(cat_logits / temperature, dim=-1)  # (batch, num_categories, max_count+1)

            # Create a copy of current tokens.
            new_tokens = cat_tokens.clone()
            # Identify positions that are still masked.
            mask_positions = cat_tokens == -1
            if mask_positions.sum() > 0:
                # Sample new tokens only for masked positions.
                sampled_new = torch.multinomial(probs[mask_positions], num_samples=1).squeeze(-1)
                new_tokens[mask_positions] = sampled_new

            # Compute confidence for the newly sampled tokens (or kept tokens).
            token_confidence = probs.gather(dim=-1, index=new_tokens.unsqueeze(-1)).squeeze(-1)
            # For positions that were just sampled (masked in previous iteration),
            # only update if the confidence exceeds the threshold; otherwise, keep them masked.
            updated_tokens = new_tokens.clone()
            updated_tokens[mask_positions] = torch.where(
                token_confidence[mask_positions] >= confidence_threshold,
                new_tokens[mask_positions],
                torch.full_like(new_tokens[mask_positions], -1),
            )
            # For positions that were already fixed, retain the previous token.
            cat_tokens = torch.where(cat_tokens != -1, cat_tokens, updated_tokens)

            if log:
                print(f"Inference iteration {it} tokens:", cat_tokens[0])

        # At the end of iterations, convert the final tokens to a one-hot style logits tensor.
        # (For tokens that remain masked, we leave their logits as -∞.)
        final_tokens = new_tokens.clone()
        # Replace any remaining masked (-1) entries with a dummy token (here 0) so scatter_ works.
        dummy_tokens = final_tokens.clone()
        dummy_tokens[dummy_tokens == -1] = 0
        one_hot_logits = torch.full(cat_logits.shape, -float("inf"), device=device)
        one_hot_logits.scatter_(2, dummy_tokens.unsqueeze(-1), 0.0)
        # Optionally, you could reset the logits for positions that remain masked to -∞.
        cat_logits = one_hot_logits

        return logits_seq[:, 1:, :]  # or try cat_logits

    def forward(self, z, target_seq=None, log=False):
        """
        Depending on whether target_seq is provided, this method either runs a
        teacher-forced (training) pass or an inference (iterative refinement) pass.

        Args:
            z (Tensor): Latent vectors, shape (batch, latent_dim)
            target_seq (Tensor or None): Ground-truth count tokens (for training).
            log (bool): If True, prints debug information.

        Returns:
            If target_seq is provided: (cat_logits, mask) where cat_logits has shape
            (batch, num_categories, max_count+1) and mask indicates which count tokens were masked.
            Otherwise, returns cat_logits from iterative refinement.
        """
        if target_seq is not None:
            return self.teacher_forced_forward(z, target_seq, log=log)
        else:
            return self.inference_forward(z, target_seq, log=log)


class AutoencoderKL_MI(nn.Module):
    def __init__(
        self,
        embed_dim,
        input_dim,
        num_categories,
        encoder_hidden_dims,
        beta,
        max_count,
        num_layers,
        mask_prob,
        num_iters,
    ):
        """
        A VAE with KL divergence, using a transformer encoder for masked imputation as the decoder.

        Args:
            embed_dim (int): Dimension of the latent code.
            input_dim (int): Dimension of the input.
            num_categories (int): Number of tokens to reconstruct.
            encoder_hidden_dims: (unused in this simplified example; in a full model, used for a deeper encoder)
            beta (float): Weight for a KL divergence loss (here KL loss is zero since we use a deterministic encoder).
            max_count (int): Maximum count value (decoding vocabulary is 0...max_count).
        """
        super().__init__()
        # A simple linear encoder.
        self._encoder = nn.Linear(input_dim, embed_dim)
        self.embed_dim = embed_dim

        # Use the transformer encoder trained via masked imputation as the decoder.
        self.mi_decoder = TransformerMaskedImputationDecoder(
            latent_dim=embed_dim,
            hidden_dim=embed_dim,
            num_categories=num_categories,
            max_count=max_count,
            num_layers=num_layers,
            mask_prob=mask_prob,
            num_iters=num_iters,
        )

        self.use_variational = True
        self.beta = beta

    def encode(self, x):
        # In this simple example the encoder predicts only a mean.
        mu = self._encoder(x)
        # A very small constant log variance.
        log_var = torch.full_like(mu, 1e-6)
        moments = torch.cat((mu, log_var), dim=1)
        posterior = DiagonalGaussianDistribution(moments, deterministic=True)
        return posterior

    def decode(self, z, outputs=None, log=False, **kwargs):
        """
        Decodes the latent vector z.

        Args:
            z (Tensor): Latent vectors of shape (batch, embed_dim).
            outputs (Tensor or None): Ground-truth tokens (if provided, teacher forcing is used).
            log (bool): For logging debug information.

        Returns:
            If outputs is provided, returns (logits, mask) from the teacher-forced pass;
            otherwise, returns logits from the inference pass.
        """
        if outputs is not None:
            outputs = outputs.long()
            return self.mi_decoder.teacher_forced_forward(z, outputs, log=log)
        else:
            return self.mi_decoder.inference_forward(z, outputs, log=log)

    def forward(self, inputs, outputs=None, disable=True):
        """
        Forward pass for the autoencoder.

        Args:
            inputs (Tensor): Input data, shape (batch, input_dim).
            outputs (Tensor or None): Ground-truth tokens; if provided, teacher forcing is used.
            disable (bool): If True, latent z is set to the posterior mean.

        Returns:
            A dictionary with keys:
              "vae_loss": Total loss.
              "vae_rec_loss": Reconstruction loss.
              "vae_kl_loss": KL divergence loss (here 0).
              "vae_reconstruction": The decoder’s output logits.
        """
        posterior = self.encode(inputs)
        if disable:
            z = posterior.mean
        else:
            z = posterior.sample()

        dec_out = self.decode(z, outputs=outputs)

        if outputs is not None:
            # When teacher forcing is used, dec_out is a tuple: (logits, mask).
            logits, mask = dec_out  # logits shape: (batch, num_categories, max_count+1)
            # Compute cross-entropy loss per time step and average only over masked positions.
            loss_per_token = F.cross_entropy(logits.transpose(1, 2), outputs.long(), reduction="none")
            rec_loss = (loss_per_token * mask.float()).sum() / (mask.float().sum() + 1e-8)
            reconstruction = logits
        else:
            # In inference mode we simply compute cross entropy loss against the argmax prediction (dummy loss).
            logits = dec_out
            rec_loss = F.cross_entropy(logits.transpose(1, 2), logits.argmax(dim=-1), reduction="mean")
            reconstruction = logits

        # Here, for illustration, we set KL loss to zero.
        loss = rec_loss  # + self.beta * kl_loss (if applicable)

        return {
            "vae_loss": loss,
            "vae_rec_loss": rec_loss,
            "vae_kl_loss": 0,
            "vae_reconstruction": reconstruction,
        }


class AutoencoderKL(nn.Module):
    def __init__(
        self,
        embed_dim,
        input_dim,
        num_bits,
        num_categories,
        output_dim=None,
        encoder_hidden_dims=[64, 32],
        decoder_hidden_dims=[32, 64],
        use_variational=True,
        beta=1.0,
        warmup_steps=None,
        annealing_steps=None,
        histogram_head_loss="l2",
        use_linear_encoder=True,
        deterministic=False,
        temperature=1.0,
    ):
        super().__init__()
        assert use_variational
        if output_dim is None:
            output_dim = input_dim
        self.use_variational = use_variational
        self.num_categories = num_categories
        VAE_ENCODER = LinearVAEEncoder if use_linear_encoder else VAEEncoder
        self._encoder = VAE_ENCODER(
            input_dim=input_dim, latent_dim=embed_dim, hidden_dims=encoder_hidden_dims
        )
        self._decoder = VAEDecoder(
            output_dim=output_dim, latent_dim=embed_dim, hidden_dims=decoder_hidden_dims
        )
        self.embed_dim = embed_dim
        self.beta = beta
        self.step = 0
        self.warmup_steps = warmup_steps if warmup_steps else 0
        self.annealing_steps = annealing_steps if annealing_steps else 0
        self.annealing_steps += self.warmup_steps
        self.histogram_head_loss = histogram_head_loss
        self.num_bits = num_bits
        self.deterministic = deterministic
        self.temperature = temperature
        if "balanced" in self.histogram_head_loss:
            self.balanced_ce_loss = nn.CrossEntropyLoss(
                weight=torch.tensor(
                    [
                        0.28054136,
                        0.0004873,
                        0.00046119,
                        0.001836,
                        0.10019334,
                        0.14765334,
                        0.28054136,
                        0.02805413,
                        0.15585631,
                        0.00337595,
                        0.2550376,
                        0.00048764,
                        0.00046633,
                        0.00781452,
                        0.00677636,
                        0.00547932,
                        0.00547932,
                        0.28054136,
                        0,
                        0,
                        0,
                    ],
                    dtype=torch.float32,
                )
            )

        if "clip" in histogram_head_loss:
            self.clip_projection = nn.Linear(embed_dim, embed_dim)
            self.clip_mlp = nn.Sequential(
                nn.Linear(num_categories, embed_dim),
                nn.ReLU(),
                nn.Linear(embed_dim, embed_dim),
                nn.ReLU(),
                nn.Linear(embed_dim, embed_dim),
                nn.ReLU(),
                nn.Linear(embed_dim, embed_dim),
            )

    def encode(self, x):
        mu, log_var = self._encoder(x)
        moments = torch.cat((mu, log_var), 1)
        posterior = DiagonalGaussianDistribution(moments, deterministic=self.deterministic)
        return posterior

    def decode(self, z, inference_method="forward"):
        if inference_method == "clip":
            assert self.histogram_head_loss == "multinomial_clip"
            history_features = self.clip_projection(z)
            if not hasattr(self, "histogram_bank"):
                df = pl.read_parquet(
                    "/storage/shared/mimic-iv/meds_v0.3.2/analysis/zero_shot/train_histograms/64_gptneox_histogram_inference_sample.parquet"
                )
                # cpu_clip_mlp = self.clip_mlp.cpu()
                with torch.no_grad():
                    self.train_histograms = torch.tensor(
                        np.vstack(df["histogram"].to_numpy()), dtype=torch.float32
                    ).to(z.device)
                    self.histogram_bank = self.clip_mlp(self.train_histograms)
                    bank_np = self.histogram_bank.cpu()
                    d = bank_np.shape[1]

                    # flat L2 index on CPU
                    self.faiss_gpu_res = faiss.StandardGpuResources()
                    self.faiss_gpu_index = faiss.GpuIndexFlatL2(self.faiss_gpu_res, d)
                    self.faiss_gpu_index.add(bank_np)

            ### Add code that uses faiss to find the closest vectors in self.histogram_bank for each history_features vector, and then
            ### uses that index to pull the actual histogram from np_histograms, and then loads this onto gpu.
            # hf_np = history_features.detach().cpu().numpy().astype('float32')
            _, I = self.faiss_gpu_index.search(history_features.cpu(), 1)
            nn_idx = I[:, 0]  # flatten to (batch_size,)
            dec = self.train_histograms[nn_idx]

        else:
            dec = self._decoder(z)
        return dec

    def forward(self, inputs, outputs=None, disable=True):
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

        kwargs = {}

        if self.histogram_head_loss == "bce":
            # Compute reconstruction loss (mean squared error)
            rec_loss = torch.nn.functional.binary_cross_entropy_with_logits(dec, outputs, reduction="mean")
        elif self.histogram_head_loss == "focal":
            bce = torch.nn.functional.binary_cross_entropy_with_logits(dec, outputs, reduction="none")
            pt = torch.exp(-bce)  # pt = probability of target class
            gamma = 2.0
            focal_loss = (1 - pt) ** gamma * bce
            rec_loss = focal_loss.mean()
        elif self.histogram_head_loss == "dice":
            pred = torch.sigmoid(dec)  # Convert logits to probabilities
            intersection = (pred * outputs).sum()
            union = pred.sum() + outputs.sum()
            rec_loss = 1 - (2 * intersection + 1e-6) / (union + 1e-6)  # Add epsilon for numerical stability
        elif self.histogram_head_loss == "l2":
            rec_loss = torch.nn.functional.mse_loss(torch.sigmoid(dec), outputs, reduction="mean")
        elif self.histogram_head_loss == "l1":
            rec_loss = torch.nn.functional.l1_loss(torch.sigmoid(dec), outputs, reduction="mean")
        elif self.histogram_head_loss == "softmax_wasserstein":
            pred_histogram, pred_count = dec[:, : -self.num_bits], dec[:, -self.num_bits :]
            target_histogram, target_count = outputs[:, : -self.num_bits], outputs[:, -self.num_bits :]
            rec_loss = torch.nn.functional.mse_loss(torch.sigmoid(pred_count), target_count, reduction="mean")
            pred_histogram = torch.nn.functional.softmax(pred_histogram)
            pred_cdf = torch.cumsum(pred_histogram, dim=-1)
            target_cdf = torch.cumsum(target_histogram, dim=-1)
            # Compute Wasserstein distance as the L1 norm of the difference in CDFs
            rec_loss += torch.abs(pred_cdf - target_cdf).sum(dim=-1).mean()
        elif self.histogram_head_loss == "softmax_chi":
            pred_histogram, pred_count = dec[:, : -self.num_bits], dec[:, -self.num_bits :]
            target_histogram, target_count = outputs[:, : -self.num_bits], outputs[:, -self.num_bits :]
            rec_loss = torch.nn.functional.mse_loss(torch.sigmoid(pred_count), target_count, reduction="mean")
            pred_histogram = torch.nn.functional.softmax(pred_histogram, dim=-1)
            eps = 1e-3
            # Compute Chi-squared loss
            chi_sq = ((pred_histogram - target_histogram) ** 2) / (target_histogram + eps)
            rec_loss += chi_sq.sum(dim=-1).mean()
        elif self.histogram_head_loss == "softmax_l2":
            pred_histogram, pred_count = dec[:, : -self.num_bits], dec[:, -self.num_bits :]
            target_histogram, target_count = outputs[:, : -self.num_bits], outputs[:, -self.num_bits :]
            rec_loss = torch.nn.functional.mse_loss(torch.sigmoid(pred_count), target_count, reduction="mean")
            pred_histogram = torch.nn.functional.softmax(pred_histogram, dim=-1)
            rec_loss += torch.nn.functional.mse_loss(pred_histogram, target_histogram, reduction="mean")
        elif self.histogram_head_loss == "softmax_l1":
            pred_histogram, pred_count = dec[:, : -self.num_bits], dec[:, -self.num_bits :]
            target_histogram, target_count = outputs[:, : -self.num_bits], outputs[:, -self.num_bits :]
            rec_loss = torch.nn.functional.mse_loss(torch.sigmoid(pred_count), target_count, reduction="mean")
            pred_histogram = torch.nn.functional.softmax(pred_histogram, dim=-1)
            rec_loss += torch.nn.functional.l1_loss(pred_histogram, target_histogram, reduction="mean")
        elif self.histogram_head_loss == "cont_softmax_multinomial":
            pred_histogram = dec
            target_histogram = outputs
            log_probs = torch.nn.functional.log_softmax(pred_histogram, dim=-1)
            rec_loss = -(target_histogram * log_probs).sum(dim=-1).mean()
        elif self.histogram_head_loss == "cont_softmax_multinomial_balanced":
            pred_histogram = dec
            target_histogram = outputs
            rec_loss = self.balanced_ce_loss(pred_histogram, target_histogram)

        elif self.histogram_head_loss == "multinomial":
            dec = dec.reshape(*outputs.shape, -1)
            rec_loss = torch.nn.functional.cross_entropy(
                dec.transpose(1, 2), outputs.long(), reduction="mean"
            )
        elif self.histogram_head_loss == "multinomial_clip":
            # Reshape dec to have an extra dimension for bins.
            dec = dec.reshape(*outputs.shape, -1)
            # Compute the base multinomial cross-entropy reconstruction loss.
            rec_loss = torch.nn.functional.cross_entropy(
                dec.transpose(1, 2).cpu().detach(), outputs.long().cpu().detach(), reduction="mean"
            )
            if outputs is not None:
                # Compute the 'history' features from z via a linear transformation.
                history_features = self.clip_projection(z)  # shape: [n, d_i]
                # Compute the 'histogram' features from outputs via an MLP.
                histogram_features = self.clip_mlp(outputs)  # shape: [n, d_t]

                # Obtain joint embeddings by L2-normalizing these features.
                history_embedding = torch.nn.functional.normalize(history_features, p=2, dim=1)
                histogram_embedding = torch.nn.functional.normalize(histogram_features, p=2, dim=1)

                # Compute scaled cosine similarity logits.
                # (Multiplying by exp(temperature) follows your pseudo-code; typically, temperature might be applied as a division factor.)
                logits = torch.matmul(history_embedding, histogram_embedding.T) * self.temperature

                # Generate labels: [0, 1, ..., n-1]
                labels = torch.arange(history_embedding.shape[0], device=logits.device)

                # Compute symmetric cross entropy loss:
                # loss over rows: each history embedding is matched to the corresponding histogram embedding.
                loss_history = torch.nn.functional.cross_entropy(logits, labels)
                # loss over columns: each histogram embedding is matched to the corresponding history embedding.
                loss_histogram = torch.nn.functional.cross_entropy(logits.T, labels)
                clip_loss = (loss_history + loss_histogram) / 2.0

                # Add the clip loss to the base reconstruction loss.
                rec_loss = rec_loss + clip_loss
                kwargs = {"vae_clip_loss": clip_loss.item(), "vae_isolated_rec_loss": rec_loss.item()}

        elif self.histogram_head_loss == "multinomial_earthmover":
            # Reshape dec to have the same shape as outputs with an extra dimension for bins.
            dec = dec.reshape(*outputs.shape, -1)  # shape: (batch, ..., num_bins)

            # Convert logits to a probability distribution
            p = torch.softmax(dec, dim=-1)

            # Compute the cumulative distribution (CDF) of the predicted probabilities
            cdf_pred = torch.cumsum(p, dim=-1)

            # Create a one-hot encoding of the target outputs.
            # outputs is assumed to be a tensor of indices with shape matching dec (without the last dim)
            target_onehot = torch.zeros_like(dec, dtype=torch.int64)
            target_onehot.scatter_(-1, outputs.unsqueeze(-1).to(torch.int64), 1)

            # Compute the CDF of the true distribution
            cdf_true = torch.cumsum(target_onehot, dim=-1)

            # Compute the Earth Mover's Distance (L1 norm between the CDFs)
            rec_loss = torch.mean(torch.abs(cdf_pred - cdf_true))
        else:
            raise ValueError(f"Invalid model.histogram_head_loss of: {self.histogram_head_loss}")

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
            "vae_kl_loss": kl_loss * self.beta,
            "vae_reconstruction": dec,
            **kwargs,
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


def get_previous_h_token_embedding(code, ntp_token):
    # Create a boolean mask where the token matches h_token.
    mask = code == ntp_token
    # Reverse the mask along the sequence dimension.
    reversed_mask = mask.flip(dims=[1])
    # Get the index of the first occurrence in the reversed mask.
    last_idx_from_end = reversed_mask.float().argmax(dim=1)
    # Convert that into the corresponding index in the original tensor.
    h_token_idx = code.size(1) - 2 - last_idx_from_end
    return h_token_idx


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
        self.train_next_token_metric = NextTokenPredictionMetric(self.cfg.vocab_size, [], False)
        self.val_next_token_metric = NextTokenPredictionMetric(
            self.cfg.vocab_size, self.cfg.top_k_acc, self.cfg.next_token_auc
        )
        self.test_next_token_metric = NextTokenPredictionMetric(self.cfg.vocab_size, [], False)

        self.train_histogram_metric = HistogramMetric()
        self.val_histogram_metric = HistogramMetric()
        self.test_histogram_metric = HistogramMetric()

        self.metadata_df = pl.read_parquet(self.cfg.augmented_code_metadata_fp)
        self.trajectory_labeler = self.cfg.get("trajectory_labeler", None)
        self.initialize_weights()

        self.h_token = self.metadata_df.filter(pl.col("code") == "[H]")["code/vocab_index"][-1]
        self.ntp_token = self.metadata_df.filter(pl.col("code") == "[NTP]")["code/vocab_index"][-1]

        self.subvocab_h_token = self.metadata_df.filter(pl.col("code") == "[H]")["code/subvocab_index"][-1]
        self.subvocab_ntp_token = self.metadata_df.filter(pl.col("code") == "[NTP]")["code/subvocab_index"][
            -1
        ]

        self.histogram_normalizer = HistogramNormalizer(
            self.subvocab_h_token,
            self.subvocab_ntp_token,
            self.cfg.subvocab_size,
            histogram_head_loss=self.cfg.histogram_head_loss,
            num_bits=self.cfg.n_bits,
            scale=self.cfg.scale,
            use_diffusion=self.cfg.use_diffusion,
            max_count=self.cfg.max_count,
            normalize=self.cfg.normalize_histogram,
        )
        histogram_dim = self.histogram_normalizer.get_normalized_size()

        if self.cfg.autoencoder_type == "autoregressive":
            self.autoencoder = AutoencoderKL_Autoregressive(
                self.cfg.encoder_dims[-1],
                self.cfg.token_dim,
                self.cfg.subvocab_size,
                self.cfg.encoder_dims,
                self.cfg.beta,
                self.cfg.max_count,
                aux_loss=self.cfg.aux_loss,
                aux_lambda=self.cfg.aux_lambda,
            )
        elif self.cfg.autoencoder_type == "vae":
            self.autoencoder = AutoencoderKL(
                use_linear_encoder=self.cfg.use_linear_encoder,
                embed_dim=self.cfg.encoder_dims[-1],
                input_dim=self.cfg.token_dim,
                num_bits=self.cfg.n_bits,
                num_categories=self.cfg.subvocab_size,
                output_dim=histogram_dim,
                encoder_hidden_dims=self.cfg.encoder_dims,
                decoder_hidden_dims=self.cfg.decoder_dims,
                use_variational=True,
                beta=self.cfg.beta,
                warmup_steps=self.cfg.warmup_steps,
                annealing_steps=self.cfg.annealing_steps,
                histogram_head_loss=self.cfg.histogram_head_loss,
                deterministic=self.cfg.deterministic,
            )
        elif self.cfg.autoencoder_type == "masked_imputation":
            self.autoencoder = AutoencoderKL_MI(
                embed_dim=self.cfg.encoder_dims[-1],
                input_dim=self.cfg.token_dim,
                num_categories=self.cfg.subvocab_size,
                encoder_hidden_dims=self.cfg.encoder_dims,
                beta=self.cfg.beta,
                max_count=self.cfg.max_count,
                num_layers=self.cfg.autoencoder_mi_num_layers,
                mask_prob=self.cfg.autoencoder_mi_mask_prob,
                num_iters=self.cfg.autoencoder_mi_num_iters,
            )
        else:
            raise ValueError(f"Unknown autoencoder type: {self.cfg.autoencoder_type}")
        self.subvocab_mapper = SubvocabMapper(metadata_df=self.metadata_df)

        from meds_torch.models.diffusion_utils.diffloss import DiffLoss

        self.diffusion = DiffLoss(
            target_channels=histogram_dim,
            z_channels=self.cfg.token_dim,
            width=self.cfg.token_dim,
            depth=12,
            num_sampling_steps="100",
            grad_checkpointing=False,
            noise_schedule=self.cfg.diffusion_noise_schedule,
        )

        EOS_TOKENS = self.metadata_df.filter(pl.col("code") == "[EOS]")["code/vocab_index"]
        if len(EOS_TOKENS) >= 1:
            self.EOS_TOKEN_ID = EOS_TOKENS[-1]
        else:
            self.EOS_TOKEN_ID = None

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
            code_logits = model_output
        else:
            code_logits = model_output[BACKBONE_TOKENS_KEY]
        # code_logits = self.code_head(all_token_embeddings)
        # histogram_mask = model_output["histogram"] > 0
        # code_logits[~histogram_mask] = -float("inf")

        return {
            CODE_LOGITS: code_logits,
        }

    @TimeableMixin.TimeAs
    def get_histogram_loss(self, prompts, histogram, embeddings, mask):
        # Simpler and identical processing:
        # ntp_mask = (prompts == self.ntp_token)
        # valid_h_mask = ntp_mask[: 1:]
        # patch_embeddings = embeddings[:, :-1][valid_h_mask]
        # num_histogram_samples = ntp_mask.sum()
        # target = histogram[ntp_mask, :]

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
            normalized_gt_histogram = self.histogram_normalizer.transform(target.detach().clone())
        # Forward pass and loss computation
        repeated_histograms = normalized_gt_histogram.detach().repeat(self.cfg.histogram_batch_mul, 1)
        loss_dict = self.autoencoder.forward(
            inputs=patch_embeddings.reshape(num_histogram_samples, -1).repeat(
                self.cfg.histogram_batch_mul, 1
            ),
            outputs=repeated_histograms,
        )
        # self.histogram_normalizer.reverse_transform(self.autoencoder.decode(self.autoencoder.encode(patch_embeddings).mean))[0][0]
        # normalized_gt_histogram[0]
        # breakpoint()
        # Why is the reverse transform incorrect, oh for the gt data it expects logits I think.
        loss = loss_dict["vae_loss"]
        loss_dict = {"MODEL//" + k: v for k, v in loss_dict.items() if k != "vae_reconstruction"}
        assert not torch.isnan(loss).any(), "histogram loss is NaN"
        return loss, loss_dict

    @TimeableMixin.TimeAs
    def forward(self, batch, keep_code_logits=False, skip_input_encoder=False):
        if skip_input_encoder:
            assert INPUT_ENCODER_TOKENS_KEY in batch
            assert INPUT_ENCODER_MASK_KEY in batch
        else:
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
        if self.cfg.histogram_loss_weight != 0:
            histogram_loss, loss_dict = self.get_histogram_loss(
                batch["code"], batch["histogram"], embeddings, batch["mask"]
            )
            model_loss = code_loss + self.cfg.histogram_loss_weight * histogram_loss
        else:
            model_loss = code_loss
            loss_dict = {}

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
            if loss_key in batch:
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

    def predict_step(self, batch):
        batch = self(batch, False)
        gen_key = GENERATE_PREFIX + str(self.cfg.generate_id)
        output = {gen_key: batch[gen_key]}
        if MODEL_PRED_STATUS_KEY in batch:
            output[MODEL_PRED_STATUS_KEY] = batch[MODEL_PRED_STATUS_KEY]
        if MODEL_PRED_PROBA_KEY in batch:
            output[MODEL_PRED_PROBA_KEY] = batch[MODEL_PRED_PROBA_KEY]
        return output

    def training_step(self, batch):
        batch = self(batch, True)
        assert not torch.isnan(batch[MODEL_BATCH_LOSS_KEY]), "Loss is NaN"
        self._log(batch, "train")
        del batch[CODE_LOGITS]
        self.train_histogram_metric(*self.get_eval_histograms(batch))
        return batch[MODEL_BATCH_LOSS_KEY]

    def on_train_epoch_end(self):
        next_token_results = self.train_next_token_metric.compute()
        for metric_name, value in next_token_results.items():
            self.log(f"test/NEXT_TOKEN/{metric_name.upper()}", value, on_epoch=True)
        self.train_next_token_metric.reset()

    def get_eval_histograms(self, batch):
        histogram_idx = get_previous_h_token_embedding(batch["code"], self.ntp_token)
        embedding = batch["MODEL//EMBEDDINGS"][torch.arange(batch["code"].shape[0]), histogram_idx, :]
        next_histogram_posterior: DiagonalGaussianDistribution = self.autoencoder.encode(embedding)
        if next_histogram_posterior.deterministic:
            mean_pred_histogram, _ = self.histogram_normalizer.reverse_transform(
                self.autoencoder.decode(next_histogram_posterior.mean, inference_method="best_of_n")
            )
            sample_pred_histogram, _ = self.histogram_normalizer.reverse_transform(
                self.autoencoder.decode(next_histogram_posterior.mean, inference_method="forward")
            )
        else:
            mean_pred_histogram, _ = self.histogram_normalizer.reverse_transform(
                self.autoencoder.decode(next_histogram_posterior.mean)
            )
            sample_pred_histogram, _ = self.histogram_normalizer.reverse_transform(
                self.autoencoder.decode(next_histogram_posterior.sample())
            )
        true_histogram = batch["histogram"][torch.arange(batch["code"].shape[0]), histogram_idx + 1, :]
        return true_histogram.float(), mean_pred_histogram.float(), sample_pred_histogram.float()

    def validation_step(self, batch):
        batch = self(batch, True)
        assert not torch.isnan(batch[MODEL_BATCH_LOSS_KEY]), "Loss is NaN"
        self._log(batch, "val")
        del batch[CODE_LOGITS]
        self.val_histogram_metric(*self.get_eval_histograms(batch))
        return batch[MODEL_BATCH_LOSS_KEY]

    def on_validation_epoch_end(self):
        next_token_results = self.val_next_token_metric.compute()
        for metric_name, value in next_token_results.items():
            self.log(f"val/NEXT_TOKEN/{metric_name.upper()}", value, on_epoch=True)
        self.val_next_token_metric.reset()

        try:
            # Compute and log the histogram metric's summed MAE
            histogram_mean_mae, histogram_sample_mae = self.val_histogram_metric.compute()
            self.log("val/HISTOGRAM_MEAN_MAE", histogram_mean_mae, on_epoch=True)
            self.log("val/HISTOGRAM_SAMPLE_MAE", histogram_sample_mae, on_epoch=True)
            # Generate the custom histogram plots
            fig, axs = self.val_histogram_metric.plot()
            # Log the plot to wandb
            import wandb

            # self.logger.experiment is the wandb run object if using WandbLogger
            self.logger.experiment.log({"val/HISTOGRAM_PLOT": wandb.Image(fig)}, commit=False)
        except:
            pass
        # Reset histogram metric state for the next epoch
        self.val_histogram_metric.reset()

    def test_step(self, batch):
        batch = self(batch, True)
        assert not torch.isnan(batch[MODEL_BATCH_LOSS_KEY]), "Loss is NaN"
        self._log(batch, "test")
        del batch[CODE_LOGITS]
        loss = batch[MODEL_BATCH_LOSS_KEY]
        self.test_histogram_metric(*self.get_eval_histograms(batch))
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
                if unknown.any().item() > 0 and labels is not None:
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
                histogram_based_logit_mask = self.subvocab_mapper.from_subvocab_histogram(prev_histogram > 0)
                logits[~histogram_based_logit_mask] = -float("inf")

                embeddings = embeddings[:, -1, :]
                full_embeddings = torch.zeros(
                    (b, embeddings.shape[1]), device=embeddings.device, dtype=embeddings.dtype
                )
                full_embeddings[orig_indices[active_indices]] = embeddings
                embeddings = full_embeddings

                if self.cfg.use_diffusion:
                    next_ae_histogram, count = self.histogram_normalizer.reverse_transform(
                        self.diffusion.sample(embeddings, temperature=temperature)
                    )
                else:
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

                    # Always allow censored token sampling
                    if self.EOS_TOKEN_ID is not None:
                        mask[~is_histogram_only_h_o_tokens, self.EOS_TOKEN_ID] = True

                    filtered_logits = filtered_logits.masked_fill(~mask, float("-inf"))
                    probs = F.softmax(filtered_logits / temperature, dim=-1)
                    sample = torch.multinomial(probs, 1)

                # TODO: handle subvocab histogram vocabulary
                subvocab_sample = self.subvocab_mapper.to_subvocab(sample)
                one_hot_sample = torch.zeros_like(prev_histogram).scatter_(1, subvocab_sample, 1)
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
        if self.cfg.use_diffusion:
            latent_next_histogram_posterior = None
            next_histogram, counts = self.histogram_normalizer.reverse_transform(
                self.diffusion.sample(last_embeddings, temperature=temperature)
            )
        else:
            latent_next_histogram_posterior = self.autoencoder.encode(last_embeddings)
            next_histogram, counts = self.histogram_normalizer.reverse_transform(
                self.autoencoder.decode(
                    latent_next_histogram_posterior.sample(), inference_method=self.cfg.inference_method
                )
            )

        # Mask token logits based on histogram
        prev_histogram = batch["histogram"][:, -1]
        histogram_based_logit_mask = self.subvocab_mapper.from_subvocab_histogram(prev_histogram > 0)
        next_token_logits[~histogram_based_logit_mask] = -float("inf")

        return next_token_logits, next_histogram, latent_next_histogram_posterior, counts, last_embeddings

    def hf_get_sample(
        self,
        logits,
        prev_histogram,
        use_histogram_multiplier: bool = False,
        ignore_histogram_for_eos: bool = False,
    ):
        """
        Get next token probabilities and sample one token.

        Args:
            logits: Logits for the current token [batch_size, seq_len, vocab_size]
            prev_histogram: Previous token histogram (either counts or binary) used to compute a mask/multiplier.
            use_histogram_multiplier: If true, multiply the probabilities by the count in the histogram.
                                Otherwise, use a binary mask (nonzero entries become 1).

        Returns:
            sample: Sampled next token [batch_size, 1]
        """
        # Use the logits for the last token only.
        logits = logits[:, -1]

        # Compute the base probability distribution with temperature scaling.
        base_probs = F.softmax(logits / self.cfg.temperature, dim=-1)

        # Get the histogram-based multiplier:
        # Either a count tensor (if use_histogram_multiplier is true) or a boolean mask.
        if use_histogram_multiplier:
            # Multiply probabilities by the actual count values.
            hist_multiplier = self.subvocab_mapper.from_subvocab_histogram_counts(prev_histogram)
            # Ensure the counts are float-compatible.
            hist_multiplier = hist_multiplier.to(base_probs.dtype)
        else:
            # Use a boolean mask indicating where counts are nonzero.
            hist_mask = self.subvocab_mapper.from_subvocab_histogram(prev_histogram > 0)
            hist_multiplier = hist_mask.to(base_probs.dtype)

        if ignore_histogram_for_eos:
            for eos_token in self.cfg.eos_tokens:
                hist_multiplier[:, eos_token] = 1.0

        # Apply the histogram multiplier to the base probabilities.
        adjusted_probs = base_probs * hist_multiplier

        # Create a custom mask that initially disables the special tokens.
        mask = torch.ones_like(logits, dtype=torch.bool)
        mask[..., self.h_token] = False
        mask[..., self.ntp_token] = False

        # Count the number of tokens with nonzero probability in the adjusted distribution.
        nonzero_probs_count = (adjusted_probs > 0).sum(dim=-1)
        is_histogram_only_h_o_tokens = nonzero_probs_count <= 2

        # Check if the special tokens have nonzero probability.
        has_h_token = adjusted_probs[:, self.h_token] > 0
        has_o_token = adjusted_probs[:, self.ntp_token] > 0

        # For batches with nearly no allowed tokens, enable the special tokens as needed.
        mask[is_histogram_only_h_o_tokens & has_h_token, self.h_token] = True
        mask[is_histogram_only_h_o_tokens & (~has_h_token) & has_o_token, self.ntp_token] = True

        # Apply the custom mask to the adjusted probabilities.
        final_probs = adjusted_probs.masked_fill(~mask, 0).clip(0, None)
        # Renormalize the probability distribution.
        final_probs = final_probs / (final_probs.sum(dim=-1, keepdim=True) + 1e-8)

        # Sample from the final probability distribution.
        sample = torch.multinomial(final_probs.clip(0, 1), 1)
        return sample

    def hf_update_histogram(
        self, embeddings: torch.Tensor, sample, prev_histogram, prev_sample, get_metadata: bool = False
    ):
        last_embeddings = embeddings[:, -1]
        subvocab_sample = self.subvocab_mapper.to_subvocab(sample)
        one_hot_sample = torch.zeros_like(prev_histogram).scatter_(1, subvocab_sample, 1)
        next_decrement_histogram = prev_histogram - one_hot_sample
        next_histogram = next_decrement_histogram
        h_token_histogram_mask = prev_sample == self.h_token
        next_histogram_posterior = None
        entropy_list = None
        likelihood_list = None
        embedding_list = None
        sample_list = None
        if h_token_histogram_mask.any():
            if self.cfg.use_diffusion:
                next_ae_histogram, count = self.histogram_normalizer.reverse_transform(
                    self.diffusion.sample(
                        last_embeddings[h_token_histogram_mask], temperature=self.cfg.temperature
                    )
                )
            else:
                next_histogram_posterior = self.autoencoder.encode(last_embeddings[h_token_histogram_mask])
                autoencoder_output = self.autoencoder.decode(
                    next_histogram_posterior.sample(),
                    inference_method=self.cfg.inference_method,
                    get_metadata=get_metadata,
                )
                if isinstance(autoencoder_output, torch.Tensor):
                    next_ae_histogram_binary = autoencoder_output
                elif isinstance(autoencoder_output, HistogramSample):
                    assert isinstance(autoencoder_output, HistogramSample)
                    next_ae_histogram_binary = autoencoder_output.tokens
                    entropy_list = autoencoder_output.entropy
                    likelihood_list = autoencoder_output.likelihood
                    h_token_histogram_mask.cumsum

                    if get_metadata:
                        entropy_iter = iter(autoencoder_output.entropy.detach().cpu())
                        entropy_list = [
                            next(entropy_iter).tolist() if m else []
                            for m in h_token_histogram_mask.detach().cpu()
                        ]
                        ll_iter = iter(autoencoder_output.likelihood.detach().cpu())
                        likelihood_list = [
                            next(ll_iter).tolist() if m else [] for m in h_token_histogram_mask.detach().cpu()
                        ]
                        embedding_iter = iter(last_embeddings.detach().cpu())
                        embedding_list = [
                            next(embedding_iter).tolist() if m else []
                            for m in h_token_histogram_mask.detach().cpu()
                        ]
                        sample_iter = iter(autoencoder_output.tokens.detach().cpu())
                        sample_list = [
                            next(sample_iter).tolist() if m else []
                            for m in h_token_histogram_mask.detach().cpu()
                        ]
                else:
                    raise ValueError(f"Autoencoder output type {type(autoencoder_output)} not supported.")
                next_ae_histogram, count = self.histogram_normalizer.reverse_transform(
                    next_ae_histogram_binary
                )

            next_histogram[h_token_histogram_mask] = next_ae_histogram.float()

        if (next_histogram == 0).all():
            raise ValueError("All histogram counts are zero somehow, this should not happen.")

        return (
            next_histogram,
            next_histogram_posterior,
            {
                "entropy": entropy_list,
                "ll": likelihood_list,
                "embedding": embedding_list,
                "sample": sample_list,
            },
        )

    # def hf_update_histogram(self, embeddings: torch.Tensor, sample, prev_histogram, prev_sample, get_metadata: bool = False):
    #     last_embeddings = embeddings[:, -1]
    #     subvocab_sample = self.subvocab_mapper.to_subvocab(sample)
    #     one_hot_sample = torch.zeros_like(prev_histogram).scatter_(1, subvocab_sample, 1)
    #     next_decrement_histogram = prev_histogram - one_hot_sample

    #     if self.cfg.use_diffusion:
    #         next_ae_histogram, count = self.histogram_normalizer.reverse_transform(
    #             self.diffusion.sample(last_embeddings, temperature=self.cfg.temperature)
    #         )
    #     else:
    #         next_histogram_posterior = self.autoencoder.encode(last_embeddings)
    #         autoencoder_output = self.autoencoder.decode(
    #             next_histogram_posterior.sample(), inference_method=self.cfg.inference_method
    #         )
    #         next_ae_histogram_binary = autoencoder_output.tokens
    #         next_ae_histogram, count = self.histogram_normalizer.reverse_transform(next_ae_histogram_binary)

    #     h_token_histogram_mask = (
    #         (prev_sample == self.h_token).reshape(-1, 1).repeat(1, next_ae_histogram.shape[1])
    #     )
    #     next_histogram = torch.where(h_token_histogram_mask, next_ae_histogram, next_decrement_histogram)

    #     if (next_histogram == 0).all():
    #         raise ValueError("All histogram counts are zero somehow, this should not happen.")

    #     return next_histogram, next_histogram_posterior, None, None

    def update_metadata(self, metadata, metadata_sample):
        for k, v in metadata.items():
            for i in range(len(v)):
                if metadata_sample[k] is None:
                    v[i].append(None)
                else:
                    v[i].append(metadata_sample[k][i])

    @torch.no_grad()
    def hf_generate(
        self,
        batch,
        use_guidance: bool = True,
        use_histogram_multiplier: bool = False,
        ignore_histogram_for_eos: bool = False,
        num_samples: None | int = None,
        get_metadata: bool = False,
        token_bin_guidance: bool = True,
    ):
        batch = self.input_encoder(batch)
        if hasattr(self.input_encoder, "prompt_mlp"):
            batch = self.model(batch, do_get_last_token=False)
            histogram_embeddings = batch[BACKBONE_EMBEDDINGS_KEY]
            if self.cfg.encode_for_prompt_tuning:
                histogram_embeddings = self.autoencoder.encode(histogram_embeddings).mean
            batch = self.input_encoder(batch, histogram_embedding=histogram_embeddings)

        gpt_model: GPTNeoXForCausalLM = self.model.model.model
        samples = batch["code"]
        input_data, input_mask = batch[INPUT_ENCODER_TOKENS_KEY], batch[INPUT_ENCODER_MASK_KEY]
        input_mask = batch["mask"].float()
        remaining_tokens = (
            self.cfg.max_seq_len - batch["code"].shape[1] if num_samples is None else num_samples
        )
        kv_cache = None
        prev_histogram = batch["histogram"][:, -1]
        from tqdm.auto import trange

        if get_metadata:
            metadata = {
                "entropy": [[] for _ in range(input_data.shape[0])],
                "ll": [[] for _ in range(input_data.shape[0])],
                "sample": [[] for _ in range(input_data.shape[0])],
                "embedding": [[] for _ in range(input_data.shape[0])],
            }
        else:
            metadata = None

        count = 1

        for _ in trange(remaining_tokens):
            if len(input_data.shape) == 2:
                kwargs = dict(input_ids=input_data)
            elif len(input_data.shape) == 3:
                kwargs = dict(inputs_embeds=input_data)
            else:
                raise ValueError(f"Invalid input_data shape: {input_data.shape}")
            output = gpt_model.forward(
                **kwargs,
                attention_mask=input_mask,
                return_dict=True,
                output_hidden_states=True,
                past_key_values=kv_cache,
                use_cache=True,
            )
            embeddings = output.hidden_states[-1]
            logits = output.logits
            kv_cache = output.past_key_values
            if use_guidance:
                if hasattr(self.input_encoder, "prompt_mlp"):
                    raise ValueError("Prompt mlp not supported for guided histogram forecasting.")
                sample = self.hf_get_sample(
                    logits, prev_histogram, use_histogram_multiplier, ignore_histogram_for_eos
                )
                prev_sample = samples[:, -1]
                prev_histogram, _, metadata_sample = self.hf_update_histogram(
                    embeddings, sample, prev_histogram, prev_sample, get_metadata
                )
                if get_metadata:
                    self.update_metadata(metadata, metadata_sample)

                # Append new tokens
                samples = torch.cat((samples, sample), dim=-1)
                if hasattr(self.input_encoder, "process_sample"):
                    next_sample_embedding = self.input_encoder.process_sample(
                        sample, prev_histogram.unsqueeze(1)
                    )
                else:
                    next_sample_embedding = sample
                input_data = next_sample_embedding
                input_mask = (
                    torch.ones(input_mask.shape[0]).to(input_mask.device, dtype=torch.float32).unsqueeze(-1)
                )
            else:
                if token_bin_guidance:
                    num_tokens_in_bin = self.cfg.token_bin_size + 2
                    if count % num_tokens_in_bin == 0:  # should generate H tokens
                        sample = torch.full_like(samples[:, -1], self.h_token).unsqueeze(-1)
                    elif count % num_tokens_in_bin == 1:  # should generate NTP tokens
                        sample = torch.full_like(samples[:, -1], self.ntp_token).unsqueeze(-1)
                    else:
                        last_logits = output.logits[:, -1]
                        last_logits[:, self.h_token] = float("-inf")
                        last_logits[:, self.ntp_token] = float("-inf")
                        probs = F.softmax(last_logits / self.cfg.temperature, dim=-1)
                        sample = torch.multinomial(probs, 1)
                else:
                    probs = F.softmax(output.logits[:, -1] / self.cfg.temperature, dim=-1)
                    sample = torch.multinomial(probs, 1)

                if get_metadata or hasattr(self.input_encoder, "process_sample"):
                    prev_sample = samples[:, -1]
                    prev_histogram, _, metadata_sample = self.hf_update_histogram(
                        embeddings, sample, prev_histogram, prev_sample, get_metadata
                    )
                    if get_metadata:
                        self.update_metadata(metadata, metadata_sample)

                # Append new tokens
                samples = torch.cat((samples, sample), dim=-1)
                if hasattr(self.input_encoder, "process_sample"):
                    kwargs = {}
                    if hasattr(self.input_encoder, "prompt_mlp"):
                        histogram_embeddings = embeddings[:, -1]
                        if self.cfg.encode_for_prompt_tuning:
                            histogram_embeddings = self.autoencoder.encode(histogram_embeddings).mean
                        kwargs["histogram_embedding"] = histogram_embeddings
                    next_sample_embedding = self.input_encoder.process_sample(
                        sample, prev_histogram.unsqueeze(1), **kwargs
                    )
                else:
                    next_sample_embedding = sample
                input_data = next_sample_embedding
                input_mask = (
                    torch.ones(input_mask.shape[0]).to(input_mask.device, dtype=torch.float32).unsqueeze(-1)
                )
            count += 1
        return samples, metadata
