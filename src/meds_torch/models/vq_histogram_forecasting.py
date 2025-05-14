import re
from collections.abc import Iterator

import numpy as np
import polars as pl
import torch
import torch.nn.functional as F
from loguru import logger
from omegaconf import DictConfig
from torchmetrics import Metric
from transformers import GPTNeoXForCausalLM

from meds_torch.utils import RankedLogger

log = RankedLogger(__name__, rank_zero_only=True)

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
    MODEL_TOKENS_KEY,
)
from meds_torch.models.base_model import BaseModule
from meds_torch.models.components.utils import TrajectoryBatch
from meds_torch.models.eic_forecasting import NextTokenPredictionMetric
from meds_torch.utils import TIME_DELTA_TOKEN

MODEL_LOSS_KEYS = [
    "MODEL//code_loss",
    "MODEL//vq_loss",
    "MODEL//vq_rec_loss",
    "MODEL//vq_aux_loss",
    "MODEL//vq_true_lp",
    "MODEL//vq_sample_lp",
    "MODEL//vq_isolated_rec_loss",
]


import torch
from omegaconf import ListConfig
from torchmetrics import Metric, MetricCollection
from torchmetrics.text import Perplexity


class TopKMulticlassAccuracy(Metric):
    """Computes the fraction of samples that yield a correct prediction within their top-k class logits.

    The torchmetrics codebase Accuracy metric does not perform what I expect it to in top-k and multi-class
    settings. This metric behaves as expected.

    See https://github.com/Lightning-AI/torchmetrics/issues/3068 for tracking.

    Args:
        top_k: A prediction for a given sequence event is correct if the true label is among the top `top_k`
            predicted logits.
        ignore_index: The index to ignore when computing the accuracy. This is useful for ignoring padding
            indices. For this metric, it must be set.

    Returns:
        A (scalar) tensor with the accuracy of the top-k predictions. If there are no valid predictions (e.g.,
        all labels are padding), returns 0.0.

    Raises:
        ValueError: If the logits and target tensors do not have the same shape in the batch (0) and sequence
            (1) dimensions.

    Examples:
        >>> code = torch.LongTensor([[3, 2], [1, 0]])
        >>> logits = torch.FloatTensor(
        ...     [
        ...         [[0.0, 0.1, 0.5, 0.4], [0.0, 0.2, 0.7, 0.1]], # Prediction orders: 2, 3, 1; 2, 1, 3
        ...         [[0.0, 0.4, 0.3, 0.3], [1.0, 0.0, 0.0, 0.0]], # Prediction orders: 1, 2 & 3; padding
        ...     ]
        ... )
        >>> TopKMulticlassAccuracy(top_k=2, ignore_index=0)(logits, code)
        tensor(1.)
        >>> TopKMulticlassAccuracy(top_k=1, ignore_index=0)(logits, code)
        tensor(0.6667)
        >>> code = torch.LongTensor([[1, 1], [1, 0]])
        >>> TopKMulticlassAccuracy(top_k=1, ignore_index=0)(logits, code)
        tensor(0.3333)
        >>> TopKMulticlassAccuracy(top_k=2, ignore_index=0)(logits, code)
        tensor(0.6667)
        >>> TopKMulticlassAccuracy(top_k=3, ignore_index=0)(logits, code)
        tensor(1.)
        >>> code = torch.LongTensor([[1, 0], [0, 0]])
        >>> TopKMulticlassAccuracy(top_k=3, ignore_index=0)(logits, code)
        tensor(1.)
        >>> TopKMulticlassAccuracy(top_k=2, ignore_index=0)(logits, code)
        tensor(0.)

    If all labels are ignored, the accuracy is 0.0:

        >>> code = torch.LongTensor([[0, 0], [0, 0]])
        >>> TopKMulticlassAccuracy(top_k=2, ignore_index=0)(logits, code)
        tensor(0.)

    Errors are raised if the shapes are misaligned:

        >>> code = torch.LongTensor([[1, 1, 0], [0, 3, 0]])
        >>> TopKMulticlassAccuracy(top_k=2, ignore_index=0)(logits, code)
        Traceback (most recent call last):
            ...
        ValueError: logits and target must have the same shape in the batch (0) and sequence (1) dimensions.
            Got torch.Size([2, 2, 4]) and torch.Size([2, 3])
    """

    top_k: int
    ignore_index: int

    def __init__(self, top_k: int, ignore_index: int = 0, **kwargs):
        super().__init__(**kwargs)

        self.top_k = top_k
        self.ignore_index = ignore_index

        self.add_state("correct", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, logits: torch.FloatTensor, target: torch.LongTensor) -> None:
        batch, seq_len = target.shape
        if logits.shape[0] != batch or logits.shape[1] != seq_len:
            raise ValueError(
                "logits and target must have the same shape in the batch (0) and sequence (1) dimensions. "
                f"Got {logits.shape} and {target.shape}"
            )

        topk = torch.topk(logits, k=self.top_k, dim=2).indices  # batch x seq_len x top_k
        correct = (topk == target.unsqueeze(2)).any(dim=2)  # batch x seq_len

        keep_mask = target != self.ignore_index

        self.correct += correct[keep_mask].sum()
        self.total += keep_mask.sum()

    def compute(self) -> torch.FloatTensor:
        safe_denom = torch.where(self.total > 0, self.total, torch.ones_like(self.total))

        return torch.where(self.total > 0, self.correct.float() / safe_denom, torch.tensor(0.0))


class NextCodeMetrics(Metric):
    """A `torchmetrics` Metric for next code prediction in Autoregressive "Everything-is-code" models.

    This module is largely a simple wrapper around `torchmetrics.MetricCollection` to enable configuration and
    code isolation and to seamlessly slice the predictions and tokens to the right shapes.

    Supported metrics:
      - Top-$k$ accuracy
      - Perplexity

    Supported Vocabulary Subdivisions:
      - All codes

    Attributes:
        accuracies: The top-$k$ accuracy metrics contained in this metric collection.
        perplexity: The perplexity metric.

    Examples:
        >>> M = NextCodeMetrics(top_k=[1, 2, 3], vocab_size=4)

    To show it in use, we'll need some codes (targets) and logits (predictions):

        >>> code = torch.LongTensor([[1, 3, 2], [2, 1, 0]])
        >>> logits = torch.FloatTensor([
        ...    [
        ...        [0.0, 0.1, 0.5, 0.4], # Label is 3; Prediction order is 2, 3, 1
        ...        [0.0, 0.2, 0.7, 0.1], # Label is 2; Prediction order is 2, 1, 3
        ...        [0.0, 1.0, 0.0, 0.0], # No label; Should be dropped.
        ...    ], [
        ...        [0.0, 0.4, 0.3, 0.3], # Label is 1; Prediction order is 1, 2 & 3
        ...        [1.0, 0.0, 0.0, 0.0], # Padding label; Should be ignored.
        ...        [0.0, 0.0, 2.0, 0.0], # No label; Should be dropped.
        ...    ]
        ... ])

    We'll make a mock batch as this metric will just use the `code` attribute:

        >>> batch = Mock(spec=MEDSTorchBatch)
        >>> batch.code = code

    Then, we can update and compute the metric values:

        >>> M(logits, batch)
        {'Accuracy/top_1': tensor(0.6667), 'Accuracy/top_2': tensor(1.), 'Accuracy/top_3': tensor(1.),
         'perplexity': tensor(3.1896)}

    You can also run on a single `top_k`:

        >>> M = NextCodeMetrics(top_k=1, vocab_size=4)

    If `top_k` is not an int or a list of ints, an error is raised:

        >>> M = NextCodeMetrics(top_k=[1, "foo"], vocab_size=4)
        Traceback (most recent call last):
            ...
        ValueError: Invalid type for top_k. Want list[int] | int, got <class 'list'> ([1, 'foo']).

    If the max `top_k` is greater than the vocab size, a warning is logged and the `top_k` is filtered to only
    those valid `top_k` values:

        >>> with print_warnings():
        ...     M = NextCodeMetrics(top_k=[1, 2, 3], vocab_size=3)
        Warning: Top-k accuracy requested for k (3 >= vocab_size (3). This is not a valid metric. Filtering to
        only requested k < 3.
        >>> sorted(M.accuracies.keys())
        ['Accuracy/top_1', 'Accuracy/top_2']

    If no valid `top_k` is requested, a warning is logged and the `top_k` is set to 1:

        >>> with print_warnings():
        ...     M = NextCodeMetrics(top_k=[], vocab_size=2)
        Warning: No valid top-k accuracy requested. Adding top-k of 1.
        >>> sorted(M.accuracies.keys())
        ['Accuracy/top_1']
    """

    def __init__(self, top_k: list[int] | int, vocab_size: int, ignore_index: int = 0, **base_metric_kwargs):
        super().__init__(**base_metric_kwargs)

        match top_k:
            case int():
                top_k = [top_k]
            case list() | ListConfig() if all(isinstance(k, int) for k in top_k):
                pass
            case _:
                raise ValueError(
                    f"Invalid type for top_k. Want list[int] | int, got {type(top_k)} ({top_k})."
                )

        if top_k and (max(top_k) >= vocab_size):
            logger.warning(
                f"Top-k accuracy requested for k ({max(top_k)} >= vocab_size ({vocab_size}). "
                f"This is not a valid metric. Filtering to only requested k < {vocab_size}."
            )
            top_k = [k for k in top_k if k < vocab_size]

        if not top_k:
            logger.warning("No valid top-k accuracy requested. Adding top-k of 1.")
            top_k = [1]

        self.accuracies = MetricCollection(
            {f"Accuracy/top_{k}": TopKMulticlassAccuracy(top_k=k, ignore_index=ignore_index) for k in top_k}
        )
        self.perplexity = Perplexity(ignore_index=ignore_index)

        self.hparams = {
            "top_k": top_k,
            "vocab_size": vocab_size,
            "ignore_index": ignore_index,
            **base_metric_kwargs,
        }

    def update(self, logits: torch.Tensor, codes: torch.Tensor):
        """Update the metric with the current batch and logits, sliced to match targets and predictions.

        Args:
            logits: The logits from the model, of shape (batch_size, sequence_length, vocab_size).
            batch: The MEDSTorchBatch containing the input codes at attribute `.code`.  Note that the `code`
                and logits are aligned such that the given code inputs are at the same position as the logits
                produced at that input -- so the logits need to be shifted to align with their prediction
                targets.
        """

        logits = logits[:, :-1]
        targets = codes[:, 1:]

        self.perplexity.update(logits, targets)
        self.accuracies.update(logits, targets)

    def compute(self):
        return {**self.accuracies.compute(), "perplexity": self.perplexity.compute()}


class HistogramMetric(Metric):
    """
    Accumulates histograms (true, mean, sample) and, when computed, produces three plots and
    returns the summed Mean Absolute Error (MAE) across all categories (using the mean predictions).

    The plot() method follows the torchmetrics v1.0.0 plotting API.
    """

    # Optional attributes for the internal _plot method (not used here because we need custom plots)
    plot_lower_bound: float | None = None
    plot_upper_bound: float | None = None

    def __init__(self, vq_num_embeddings, dist_sync_on_step: bool = False):
        super().__init__(dist_sync_on_step=dist_sync_on_step)
        self.add_state("true_hist", default=[], dist_reduce_fx=None)
        self.add_state("sample_hist", default=[], dist_reduce_fx=None)
        self.add_state(
            "counts", default=torch.zeros(vq_num_embeddings, dtype=torch.long), dist_reduce_fx="sum"
        )

    def sync(self, *args, **kwargs):
        # Disable distributed synchronization on CPU to avoid the error.
        return

    def update(self, true_hist: torch.Tensor, sample_hist: torch.Tensor, codes: torch.Tensor):
        """
        Update the metric state with a batch of histograms.
        Each input is expected to be a tensor of shape (batch_size, num_categories).
        """
        self.true_hist.append(true_hist.detach().cpu())
        self.sample_hist.append(sample_hist.detach().cpu())
        self.counts += torch.bincount(codes.view(-1).long(), minlength=self.counts.numel())

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
        sample_hist = torch.cat(self.sample_hist, dim=0)
        # print("$$$$$$$$$$$$$$$$$$$$$$$$$$$$HISTOGRAM_DEBUG$$$$$$$$$$$$$$$$$$$$$$$$$$$$")
        # print(true_hist[0])
        # print(mean_hist[0])
        # print(sample_hist[0])

        num_categories = true_hist.shape[1]

        # Plot 1 data: MAE per category & average true count
        mae_sample = torch.mean(torch.abs(sample_hist - true_hist), dim=0).numpy()

        # Plot 2 data: Pearson correlation per category
        true_np = true_hist.numpy()
        sample_np = sample_hist.numpy()
        corr_sample = []
        for i in range(num_categories):
            if np.std(true_np[:, i]) > 0 and np.std(sample_np[:, i]) > 0:
                corr_sample.append(np.corrcoef(sample_np[:, i], true_np[:, i])[0, 1])
            else:
                corr_sample.append(np.nan)
        corr_sample = np.nanmean(corr_sample)

        # Plot 3 data: Means and stds per category for each histogram type
        true_mean_val = torch.mean(true_hist, dim=0).numpy()
        true_std_val = torch.std(true_hist, dim=0).numpy()
        sample_mean_val = torch.mean(sample_hist, dim=0).numpy()
        sample_std_val = torch.std(sample_hist, dim=0).numpy()

        # Summed MAE across categories for mean histogram predictions
        mae_sample_sum = float(mae_sample.sum())
        centroid_coverage = torch.nonzero(self.counts > 0, as_tuple=False).flatten().float().mean().item()

        return {
            "MODEL//HISTOGRAM//mae_sample": mae_sample.mean(),
            "MODEL//HISTOGRAM//corr_sample": corr_sample.mean(),
            "MODEL//HISTOGRAM//true_mean": true_mean_val.mean(),
            "MODEL//HISTOGRAM//true_std": true_std_val.mean(),
            "MODEL//HISTOGRAM//sample_mean": sample_mean_val.mean(),
            "MODEL//HISTOGRAM//sample_std": sample_std_val.mean(),
            "MODEL//HISTOGRAM//mae_sample_sum": mae_sample_sum,
            "MODEL//HISTOGRAM//num_categories": num_categories,
            "MODEL//HISTOGRAM//centroid_coverage": centroid_coverage,
        }

    def compute(self) -> float:
        """
        Computes and returns the summed MAE across all categories (using mean histogram predictions).
        Also clears the internal states if needed.
        """
        return self._aggregate()


def eval_decorator(fn):
    def inner(self, *args, **kwargs):
        was_training = self.model.model.training
        self.model.model.eval()
        out = fn(self, *args, **kwargs)
        self.model.model.train(was_training)
        return out

    return inner


CODE_LOGITS = "EIC_MODEL//CODE_LOGITS"

import torch
import torch.nn.functional as F


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


class VQHistogramForecastingModule(BaseModule):
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
        self.val_next_token_metric = NextCodeMetrics(self.cfg.top_k_acc, self.cfg.vocab_size)
        self.test_next_token_metric = NextTokenPredictionMetric(self.cfg.vocab_size, [], False)

        self.train_histogram_metric = HistogramMetric(
            vq_num_embeddings=self.input_encoder.cfg.vq_num_embeddings
        )
        self.val_histogram_metric = HistogramMetric(
            vq_num_embeddings=self.input_encoder.cfg.vq_num_embeddings
        )
        self.test_histogram_metric = HistogramMetric(
            vq_num_embeddings=self.input_encoder.cfg.vq_num_embeddings
        )

        self.metadata_df = pl.read_parquet(self.cfg.augmented_code_metadata_fp)
        self.trajectory_labeler = self.cfg.get("trajectory_labeler", None)
        self.initialize_weights()

        self.h_token = self.metadata_df.filter(pl.col("code") == "[H]")["code/vocab_index"][-1]
        self.ntp_token = self.metadata_df.filter(pl.col("code") == "[NTP]")["code/vocab_index"][-1]

        self.subvocab_h_token = self.metadata_df.filter(pl.col("code") == "[H]")["code/subvocab_index"][-1]
        self.subvocab_ntp_token = self.metadata_df.filter(pl.col("code") == "[NTP]")["code/subvocab_index"][
            -1
        ]
        self.subvocab_mapper = SubvocabMapper(metadata_df=self.metadata_df)

        EOS_TOKENS = self.metadata_df.filter(pl.col("code") == "[EOS]")["code/vocab_index"]
        if len(EOS_TOKENS) > 1:
            raise ValueError(
                "Multiple EOS tokens found in metadata. Please ensure there is only one EOS token in the metadata."
            )
        else:
            self.EOS_TOKEN_ID = None

        self.freeze_vector_quantizer_parameters()

    def freeze_vector_quantizer_parameters(self):
        if not self.cfg.pretrain:
            log.info("Freezing params:")
            for n, p in self.input_encoder.named_parameters():
                p.requires_grad = False
                log.info(n)

    @staticmethod
    def _is_norm_bias_param(name: str) -> bool:
        # matches "bias" or "layernorm.weight" / "LayerNorm.bias" etc.
        return bool(re.search(r"(bias|layer(_?)norm(\d*)\.weight)", name, re.IGNORECASE))

    def _norm_bias_params(self) -> Iterator[torch.nn.Parameter]:
        for n, p in self.named_parameters():
            if p.requires_grad and self._is_norm_bias_param(n):
                yield p

    def _decay_params(self) -> Iterator[torch.nn.Parameter]:
        for n, p in self.named_parameters():
            if p.requires_grad and not self._is_norm_bias_param(n):
                yield p

    def configure_optimizers(self):
        trainable_names = [n for n, p in self.named_parameters() if p.requires_grad]

        param_groups = [
            {"params": list(self._decay_params()), "weight_decay": self.cfg.weight_decay},
            {"params": list(self._norm_bias_params()), "weight_decay": 0.0},
        ]

        optimizer = self.optimizer(param_groups)
        if self.scheduler is not None:
            scheduler = self.scheduler.instantiate(optimizer)
            return {
                "optimizer": optimizer,
                "lr_scheduler": {"scheduler": scheduler, **self.scheduler.extra_kwargs},
            }
        return optimizer

    def get_loss(self, batch):
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

    def forward(self, batch, keep_code_logits=False):
        batch = self.input_encoder(batch)
        if self.cfg.pretrain:
            return batch
        else:
            model_output = self.model(batch, do_get_last_token=False)
            embeddings = model_output[BACKBONE_EMBEDDINGS_KEY]

            if self.cfg.return_tokens:
                batch[MODEL_TOKENS_KEY] = model_output[BACKBONE_TOKENS_KEY]
            batch[MODEL_EMBEDDINGS_KEY] = model_output[BACKBONE_EMBEDDINGS_KEY]
            model_output[CODE_LOGITS] = model_output[BACKBONE_TOKENS_KEY]
            if self.cfg.return_logits:
                batch[MODEL_LOGITS_SEQUENCE_KEY] = model_output[CODE_LOGITS]
            batch[CODE_LOGITS] = model_output[CODE_LOGITS]

            code_loss = self.get_loss(batch)

            batch[MODEL_LOSS_KEY] = code_loss
            batch[MODEL_BATCH_LOSS_KEY] = code_loss

            if not keep_code_logits:
                del batch[CODE_LOGITS]
            return batch

    def _log(self, batch, split):
        on_step = split == "train"
        for loss_key in (
            MODEL_LOSS_KEYS
            + [MODEL_LOSS_KEY]
            + [key for key in batch.keys() if key.startswith("MODEL//VQ//")]
        ):
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
        if not self.cfg.pretrain:
            if split == "train":
                self.train_next_token_metric.update(batch[CODE_LOGITS], batch["code"], batch["mask"])
            elif split == "val":
                self.val_next_token_metric.update(batch[CODE_LOGITS], batch["code"])
            elif split == "test":
                self.test_next_token_metric.update(batch[CODE_LOGITS], batch["code"], batch["mask"])
            else:
                raise ValueError(f"Invalid split: {split}")

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
        if CODE_LOGITS in batch:
            del batch[CODE_LOGITS]
        return batch[MODEL_BATCH_LOSS_KEY]

    def on_train_epoch_end(self):
        if not self.cfg.pretrain:
            next_token_results = self.train_next_token_metric.compute()
            for metric_name, value in next_token_results.items():
                self.log(f"test/NEXT_TOKEN/{metric_name.upper()}", value, on_epoch=True)
            self.train_next_token_metric.reset()

    def validation_step(self, batch):
        batch = self(batch, True)
        assert not torch.isnan(batch[MODEL_BATCH_LOSS_KEY]), "Loss is NaN"
        if self.cfg.pretrain:
            self.val_histogram_metric(
                batch["vq_true_hist"], batch[BACKBONE_TOKENS_KEY], batch[BACKBONE_EMBEDDINGS_KEY]
            )
        self._log(batch, "val")
        if CODE_LOGITS in batch:
            del batch[CODE_LOGITS]
        return batch[MODEL_BATCH_LOSS_KEY]

    def on_validation_epoch_end(self):
        if self.cfg.pretrain:
            histogram_sample_mae = self.val_histogram_metric.compute()
            for metric_name, value in histogram_sample_mae.items():
                self.log(f"val/HISTOGRAM/{metric_name}", value, on_epoch=True)
            # Reset histogram metric state for the next epoch
            self.val_histogram_metric.reset()
        else:
            next_token_results = self.val_next_token_metric.compute()
            for metric_name, value in next_token_results.items():
                self.log(f"val/NEXT_TOKEN/{metric_name.upper()}", value, on_epoch=True)
            self.val_next_token_metric.reset()

    def test_step(self, batch):
        batch = self(batch, True)
        assert not torch.isnan(batch[MODEL_BATCH_LOSS_KEY]), "Loss is NaN"
        self._log(batch, "test")
        if CODE_LOGITS in batch:
            del batch[CODE_LOGITS]
        loss = batch[MODEL_BATCH_LOSS_KEY]
        if self.cfg.pretrain:
            self.test_histogram_metric(
                batch["vq_true_hist"], batch[BACKBONE_TOKENS_KEY], batch[BACKBONE_EMBEDDINGS_KEY]
            )
        return loss

    def on_test_epoch_end(self):
        if not self.cfg.pretrain:
            next_token_results = self.test_next_token_metric.compute()
            for metric_name, value in next_token_results.items():
                self.log(f"test/NEXT_TOKEN/{metric_name.upper()}", value, on_epoch=True)
            self.test_next_token_metric.reset()

    def get_metadata_means(self, metadata_df):
        if "values/sum" not in metadata_df or "values/n_occurrences" not in metadata_df:
            raise ValueError("Missing 'values/sum' and/or 'values/n_occurrences' columns in metadata_df")
        metadata_df = metadata_df.with_columns(
            (pl.col("values/sum") / pl.col("values/n_occurrences")).alias("values/mean")
        )
        return metadata_df

    def get_code_to_time_map(self, metadata_df) -> dict:
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
        metadata_df = self.get_metadata_means(metadata_df)
        # Assuming we know the vocab size
        code_to_time_map = torch.zeros(
            self.cfg.vocab_size
        )  # +2 since indices start at 1 and EOS token is added

        # Set values using the indices
        time_mask = pl.col("code").str.starts_with(TIME_DELTA_TOKEN)
        vocab_indices = metadata_df.filter(time_mask)["code/vocab_index"]
        time_values = metadata_df.filter(time_mask)["values/mean"]

        code_to_time_map[vocab_indices.to_list()] = time_values.to_torch().to(code_to_time_map.dtype)
        return code_to_time_map

    def get_code_to_numeric_value_map(self, metadata_df, get_raw_values=True) -> dict:
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

        # Create a tensor filled with NaN values
        result = torch.full((self.cfg.vocab_size,), float("nan"))
        # TODO(Oufattole) remove this and enforce that metadata_df must include the values/min
        ordered_quantiles = [field.name for field in metadata_df.schema["values/quantiles"].fields]
        percentiles = [0, *[float(q.split("/")[-1]) for q in ordered_quantiles], 1]
        if "values/min" not in metadata_df.columns or "values/max" not in metadata_df.columns:
            raise ValueError("Missing values/min and/or values/max values in metadata_df")
        metadata_df = self.get_metadata_means(metadata_df)

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

    def to_trajectory_batch(
        self,
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
            code_to_time_map = self.get_code_to_time_map(metadata_df)
        if not code_to_numeric_value_map:
            code_to_numeric_value_map = self.get_code_to_numeric_value_map(metadata_df)
        # Initialize lists to store the DataFrame rows
        time = torch.cumsum(code_to_time_map[code], dim=1)
        numeric_value = code_to_numeric_value_map[code]
        numeric_value_mask = ~numeric_value.isnan()
        time += prediction_time_offset_years.unsqueeze(1)
        return TrajectoryBatch(time, code, mask, numeric_value, numeric_value_mask, metadata_df)

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
        num_samples: None | int = None,
        token_bin_guidance: bool = False,
        get_metadata: bool = False,
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
        from tqdm.auto import trange

        metadata = {}

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
            # embeddings = output.hidden_states[-1]
            # logits = output.logits
            kv_cache = output.past_key_values
            if token_bin_guidance:
                num_tokens_in_bin = self.cfg.max_count
                last_logits = output.logits[:, -1]
                last_logits[:, self.ntp_token] = float("-inf")
                assert (
                    self.ntp_token == self.cfg.no_cluster_vocab_size - 1
                ), "ntp_token is not the last token in the metadata, which is assumed for token bin guidance"
                if count % num_tokens_in_bin == 0:  # should generate H tokens
                    sample = torch.full_like(samples[:, -1], self.h_token).unsqueeze(-1)
                elif count % num_tokens_in_bin == 1:  # should generate NTP tokens
                    last_logits[:, : self.ntp_token] = float("-inf")
                    probs = F.softmax(last_logits / self.cfg.temperature, dim=-1)
                    sample = torch.multinomial(probs, 1)
                else:
                    last_logits = output.logits[:, -1]
                    last_logits[:, self.h_token] = float("-inf")
                    last_logits[:, self.ntp_token :] = float("-inf")
                    last_logits[:, self.cfg.vocab_size :] = float("-inf")
                    probs = F.softmax(last_logits / self.cfg.temperature, dim=-1)
                    sample = torch.multinomial(probs, 1)
            else:
                probs = F.softmax(output.logits[:, -1] / self.cfg.temperature, dim=-1)
                sample = torch.multinomial(probs, 1)

            if get_metadata:
                # Store embeddings
                pass

            # Append new tokens
            samples = torch.cat((samples, sample), dim=-1)
            next_sample_embedding = sample
            input_data = next_sample_embedding
            input_mask = (
                torch.ones(input_mask.shape[0]).to(input_mask.device, dtype=torch.float32).unsqueeze(-1)
            )
            count += 1
        return samples, metadata
