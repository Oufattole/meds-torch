import numpy as np
import polars as pl
import torch
import torch.nn.functional as F
import torch.utils
from clinical_zeroshot_labeler.labeler import SequenceLabeler, WindowStatus
from loguru import logger
from mixins import TimeableMixin
from omegaconf import DictConfig
from torchmetrics import Metric, MetricCollection
from torchmetrics.classification import MulticlassAccuracy, MulticlassAUROC
from x_transformers import Decoder, TransformerWrapper
from x_transformers.autoregressive_wrapper import eval_decorator

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
from meds_torch.models.components.utils import (
    BaseGenerativeModel,
    TrajectoryBatch,
    get_time_days_delta,
)
from meds_torch.utils.custom_time_token import TIME_DELTA_TOKEN


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
        >>> assert batch['code'].shape == (2, 3)  # batch_size=2, seq_len=3
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
            ],
            "code/vocab_index": [0, 1, 2, 3, 4, 5, 6, 7],
            "values/min": [None, None, 2.0, 0.0, 1.0, 0, 1, 2],
            "values/max": [None, None, 3.0, 1.0, 2.0, 1, 2, 3],
            "values/sum": [None, None, 0.5, 1.5, 2.5, 0.5, 1.5, 2.5],
            "values/n_occurrences": [None, None, 1, 1, 1, 1, 1, 1],
            "values/quantiles": [
                {"values/quantile/0.5": None},
                {"values/quantile/0.5": None},
                {"values/quantile/0.5": 1},
                {"values/quantile/0.5": 1},
                {"values/quantile/0.5": 1},
                {"values/quantile/0.5": 1},
                {"values/quantile/0.5": 1},
                {"values/quantile/0.5": 1},
            ],
        }
    )

    # Create sample input batch
    # Sequence 1: Hospital discharge -> High lab value -> Time token
    # Sequence 2: Hospital discharge -> Normal lab value -> Time token
    # Note: All indices should be < len(metadata_df)
    batch = {
        "code": torch.tensor([[1, 2, 4], [1, 3, 4]]),  # Using vocab indices
        "mask": torch.ones(2, 3, dtype=torch.bool),
        "subject_id": torch.tensor([1, 2]),
        "prediction_time": [datetime(2020, 1, 1), datetime(2020, 1, 1)],
        "end_time": [datetime(2020, 1, 1), datetime(2020, 1, 1)],
    }

    # Initialize sequence labeler
    from functools import partial

    labeler = partial(SequenceLabeler.from_yaml_str, yaml_str=task_config)

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
        ...     df = pl.DataFrame({"code": ["A"], "code/vocab_index": [0]})
        ...     df.write_parquet(temp_file.name)
        ...     cfg = create_model_config(temp_file.name)
        >>> assert cfg.vocab_size == 2  # Original size + pad token
    """
    from hydra.utils import instantiate

    vocab_size = pl.read_parquet(metadata_df_path).height + 1
    cfg = {
        "code_metadata_fp": metadata_df_path,
        "backbone": {"_target_": "meds_torch.models.eic_forecasting.DummyModel"},
        "vocab_size": vocab_size,  # Add 1 for pad token
        "generate_id": None,
        "store_generated_trajectory": True,
        "max_seq_len": 10,
        "temperature": 1.0,
        "eos_tokens": None,
        "optimizer": {"_target_": "meds_torch.models.eic_forecasting.DummyOptimizer", "_partial_": True},
        "scheduler": {"_target_": "meds_torch.models.eic_forecasting.DummyScheduler", "_partial_": True},
        "input_encoder": {"_target_": "meds_torch.models.eic_forecasting.DummyEncoder"},
        "code_head": {
            "_target_": "meds_torch.models.eic_forecasting.DummyCodeHead",
            "vocab_size": vocab_size,
        },
        "compile": False,
        "top_k_acc": [1],
        "next_token_auc": False,
        "max_tokens_budget": 10,
        "return_tokens": False,
        "return_logits": False,
    }
    return instantiate(cfg)


class DummyModel:
    """Dummy model that generates two fixed sequences."""

    cfg = DictConfig(dict(token_emb=None))

    def __init__(self):
        self.model = TransformerWrapper(
            num_tokens=5,
            max_seq_len=10,
            attn_layers=Decoder(dim=8, depth=1, heads=2, rotary_pos_emb=True),
            use_abs_pos_emb=False,
        )

    def __call__(self, batch):
        B, S = batch["code"].shape
        return {BACKBONE_TOKENS_KEY: torch.ones(B, S, 32), BACKBONE_EMBEDDINGS_KEY: None}

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
    def __call__(self, batch):
        batch[INPUT_ENCODER_TOKENS_KEY] = batch["code"]
        batch[INPUT_ENCODER_MASK_KEY] = torch.ones_like(batch["mask"]).bool()
        return batch


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


# Function to pad a single array
def pad_array(arr, max_len):
    pad_width = ((0, 0), (0, max_len - arr.shape[1]))
    if arr.dtype == bool:
        return np.pad(arr, pad_width, mode="constant", constant_values=False)
    else:
        return np.pad(arr, pad_width, mode="constant", constant_values=0)


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


class EicForecastingModule(BaseModule, TimeableMixin, BaseGenerativeModel):
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
            - code_metadata_fp: Path to code metadata file

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
        >>> model = EicForecastingModule(cfg)
        >>> loss = model.training_step(batch)
        >>> assert loss.isfinite().all()

        >>> # Test workflow 2: Data generation without labeling
        >>> cfg.generate_id = 1
        >>> model = EicForecastingModule(cfg)
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
        >>> generated_df.shape[0]
        20

        >>> # Test workflow 3: Generation with zero-shot labeling
        >>> cfg.generate_id = 1
        >>> model = EicForecastingModule(cfg)
        >>> model.trajectory_labeler = trajectory_labeler
        >>> output = model.forward(batch)
        >>> # Check labeling output
        >>> assert MODEL_PRED_PROBA_KEY in output
        >>> assert MODEL_PRED_STATUS_KEY in output
        >>> assert output[MODEL_PRED_PROBA_KEY].shape == (2,)  # Binary prediction per sequence
        >>> assert output[MODEL_PRED_STATUS_KEY].shape == (2,)  # Status per sequence
        >>> # Verify status progression works
        >>> status_vals = output[MODEL_PRED_STATUS_KEY]
        >>> assert (status_vals == WindowStatus.SATISFIED.value).any(), status_vals  # Some sequences complete
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

        self.metadata_df = pl.read_parquet(self.cfg.code_metadata_fp)
        self.trajectory_labeler = self.cfg.get("trajectory_labeler", None)

    def get_loss(self, batch):
        code_logits = batch[CODE_LOGITS]
        assert not torch.isnan(code_logits).any(), "code_logits is NaN"

        # Code Mask
        mask = batch["mask"]
        code_target = batch["code"]

        # Shift the target to predict the next token
        shifted_code_target = code_target[:, 1:]  # Remove the first token
        shifted_mask = mask[:, 1:]  # Remove the first position from the mask too

        # Apply the mask to code_logits and shifted_code_target
        masked_code_logits = code_logits[:, :-1] * shifted_mask.unsqueeze(-1)  # Remove the last prediction
        masked_code_target = shifted_code_target * shifted_mask

        # Code Loss
        code_loss = F.cross_entropy(
            masked_code_logits.transpose(1, 2),
            masked_code_target.to(dtype=torch.long),
            reduction="none",
        ).mean(dim=-1)

        assert not torch.isnan(code_loss).any(), "code_loss is NaN"

        return code_loss

    def get_forecast_logits(self, model_output):
        if isinstance(model_output, torch.Tensor):
            all_token_embeddings = model_output
        else:
            all_token_embeddings = model_output[BACKBONE_TOKENS_KEY]
        code_logits = self.code_head(all_token_embeddings)
        return {
            CODE_LOGITS: code_logits,
        }

    def forward(self, batch, keep_code_logits=False):
        batch = self.input_encoder(batch)
        model_output = self.model(batch)

        if self.cfg.return_tokens:
            batch[MODEL_TOKENS_KEY] = model_output[BACKBONE_TOKENS_KEY]
        batch[MODEL_EMBEDDINGS_KEY] = model_output[BACKBONE_EMBEDDINGS_KEY]
        forecast = self.get_forecast_logits(model_output)
        if self.cfg.return_logits:
            batch[MODEL_LOGITS_SEQUENCE_KEY] = forecast[CODE_LOGITS]
        batch[CODE_LOGITS] = forecast[CODE_LOGITS]

        code_loss = self.get_loss(batch)
        batch[MODEL_LOSS_KEY] = code_loss
        batch[MODEL_BATCH_LOSS_KEY] = code_loss.mean()
        batch = self._generate(batch)

        if not keep_code_logits:
            del batch[CODE_LOGITS]
        return batch

    def _log(self, batch, split):
        self.log(split + "/loss", batch[MODEL_BATCH_LOSS_KEY])
        if split == "train":
            self.train_next_token_metric.update(batch[CODE_LOGITS], batch["code"], batch["mask"])
        elif split == "val":
            self.val_next_token_metric.update(batch[CODE_LOGITS], batch["code"], batch["mask"])
        elif split == "test":
            self.test_next_token_metric.update(batch[CODE_LOGITS], batch["code"], batch["mask"])

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
        >>> EicForecastingModule.get_code_to_time_map(metadata_df)
        tensor([0., 0., 0., 1., 0.])
        """
        metadata_df = cls.get_metadata_means(metadata_df)
        assert metadata_df["code/vocab_index"].is_sorted()
        code_to_time_map = (
            metadata_df.select(
                pl.when(pl.col("code").str.starts_with(TIME_DELTA_TOKEN))
                .then(pl.col("values/mean"))
                .otherwise(pl.lit(0.0))
            )
            .to_torch(dtype=pl.Float32)
            .flatten()
        )
        code_to_time_map = torch.cat([code_to_time_map, torch.zeros(1)])
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
        >>> EicForecastingModule.get_code_to_numeric_value_map(metadata_df, get_raw_values=True).tolist()
        [nan, 0.5, 1.5, 2.5, 3.5, nan, nan]
        >>> EicForecastingModule.get_code_to_numeric_value_map(metadata_df, get_raw_values=False).tolist()
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
                    result[vocab_idx] = mean_value
                    # result[vocab_idx] = sum([raw_quantiles[rank], raw_quantiles[rank + 1]]) / 2
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
        >>> EicForecastingModule.to_trajectory_batch(code, mask, metadata_df, prediction_time_offset_years
        ...     ).to_meds(subject_ids, prediction_times).columns
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

            >>> model = EicForecastingModule(cfg)
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

            >>> # Test second step shows active status
            >>> time, status, is_finished, ended = model.update_generation_state(
            ...     tokens=tokens,
            ...     cumulative_time=time,
            ...     trajectory_labeler=labeler,
            ... )
            >>> assert (status == WindowStatus.ACTIVE.value).all()
            >>> assert not is_finished
            >>> assert not ended.any()

            >>> # Test third step shows satisfied status and finished
            >>> time, status, is_finished, ended = model.update_generation_state(
            ...     tokens=tokens,
            ...     cumulative_time=time,
            ...     trajectory_labeler=labeler,
            ... )
            >>> assert (status == WindowStatus.SATISFIED.value).all()
            >>> assert is_finished
            >>> assert ended.all()

            >>> # Test without trajectory labeler
            >>> time, status, is_finished, ended = model.update_generation_state(
            ...     tokens=tokens,
            ...     cumulative_time=time,
            ...     trajectory_labeler=None,
            ... )
            >>> assert time.shape == (B,)
            >>> assert status is None
            >>> assert not is_finished
            >>> assert not ended.any()
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

    @torch.no_grad()
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
        if self.cfg.backbone.cfg.token_emb:
            raise NotImplementedError(
                "Token embeddings not supported, use x-transformers library for token embeddings"
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
            out, out_lengths, metadata = self.generate(
                prompts=prompts,
                mask=mask,
                trajectory_labeler=self.trajectory_labeler(
                    batch_size=prompts.shape[0], metadata_df=self.metadata_df
                )
                if self.trajectory_labeler is not None
                else None,
                time_offset_years=prediction_time_offset_years,
                temperature=self.cfg.temperature,
                eos_tokens=self.cfg.eos_tokens,
                log_progress=self.cfg.get("log_progress", False),
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
                logger.info(f"Completed zero-shot labeling for sample {self.cfg.generate_id}")
        return input_batch
