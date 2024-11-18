from collections.abc import Callable

import numpy as np
import polars as pl
import torch
import torch.nn.functional as F
import torch.utils
from loguru import logger
from mixins import TimeableMixin
from omegaconf import DictConfig
from torchmetrics import Metric, MetricCollection
from torchmetrics.classification import MulticlassAccuracy, MulticlassAUROC
from x_transformers.autoregressive_wrapper import eval_decorator

from meds_torch.input_encoder import INPUT_ENCODER_MASK_KEY, INPUT_ENCODER_TOKENS_KEY
from meds_torch.input_encoder.eic_encoder import EicEncoder
from meds_torch.models import (
    BACKBONE_TOKENS_KEY,
    GENERATE_PREFIX,
    MODEL_BATCH_LOSS_KEY,
    MODEL_LOGITS_SEQUENCE_KEY,
    MODEL_LOSS_KEY,
    MODEL_PRED_PROBA_KEY,
    MODEL_TOKENS_KEY,
)
from meds_torch.models.base_model import BaseModule
from meds_torch.models.components import AUTOREGRESSIVE_MODELS
from meds_torch.models.zero_shot_labeler.time_to_event_labeler import TaskLabeler
from meds_torch.models.zero_shot_labeler.utils import (
    TrajectoryBatch,
    get_time_days_delta,
)

# Create dummy components for testing


class DummyModel:
    cfg = DictConfig(dict(token_emb=None))

    def __call__(self, batch):
        # Simulate backbone output
        B, S = batch[INPUT_ENCODER_TOKENS_KEY].shape
        return {BACKBONE_TOKENS_KEY: torch.randn(B, S, 32)}

    def generate(
        self,
        prompts: torch.Tensor,
        mask: torch.Tensor | None = None,
        get_next_token_time: Callable[[torch.Tensor, torch.Tensor], torch.Tensor] | None = None,
        temperature: float = 1.0,
        filter_logits_fn: str | Callable = torch.nn.Identity(),
    ) -> torch.Tensor:
        B, S = prompts.shape
        out = torch.randint(0, 4, (B, S))
        return out


class DummyCodeHead:
    def __call__(self, x):
        # Convert embeddings to logits
        B, S, _ = x.shape
        return torch.randn(B, S, 4)  # 4 possible codes


class DummyEncoder:
    def __call__(self, batch):
        # Add encoder fields to batch
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


CODE_LOGITS = "MODEL//CODE_LOGITS"

# Time quantiles for the EIC dataset
TIME_QUANTILE_VALUES = [
    0,
    0.00000190258,
    0.00000951293,
    0.00001902587,
    0.00005707762,
    0.00011415525,
    0.00034246575,
    0.0006849315,
    0.00136986301,
    0.00273972602,
    0.00547945205,
    0.0109589041,
    0.01917808219,
    0.03835616438,
    0.08219178082,
    0.16438356164,
    0.32876712328,
    1,
    2,
    5,
    10,
    20,
    40,
]

TIME_QUANTILE_NAMES = [
    "TIME//DELTA//TOKEN",
    "TIME//DELTA//TOKEN//_Q_1",
    "TIME//DELTA//TOKEN//_Q_2",
    "TIME//DELTA//TOKEN//_Q_3",
    "TIME//DELTA//TOKEN//_Q_4",
    "TIME//DELTA//TOKEN//_Q_5",
    "TIME//DELTA//TOKEN//_Q_6",
    "TIME//DELTA//TOKEN//_Q_7",
    "TIME//DELTA//TOKEN//_Q_8",
    "TIME//DELTA//TOKEN//_Q_9",
    "TIME//DELTA//TOKEN//_Q_10",
    "TIME//DELTA//TOKEN//_Q_11",
    "TIME//DELTA//TOKEN//_Q_12",
    "TIME//DELTA//TOKEN//_Q_13",
    "TIME//DELTA//TOKEN//_Q_14",
    "TIME//DELTA//TOKEN//_Q_15",
    "TIME//DELTA//TOKEN//_Q_16",
    "TIME//DELTA//TOKEN//_Q_17",
    "TIME//DELTA//TOKEN//_Q_18",
    "TIME//DELTA//TOKEN//_Q_19",
    "TIME//DELTA//TOKEN//_Q_20",
    "TIME//DELTA//TOKEN//_Q_21",
    "TIME//DELTA//TOKEN//_Q_22",
]


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

    def __init__(self, vocab_size: int, top_k_acc: list[int], dist_sync_on_step=False):
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

        self.auroc = MulticlassAUROC(num_classes=vocab_size, average="weighted", thresholds=100)
        self.top_k_acc = top_k_acc
        self.accuracy_metrics = MetricCollection(
            {f"top_{k}_accuracy": MulticlassAccuracy(num_classes=vocab_size, top_k=k) for k in top_k_acc}
        )

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

        # Update AUROC
        self.auroc.update(flat_logits, flat_targets)

        # Update top-n accuracy
        self.accuracy_metrics.update(flat_logits, flat_targets)

    def compute(self):
        """
        Compute the AUROC and top-n accuracy based on accumulated statistics.

        Returns:
            dict: A dictionary containing the computed AUROC and top-n accuracy for each n in top_n.
        """
        results = {
            "auroc": self.auroc.compute(),
        }
        results.update(self.accuracy_metrics.compute())
        return results


class EicForecastingModule(BaseModule, TimeableMixin):
    """EIC token based GPT Forecasting Model.

    This model has three main capabilities:
    1. Autoregressive training (learning to predict next tokens)
    2. Data generation (creating synthetic medical event sequences)
    3. Zero-shot prediction (using generated sequences for prediction)

    Args:
        cfg (DictConfig): Configuration object containing:
            - vocab_size: Size of the vocabulary
            - num_samples: Number of sequences to generate (0 for training only)
            - max_seq_len: Maximum sequence length
            - zero_shot_labeler: Optional function for zero-shot prediction
            - code_metadata_fp: Path to code metadata file

    Examples:
    >>> import tempfile
    >>> import datetime
    >>> from hydra.utils import instantiate
    >>> # Create temporary metadata file
    >>> temp_file = tempfile.NamedTemporaryFile(suffix='.parquet')
    >>> metadata_df = pl.DataFrame({
    ...     "code": ["A", "B", "TIME//DELTA//TOKEN//_Q_17", "C"],
    ...     "code/vocab_index": [0, 1, 2, 3],
    ...     "values/min": [0.0, 1.0, None, 2.0],
    ...     "values/max": [1.0, 2.0, None, 3.0],
    ...     "values/quantiles": [
    ...         {"values/quantile/0.5": 0.5},
    ...         {"values/quantile/0.5": 1.5},
    ...         {"values/quantile/0.5": None},
    ...         {"values/quantile/0.5": 2.5},
    ...     ]
    ... })
    >>> metadata_df.write_parquet(temp_file.name)
    >>> # Create config
    >>> cfg = {
    ...     "code_metadata_fp": temp_file.name,
    ...     "backbone": {"_target_": "meds_torch.models.eic_forecasting.DummyModel"},
    ...     "vocab_size": 4,
    ...     "num_samples": 0,
    ...     "max_seq_len": 10,
    ...     "zero_shot_labeler": None,
    ...     'temperature': 1.0,
    ...     "optimizer": {
    ...         "_target_": "meds_torch.models.eic_forecasting.DummyOptimizer",
    ...         "_partial_": True
    ...     },
    ...     "scheduler": {
    ...         "_target_": "meds_torch.models.eic_forecasting.DummyScheduler",
    ...         "_partial_": True
    ...     },
    ...     "input_encoder": {"_target_": "meds_torch.models.eic_forecasting.DummyEncoder"},
    ...     "code_head": {"_target_": "meds_torch.models.eic_forecasting.DummyCodeHead"},
    ...     "compile": False,
    ...     "top_k_acc": [1],
    ... }
    >>> cfg = instantiate(cfg)
    >>> # Create input batch
    >>> batch = {
    ...     'code': torch.tensor([[0, 1, 2], [1, 2, 3]]),
    ...     'mask': torch.ones(2, 3).bool()
    ... }
    >>>
    >>> # Test workflow 1: Autoregressive training
    >>> model = EicForecastingModule(cfg)
    >>> loss = model.training_step(batch)
    >>> assert loss.isfinite().all()
    >>>
    >>> # Test workflow 2: Data generation
    >>> cfg.num_samples = 2  # Enable generation
    >>> model = EicForecastingModule(cfg)
    >>>
    >>> # Test generation
    >>> output = model.forward(batch)
    >>> print(f"Generated sequences shape: {output['GENERATE//0']['code'].shape}")
    Generated sequences shape: torch.Size([2, 3])
    >>> # Test workflow 3: Zero-shot prediction
    >>> cfg.zero_shot_labeler = lambda x: (torch.tensor([0.7, 0.3]), torch.tensor([False, True]))
    >>> model = EicForecastingModule(cfg)
    >>> batch['prediction_time'] = [datetime.datetime(1997,1,1), datetime.datetime(1997,1,1)]
    >>> batch['end_time'] = [datetime.datetime(1997,1,1), datetime.datetime(1997,1,1)]
    >>> output = model.forward(batch)
    >>> print(f"Zero-shot predictions shape: {output[MODEL_PRED_PROBA_KEY].shape}")
    Zero-shot predictions shape: torch.Size([2])
    >>> output[MODEL_PRED_PROBA_KEY]
    tensor([0.7000, 0.5000])
    >>> # Test generation with real model
    >>> cfg.num_samples = 2  # Enable generation
    >>> B, S, L = 2, 5, 8 # [batch_size, input_sequence_length, token_dim]
    >>> vocab_size = 4
    >>> max_seq_len = 10
    >>> cfg.temperature = 1000.0 # Raise the temperature to randomize the predictions
    >>> model = EicForecastingModule(cfg)
    >>> _ = torch.manual_seed(42)
    >>> model.model = instantiate({
    ...     '_target_': "meds_torch.models.components.transformer_decoder.TransformerDecoderModel.initialize",
    ...     'token_dim': L,
    ...     'vocab_size': vocab_size,
    ...     'max_seq_len': max_seq_len,
    ...     'get_last_token': True,
    ...     'token_emb': None,
    ...     'generation_budget': {
    ...         '_target_': ("meds_torch.models.components.transformer_decoder."
    ...                      "GenerationBudget.from_time_len"),
    ...         "value": .5,
    ...     },
    ...     'model': {
    ...         '_target_': 'x_transformers.TransformerWrapper',
    ...         'num_tokens': vocab_size,
    ...         'max_seq_len': max_seq_len,
    ...         'attn_layers': {
    ...             '_target_': 'x_transformers.Decoder',
    ...             'dim': L,
    ...             'depth': 2,
    ...             'heads': 2,
    ...         }
    ...     }
    ... })
    >>>
    >>> # Create input batch
    >>> output = model.forward(batch)
    >>> codes = output['GENERATE//0']['code']
    >>> codes.shape[0] == 2
    True
    >>> codes.shape[1] >= 1
    True
    >>> # Note that the time token is code/vocab_index 2 in the medadata
    >>>  # check that 1 time token was generated for the shortest time trajectory
    >>> (codes == 2).sum(dim=1).min().item()
    1
    >>> temp_file.close()
    """

    def __init__(self, cfg: DictConfig):
        super().__init__(cfg)
        if not isinstance(self.model, AUTOREGRESSIVE_MODELS + (DummyModel,)):
            raise ValueError(
                f"Unsupported model type: {type(self.model)}, choose one from {AUTOREGRESSIVE_MODELS}"
            )
        if not isinstance(self.input_encoder, (EicEncoder, DummyEncoder)):
            raise NotImplementedError(f"Unsupported input encoder type: {type(self.input_encoder)}")
        self.code_head = self.cfg.code_head

        num_future_codes = self.cfg.get("num_future_codes", None)
        if num_future_codes is not None:
            logger.info(f"Using {num_future_codes} future codes for forecasting")
        self.train_next_token_metric = NextTokenPredictionMetric(self.cfg.vocab_size, self.cfg.top_k_acc)
        self.val_next_token_metric = NextTokenPredictionMetric(self.cfg.vocab_size, self.cfg.top_k_acc)
        self.test_next_token_metric = NextTokenPredictionMetric(self.cfg.vocab_size, self.cfg.top_k_acc)

        self.metadata_df = pl.read_parquet(self.cfg.code_metadata_fp)

    def get_loss(self, batch):
        code_logits = batch[CODE_LOGITS]
        assert not torch.isnan(code_logits).any(), "code_logits is NaN"

        # Code Mask
        mask = batch["mask"]
        code_target = batch["code"]

        # Shift the target to predict the next token
        shifted_code_target = code_target[:, 1:]  # Remove the first token
        shifted_mask = mask[:, :-1]  # Remove the last position from the mask

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

    def forward(self, batch):
        batch = self.input_encoder(batch)
        model_output = self.model(batch)

        batch[MODEL_TOKENS_KEY] = model_output[BACKBONE_TOKENS_KEY]
        forecast = self.get_forecast_logits(model_output)
        batch[CODE_LOGITS] = forecast[CODE_LOGITS]
        batch[MODEL_LOGITS_SEQUENCE_KEY] = forecast[CODE_LOGITS]

        code_loss = self.get_loss(batch)
        batch[MODEL_LOSS_KEY] = code_loss
        batch[MODEL_BATCH_LOSS_KEY] = code_loss.mean()
        batch = self._generate(batch)
        return batch

    def _log(self, batch, split):
        self.log(split + "/loss", batch[MODEL_BATCH_LOSS_KEY])

    def _generate(self, batch):
        if self.cfg.num_samples > 0:
            return self.generate_evaluation(batch)
        else:
            return batch

    def training_step(self, batch):
        batch = self(batch)
        assert not torch.isnan(batch[MODEL_BATCH_LOSS_KEY]), "Loss is NaN"
        self._log(batch, "train")
        self.train_next_token_metric.update(batch[CODE_LOGITS], batch["code"], batch["mask"])
        return batch[MODEL_BATCH_LOSS_KEY]

    def on_train_epoch_end(self):
        next_token_results = self.train_next_token_metric.compute()
        for metric_name, value in next_token_results.items():
            self.log(f"test/NEXT_TOKEN/{metric_name.upper()}", value, on_epoch=True)
        self.train_next_token_metric.reset()

    def validation_step(self, batch):
        batch = self(batch)
        assert not torch.isnan(batch[MODEL_BATCH_LOSS_KEY]), "Loss is NaN"
        self._log(batch, "val")
        self.val_next_token_metric.update(batch[CODE_LOGITS], batch["code"], batch["mask"])
        return batch[MODEL_BATCH_LOSS_KEY]

    def on_validation_epoch_end(self):
        next_token_results = self.val_next_token_metric.compute()
        for metric_name, value in next_token_results.items():
            self.log(f"test/NEXT_TOKEN/{metric_name.upper()}", value, on_epoch=True)
        self.val_next_token_metric.reset()

    def test_step(self, batch):
        batch = self(batch)
        assert not torch.isnan(batch[MODEL_BATCH_LOSS_KEY]), "Loss is NaN"
        self._log(batch, "test")
        loss = batch[MODEL_BATCH_LOSS_KEY]
        self.test_next_token_metric.update(batch[CODE_LOGITS], batch["code"], batch["mask"])
        return loss

    def on_test_epoch_end(self):
        next_token_results = self.test_next_token_metric.compute()
        for metric_name, value in next_token_results.items():
            self.log(f"test/NEXT_TOKEN/{metric_name.upper()}", value, on_epoch=True)
        self.test_next_token_metric.reset()

    @staticmethod
    def get_code_to_time_map(metadata_df) -> dict:
        """Convert the metadata DataFrame to a dictionary mapping code to time.

        Args:
            metadata_df: Polars DataFrame containing code metadata
                (includes 'code' and 'code/vocab_index' columns)

        Returns:
            dict: Mapping code to time in years

        Example:
        >>> metadata_df = pl.DataFrame({
        ...     "code": ["A", "B", "C", "TIME//DELTA//TOKEN//_Q_17"],
        ...     "code/vocab_index": [0, 1, 2, 3]
        ... })
        >>> # Note that the code "TIME//DELTA//TOKEN//_Q_17" maps to 1 year
        >>> EicForecastingModule.get_code_to_time_map(metadata_df)
        tensor([0., 0., 0., 1., 0.])
        """
        assert metadata_df["code/vocab_index"].is_sorted()
        code_to_time_map = torch.tensor(
            [
                TIME_QUANTILE_VALUES[TIME_QUANTILE_NAMES.index(code)]
                if code in set(TIME_QUANTILE_NAMES)
                else 0
                for code in metadata_df["code"]
            ]
        )
        code_to_time_map = torch.cat([code_to_time_map, torch.zeros(1)])
        return code_to_time_map

    @staticmethod
    def get_code_to_numeric_value_map(metadata_df, get_raw_values=False) -> dict:
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
        ordered_quantiles = [field.name for field in metadata_df.schema["values/quantiles"].fields]
        percentiles = [0, *[float(q.split("/")[-1]) for q in ordered_quantiles], 1]

        # Process each row in the DataFrame
        for row in metadata_df.iter_rows(named=True):
            vocab_idx = row["code/vocab_index"]
            code = row["code"]
            min_value = row["values/min"]
            max_value = row["values/max"]
            raw_quantiles = [row["values/quantiles"][each] for each in ordered_quantiles]
            raw_quantiles = [min_value, *raw_quantiles, max_value]

            # Check if this is a quarterly code (contains "//_Q_")
            if code and "//_Q_" in code and not code.startswith("TIME//DELTA//TOKEN"):
                # Extract the number of quantiles the value is greater than, 0 for Q_1, 1 for Q_2, etc.
                rank = int(code.split("//_Q_")[1]) - 1
                # We estimate the numeric value is the average of the bordering quantiles it is between
                if get_raw_values:
                    result[vocab_idx] = sum([raw_quantiles[rank], raw_quantiles[rank + 1]]) / 2
                else:
                    result[vocab_idx] = sum([percentiles[rank], percentiles[rank + 1]]) / 2

            # For non-quarterly codes, leave as NaN
            # This handles both the base code (e.g., "A") and any other non-quarterly codes
        return torch.cat([result, torch.Tensor([np.nan])])  # postpend a zero in case EOS token is postpended

    @classmethod
    def to_meds(cls, code_tensors: list[torch.Tensor], metadata_df: pl.DataFrame) -> pl.DataFrame:
        """Convert the model output to MEDS format.

        Args:
            code_tensors: List of torch tensors containing generated code sequences
            metadata_df: Polars DataFrame containing code metadata (includes 'code' column)

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
        >>> code_tensors = [
        ...     {'code': torch.tensor([[0, 2, 5, 5]]), 'subject_id': ['1'],
        ...      'mask': torch.tensor([[1, 1, 1, 1]]), 'prediction_time': [datetime(1997, 1, 1)],
        ...      'end_time': [datetime(1996, 12, 31)]},
        ...     {'code': torch.tensor([[2, 3, 4, 5], [5, 5, 0, 1]]),
        ...      'mask': torch.tensor([[1, 1, 1, 0], [1, 1, 1, 0]]),
        ...      'prediction_time': [datetime(1998, 1, 1), datetime(1999, 1, 1)],
        ...      'subject_id': ['2','3'], 'end_time': [datetime(1997, 12, 31), datetime(1996, 12, 31)],},
        ... ]
        >>> EicForecastingModule.to_meds(code_tensors, metadata_df)
        shape: (10, 5)
        ┌──────┬───────────────┬────────────┬─────────────────────┬────────────────────────────┐
        │ code ┆ numeric_value ┆ subject_id ┆ prediction_time     ┆ time                       │
        │ ---  ┆ ---           ┆ ---        ┆ ---                 ┆ ---                        │
        │ i64  ┆ f32           ┆ i64        ┆ datetime[μs]        ┆ datetime[μs]               │
        ╞══════╪═══════════════╪════════════╪═════════════════════╪════════════════════════════╡
        │ 0    ┆ NaN           ┆ 1          ┆ 1997-01-01 00:00:00 ┆ 1996-12-31 00:00:00        │
        │ 2    ┆ 0.375         ┆ 1          ┆ 1997-01-01 00:00:00 ┆ 1996-12-31 00:00:00        │
        │ 5    ┆ NaN           ┆ 1          ┆ 1997-01-01 00:00:00 ┆ 1997-12-31 05:48:44.315009 │
        │ 5    ┆ NaN           ┆ 1          ┆ 1997-01-01 00:00:00 ┆ 1998-12-31 11:37:28.630018 │
        │ 2    ┆ 0.375         ┆ 2          ┆ 1998-01-01 00:00:00 ┆ 1997-12-31 00:00:00        │
        │ 3    ┆ 0.625         ┆ 2          ┆ 1998-01-01 00:00:00 ┆ 1997-12-31 00:00:00        │
        │ 4    ┆ 0.875         ┆ 2          ┆ 1998-01-01 00:00:00 ┆ 1997-12-31 00:00:00        │
        │ 5    ┆ NaN           ┆ 3          ┆ 1999-01-01 00:00:00 ┆ 1997-12-31 05:48:44.315009 │
        │ 5    ┆ NaN           ┆ 3          ┆ 1999-01-01 00:00:00 ┆ 1998-12-31 11:37:28.630018 │
        │ 0    ┆ NaN           ┆ 3          ┆ 1999-01-01 00:00:00 ┆ 1998-12-31 11:37:28.630018 │
        └──────┴───────────────┴────────────┴─────────────────────┴────────────────────────────┘
        """
        code_to_time_map = cls.get_code_to_time_map(metadata_df)
        code_to_numeric_value_map = cls.get_code_to_numeric_value_map(metadata_df)
        # Initialize lists to store the DataFrame rows
        dfs = []
        for item in code_tensors:
            time_delta_years = torch.cumsum(code_to_time_map[item["code"]], dim=1)
            numeric_values = code_to_numeric_value_map[item["code"]]
            subject_id = item["subject_id"]
            if isinstance(subject_id, torch.Tensor):
                subject_id = subject_id.numpy()
            data = dict(
                time_delta_days=time_delta_years.numpy() * 365.2422,
                code=item["code"].numpy(),
                numeric_value=numeric_values.numpy(),
                subject_id=subject_id,
                mask=item["mask"].numpy(),
                end_time=item["end_time"],
                prediction_time=item["prediction_time"],
            )

            df = pl.from_dict(data)
            df = (
                df.explode("time_delta_days", "code", "numeric_value", "mask")
                .filter(pl.col("mask").cast(pl.Boolean))
                .with_columns(pl.col("subject_id").cast(pl.Int64))
                .drop("mask")
            )
            time_expr = (pl.col("time_delta_days") * 24 * 60 * 60 * 1_000_000_000).cast(
                pl.Duration(time_unit="ns")
            )
            df = df.with_columns((time_expr + pl.col("end_time")).alias("time"))
            df = df.drop("time_delta_days", "end_time")
            dfs.append(df)
        return pl.concat(dfs)

    @classmethod
    def to_trajectory_batch(
        cls,
        code,
        mask,
        metadata_df,
        prediction_time_offset_days: torch.Tensor,
        code_to_time_map: torch.Tensor = None,
        code_to_numeric_value_map: torch.Tensor = None,
    ):
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
        >>> prediction_time_offset_days = torch.tensor([0.0, 1.0, 2.0])
        >>> from pprint import pprint, pformat
        >>> EicForecastingModule.to_trajectory_batch(code, mask, metadata_df, prediction_time_offset_days)
        TrajectoryBatch(time=tensor([[0., 0., 1., 2.],
                [1., 1., 1., 2.],
                [3., 4., 4., 4.]]), code=tensor([[0, 2, 5, 5],
                [2, 3, 4, 5],
                [5, 5, 0, 1]]), mask=tensor([[1, 1, 1, 1],
                [1, 1, 1, 0],
                [1, 1, 1, 0]]), numeric_value=tensor([[   nan, 0.3750,    nan,    nan],
                [0.3750, 0.6250, 0.8750,    nan],
                [   nan,    nan,    nan, 0.1250]]), numeric_value_mask=tensor([[ True, False,  True,  True],
                [False, False, False,  True],
                [ True,  True,  True, False]]))
        """
        if not code_to_time_map:
            code_to_time_map = cls.get_code_to_time_map(metadata_df)
        if not code_to_numeric_value_map:
            code_to_numeric_value_map = cls.get_code_to_numeric_value_map(metadata_df)
        # Initialize lists to store the DataFrame rows
        time = torch.cumsum(code_to_time_map[code], dim=1)
        numeric_value = code_to_numeric_value_map[code]
        numeric_value_mask = numeric_value.isnan()
        time += prediction_time_offset_days.unsqueeze(1)
        return TrajectoryBatch(time, code, mask, numeric_value, numeric_value_mask)

    @torch.no_grad()
    @eval_decorator
    @TimeableMixin.TimeAs
    def generate_evaluation(
        self,
        input_batch,
        **kwargs,
    ):
        """Generate evaluation metrics for the model."""
        if self.cfg.backbone.cfg.token_emb:
            raise NotImplementedError(
                "Token embeddings not supported, use x-transformers library for token embeddings"
            )
        else:
            prompts, mask = input_batch[INPUT_ENCODER_TOKENS_KEY], input_batch[INPUT_ENCODER_MASK_KEY]

        self.time_quantile_map = torch.tensor(
            [
                TIME_QUANTILE_VALUES[TIME_QUANTILE_NAMES.index(code)]
                if code in set(TIME_QUANTILE_NAMES)
                else 0
                for code in self.metadata_df["code"]
            ],
            device=self.device,
        )
        self.time_quantile_map = torch.cat([self.time_quantile_map, torch.zeros(1, device=self.device)])

        def get_next_token_time(x):
            return self.time_quantile_map[x.squeeze()]

        for i in range(self.cfg.num_samples):
            out = self.model.generate(
                prompts=prompts,
                mask=mask,
                get_next_token_time=get_next_token_time,
                temperature=self.cfg.temperature,
                **kwargs,
            )
            out_mask = torch.ones_like(out).bool()

            # Store generated data
            null_data = torch.zeros_like(out).cpu()
            # Convert codes to time deltas
            time_deltas = self.time_quantile_map.to(out.device)[out]
            generated_data = {
                "code": out.cpu(),
                "mask": out_mask.cpu(),
                "numeric_value": null_data,
                "numeric_value_mask": null_data,
                "static_mask": null_data,
                "time_delta_days": time_deltas.cpu(),
            }
            input_batch[GENERATE_PREFIX + str(i)] = generated_data

            if self.cfg.get("zero_shot_labeler") is not None:
                if "prediction_time" not in input_batch or "end_time" not in input_batch:
                    raise ValueError(
                        "Prediction time and end time must be provided for zero-shot labeling. "
                        "Enable the flags do_include_prediction_time and do_include_end_time."
                    )
                prediction_time_offset_days = get_time_days_delta(
                    input_batch["prediction_time"], input_batch["end_time"], generated_data["code"].device
                )
                trajectory_batch = self.to_trajectory_batch(
                    generated_data["code"],
                    generated_data["mask"],
                    self.metadata_df,
                    prediction_time_offset_days,
                )
                input_batch.setdefault(MODEL_PRED_PROBA_KEY, torch.zeros(prompts.shape[0]))
                if isinstance(self.cfg.zero_shot_labeler, DictConfig):
                    task_labeler = TaskLabeler(**self.cfg.zero_shot_labeler)
                else:
                    task_labeler = self.cfg.zero_shot_labeler
                pred_labels, unknown_pred = task_labeler(trajectory_batch)
                # Handle unknown values by setting their probability to 0.5
                pred_labels[unknown_pred] = 0.5
                input_batch[MODEL_PRED_PROBA_KEY] += pred_labels
                logger.info(f"Completed zero-shot labeling for sample {i+1}")
        if MODEL_PRED_PROBA_KEY in input_batch:
            input_batch[MODEL_PRED_PROBA_KEY] /= self.cfg.num_samples
        return input_batch
