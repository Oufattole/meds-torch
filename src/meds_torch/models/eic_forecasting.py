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
    BACKBONE_EMBEDDINGS_KEY,
    BACKBONE_TOKENS_KEY,
    GENERATE_PREFIX,
    MODEL_BATCH_LOSS_KEY,
    MODEL_EMBEDDINGS_KEY,
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
        return {BACKBONE_TOKENS_KEY: torch.randn(B, S, 32), BACKBONE_EMBEDDINGS_KEY: None}

    def generate(
        self,
        prompts: torch.Tensor,
        mask: torch.Tensor | None = None,
        get_next_token_time: Callable[[torch.Tensor, torch.Tensor], torch.Tensor] | None = None,
        eos_tokens=None,
        time_offset_years: torch.Tensor | None = None,
        temperature: float = 1.0,
        filter_logits_fn: str | Callable = torch.nn.Identity(),
    ) -> torch.Tensor:
        B, S = prompts.shape
        out = torch.randint(0, 4, (B, S))
        out_lengths = S * torch.ones(B)
        return out, out_lengths


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
    1.8697990981546083e-06,
    6.042738570007656e-06,
    1.5023761280952375e-05,
    3.7040579000812096e-05,
    9.407141472104954e-05,
    0.00021470011508317946,
    0.000483380440410716,
    0.0010109219329935046,
    0.0018344958495578154,
    0.0041425300336417024,
    0.008303153066089282,
    0.015989647084188808,
    0.029314912679925205,
    0.0594034167939482,
    0.11676680580396949,
    0.23320547561943775,
    0.5373471820293758,
    1.354175718970152,
    3.012664764359352,
    6.511592621764305,
    18.78151017351385,
    29.61825810321306,
    64.47917702102863,
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
    "TIME//DELTA//TOKEN//_Q_23",
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
        self.next_token_metrics = MetricCollection(
            {f"top_{k}_accuracy": MulticlassAccuracy(num_classes=vocab_size, top_k=k) for k in top_k_acc}
        )
        if next_token_auc:
            self.next_token_metrics["auroc"] = MulticlassAUROC(
                num_classes=vocab_size, average="macro", thresholds=100
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


class EicForecastingModule(BaseModule, TimeableMixin):
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
    ...     "generate_id": None,
    ...     "store_generated_trajectory": True,
    ...     "max_seq_len": 10,
    ...     "zero_shot_labeler": None,
    ...     'temperature': 1.0,
    ...     'eos_tokens': [4,],
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
    ...     "next_token_auc": False,
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
    >>> cfg.generate_id = 0  # Enable generation
    >>> model = EicForecastingModule(cfg)
    >>>
    >>> # Test generation
    >>> batch['subject_id'] = torch.tensor([1, 2])
    >>> batch['prediction_time'] = [datetime.datetime(1997,1,1), datetime.datetime(1997,1,1)]
    >>> batch['end_time'] = [datetime.datetime(1997,1,1), datetime.datetime(1997,1,1)]
    >>> output = model.forward(batch)
    >>> output['GENERATE//0'].columns
    ['time', 'code', 'numeric_value', 'subject_id', 'prediction_time']
    >>> assert output['GENERATE//0'].shape[0] == 6
    >>> # Test workflow 3: Zero-shot prediction
    >>> cfg.zero_shot_labeler = lambda x: (torch.tensor([0.7, 0.3]), torch.tensor([False, True]))
    >>> model = EicForecastingModule(cfg)
    >>> output = model.forward(batch)
    >>> print(f"Zero-shot predictions shape: {output[MODEL_PRED_PROBA_KEY].shape}")
    Zero-shot predictions shape: torch.Size([2])
    >>> output[MODEL_PRED_PROBA_KEY]
    tensor([0.7000, 0.5000])
    >>> # Test generation with real model
    >>> cfg.generate_id = 1  # Enable generation
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
    >>> output['GENERATE//1'].columns
    ['time', 'code', 'numeric_value', 'subject_id', 'prediction_time']
    >>> output['GENERATE//1'].shape[0] > 0
    True
    >>>  # Note that the time token is code/vocab_index 2 in the metadata
    >>>  # check that 1 time token was generated for the shortest time trajectory
    >>> output['GENERATE//1'].filter(pl.col('subject_id').eq(1))['code'][-1] == 2
    True
    >>> output['GENERATE//1'].filter(pl.col('subject_id').eq(2))['code'][-1] == 2
    True
    >>> temp_file.close()
    """

    TIME_QUANTILE_VALUES = TIME_QUANTILE_VALUES
    TIME_QUANTILE_NAMES = TIME_QUANTILE_NAMES

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

    def forward(self, batch):
        batch = self.input_encoder(batch)
        model_output = self.model(batch)

        batch[MODEL_TOKENS_KEY] = model_output[BACKBONE_TOKENS_KEY]
        batch[MODEL_EMBEDDINGS_KEY] = model_output[BACKBONE_EMBEDDINGS_KEY]
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
        if self.cfg.generate_id is not None:
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
        ...     "code/vocab_index": [0, 1, 2, 3]
        ... })
        >>> # Note that the code "TIME//DELTA//TOKEN//_Q_17" maps to 1 year
        >>> EicForecastingModule.TIME_QUANTILE_VALUES = [1.0 for _ in range(24)]
        >>> EicForecastingModule.get_code_to_time_map(metadata_df)
        tensor([0., 0., 0., 1., 0.])
        """
        assert metadata_df["code/vocab_index"].is_sorted()
        code_to_time_map = torch.tensor(
            [
                cls.TIME_QUANTILE_VALUES[cls.TIME_QUANTILE_NAMES.index(code)]
                if code in set(TIME_QUANTILE_NAMES)
                else 0
                for code in metadata_df["code"]
            ]
        )
        code_to_time_map = torch.cat([code_to_time_map, torch.zeros(1)])
        return code_to_time_map

    @classmethod
    def get_code_to_numeric_value_map(cls, metadata_df, get_raw_values=False) -> dict:
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
        # TODO(Oufattole) remove this and enforce that metadata_df must include the values/min
        try:
            ordered_quantiles = [field.name for field in metadata_df.schema["values/quantiles"].fields]
            percentiles = [0, *[float(q.split("/")[-1]) for q in ordered_quantiles], 1]
            if "values/min" not in metadata_df.columns or "values/max" not in metadata_df.columns:
                logger.warning("Missing values/min and/or values/max values in metadata_df")

            # Process each row in the DataFrame
            for row in metadata_df.iter_rows(named=True):
                vocab_idx = row["code/vocab_index"]
                code = row["code"]
                raw_quantiles = [row["values/quantiles"][each] for each in ordered_quantiles]
                if "values/min" in row:
                    min_value = row["values/min"]
                else:
                    min_value = raw_quantiles[0]
                if "values/max" in row:
                    max_value = row["values/max"]
                else:
                    max_value = raw_quantiles[-1]
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
        except:  # noqa: E722
            pass
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
        >>> prediction_time_offset_years = torch.tensor([0.0, 1.0, 2.0])
        >>> from pprint import pprint, pformat
        >>> EicForecastingModule.to_trajectory_batch(code, mask, metadata_df, prediction_time_offset_years)
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
                [ True,  True,  True, False]]), time_scale='Y')
        """
        if not code_to_time_map:
            code_to_time_map = cls.get_code_to_time_map(metadata_df)
        if not code_to_numeric_value_map:
            code_to_numeric_value_map = cls.get_code_to_numeric_value_map(metadata_df)
        # Initialize lists to store the DataFrame rows
        time = torch.cumsum(code_to_time_map[code], dim=1)
        numeric_value = code_to_numeric_value_map[code]
        numeric_value_mask = numeric_value.isnan()
        time += prediction_time_offset_years.unsqueeze(1)
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
            out, out_lengths = self.model.generate(
                prompts=prompts,
                mask=mask,
                get_next_token_time=get_next_token_time,
                time_offset_years=prediction_time_offset_years,
                temperature=self.cfg.temperature,
                eos_tokens=self.cfg.eos_tokens,
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

            if self.cfg.get("zero_shot_labeler") is not None:
                if isinstance(self.cfg.zero_shot_labeler, DictConfig):
                    task_labeler = TaskLabeler(**self.cfg.zero_shot_labeler)
                else:
                    task_labeler = self.cfg.zero_shot_labeler
                pred_labels, unknown_pred = task_labeler(trajectory_batch)
                # Handle unknown values by setting their probability to 0.5
                if unknown_pred.sum().item() > 0:
                    logger.warning(f"Found {unknown_pred.sum().item()} unknown zero-shot predictions")
                pred_labels[unknown_pred] = 0.5
                input_batch[MODEL_PRED_PROBA_KEY] = pred_labels
                logger.info(f"Completed zero-shot labeling for sample {self.cfg.generate_id}")
        return input_batch
