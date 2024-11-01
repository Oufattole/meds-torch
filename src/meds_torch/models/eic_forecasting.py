import numpy as np
import polars as pl
import torch
import torch.nn.functional as F
from loguru import logger
from mixins import TimeableMixin
from omegaconf import DictConfig
from torchmetrics import Metric
from torchmetrics.classification import MulticlassAccuracy, MulticlassAUROC
from x_transformers import AutoregressiveWrapper
from x_transformers.autoregressive_wrapper import eval_decorator

from meds_torch.input_encoder import INPUT_ENCODER_MASK_KEY, INPUT_ENCODER_TOKENS_KEY
from meds_torch.input_encoder.eic_encoder import EicEncoder
from meds_torch.models import BACKBONE_TOKENS_KEY, GENERATE_PREFIX, MODEL_LOSS_KEY
from meds_torch.models.base_model import BaseModule
from meds_torch.models.components import AUTOREGRESSIVE_MODELS

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

    def __init__(self, vocab_size: int, dist_sync_on_step=False):
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
        self.top_1_accuracy = MulticlassAccuracy(num_classes=vocab_size, top_k=1)
        self.top_5_accuracy = MulticlassAccuracy(num_classes=vocab_size, top_k=5)
        self.top_10_accuracy = MulticlassAccuracy(num_classes=vocab_size, top_k=10)

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
        self.top_1_accuracy.update(flat_logits, flat_targets)
        self.top_5_accuracy.update(flat_logits, flat_targets)
        self.top_10_accuracy.update(flat_logits, flat_targets)

    def compute(self):
        """
        Compute the AUROC and top-n accuracy based on accumulated statistics.

        Returns:
            dict: A dictionary containing the computed AUROC and top-n accuracy for each n in top_n.
        """
        results = {
            "auroc": self.auroc.compute(),
        }
        results["top_1_accuracy"] = self.top_1_accuracy.compute()
        results["top_5_accuracy"] = self.top_5_accuracy.compute()
        results["top_10_accuracy"] = self.top_10_accuracy.compute()
        return results


class EicForecastingModule(BaseModule, TimeableMixin):
    """EIC token based GPT Forecasting Model."""

    def __init__(self, cfg: DictConfig):
        super().__init__(cfg)
        if not isinstance(self.model, AUTOREGRESSIVE_MODELS):
            raise ValueError(
                f"Unsupported model type: {type(self.model)}, choose one from {AUTOREGRESSIVE_MODELS}"
            )
        if not isinstance(self.input_encoder, EicEncoder):
            raise NotImplementedError(f"Unsupported input encoder type: {type(self.input_encoder)}")
        self.code_head = self.cfg.code_head

        num_future_codes = self.cfg.get("num_future_codes", None)
        if num_future_codes is not None:
            logger.info(f"Using {num_future_codes} future codes for forecasting")
        self.train_next_token_metric = NextTokenPredictionMetric(vocab_size=self.cfg.vocab_size)
        self.val_next_token_metric = NextTokenPredictionMetric(vocab_size=self.cfg.vocab_size)
        self.test_next_token_metric = NextTokenPredictionMetric(vocab_size=self.cfg.vocab_size)

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
            masked_code_logits.view(-1, masked_code_logits.size(-1)),
            masked_code_target.view(-1).to(dtype=torch.long),
            reduction="mean",
        )

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

        forecast = self.get_forecast_logits(model_output)
        batch[CODE_LOGITS] = forecast[CODE_LOGITS]

        code_loss = self.get_loss(batch)
        batch[MODEL_LOSS_KEY] = code_loss
        batch = self._generate(batch)
        return batch

    def _log(self, batch, split):
        self.log(split + "/loss", batch[MODEL_LOSS_KEY])

    def _generate(self, batch):
        if self.cfg.num_samples > 0:
            return self.generate_evaluation(batch)
        else:
            return batch

    def training_step(self, batch):
        batch = self(batch)
        assert not torch.isnan(batch[MODEL_LOSS_KEY]), "Loss is NaN"
        self._log(batch, "train")
        self.train_next_token_metric.update(batch[CODE_LOGITS], batch["code"], batch["mask"])
        return batch[MODEL_LOSS_KEY]

    def on_train_epoch_end(self):
        next_token_results = self.train_next_token_metric.compute()
        for metric_name, value in next_token_results.items():
            self.log(f"test/NEXT_TOKEN/{metric_name.upper()}", value, on_epoch=True)
        self.train_next_token_metric.reset()

    def validation_step(self, batch):
        batch = self(batch)
        assert not torch.isnan(batch[MODEL_LOSS_KEY]), "Loss is NaN"
        self._log(batch, "val")
        self.val_next_token_metric.update(batch[CODE_LOGITS], batch["code"], batch["mask"])
        return batch[MODEL_LOSS_KEY]

    def on_validation_epoch_end(self):
        next_token_results = self.val_next_token_metric.compute()
        for metric_name, value in next_token_results.items():
            self.log(f"test/NEXT_TOKEN/{metric_name.upper()}", value, on_epoch=True)
        self.val_next_token_metric.reset()

    def test_step(self, batch):
        batch = self(batch)
        assert not torch.isnan(batch[MODEL_LOSS_KEY]), "Loss is NaN"
        self._log(batch, "test")
        loss = batch[MODEL_LOSS_KEY]
        self.test_next_token_metric.update(batch[CODE_LOGITS], batch["code"], batch["mask"])
        return loss

    def on_test_epoch_end(self):
        next_token_results = self.test_next_token_metric.compute()
        for metric_name, value in next_token_results.items():
            self.log(f"test/NEXT_TOKEN/{metric_name.upper()}", value, on_epoch=True)
        self.test_next_token_metric.reset()

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

        # Compute bounds
        model = AutoregressiveWrapper(self.model.model)

        # Calculate actual lengths of prompts using the mask
        prompt_lengths = mask.sum(dim=1)

        logger.info("Generate output using the history")
        self.time_quantile_map = torch.tensor(
            [
                TIME_QUANTILE_VALUES[TIME_QUANTILE_NAMES.index(code)]
                if code in set(TIME_QUANTILE_NAMES)
                else 0
                for code in self.metadata_df["code"]
            ],
            device=self.device,
        )
        self.time_quantile_map = torch.cat([self.time_quantile_map, torch.zeros(1)])

        for i in range(self.cfg.num_samples):
            out = model.generate(
                prompts,
                self.cfg.max_seq_len,
                prompt_lens=prompt_lengths,
                temperature=self.cfg.temperature,
                eos_token=self.cfg.eos_token_id,
                context_mask=mask,
                **kwargs,
            )[
                :, prompts.shape[1] :
            ]  # Remove the prompt

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

        return input_batch
