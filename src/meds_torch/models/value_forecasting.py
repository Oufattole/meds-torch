import torch
import torchmetrics
from loguru import logger
from omegaconf import DictConfig
from torch import nn

from meds_torch.models import BACKBONE_EMBEDDINGS_KEY, MODEL_BATCH_LOSS_KEY
from meds_torch.models.base_model import BaseModule


class ValueForecastingModule(BaseModule):
    def __init__(self, cfg: DictConfig):
        super().__init__(cfg)
        # pretraining model components
        self.presence_projection = nn.Linear(cfg.token_dim, cfg.vocab_size)
        self.presence_criterion = nn.MSELoss()

        # logging components
        self.train_presence_mse = torchmetrics.MeanSquaredError()
        self.train_value_mse = torchmetrics.MeanSquaredError()

        self.val_presence_mse = torchmetrics.MeanSquaredError()
        self.val_value_mse = torchmetrics.MeanSquaredError()

        self.test_presence_mse = torchmetrics.MeanSquaredError()
        self.test_value_mse = torchmetrics.MeanSquaredError()

        self.value_projection = nn.Linear(cfg.token_dim, cfg.vocab_size)
        self.value_criterion = nn.MSELoss()

    @staticmethod
    def set_target(empty_target, row_indices, col_indices, value=1):
        try:
            empty_target[row_indices, col_indices] = 1
        except IndexError:
            logger.warning("Index out of bounds, doing inefficient loop")
            for i, j in zip(row_indices, col_indices):
                empty_target[i, j] = 1

    def forward(self, batch):
        forecast_window_data = batch[self.cfg.forecast_window_name]
        batch[self.cfg.input_window_name] = self.model(self.input_encoder(batch[self.cfg.input_window_name]))

        numeric_values = forecast_window_data["numeric_value"]
        codes = forecast_window_data["code"]
        vocab_size = self.cfg.vocab_size

        with torch.no_grad():
            # create presence target and value target
            presence_target = torch.zeros(
                (codes.shape[0], vocab_size), dtype=torch.float32, device=codes.device
            )
            presence_target = presence_target.scatter_(
                dim=1,
                index=codes.to(torch.int64),
                src=torch.ones_like(codes, dtype=torch.float32),
                reduce="add",
            ).clamp_(max=1)

            # create value target
            numeric_value_mask = forecast_window_data["numeric_value_mask"]
            numeric_value_codes = codes * numeric_value_mask
            value_target = torch.zeros(
                (numeric_value_codes.shape[0], vocab_size), dtype=torch.float32, device=codes.device
            )
            value_target = value_target.scatter_(
                dim=1, index=codes.to(torch.int64), src=numeric_values, reduce="add"
            )
            count_target = torch.zeros_like(value_target)
            count_target = count_target.scatter_(
                dim=1, index=codes.to(torch.int64), src=torch.ones_like(numeric_values), reduce="add"
            )
            value_target = torch.where(count_target > 0, value_target / count_target, value_target)

            # Set the 0 index to zero to ignore mask tokens
            value_target[:, 0] = 0

        value_forecast = self.value_projection(batch[self.cfg.input_window_name][BACKBONE_EMBEDDINGS_KEY])
        presence_forecast = self.presence_projection(
            batch[self.cfg.input_window_name][BACKBONE_EMBEDDINGS_KEY]
        )

        value_loss = self.value_criterion(value_forecast, value_target)
        presence_loss = self.presence_criterion(presence_forecast, presence_target)
        loss = value_loss + presence_loss

        batch["MODEL//VALUE_TARGET"] = value_target
        batch["MODEL//PRESENCE_TARGET"] = presence_target
        batch["MODEL//VALUE_FORECAST"] = value_forecast
        batch["MODEL//PRESENCE_FORECAST"] = presence_forecast
        batch["MODEL//VALUE_LOSS"] = value_loss
        batch["MODEL//PRESENCE_LOSS"] = presence_loss
        batch[MODEL_BATCH_LOSS_KEY] = loss

        return batch

    def training_step(self, batch):
        output = self.forward(batch)

        self.train_presence_mse(output["MODEL//PRESENCE_FORECAST"], output["MODEL//PRESENCE_TARGET"])
        self.train_value_mse(output["MODEL//VALUE_FORECAST"], output["MODEL//VALUE_TARGET"])

        self.log("train/loss", output[MODEL_BATCH_LOSS_KEY], batch_size=self.cfg.batch_size)
        return output[MODEL_BATCH_LOSS_KEY]

    def validation_step(self, batch):
        output = self.forward(batch)

        self.val_presence_mse(output["MODEL//PRESENCE_FORECAST"], output["MODEL//PRESENCE_TARGET"])
        self.val_value_mse(output["MODEL//VALUE_FORECAST"], output["MODEL//VALUE_TARGET"])

        self.log("val/loss", output[MODEL_BATCH_LOSS_KEY], batch_size=self.cfg.batch_size)
        return output[MODEL_BATCH_LOSS_KEY]

    def test_step(self, batch):
        output = self.forward(batch)

        self.test_presence_mse(output["MODEL//PRESENCE_FORECAST"], output["MODEL//PRESENCE_TARGET"])
        self.test_value_mse(output["MODEL//VALUE_FORECAST"], output["MODEL//VALUE_TARGET"])

        self.log("test/loss", output[MODEL_BATCH_LOSS_KEY], batch_size=self.cfg.batch_size)

        return output[MODEL_BATCH_LOSS_KEY]

    def on_train_epoch_end(self):
        self.log(
            "train_presence_mse",
            self.train_presence_mse,
            on_epoch=True,
        )
        self.log(
            "train_value_mse",
            self.train_value_mse,
            on_epoch=True,
        )

    def on_val_epoch_end(self):
        self.log(
            "val_presence_mse",
            self.val_presence_mse,
            on_epoch=True,
        )
        self.log(
            "val_value_mse",
            self.val_value_mse,
            on_epoch=True,
        )

    def on_test_epoch_end(self):
        self.log(
            "test_presence_mse",
            self.test_presence_mse,
            on_epoch=True,
        )
        self.log(
            "test_value_mse",
            self.test_value_mse,
            on_epoch=True,
        )
