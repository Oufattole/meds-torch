import torch
import torchmetrics
from omegaconf import DictConfig
from torch import nn

from meds_torch.models import BACKBONE_EMBEDDINGS_KEY, MODEL_LOSS_KEY
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

    def forward(self, batch):
        forecast_window_data = batch[self.cfg.forecast_window_name]
        batch = self.model(self.input_encoder(batch[self.cfg.input_window_name]))

        numerical_values = forecast_window_data["numeric_value"]
        codes = forecast_window_data["code"]
        vocab_size = self.cfg.vocab_size

        with torch.no_grad():
            # create presence target and value target
            presence_target = torch.zeros(
                (codes.shape[0], vocab_size), dtype=torch.float32, device=codes.device
            )
            row_indices = torch.arange(codes.shape[0]).unsqueeze(-1).expand_as(codes).reshape(-1)
            col_indices = codes.reshape(-1)
            presence_target[row_indices, col_indices] = 1

            # create value target
            numerical_value_mask = forecast_window_data["numerical_value_mask"]
            numerical_value_codes = codes * numerical_value_mask
            value_target = torch.zeros(
                (numerical_value_codes.shape[0], vocab_size), dtype=torch.float32, device=codes.device
            )
            row_indices = (
                torch.arange(numerical_value_codes.shape[0])
                .unsqueeze(-1)
                .expand_as(numerical_value_codes)
                .reshape(-1)
            )
            col_indices = codes.reshape(-1)
            numerical_value_indices = (
                torch.arange(numerical_value_codes.shape[1]).expand_as(numerical_value_codes).reshape(-1)
            )
            value_target[row_indices, col_indices] = numerical_values[row_indices, numerical_value_indices]

        value_forecast = self.value_projection(batch[BACKBONE_EMBEDDINGS_KEY])
        presence_forecast = self.presence_projection(batch[BACKBONE_EMBEDDINGS_KEY])

        value_loss = self.value_criterion(value_forecast, value_target)
        presence_loss = self.presence_criterion(presence_forecast, presence_target)
        loss = value_loss + presence_loss

        output = batch
        output["MODEL//VALUE_TARGET"] = value_target
        output["MODEL//PRESENCE_TARGET"] = presence_target
        output["MODEL//VALUE_FORECAST"] = value_forecast
        output["MODEL//PRESENCE_FORECAST"] = presence_forecast
        output["MODEL//VALUE_LOSS"] = value_loss
        output["MODEL//PRESENCE_LOSS"] = presence_loss
        output[MODEL_LOSS_KEY] = loss

        return output

    def training_step(self, batch):
        output = self.forward(batch)

        self.train_presence_mse(output["MODEL//PRESENCE_FORECAST"], output["MODEL//PRESENCE_TARGET"])
        self.train_value_mse(output["MODEL//VALUE_FORECAST"], output["MODEL//VALUE_TARGET"])

        self.log("train/loss", output[MODEL_LOSS_KEY], batch_size=self.cfg.batch_size)
        return output[MODEL_LOSS_KEY]

    def validation_step(self, batch):
        output = self.forward(batch)

        self.val_presence_mse(output["MODEL//PRESENCE_FORECAST"], output["MODEL//PRESENCE_TARGET"])
        self.val_value_mse(output["MODEL//VALUE_FORECAST"], output["MODEL//VALUE_TARGET"])

        self.log("val/loss", output[MODEL_LOSS_KEY], batch_size=self.cfg.batch_size)
        return output[MODEL_LOSS_KEY]

    def test_step(self, batch):
        output = self.forward(batch)

        self.test_presence_mse(output["MODEL//PRESENCE_FORECAST"], output["MODEL//PRESENCE_TARGET"])
        self.test_value_mse(output["MODEL//VALUE_FORECAST"], output["MODEL//VALUE_TARGET"])

        self.log("test/loss", output[MODEL_LOSS_KEY], batch_size=self.cfg.batch_size)

        return output[MODEL_LOSS_KEY]

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
