import dataclasses
import enum

import torch
from omegaconf import DictConfig
from torch import nn

from meds_torch.models.base_model import BaseModule
from meds_torch.models.utils import OutputBase


class SupervisedView(enum.Enum):
    PRE = "pre"
    POST = "post"
    PRE_AND_POST = "pre_and_post"


@dataclasses.dataclass
class ValueForecastingOutput(OutputBase):
    loss: torch.Tensor
    forecast: torch.Tensor
    outputs: torch.Tensor
    attn: torch.Tensor = None


class ValueForecastingModule(BaseModule):
    def __init__(self, cfg: DictConfig):
        super().__init__(cfg)
        # pretraining model components
        self.forecast_projection = nn.Linear(cfg.token_dim, cfg.vocab_size)
        self.forecast_criterion = nn.MSELoss(
            reduction="none"
        )  # this computes the loss for each element in the batch instead of averaging the squared error.
        # we do this because we will mask out values that are not present in the prediction window and
        # ignore them for the MSE calculation

    def pretrain_forward(self, batch):
        outputs = self.pre_model(batch[SupervisedView.EARLY_FUSION.value]).rep
        forecast_target = batch["forecast_target"]
        forecast_target_mask = batch["forecast_target_mask"]  # mask 1s are the ones we want to predict
        forecast = self.forecast_projection(outputs)

        loss = self.forecast_criterion(forecast, forecast_target)  # (loss * forecast_target_mask)
        loss = (
            ((loss * forecast_target_mask).T / forecast_target_mask.sum(dim=-1)).sum(dim=0).mean()
        )  # gives mean squred error over unmasked elements

        return ValueForecastingOutput(
            forecast=forecast,
            loss=loss,
            outputs=outputs,
        )

    def pretrain_training_step(self, batch):
        output: ValueForecastingOutput = self.forward(batch)
        return output

    def pretrain_validation_step(self, batch):
        output: ValueForecastingOutput = self.forward(batch)
        return output

    def pretrain_test_step(self, batch):
        output: ValueForecastingOutput = self.forward(batch)
        return output

    def pretrain_on_train_epoch_end(self):
        pass

    def pretrain_on_val_epoch_end(self):
        pass

    def pretrain_on_test_epoch_end(self):
        pass
