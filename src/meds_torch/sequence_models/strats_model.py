import dataclasses
import enum

import torch
from omegaconf import DictConfig
from torch import nn

from meds_torch.model.supervised_model import SupervisedModule
from meds_torch.model.utils import OutputBase


class SupervisedView(enum.Enum):
    PRE = "pre"
    POST = "post"
    PRE_AND_POST = "pre_and_post"


@dataclasses.dataclass
class StratsPretrainOutput(OutputBase):
    loss: torch.Tensor
    forecast: torch.Tensor
    outputs: torch.Tensor
    attn: torch.Tensor = None


class StratsModule(SupervisedModule):
    def __init__(self, cfg: DictConfig):
        super().__init__(cfg)
        self.cfg = cfg
        if cfg.model_config.pretrain:
            assert self.cfg.dataloader_config.modalities == SupervisedView.EARLY_FUSION

        # pretraining model components
        self.forecast_dim = (
            cfg.dataset_config.n_variable - cfg.dataset_config.n_cat_variable + cfg.dataset_config.n_cat_value
        )
        self.forecast_projection = nn.Linear(cfg.model_config.embed_size, self.forecast_dim)
        self.forecast_criterion = nn.MSELoss(
            reduction="none"
        )  # this computes the loss for each element in the batch instead of averaging the squared error.
        # we do this because we will mask out values that are not present in the prediction window and
        # ignore them for the MSE calculation

    def pretrain_forward(self, batch):
        assert self.cfg.dataloader_config.modalities == SupervisedView.EARLY_FUSION
        outputs = self.pre_model(batch[SupervisedView.EARLY_FUSION.value]).rep
        forecast_target = batch["forecast_target"]
        forecast_target_mask = batch["forecast_target_mask"]  # mask 1s are the ones we want to predict
        forecast = self.forecast_projection(outputs)

        loss = self.forecast_criterion(forecast, forecast_target)  # (loss * forecast_target_mask)
        loss = (
            ((loss * forecast_target_mask).T / forecast_target_mask.sum(dim=-1)).sum(dim=0).mean()
        )  # gives mean squred error over unmasked elements

        return StratsPretrainOutput(
            forecast=forecast,
            loss=loss,
            outputs=outputs,
        )

    def pretrain_training_step(self, batch):
        output: StratsPretrainOutput = self.forward(batch)
        return output

    def pretrain_validation_step(self, batch):
        output: StratsPretrainOutput = self.forward(batch)
        return output

    def pretrain_test_step(self, batch):
        output: StratsPretrainOutput = self.forward(batch)
        return output

    def pretrain_on_train_epoch_end(self):
        pass

    def pretrain_on_val_epoch_end(self):
        pass

    def pretrain_on_test_epoch_end(self):
        pass

    def configure_optimizers(self):
        if self.cfg.model_config.half_dtype:
            eps = 1e-4
        else:
            eps = 1e-8
        optimizer = torch.optim.Adam(self.parameters(), lr=self.cfg.model_config.lr, eps=eps)
        return optimizer

    @classmethod
    def initialize_pretrain(cls, cfg: DictConfig):
        assert cfg.model_config.pretrain
        return cls(cfg)
