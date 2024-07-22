import dataclasses

import lightning as L
import torch
import torchmetrics
from loguru import logger
from omegaconf import DictConfig
from torch import nn

from meds_torch.sequence_models import SEQUENCE_MODEL_EMBEDDINGS_KEY
from meds_torch.sequence_models.utils import OutputBase
from meds_torch.utils.module_class import Module


@dataclasses.dataclass
class SupervisedOutput(OutputBase):
    embeddings: torch.Tensor
    logits: torch.Tensor
    loss: torch.Tensor


class SupervisedModule(L.LightningModule, Module):
    def __init__(
        self,
        cfg: DictConfig,  # , embedder: nn.Module, architecture: nn.Module, projection: nn.Module = None
    ):
        super().__init__()
        self.cfg = cfg
        self.task_name = cfg.task_name
        if cfg.task_name is None:
            raise ValueError("task name must be specified")
        # shared components
        self.optimizer = cfg.optimizer
        self.scheduler = cfg.scheduler
        self.model = cfg.backbone
        self.projection = nn.Linear(cfg.token_dim, 1)
        self.input_encoder = cfg.input_encoder

        # metrics
        self.train_acc = torchmetrics.Accuracy(task="binary")
        self.train_auc = torchmetrics.AUROC(task="binary")
        self.train_apr = torchmetrics.AveragePrecision(task="binary")

        self.val_acc = torchmetrics.Accuracy(task="binary")
        self.val_auc = torchmetrics.AUROC(task="binary")
        self.val_apr = torchmetrics.AveragePrecision(task="binary")

        self.test_acc = torchmetrics.Accuracy(task="binary")
        self.test_auc = torchmetrics.AUROC(task="binary")
        self.test_apr = torchmetrics.AveragePrecision(task="binary")

        self.criterion = torch.nn.BCEWithLogitsLoss()

    def forward(self, batch) -> OutputBase:
        batch = self.input_encoder(batch)
        batch = self.model(batch)
        embeddings = batch[SEQUENCE_MODEL_EMBEDDINGS_KEY]
        logits = self.projection(embeddings)
        if self.cfg.get_representations:
            loss = None
        else:
            loss = self.criterion(logits.squeeze(), batch[self.task_name].float())
        return SupervisedOutput(
            embeddings=embeddings,  # TODO: Add embeddings
            logits=logits,
            loss=loss,
        )

    def training_step(self, batch):
        output: SupervisedOutput = self.forward(batch)
        # logs metrics for each training_step, and the average across the epoch
        self.train_acc.update(output.logits.squeeze(), batch[self.task_name].float())
        self.train_auc.update(output.logits.squeeze(), batch[self.task_name].float())
        self.train_apr.update(output.logits.squeeze(), batch[self.task_name].int())
        self.log("train_loss", output.loss, batch_size=self.cfg.batch_size)

        assert not torch.isnan(output.loss), "Loss is NaN"
        return output.loss

    def on_train_epoch_end(self):
        self.log(
            "train_auc",
            self.train_auc,
            on_epoch=True,
            batch_size=self.cfg.batch_size,
        )
        self.log(
            "train_acc",
            self.train_acc,
            on_epoch=True,
            batch_size=self.cfg.batch_size,
        )
        self.log(
            "train_apr",
            self.train_apr,
            on_epoch=True,
            batch_size=self.cfg.batch_size,
        )

    def validation_step(self, batch):
        output: OutputBase = self.forward(batch)
        # logs metrics for each training_step, and the average across the epoch
        self.val_acc.update(output.logits.squeeze(), batch[self.task_name].float())
        self.val_auc.update(output.logits.squeeze(), batch[self.task_name].float())
        self.val_apr.update(output.logits.squeeze(), batch[self.task_name].int())

        self.log(
            "val_loss",
            output.loss,
            on_epoch=True,
            batch_size=self.cfg.batch_size,
        )
        return output.loss

    def on_validation_epoch_end(self):
        self.log(
            "val_auc",
            self.val_auc,
            on_epoch=True,
            batch_size=self.cfg.batch_size,
        )
        self.log(
            "val_acc",
            self.val_acc,
            on_epoch=True,
            batch_size=self.cfg.batch_size,
        )
        self.log(
            "val_apr",
            self.val_apr,
            on_epoch=True,
            batch_size=self.cfg.batch_size,
        )
        logger.info(
            "val_auc",
            self.val_auc.compute(),
            "val_acc",
            self.val_acc.compute(),
            "val_apr",
            self.val_apr.compute(),
        )

    def test_step(self, batch, batch_idx):
        output: OutputBase = self.forward(batch)
        # logs metrics for each training_step, and the average across the epoch
        self.test_acc.update(output.logits.squeeze(), batch[self.task_name].float())
        self.test_auc.update(output.logits.squeeze(), batch[self.task_name].float())
        self.test_apr.update(output.logits.squeeze(), batch[self.task_name].int())

        self.log("test_loss", output.loss, batch_size=self.cfg.batch_size)
        return output.loss

    def on_test_epoch_end(self):
        self.log(
            "test_auc",
            self.test_auc,
            on_epoch=True,
            batch_size=self.cfg.batch_size,
        )
        self.log(
            "test_acc",
            self.test_acc,
            on_epoch=True,
            batch_size=self.cfg.batch_size,
        )
        self.log(
            "test_apr",
            self.test_apr,
            on_epoch=True,
            batch_size=self.cfg.batch_size,
        )
        logger.info(
            "test_auc",
            self.test_auc.compute(),
            "test_acc",
            self.test_acc.compute(),
            "test_apr",
            self.test_apr.compute(),
        )

    def configure_optimizers(self):
        optimizer = self.optimizer(self.parameters())
        return optimizer
