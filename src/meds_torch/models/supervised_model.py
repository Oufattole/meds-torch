import torch
import torchmetrics
from loguru import logger
from omegaconf import DictConfig
from torch import nn

from meds_torch.models import (
    BACKBONE_EMBEDDINGS_KEY,
    MODEL_EMBEDDINGS_KEY,
    MODEL_LOGITS_KEY,
    MODEL_LOSS_KEY,
)
from meds_torch.models.base_model import BaseModule
from meds_torch.models.utils import OutputBase


class SupervisedModule(BaseModule):
    def __init__(
        self,
        cfg: DictConfig,  # , embedder: nn.Module, architecture: nn.Module, projection: nn.Module = None
    ):
        super().__init__(cfg)
        self.task_name = cfg.task_name
        if cfg.task_name is None:
            raise ValueError("task name must be specified")
        # shared components
        self.projection = nn.Linear(cfg.token_dim, 1)

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
        embeddings = batch[BACKBONE_EMBEDDINGS_KEY]
        logits = self.projection(embeddings)
        if self.cfg.get_representations:
            loss = None
        else:
            loss = self.criterion(logits.squeeze(dim=-1), batch[self.task_name].float())
        batch[MODEL_EMBEDDINGS_KEY] = embeddings
        batch[MODEL_LOGITS_KEY] = logits
        batch[MODEL_LOSS_KEY] = loss
        return batch

    def training_step(self, batch):
        output = self.forward(batch)
        # logs metrics for each training_step, and the average across the epoch
        self.train_acc.update(output[MODEL_LOGITS_KEY].squeeze(), batch[self.task_name].float())
        self.train_auc.update(output[MODEL_LOGITS_KEY].squeeze(), batch[self.task_name].float())
        self.train_apr.update(output[MODEL_LOGITS_KEY].squeeze(), batch[self.task_name].int())
        self.log("train/step_loss", output[MODEL_LOSS_KEY], on_step=True, batch_size=self.cfg.batch_size)
        self.log("train/loss", output[MODEL_LOSS_KEY], on_epoch=True, batch_size=self.cfg.batch_size)

        assert not torch.isnan(output[MODEL_LOSS_KEY]), "Loss is NaN"
        return output[MODEL_LOSS_KEY]

    def on_train_epoch_end(self):
        self.log(
            "train/auc",
            self.train_auc,
            on_epoch=True,
            batch_size=self.cfg.batch_size,
        )
        self.log(
            "train/acc",
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
        output = self.forward(batch)
        # logs metrics for each training_step, and the average across the epoch
        self.val_acc.update(output[MODEL_LOGITS_KEY].squeeze(), batch[self.task_name].float())
        self.val_auc.update(output[MODEL_LOGITS_KEY].squeeze(), batch[self.task_name].float())
        self.val_apr.update(output[MODEL_LOGITS_KEY].squeeze(), batch[self.task_name].int())

        self.log(
            "val/loss",
            output[MODEL_LOSS_KEY],
            on_epoch=True,
            batch_size=self.cfg.batch_size,
        )
        return output[MODEL_LOSS_KEY]

    def on_validation_epoch_end(self):
        self.log(
            "val/auc",
            self.val_auc,
            on_epoch=True,
            batch_size=self.cfg.batch_size,
        )
        self.log(
            "val/acc",
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
            "val/auc",
            self.val_auc.compute(),
            "val/acc",
            self.val_acc.compute(),
            "val_apr",
            self.val_apr.compute(),
        )

    def test_step(self, batch, batch_idx):
        output = self.forward(batch)
        # logs metrics for each training_step, and the average across the epoch
        self.test_acc.update(output[MODEL_LOGITS_KEY].squeeze(), batch[self.task_name].float())
        self.test_auc.update(output[MODEL_LOGITS_KEY].squeeze(), batch[self.task_name].float())
        self.test_apr.update(output[MODEL_LOGITS_KEY].squeeze(), batch[self.task_name].int())

        self.log("test/loss", output[MODEL_LOSS_KEY], batch_size=self.cfg.batch_size)
        return output[MODEL_LOSS_KEY]

    def on_test_epoch_end(self):
        self.log(
            "test/auc",
            self.test_auc,
            on_epoch=True,
            batch_size=self.cfg.batch_size,
        )
        self.log(
            "test/acc",
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
            "test/auc",
            self.test_auc.compute(),
            "test/acc",
            self.test_acc.compute(),
            "test_apr",
            self.test_apr.compute(),
        )
