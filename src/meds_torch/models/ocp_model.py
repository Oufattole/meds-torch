import dataclasses
import enum

import torch
import torchmetrics
from omegaconf import DictConfig
from torch import nn

from meds_torch.models.base_model import BaseModule
from meds_torch.models.utils import OutputBase


class SupervisedView(enum.Enum):
    PRE = "pre"
    POST = "post"
    PRE_AND_POST = "pre_and_post"


@dataclasses.dataclass
class OCPPretrainOutput(OutputBase):
    loss: torch.Tensor
    logits: torch.Tensor
    outputs: torch.Tensor
    attn: torch.Tensor = None


class OCPModule(BaseModule):
    def __init__(self, cfg: DictConfig):
        super().__init__(cfg)

        # pretrain metrics
        self.train_pretrain_acc = torchmetrics.Accuracy(num_classes=cfg.batch_size, task="binary")
        self.train_pretrain_auc = torchmetrics.AUROC(num_classes=cfg.batch_size, task="binary")

        self.val_pretrain_acc = torchmetrics.Accuracy(num_classes=cfg.batch_size, task="binary")
        self.val_pretrain_auc = torchmetrics.AUROC(num_classes=cfg.batch_size, task="binary")

        # pretraining model components
        self.pretrain_projection = nn.Linear(cfg.token_dim, 1)
        self.pretrain_criterion = torch.nn.BCEWithLogitsLoss()

    def pretrain_forward(self, batch):
        outputs = self.pre_model(batch[SupervisedView.EARLY_FUSION.value]).rep

        logits = self.pretrain_projection(outputs)
        loss = self.pretrain_criterion(logits.squeeze(), batch["flip"].float())

        return OCPPretrainOutput(
            logits=logits,
            loss=loss,
            outputs=outputs,
        )

    def pretrain_training_step(self, batch):
        output: OCPPretrainOutput = self.forward(batch)
        # pretrain metrics
        # pre metrics
        labels = batch["flip"].float()
        self.train_pretrain_acc.update(output.logits.squeeze(), labels)
        self.train_pretrain_auc.update(output.logits.squeeze(), labels)
        return output

    def pretrain_validation_step(self, batch):
        output: OCPPretrainOutput = self.forward(batch)
        # pretrain metrics
        labels = batch["flip"].float()
        self.val_pretrain_acc.update(output.logits.squeeze(), labels)
        self.val_pretrain_auc.update(output.logits.squeeze(), labels)
        return output

    def pretrain_test_step(self, batch):
        output: OCPPretrainOutput = self.forward(batch)
        # pretrain metrics
        labels = batch["flip"].float()
        self.test_pretrain_acc.update(output.logits.squeeze(), labels)
        self.test_pretrain_auc.update(output.logits.squeeze(), labels)
        return output

    def pretrain_on_train_epoch_end(self):
        self.log(
            "train_pretrain_acc",
            self.train_pretrain_acc,
            on_epoch=True,
            batch_size=self.cfg.batch_size,
        )
        self.log(
            "train_pretrain_auc",
            self.train_pretrain_auc,
            on_epoch=True,
            batch_size=self.cfg.batch_size,
        )

    def pretrain_on_val_epoch_end(self):
        self.log(
            "val_pretrain_acc",
            self.val_pretrain_acc,
            on_epoch=True,
            batch_size=self.cfg.batch_size,
        )
        self.log(
            "val_pretrain_auc",
            self.val_pretrain_auc,
            on_epoch=True,
            batch_size=self.cfg.batch_size,
        )
        print(
            "val_pretrain_acc",
            self.val_pretrain_acc.compute(),
            "val_pretrain_auc",
            self.val_pretrain_auc.compute(),
        )

    def pretrain_on_test_epoch_end(self):
        self.log(
            "test_pretrain_acc",
            self.test_pretrain_acc,
            on_epoch=True,
            batch_size=self.cfg.batch_size,
        )
        self.log(
            "test_pretrain_auc",
            self.test_pretrain_auc,
            on_epoch=True,
            batch_size=self.cfg.batch_size,
        )
        print(
            "test_pretrain_acc",
            self.test_pretrain_acc.compute(),
            "test_pretrain_auc",
            self.test_pretrain_auc.compute(),
        )
