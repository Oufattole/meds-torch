import dataclasses

import torch
import torchmetrics
from omegaconf import DictConfig
from torch import nn

from meds_torch.models import (
    BACKBONE_EMBEDDINGS_KEY,
    MODEL_EMBEDDINGS_KEY,
    MODEL_LOGITS_KEY,
    MODEL_LOSS_KEY,
    MODEL_TOKENS_KEY,
)
from meds_torch.models.base_model import BaseModule
from meds_torch.models.utils import OutputBase


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
        self.train_acc = torchmetrics.Accuracy(num_classes=cfg.batch_size, task="binary")
        self.train_auc = torchmetrics.AUROC(num_classes=cfg.batch_size, task="binary")

        self.val_acc = torchmetrics.Accuracy(num_classes=cfg.batch_size, task="binary")
        self.val_auc = torchmetrics.AUROC(num_classes=cfg.batch_size, task="binary")

        # pretraining model components
        self.projection = nn.Linear(cfg.token_dim * 2, 1)
        self.criterion = torch.nn.BCEWithLogitsLoss()

    def forward(self, batch):
        pre_batch = batch[self.cfg.pre_window_name]
        pre_batch = self.input_encoder(pre_batch)
        pre_batch = self.model(pre_batch)
        pre_outputs = pre_batch[BACKBONE_EMBEDDINGS_KEY]

        post_batch = batch[self.cfg.post_window_name]
        post_batch = self.input_encoder(post_batch)
        post_batch = self.model(post_batch)
        post_outputs = post_batch[BACKBONE_EMBEDDINGS_KEY]

        random_flips = torch.randint(0, 2, (pre_outputs.shape[0], 1)).bool()
        shuffled_pre_outputs = torch.where(random_flips, post_outputs, pre_outputs)
        shuffled_post_outputs = torch.where(random_flips, pre_outputs, post_outputs)
        classifier_inputs = torch.concat([shuffled_pre_outputs, shuffled_post_outputs], dim=1)

        logits = self.projection(classifier_inputs)
        loss = self.criterion(logits, random_flips.float())

        output = dict(
            pre=pre_batch,
            post=post_batch,
        )
        output[MODEL_EMBEDDINGS_KEY] = classifier_inputs
        output[MODEL_TOKENS_KEY] = None
        output[MODEL_LOSS_KEY] = loss
        output[MODEL_LOGITS_KEY] = logits
        output["MODEL//LABELS"] = random_flips

        return output

    def training_step(self, batch):
        output: OCPPretrainOutput = self.forward(batch)
        # pretrain metrics
        # pre metrics
        labels = output["MODEL//LABELS"].float()
        self.train_acc.update(output[MODEL_LOGITS_KEY], labels)
        self.train_auc.update(output[MODEL_LOGITS_KEY], labels)
        output[MODEL_LOSS_KEY]

    def validation_step(self, batch):
        output: OCPPretrainOutput = self.forward(batch)
        # pretrain metrics
        labels = output["MODEL//LABELS"].float()
        self.val_acc.update(output[MODEL_LOGITS_KEY], labels)
        self.val_auc.update(output[MODEL_LOGITS_KEY], labels)
        output[MODEL_LOSS_KEY]

    def test_step(self, batch):
        output: OCPPretrainOutput = self.forward(batch)
        # pretrain metrics
        labels = output["MODEL//LABELS"].float()
        self.test_acc.update(output[MODEL_LOGITS_KEY], labels)
        self.test_auc.update(output[MODEL_LOGITS_KEY], labels)
        return output[MODEL_LOSS_KEY]

    def on_train_epoch_end(self):
        self.log(
            "train_acc",
            self.train_acc,
            on_epoch=True,
            batch_size=self.cfg.batch_size,
        )
        self.log(
            "train_auc",
            self.train_auc,
            on_epoch=True,
            batch_size=self.cfg.batch_size,
        )

    def on_val_epoch_end(self):
        self.log(
            "val_acc",
            self.val_acc,
            on_epoch=True,
            batch_size=self.cfg.batch_size,
        )
        self.log(
            "val_auc",
            self.val_auc,
            on_epoch=True,
            batch_size=self.cfg.batch_size,
        )
        print(
            "val_acc",
            self.val_acc.compute(),
            "val_auc",
            self.val_auc.compute(),
        )

    def on_test_epoch_end(self):
        self.log(
            "test_acc",
            self.test_acc,
            on_epoch=True,
            batch_size=self.cfg.batch_size,
        )
        self.log(
            "test_auc",
            self.test_auc,
            on_epoch=True,
            batch_size=self.cfg.batch_size,
        )
        print(
            "test_acc",
            self.test_acc.compute(),
            "test_auc",
            self.test_auc.compute(),
        )
