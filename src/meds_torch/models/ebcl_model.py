import dataclasses
import enum

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


class SupervisedView(enum.Enum):
    PRE = "pre"
    POST = "post"
    PRE_AND_POST = "pre_and_post"


@dataclasses.dataclass
class EBCLOutput(OutputBase):
    loss: float
    logits_per_pre: torch.Tensor
    logits_per_post: torch.Tensor
    post_embeds: torch.Tensor
    pre_embeds: torch.Tensor
    pre_norm_embeds: torch.Tensor
    post_norm_embeds: torch.Tensor
    post_model_output: torch.Tensor
    pre_model_output: torch.Tensor
    pre_attn: torch.Tensor = None
    post_attn: torch.Tensor = None


class EBCLModule(BaseModule):
    def __init__(self, cfg: DictConfig):
        super().__init__(cfg)
        batch_size = cfg.batch_size
        self.optimizer = cfg.optimizer
        self.scheduler = cfg.scheduler
        self.pre_model = self.model
        self.post_model = self.model
        #  metrics
        self.train_pre_acc = torchmetrics.Accuracy(num_classes=batch_size, task="multiclass")
        self.train_pre_auc = torchmetrics.AUROC(num_classes=batch_size, task="multiclass")

        self.train_post_acc = torchmetrics.Accuracy(num_classes=batch_size, task="multiclass")
        self.train_post_auc = torchmetrics.AUROC(num_classes=batch_size, task="multiclass")

        self.val_pre_acc = torchmetrics.Accuracy(num_classes=batch_size, task="multiclass")
        self.val_pre_auc = torchmetrics.AUROC(num_classes=batch_size, task="multiclass")

        self.val_post_acc = torchmetrics.Accuracy(num_classes=batch_size, task="multiclass")
        self.val_post_auc = torchmetrics.AUROC(num_classes=batch_size, task="multiclass")

        # Model components
        self.pre_projection = nn.Linear(cfg.token_dim, cfg.token_dim)
        self.post_projection = nn.Linear(cfg.token_dim, cfg.token_dim)

        self.t = nn.Linear(1, 1)
        self.criterion = torch.nn.CrossEntropyLoss()

    def forward(self, batch):
        pre_batch = batch[SupervisedView.PRE.value]
        pre_batch = self.input_encoder(pre_batch)
        pre_batch = self.pre_model(pre_batch)
        pre_outputs = pre_batch[BACKBONE_EMBEDDINGS_KEY]

        post_batch = batch[SupervisedView.POST.value]
        post_batch = self.input_encoder(post_batch)
        post_batch = self.pre_model(post_batch)
        post_outputs = post_batch[BACKBONE_EMBEDDINGS_KEY]

        pre_embeds = self.pre_projection(pre_outputs)
        post_embeds = self.post_projection(post_outputs)

        pre_norm_embeds = pre_embeds
        post_norm_embeds = post_embeds

        pre_norm_embeds = pre_embeds / pre_embeds.norm(dim=-1, keepdim=True)
        post_norm_embeds = post_embeds / post_embeds.norm(dim=-1, keepdim=True)

        logits = torch.mm(post_norm_embeds, pre_norm_embeds.T) * torch.exp(self.t.weight)
        labels = torch.arange(pre_norm_embeds.shape[0], device=pre_norm_embeds.device)
        logits_per_post = logits
        logits_per_pre = logits.T
        loss_post = self.criterion(logits_per_post, labels)
        loss_pre = self.criterion(logits_per_pre, labels)
        loss = (loss_pre + loss_post) / 2
        output = dict(
            pre=pre_outputs,
            post=post_outputs,
        )
        output[MODEL_EMBEDDINGS_KEY] = torch.concat([pre_norm_embeds, post_norm_embeds], dim=0)
        output[MODEL_TOKENS_KEY] = None
        output[MODEL_LOSS_KEY] = loss
        output[MODEL_LOGITS_KEY] = logits
        return output

    def training_step(self, batch):
        output = self.forward(batch)
        # pretrain metrics
        # pre metrics
        labels = torch.arange(self.cfg.batch_size, device=output[MODEL_LOGITS_KEY].device)
        self.train_pre_acc.update(output[MODEL_LOGITS_KEY], labels)
        self.train_pre_auc.update(output[MODEL_LOGITS_KEY], labels)

        # post metrics
        self.train_post_acc.update(output[MODEL_LOGITS_KEY].T, labels)
        self.train_post_auc.update(output[MODEL_LOGITS_KEY].T, labels)

        self.log("train_loss", output[MODEL_LOSS_KEY], batch_size=self.cfg.batch_size)
        return output[MODEL_LOSS_KEY]

    def validation_step(self, batch):
        output: EBCLOutput = self.forward(batch)
        # pretrain metrics
        # pre metrics
        labels = torch.arange(self.cfg.batch_size, device=output[MODEL_LOGITS_KEY].device)
        self.val_pre_acc.update(output[MODEL_LOGITS_KEY], labels)
        self.val_pre_auc.update(output[MODEL_LOGITS_KEY], labels)

        # post metrics
        self.val_post_acc.update(output[MODEL_LOGITS_KEY].T, labels)
        self.val_post_auc.update(output[MODEL_LOGITS_KEY].T, labels)
        self.log("val_loss", output[MODEL_LOSS_KEY], batch_size=self.cfg.batch_size)
        return output[MODEL_LOSS_KEY]

    def test_step(self, batch):
        output: EBCLOutput = self.forward(batch)
        # pretrain metrics
        # pre metrics
        labels = torch.arange(self.cfg.batch_size, device=output[MODEL_LOGITS_KEY].device)
        self.test_pre_acc.update(output[MODEL_LOGITS_KEY], labels)
        self.test_pre_auc.update(output[MODEL_LOGITS_KEY], labels)

        # post metrics
        self.test_post_acc.update(output[MODEL_LOGITS_KEY].T, labels)
        self.test_post_auc.update(output[MODEL_LOGITS_KEY].T, labels)
        self.log("test_loss", output[MODEL_LOSS_KEY], batch_size=self.cfg.batch_size)
        return output[MODEL_LOSS_KEY]

    def on_train_epoch_end(self):
        self.log(
            "train_pre_acc",
            self.train_pre_acc,
            on_epoch=True,
            batch_size=self.cfg.batch_size,
        )
        self.log(
            "train_pre_auc",
            self.train_pre_auc,
            on_epoch=True,
            batch_size=self.cfg.batch_size,
        )

        self.log(
            "train_post_acc",
            self.train_post_acc,
            on_epoch=True,
            batch_size=self.cfg.batch_size,
        )
        self.log(
            "train_post_auc",
            self.train_post_auc,
            on_epoch=True,
            batch_size=self.cfg.batch_size,
        )

    def on_val_epoch_end(self):
        self.log(
            "val_pre_acc",
            self.val_pre_acc,
            on_epoch=True,
            batch_size=self.cfg.batch_size,
        )
        self.log(
            "val_pre_auc",
            self.val_pre_auc,
            on_epoch=True,
            batch_size=self.cfg.batch_size,
        )

        self.log(
            "val_post_acc",
            self.val_post_acc,
            on_epoch=True,
            batch_size=self.cfg.batch_size,
        )
        self.log(
            "val_post_auc",
            self.val_post_auc,
            on_epoch=True,
            batch_size=self.cfg.batch_size,
        )

        print(
            "val_pre_acc",
            self.val_pre_acc.compute(),
            "val_pre_auc",
            self.val_pre_auc.compute(),
        )
        print(
            "val_post_acc",
            self.val_post_acc.compute(),
            "val_post_auc",
            self.val_post_auc.compute(),
        )

    def on_test_epoch_end(self):
        self.log(
            "test_pre_acc",
            self.test_pre_acc,
            on_epoch=True,
            batch_size=self.cfg.batch_size,
        )
        self.log(
            "test_pre_auc",
            self.test_pre_auc,
            on_epoch=True,
            batch_size=self.cfg.batch_size,
        )

        self.log(
            "test_post_acc",
            self.test_post_acc,
            on_epoch=True,
            batch_size=self.cfg.batch_size,
        )
        self.log(
            "test_post_auc",
            self.test_post_auc,
            on_epoch=True,
            batch_size=self.cfg.batch_size,
        )
        print(
            "test_pre_acc",
            self.test_pre_acc.compute(),
            "test_pre_auc",
            self.test_pre_auc.compute(),
        )
        print(
            "test_post_acc",
            self.test_post_acc.compute(),
            "test_post_auc",
            self.test_post_auc.compute(),
        )
