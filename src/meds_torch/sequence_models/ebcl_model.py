import dataclasses
import enum

import torch
import torchmetrics
from omegaconf import DictConfig
from torch import nn

from meds_torch.model.supervised_model import SupervisedModule
from meds_torch.model.utils import OutputBase


class SupervisedView(enum.Enum):
    PRE = "pre"
    POST = "post"
    PRE_AND_POST = "pre_and_post"


@dataclasses.dataclass
class EBCLPretrainOutput(OutputBase):
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


class EBCLModule(SupervisedModule):
    def __init__(self, cfg: DictConfig):
        super().__init__(cfg)
        batch_size = cfg.dataloader_config.batch_size
        # pretrain metrics
        self.train_pretrain_pre_acc = torchmetrics.Accuracy(num_classes=batch_size, task="multiclass")
        self.train_pretrain_pre_auc = torchmetrics.AUROC(num_classes=batch_size, task="multiclass")

        self.train_pretrain_post_acc = torchmetrics.Accuracy(num_classes=batch_size, task="multiclass")
        self.train_pretrain_post_auc = torchmetrics.AUROC(num_classes=batch_size, task="multiclass")

        self.val_pretrain_pre_acc = torchmetrics.Accuracy(num_classes=batch_size, task="multiclass")
        self.val_pretrain_pre_auc = torchmetrics.AUROC(num_classes=batch_size, task="multiclass")

        self.val_pretrain_post_acc = torchmetrics.Accuracy(num_classes=batch_size, task="multiclass")
        self.val_pretrain_post_auc = torchmetrics.AUROC(num_classes=batch_size, task="multiclass")

        # pretraining model components
        self.t = nn.Linear(1, 1)
        self.pretrain_criterion = torch.nn.CrossEntropyLoss()

    def pretrain_forward(self, batch):
        assert self.cfg.dataloader_config.modalities == SupervisedView.PRE_AND_POST
        pre_outputs = self.pre_model(batch[SupervisedView.PRE.value]).rep
        post_outputs = self.post_model(batch[SupervisedView.POST.value]).rep

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
        loss_post = self.pretrain_criterion(logits_per_post, labels)
        loss_pre = self.pretrain_criterion(logits_per_pre, labels)
        loss = (loss_pre + loss_post) / 2
        return EBCLPretrainOutput(
            logits_per_pre=logits_per_pre,
            logits_per_post=logits_per_post,
            post_embeds=post_embeds,
            pre_embeds=pre_embeds,
            pre_norm_embeds=pre_norm_embeds,
            post_norm_embeds=post_norm_embeds,
            post_model_output=post_outputs,
            pre_model_output=pre_outputs,
            loss=loss,
        )

    def pretrain_training_step(self, batch):
        output: EBCLPretrainOutput = self.forward(batch)
        # pretrain metrics
        # pre metrics
        labels = torch.arange(self.cfg.dataloader_config.batch_size, device=output.logits_per_pre.device)
        self.train_pretrain_pre_acc.update(output.logits_per_pre, labels)
        self.train_pretrain_pre_auc.update(output.logits_per_pre, labels)

        # post metrics
        self.train_pretrain_post_acc.update(output.logits_per_post, labels)
        self.train_pretrain_post_auc.update(output.logits_per_post, labels)
        return output

    def pretrain_validation_step(self, batch):
        output: EBCLPretrainOutput = self.forward(batch)
        # pretrain metrics
        # pre metrics
        labels = torch.arange(self.cfg.dataloader_config.batch_size, device=output.logits_per_pre.device)
        self.val_pretrain_pre_acc.update(output.logits_per_pre, labels)
        self.val_pretrain_pre_auc.update(output.logits_per_pre, labels)

        # post metrics
        self.val_pretrain_post_acc.update(output.logits_per_post, labels)
        self.val_pretrain_post_auc.update(output.logits_per_post, labels)
        return output

    def pretrain_test_step(self, batch):
        output: EBCLPretrainOutput = self.forward(batch)
        # pretrain metrics
        # pre metrics
        labels = torch.arange(self.cfg.dataloader_config.batch_size, device=output.logits_per_pre.device)
        self.test_pretrain_pre_acc.update(output.logits_per_pre, labels)
        self.test_pretrain_pre_auc.update(output.logits_per_pre, labels)

        # post metrics
        self.test_pretrain_post_acc.update(output.logits_per_post, labels)
        self.test_pretrain_post_auc.update(output.logits_per_post, labels)
        return output

    def pretrain_on_train_epoch_end(self):
        self.log(
            "train_pretrain_pre_acc",
            self.train_pretrain_pre_acc,
            on_epoch=True,
            batch_size=self.cfg.dataloader_config.batch_size,
        )
        self.log(
            "train_pretrain_pre_auc",
            self.train_pretrain_pre_auc,
            on_epoch=True,
            batch_size=self.cfg.dataloader_config.batch_size,
        )

        self.log(
            "train_pretrain_post_acc",
            self.train_pretrain_post_acc,
            on_epoch=True,
            batch_size=self.cfg.dataloader_config.batch_size,
        )
        self.log(
            "train_pretrain_post_auc",
            self.train_pretrain_post_auc,
            on_epoch=True,
            batch_size=self.cfg.dataloader_config.batch_size,
        )

    def pretrain_on_val_epoch_end(self):
        self.log(
            "val_pretrain_pre_acc",
            self.val_pretrain_pre_acc,
            on_epoch=True,
            batch_size=self.cfg.dataloader_config.batch_size,
        )
        self.log(
            "val_pretrain_pre_auc",
            self.val_pretrain_pre_auc,
            on_epoch=True,
            batch_size=self.cfg.dataloader_config.batch_size,
        )

        self.log(
            "val_pretrain_post_acc",
            self.val_pretrain_post_acc,
            on_epoch=True,
            batch_size=self.cfg.dataloader_config.batch_size,
        )
        self.log(
            "val_pretrain_post_auc",
            self.val_pretrain_post_auc,
            on_epoch=True,
            batch_size=self.cfg.dataloader_config.batch_size,
        )

        print(
            "val_pretrain_pre_acc",
            self.val_pretrain_pre_acc.compute(),
            "val_pretrain_pre_auc",
            self.val_pretrain_pre_auc.compute(),
        )
        print(
            "val_pretrain_post_acc",
            self.val_pretrain_post_acc.compute(),
            "val_pretrain_post_auc",
            self.val_pretrain_post_auc.compute(),
        )

    def pretrain_on_test_epoch_end(self):
        self.log(
            "test_pretrain_pre_acc",
            self.test_pretrain_pre_acc,
            on_epoch=True,
            batch_size=self.cfg.dataloader_config.batch_size,
        )
        self.log(
            "test_pretrain_pre_auc",
            self.test_pretrain_pre_auc,
            on_epoch=True,
            batch_size=self.cfg.dataloader_config.batch_size,
        )

        self.log(
            "test_pretrain_post_acc",
            self.test_pretrain_post_acc,
            on_epoch=True,
            batch_size=self.cfg.dataloader_config.batch_size,
        )
        self.log(
            "test_pretrain_post_auc",
            self.test_pretrain_post_auc,
            on_epoch=True,
            batch_size=self.cfg.dataloader_config.batch_size,
        )
        print(
            "test_pretrain_pre_acc",
            self.test_pretrain_pre_acc.compute(),
            "test_pretrain_pre_auc",
            self.test_pretrain_pre_auc.compute(),
        )
        print(
            "test_pretrain_post_acc",
            self.test_pretrain_post_acc.compute(),
            "test_pretrain_post_auc",
            self.test_pretrain_post_auc.compute(),
        )

    def configure_optimizers(self):
        if self.cfg.model_config.half_dtype:
            eps = 1e-4
        else:
            eps = 1e-8
        optimizer = torch.optim.Adam(self.parameters(), lr=self.cfg.model_config.lr, eps=eps)
        return optimizer

    @classmethod
    def initialize_pretrain(cls, cfg: DictConfig):
        return cls(cfg)
