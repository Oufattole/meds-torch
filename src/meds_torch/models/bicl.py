import os

import torch
import torchmetrics
from omegaconf import DictConfig
from torch import nn

from transformers import AutoTokenizer

from meds_torch.models import (
    BACKBONE_EMBEDDINGS_KEY,
    MODEL_EMBEDDINGS_KEY,
    MODEL_LOGITS_KEY,
    MODEL_LOSS_KEY,
    MODEL_TOKENS_KEY,
)
from meds_torch.models.base_model import BaseModule
from meds_torch.models.components.text_encoder import TextEncoderModel
from meds_torch.utils.zeroshot_eval import CLIPZeroShotAUROC


class BICLModule(BaseModule):
    def __init__(self, cfg: DictConfig):
        super().__init__(cfg)
        batch_size = cfg.batch_size
        self.optimizer = cfg.optimizer
        self.scheduler = cfg.scheduler
        self.meas_model = self.model
        self.text_model = cfg.text_encoder  
        self.task_names = cfg.task_names
        self.zeroshot_templates = list(cfg.zeroshot_templates)
        os.environ["TOKENIZERS_PARALLELISM"] = "false"
        self.tokenizer = AutoTokenizer.from_pretrained(cfg.tokenizer)
        #  metrics
        self.train_meas_acc = torchmetrics.Accuracy(num_classes=batch_size, task="multiclass")
        self.train_meas_auc = torchmetrics.AUROC(num_classes=batch_size, task="multiclass")

        self.train_text_acc = torchmetrics.Accuracy(num_classes=batch_size, task="multiclass")
        self.train_text_auc = torchmetrics.AUROC(num_classes=batch_size, task="multiclass")

        self.val_zero_in_hospital_auc = CLIPZeroShotAUROC(num_classes=1)
        self.val_zero_in_icu_auc = CLIPZeroShotAUROC(num_classes=1)

        self.val_meas_acc = torchmetrics.Accuracy(num_classes=batch_size, task="multiclass")
        self.val_meas_auc = torchmetrics.AUROC(num_classes=batch_size, task="multiclass")

        self.val_text_acc = torchmetrics.Accuracy(num_classes=batch_size, task="multiclass")
        self.val_text_auc = torchmetrics.AUROC(num_classes=batch_size, task="multiclass")

        self.test_zero_in_hospital_auc = CLIPZeroShotAUROC(num_classes=1)
        self.test_zero_in_icu_auc = CLIPZeroShotAUROC(num_classes=1)

        self.test_meas_acc = torchmetrics.Accuracy(num_classes=batch_size, task="multiclass")
        self.test_meas_auc = torchmetrics.AUROC(num_classes=batch_size, task="multiclass")

        self.test_text_acc = torchmetrics.Accuracy(num_classes=batch_size, task="multiclass")
        self.test_text_auc = torchmetrics.AUROC(num_classes=batch_size, task="multiclass")

        # Model components
        self.meas_projection = nn.Linear(cfg.token_dim, cfg.token_dim)
        self.text_projection = nn.Linear(cfg.token_dim, cfg.token_dim)

        self.t = nn.Parameter(torch.ones(1).reshape(-1, 1) * cfg.tau)
        self.criterion = torch.nn.CrossEntropyLoss()
        self.zeroshot_criterion = torch.nn.BCEWithLogitsLoss()

        self.update_label_embeddings()

    def forward(self, batch):
        # pre_batch = batch[self.cfg.meas_window_name]
        meas_batch = batch
        meas_batch = self.input_encoder(meas_batch)
        meas_batch = self.meas_model(meas_batch)
        meas_outputs = meas_batch[BACKBONE_EMBEDDINGS_KEY]

        text_batch = batch["notes"]
        text_outputs = self.text_model(text_batch)

        meas_embeds = self.meas_projection(meas_outputs)
        text_embeds = self.text_projection(text_outputs)

        meas_norm_embeds = meas_embeds
        text_norm_embeds = text_embeds

        meas_norm_embeds = meas_embeds / meas_embeds.norm(dim=-1, keepdim=True)
        text_norm_embeds = text_embeds / text_embeds.norm(dim=-1, keepdim=True)

        logits = torch.mm(text_norm_embeds, meas_norm_embeds.T) * torch.exp(self.t)
        labels = torch.arange(meas_norm_embeds.shape[0], device=meas_norm_embeds.device)
        logits_per_text = logits
        logits_per_meas = logits.T
        loss_text = self.criterion(logits_per_text, labels)
        loss_meas = self.criterion(logits_per_meas, labels)
        loss = (loss_meas + loss_text) / 2
        output = dict(
            meas=meas_batch,
            text=text_batch,
        )
        output[MODEL_EMBEDDINGS_KEY] = torch.concat([meas_norm_embeds, text_norm_embeds], dim=0)
        output[MODEL_TOKENS_KEY] = None
        output[MODEL_LOSS_KEY] = loss
        output[MODEL_LOGITS_KEY] = logits
        return output

    def update_label_embeddings(self):
        tokenized_templates = self.tokenizer(
            self.zeroshot_templates,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=512,
        ).to(self.t.device)
        self.zeroshot_label_emb = self.text_model(tokenized_templates)
        self.zeroshot_label_emb /= self.zeroshot_label_emb.norm(dim=-1, keepdim=True)


    def training_step(self, batch):
        output = self.forward(batch)
        # pretrain metrics
        # measurement metrics
        labels = torch.arange(output[MODEL_LOGITS_KEY].shape[0], device=output[MODEL_LOGITS_KEY].device)
        self.train_meas_acc.update(torch.diag(output[MODEL_LOGITS_KEY]), labels)
        self.train_meas_auc.update(output[MODEL_LOGITS_KEY], labels)

        # text metrics
        self.train_text_acc.update(torch.diag(output[MODEL_LOGITS_KEY]), labels)
        self.train_text_auc.update(output[MODEL_LOGITS_KEY], labels)

        self.log("train/loss", output[MODEL_LOSS_KEY], batch_size=self.cfg.batch_size)
        return output[MODEL_LOSS_KEY]


    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        if dataloader_idx > 0:
            zeroshot_idx = dataloader_idx - 1
            if batch_idx == 0:
                self.update_label_embeddings()
            batch = self.input_encoder(batch)
            batch = self.meas_model(batch)
            meas_embeds = batch[BACKBONE_EMBEDDINGS_KEY]

            meas_norm_embeds = meas_embeds / meas_embeds.norm(dim=-1, keepdim=True)

            self.val_zero_in_hospital_auc.update(
                meas_norm_embeds,
                self.zeroshot_label_emb,
                batch[self.task_names[zeroshot_idx]],
                tau=self.t
            )
            logits = torch.matmul(meas_norm_embeds, self.zeroshot_label_emb.T) * torch.exp(self.t) # Shape: (batch_size, num_classes)
            loss = self.zeroshot_criterion(logits[:,0], batch[self.task_names[zeroshot_idx]])
            self.log(
                "val/zero/in/hospital/auc",
                self.val_zero_in_hospital_auc,
                on_epoch=True,
                batch_size=self.cfg.batch_size,
                metric_attribute="val_zero_in_hospital_auc",
            )
            return loss
        else:
            output = self.forward(batch)
            # pretrain metrics
            # measurement metrics
            labels = torch.arange(output[MODEL_LOGITS_KEY].shape[0], device=output[MODEL_LOGITS_KEY].device)
            self.val_meas_acc.update(torch.diag(output[MODEL_LOGITS_KEY]), labels)
            if output[MODEL_LOGITS_KEY].shape[0] == self.cfg.batch_size:
                self.val_meas_auc.update(output[MODEL_LOGITS_KEY], labels)

            # text metrics
            self.val_text_acc.update(torch.diag(output[MODEL_LOGITS_KEY]), labels)
            
            if output[MODEL_LOGITS_KEY].shape[0] == self.cfg.batch_size:
                self.val_text_auc.update(output[MODEL_LOGITS_KEY], labels)
            self.log("val/loss", output[MODEL_LOSS_KEY], batch_size=self.cfg.batch_size)
            return output[MODEL_LOSS_KEY]

    def test_step(self, batch, batch_idx, dataloader_idx=0):
        if dataloader_idx > 0:
            zeroshot_idx = dataloader_idx - 1
            self.update_label_embeddings()
            batch = self.input_encoder(batch)
            batch = self.meas_model(batch)
            meas_embeds = batch[BACKBONE_EMBEDDINGS_KEY]

            meas_norm_embeds = meas_embeds / meas_embeds.norm(dim=-1, keepdim=True)

            self.test_zero_in_hospital_auc.update(
                meas_norm_embeds,
                self.zeroshot_label_emb,
                batch[self.task_names[zeroshot_idx]],
                tau=self.t
            )
            logits = torch.matmul(meas_norm_embeds, self.zeroshot_label_emb.T) * torch.exp(self.t) # Shape: (batch_size, num_classes)
            loss = self.zeroshot_criterion(logits[:,0], batch[self.task_names[zeroshot_idx]])
            self.log(
                "test/zero/in/hospital/auc",
                self.test_zero_in_hospital_auc,
                on_epoch=True,
                batch_size=self.cfg.batch_size,
                metric_attribute="test_zero_in_hospital_auc",
            )
            return loss
        else:
            output = self.forward(batch)
            # pretrain metrics
            # measurement metrics
            labels = torch.arange(output[MODEL_LOGITS_KEY].shape[0], device=output[MODEL_LOGITS_KEY].device)
            self.test_meas_acc.update(torch.diag(output[MODEL_LOGITS_KEY]), labels)
            if output[MODEL_LOGITS_KEY].shape[0] == self.cfg.batch_size:
                self.test_meas_auc.update(output[MODEL_LOGITS_KEY], labels)

            # text metrics
            self.test_text_acc.update(torch.diag(output[MODEL_LOGITS_KEY]), labels)
            if output[MODEL_LOGITS_KEY].shape[0] == self.cfg.batch_size:
                self.test_text_auc.update(output[MODEL_LOGITS_KEY], labels)
            self.log("test/loss", output[MODEL_LOSS_KEY], batch_size=self.cfg.batch_size)
            return output[MODEL_LOSS_KEY]

    def on_train_epoch_end(self):
        self.log(
            "train/meas/acc",
            self.train_meas_acc,
            on_epoch=True,
            batch_size=self.cfg.batch_size,
        )
        self.log(
            "train/meas/auc",
            self.train_meas_auc,
            on_epoch=True,
            batch_size=self.cfg.batch_size,
        )

        self.log(
            "train/text/acc",
            self.train_text_acc,
            on_epoch=True,
            batch_size=self.cfg.batch_size,
        )
        self.log(
            "train/text/auc",
            self.train_text_auc,
            on_epoch=True,
            batch_size=self.cfg.batch_size,
        )

    def on_val_epoch_end(self):
        self.log(
            "val/meas/acc",
            self.val_meas_acc,
            on_epoch=True,
            batch_size=self.cfg.batch_size,
        )
        self.log(
            "val/meas/auc",
            self.val_meas_auc,
            on_epoch=True,
            batch_size=self.cfg.batch_size,
        )

        self.log(
            "val/text/acc",
            self.val_text_acc,
            on_epoch=True,
            batch_size=self.cfg.batch_size,
        )
        self.log(
            "val/text/auc",
            self.val_text_auc,
            on_epoch=True,
            batch_size=self.cfg.batch_size,
        )
        self.log(
            "val/zero/in/hospital/auc",
            self.val_zeroshot,
            on_epoch=True,
            batch_size=self.cfg.batch_size,
            metric_attribute="val_zero_in_hospital_auc",
        )
        self.log(
            "val/zero/in/icu/auc",
            self.val_zeroshot,
            on_epoch=True,
            batch_size=self.cfg.batch_size,
            metric_attribute="val_zero_in_icu_auc",
        )
        print(
            "val/meas/acc",
            self.val_meas_acc.compute(),
            "val/meas/auc",
            self.val_meas_auc.compute(),
        )
        print(
            "val/text/acc",
            self.val_text_acc.compute(),
            "val/text/auc",
            self.val_text_auc.compute(),
        )
        print(
            "val/zero/in/hospital/auc",
            self.val_zero_in_hospital_auc.compute(),
            "val/zero/in/icu/auc",
            self.val_zero_in_icu_auc.compute(),
        )

    def on_test_epoch_end(self):
        self.log(
            "test/meas/acc",
            self.test_meas_acc,
            on_epoch=True,
            batch_size=self.cfg.batch_size,
        )
        self.log(
            "test/meas/auc",
            self.test_meas_auc,
            on_epoch=True,
            batch_size=self.cfg.batch_size,
        )

        self.log(
            "test/text/acc",
            self.test_text_acc,
            on_epoch=True,
            batch_size=self.cfg.batch_size,
        )
        self.log(
            "test/text/auc",
            self.test_text_auc,
            on_epoch=True,
            batch_size=self.cfg.batch_size,
        )
        self.log(
            "test/zero/in/hospital/auc",
            self.test_zero_in_hospital_auc,
            on_epoch=True,
            batch_size=self.cfg.batch_size,
            metric_attribute="test_zero_in_hospital_auc",
        )
        self.log(
            "test/zero/in/icu/auc",
            self.test_zero_in_icu_auc,
            on_epoch=True,
            batch_size=self.cfg.batch_size,
            metric_attribute="test_zero_in_icu_auc",
        )
        print(
            "test/meas/acc",
            self.test_meas_acc.compute(),
            "test/meas/auc",
            self.test_meas_auc.compute(),
        )
        print(
            "test/text/acc",
            self.test_text_acc.compute(),
            "test/text/auc",
            self.test_text_auc.compute(),
        )
        print(
            "test/zero/in/hospital/auc",
            self.test_zero_in_hospital_auc.compute(),
            "test/zero/in/icu/auc",
            self.test_zero_in_icu_auc.compute(),
        )

    def configure_optimizers(self):
        optimizer = self.optimizer(self.parameters())
        scheduler = self.scheduler(optimizer)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step"  # or "epoch"
            }
        }

