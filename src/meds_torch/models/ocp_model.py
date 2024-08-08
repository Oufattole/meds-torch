import torch
import torch.nn.functional as F
import torchmetrics
from omegaconf import DictConfig
from torch import nn

from meds_torch.input_encoder import INPUT_ENCODER_MASK_KEY, INPUT_ENCODER_TOKENS_KEY
from meds_torch.models import (
    BACKBONE_EMBEDDINGS_KEY,
    MODEL_EMBEDDINGS_KEY,
    MODEL_LOGITS_KEY,
    MODEL_LOSS_KEY,
    MODEL_TOKENS_KEY,
)
from meds_torch.models.base_model import BaseModule


class OCPModule(BaseModule):
    def __init__(self, cfg: DictConfig):
        if cfg.early_fusion:
            # double the sequence length for early fusion as we concatenate the pre and post tokens
            cfg = cfg.copy()
            cfg.max_seq_len = cfg.max_seq_len * 2
        super().__init__(cfg)

        # pretrain metrics
        self.train_acc = torchmetrics.Accuracy(num_classes=cfg.batch_size, task="binary")
        self.train_auc = torchmetrics.AUROC(num_classes=cfg.batch_size, task="binary")

        self.val_acc = torchmetrics.Accuracy(num_classes=cfg.batch_size, task="binary")
        self.val_auc = torchmetrics.AUROC(num_classes=cfg.batch_size, task="binary")

        self.test_acc = torchmetrics.Accuracy(num_classes=cfg.batch_size, task="binary")
        self.test_auc = torchmetrics.AUROC(num_classes=cfg.batch_size, task="binary")

        # pretraining model components
        if cfg.early_fusion:
            self.projection = nn.Linear(cfg.token_dim, 1)
        else:
            self.projection = nn.Linear(cfg.token_dim * 2, 1)
        self.criterion = torch.nn.BCEWithLogitsLoss()

    @classmethod
    def early_fusion_pad(cls, pre, post):
        """Determines the maximum sequence length and pad the sequences to it."""
        max_length = max(pre.size(-1), post.size(-1))
        pre_padded = F.pad(pre, (0, max_length - pre.size(-1)))
        post_padded = F.pad(post, (0, max_length - post.size(-1)))
        return pre_padded, post_padded

    @classmethod
    def shuffled_concat(cls, pre, post, random_flips):
        """Shuffles the pre and post sequences and concatenates them."""
        shuffled_pre_data = torch.where(random_flips, post, pre)
        shuffled_post_data = torch.where(random_flips, pre, post)
        shuffled_data = torch.concat([shuffled_pre_data, shuffled_post_data], dim=-1)
        return shuffled_data

    def forward(self, batch):
        if self.cfg.early_fusion:
            pre_batch = batch[self.cfg.pre_window_name]
            pre_batch = self.input_encoder(pre_batch)
            post_batch = batch[self.cfg.post_window_name]
            post_batch = self.input_encoder(post_batch)

            pre_padded_mask, post_padded_mask = self.early_fusion_pad(
                pre_batch[INPUT_ENCODER_MASK_KEY], post_batch[INPUT_ENCODER_MASK_KEY]
            )
            random_flips = torch.randint(
                0, 2, (pre_batch[INPUT_ENCODER_MASK_KEY].shape[0], 1), device=pre_padded_mask.device
            ).bool()
            fusion_mask = self.shuffled_concat(pre_padded_mask, post_padded_mask, random_flips)

            pre_padded_tokens, post_padded_tokens = self.early_fusion_pad(
                pre_batch[INPUT_ENCODER_TOKENS_KEY], post_batch[INPUT_ENCODER_TOKENS_KEY]
            )
            fusion_tokens = self.shuffled_concat(
                pre_padded_tokens, post_padded_tokens, random_flips.unsqueeze(-1)
            )

            # Repeat the same procedure for the token embeddings
            fused_batch = {INPUT_ENCODER_MASK_KEY: fusion_mask, INPUT_ENCODER_TOKENS_KEY: fusion_tokens}
            batch = self.model(fused_batch)
            output = batch
            classifier_inputs = batch[BACKBONE_EMBEDDINGS_KEY]
        else:
            pre_batch = batch[self.cfg.pre_window_name]
            pre_batch = self.input_encoder(pre_batch)
            pre_batch = self.model(pre_batch)
            pre_outputs = pre_batch[BACKBONE_EMBEDDINGS_KEY]

            post_batch = batch[self.cfg.post_window_name]
            post_batch = self.input_encoder(post_batch)
            post_batch = self.model(post_batch)
            post_outputs = post_batch[BACKBONE_EMBEDDINGS_KEY]

            random_flips = torch.randint(0, 2, (pre_outputs.shape[0], 1), device=pre_outputs.device).bool()
            shuffled_pre_outputs = torch.where(random_flips, post_outputs, pre_outputs)
            shuffled_post_outputs = torch.where(random_flips, pre_outputs, post_outputs)
            classifier_inputs = torch.concat([shuffled_pre_outputs, shuffled_post_outputs], dim=1)
            output = dict(
                pre=pre_batch,
                post=post_batch,
            )
        logits = self.projection(classifier_inputs)
        loss = self.criterion(logits, random_flips.float())

        output[MODEL_EMBEDDINGS_KEY] = classifier_inputs
        output[MODEL_TOKENS_KEY] = None
        output[MODEL_LOSS_KEY] = loss
        output[MODEL_LOGITS_KEY] = logits
        output["MODEL//LABELS"] = random_flips

        return output

    def training_step(self, batch):
        output = self.forward(batch)
        # pretrain metrics
        # pre metrics
        labels = output["MODEL//LABELS"].float()
        self.train_acc.update(output[MODEL_LOGITS_KEY], labels)
        self.train_auc.update(output[MODEL_LOGITS_KEY], labels)
        self.log("train/loss", output[MODEL_LOSS_KEY], batch_size=self.cfg.batch_size)
        return output[MODEL_LOSS_KEY]

    def validation_step(self, batch):
        output = self.forward(batch)
        # pretrain metrics
        labels = output["MODEL//LABELS"].float()
        self.val_acc.update(output[MODEL_LOGITS_KEY], labels)
        self.val_auc.update(output[MODEL_LOGITS_KEY], labels)
        self.log("val/loss", output[MODEL_LOSS_KEY], batch_size=self.cfg.batch_size)
        return output[MODEL_LOSS_KEY]

    def test_step(self, batch):
        output = self.forward(batch)
        # pretrain metrics
        labels = output["MODEL//LABELS"].float()
        self.test_acc.update(output[MODEL_LOGITS_KEY], labels)
        self.test_auc.update(output[MODEL_LOGITS_KEY], labels)
        self.log("test/loss", output[MODEL_LOSS_KEY], batch_size=self.cfg.batch_size)
        return output[MODEL_LOSS_KEY]

    def on_train_epoch_end(self):
        self.log(
            "train/acc",
            self.train_acc,
            on_epoch=True,
            batch_size=self.cfg.batch_size,
        )
        self.log(
            "train/auc",
            self.train_auc,
            on_epoch=True,
            batch_size=self.cfg.batch_size,
        )

    def on_val_epoch_end(self):
        self.log(
            "val/acc",
            self.val_acc,
            on_epoch=True,
            batch_size=self.cfg.batch_size,
        )
        self.log(
            "val/auc",
            self.val_auc,
            on_epoch=True,
            batch_size=self.cfg.batch_size,
        )
        print(
            "val/acc",
            self.val_acc.compute(),
            "val/auc",
            self.val_auc.compute(),
        )

    def on_test_epoch_end(self):
        self.log(
            "test/acc",
            self.test_acc,
            on_epoch=True,
            batch_size=self.cfg.batch_size,
        )
        self.log(
            "test/auc",
            self.test_auc,
            on_epoch=True,
            batch_size=self.cfg.batch_size,
        )
        print(
            "test/acc",
            self.test_acc.compute(),
            "test/auc",
            self.test_auc.compute(),
        )
