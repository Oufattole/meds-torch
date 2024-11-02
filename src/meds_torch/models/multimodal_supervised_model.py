import dataclasses
import torch
import torch.nn.functional as F
import torchmetrics
from loguru import logger
from omegaconf import DictConfig
from torch import nn

from meds_torch.models.base_model import BaseModule
from meds_torch.models.utils import OutputBase

@dataclasses.dataclass
class SupervisedOutput(OutputBase):
    embeddings: torch.Tensor
    logits: torch.Tensor
    loss: torch.Tensor
    contrastive_loss: torch.Tensor = None  # Add contrastive loss to the output

class ContrastiveLossWithCLIP(nn.Module):
    """Contrastive loss similar to the one used in CLIP, with cross-entropy over cosine similarities."""
    def __init__(self, temperature=0.07):
        super(ContrastiveLossWithCLIP, self).__init__()
        self.temperature = temperature

    def forward(self, text_embeddings, triplet_embeddings):
        # Normalize embeddings
        text_embeddings = F.normalize(text_embeddings, p=2, dim=-1)
        triplet_embeddings = F.normalize(triplet_embeddings, p=2, dim=-1)

        # Compute cosine similarity between all text and triplet embeddings
        logits_per_text = torch.matmul(text_embeddings, triplet_embeddings.T) / self.temperature
        logits_per_triplet = logits_per_text.T

        # Labels for contrastive loss
        batch_size = text_embeddings.shape[0]
        labels = torch.arange(batch_size, device=text_embeddings.device)

        # Cross-entropy loss for both directions
        loss_text_to_triplet = F.cross_entropy(logits_per_text, labels)
        loss_triplet_to_text = F.cross_entropy(logits_per_triplet, labels)

        # Average the two losses
        loss = (loss_text_to_triplet + loss_triplet_to_text) / 2
        return loss

class ContrastiveSupervisedModule(BaseModule):
    """Supervised model with contrastive loss to align embeddings across two modalities."""
    def __init__(self, cfg: DictConfig):
        super().__init__(cfg)
        self.task_name = cfg.task_name
        if self.task_name is None:
            raise ValueError("Task name must be specified")

        # Projection layer for supervised task
        self.projection = nn.Linear(cfg.token_dim, 1)

        # Metrics for supervised task
        self.train_acc = torchmetrics.Accuracy(task="binary")
        self.train_auc = torchmetrics.AUROC(task="binary")
        self.train_apr = torchmetrics.AveragePrecision(task="binary")

        self.val_acc = torchmetrics.Accuracy(task="binary")
        self.val_auc = torchmetrics.AUROC(task="binary")
        self.val_apr = torchmetrics.AveragePrecision(task="binary")

        self.test_acc = torchmetrics.Accuracy(task="binary")
        self.test_auc = torchmetrics.AUROC(task="binary")
        self.test_apr = torchmetrics.AveragePrecision(task="binary")

        # Loss functions
        self.criterion = torch.nn.BCEWithLogitsLoss()
        self.contrastive_loss_fn = ContrastiveLossWithCLIP(temperature=0.07)

    def forward(self, batch) -> SupervisedOutput:
        # Get modality-specific embeddings from the batch
        triplet_embedding = batch["input_encoder_tokens_triplet"]
        text_embedding = batch["input_encoder_tokens_text"]

        # Compute contrastive loss to align the modalities
        contrastive_loss = self.contrastive_loss_fn(text_embedding, triplet_embedding)

        # Combine embeddings for supervised task (e.g., concatenate or sum)
        combined_embedding = triplet_embedding + text_embedding  # Summing for simplicity

        # Apply projection for supervised task
        logits = self.projection(combined_embedding)
        
        if self.cfg.get_representations:
            supervised_loss = None
        else:
            supervised_loss = self.criterion(logits.squeeze(), batch[self.task_name].float())

        # Total loss includes both supervised and contrastive losses
        total_loss = supervised_loss + contrastive_loss if supervised_loss is not None else contrastive_loss

        return SupervisedOutput(
            embeddings=combined_embedding,
            logits=logits,
            loss=total_loss,
            contrastive_loss=contrastive_loss,
        )

    def training_step(self, batch):
        output: SupervisedOutput = self.forward(batch)

        # Log metrics for the supervised classification task
        self.train_acc.update(output.logits.squeeze(), batch[self.task_name].float())
        self.train_auc.update(output.logits.squeeze(), batch[self.task_name].float())
        self.train_apr.update(output.logits.squeeze(), batch[self.task_name].int())
        
        # Log losses
        self.log("train/step_loss", output.loss, on_step=True, batch_size=self.cfg.batch_size)
        self.log("train/contrastive_loss", output.contrastive_loss, on_step=True, batch_size=self.cfg.batch_size)

        assert not torch.isnan(output.loss), "Loss is NaN"
        return output.loss

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
        output: SupervisedOutput = self.forward(batch)

        # Log metrics for validation
        self.val_acc.update(output.logits.squeeze(), batch[self.task_name].float())
        self.val_auc.update(output.logits.squeeze(), batch[self.task_name].float())
        self.val_apr.update(output.logits.squeeze(), batch[self.task_name].int())

        self.log("val/loss", output.loss, on_epoch=True, batch_size=self.cfg.batch_size)
        self.log("val/contrastive_loss", output.contrastive_loss, on_epoch=True, batch_size=self.cfg.batch_size)
        return output.loss

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
        output: SupervisedOutput = self.forward(batch)

        # Log metrics for testing
        self.test_acc.update(output.logits.squeeze(), batch[self.task_name].float())
        self.test_auc.update(output.logits.squeeze(), batch[self.task_name].float())
        self.test_apr.update(output.logits.squeeze(), batch[self.task_name].int())

        self.log("test/loss", output.loss, batch_size=self.cfg.batch_size)
        self.log("test/contrastive_loss", output.contrastive_loss, batch_size=self.cfg.batch_size)
        return output.loss

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