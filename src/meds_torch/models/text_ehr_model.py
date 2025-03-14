import dataclasses
import torch
import torch.nn.functional as F
import torchmetrics
from loguru import logger
from omegaconf import DictConfig
from torch import nn

from meds_torch.models.base_model import BaseModule
from meds_torch.models.utils import OutputBase
from meds_torch.models import BACKBONE_EMBEDDINGS_KEY

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
        self.contrastive_loss_fn = ContrastiveLossWithCLIP(temperature=cfg.get("temperature", 0.07))

    def forward(self, batch) -> SupervisedOutput:
        """
        Process the batch through input_encoder and backbone, then apply contrastive and supervised losses.
        """
        # Process through input encoder
        batch = self.input_encoder(batch)
        
        # Process through backbone model
        batch = self.model(batch)
        
        # Get embeddings from backbone
        combined_embedding = batch[BACKBONE_EMBEDDINGS_KEY]
        
        # Get modality-specific embeddings from input encoder
        if "input_encoder_tokens_text" not in batch or "input_encoder_tokens_triplet" not in batch:
            raise ValueError("Batch missing required modality inputs. "
                            "Ensure that both 'input_encoder_tokens_text' and 'input_encoder_tokens_triplet' are provided.")

        text_embedding = batch["input_encoder_tokens_text"]
        triplet_embedding = batch["input_encoder_tokens_triplet"]

        # Compute contrastive loss to align the text and triplet modalities
        contrastive_loss = self.contrastive_loss_fn(text_embedding, triplet_embedding)

        # Supervised projection to task space
        logits = self.projection(combined_embedding)
        
        if self.cfg.get_representations:
            supervised_loss = None
        else:
            # Assume that batch contains the supervision target under self.task_name key
            supervised_loss = self.criterion(logits.squeeze(), batch[self.task_name].float())

        # Combine supervised and contrastive losses
        if supervised_loss is not None:
            # Weight between supervised and contrastive loss can be configured
            contrastive_weight = self.cfg.get("contrastive_weight", 0.5)
            total_loss = (1 - contrastive_weight) * supervised_loss + contrastive_weight * contrastive_loss
        else:
            total_loss = contrastive_loss

        return SupervisedOutput(
            embeddings=combined_embedding,
            logits=logits,
            loss=total_loss,
            contrastive_loss=contrastive_loss
        )

    def training_step(self, batch, batch_idx):
        """Training step for the model."""
        outputs = self.forward(batch)
        
        # Log losses
        self.log("train/loss", outputs.loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log("train/contrastive_loss", outputs.contrastive_loss, on_step=True, on_epoch=True)
        
        # Update and log metrics if not in representation mode
        if not self.cfg.get_representations:
            preds = torch.sigmoid(outputs.logits.squeeze())
            targets = batch[self.task_name].float()
            
            self.train_acc(preds, targets)
            self.train_auc(preds, targets)
            self.train_apr(preds, targets)
            
            self.log("train/accuracy", self.train_acc, on_step=True, on_epoch=True)
            self.log("train/auc", self.train_auc, on_step=True, on_epoch=True)
            self.log("train/average_precision", self.train_apr, on_step=True, on_epoch=True)
        
        return outputs.loss

    def validation_step(self, batch, batch_idx):
        """Validation step for the model."""
        outputs = self.forward(batch)
        
        # Log losses
        self.log("val/loss", outputs.loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val/contrastive_loss", outputs.contrastive_loss, on_step=False, on_epoch=True)
        
        # Update and log metrics if not in representation mode
        if not self.cfg.get_representations:
            preds = torch.sigmoid(outputs.logits.squeeze())
            targets = batch[self.task_name].float()
            
            self.val_acc(preds, targets)
            self.val_auc(preds, targets)
            self.val_apr(preds, targets)
            
            self.log("val/accuracy", self.val_acc, on_step=False, on_epoch=True)
            self.log("val/auc", self.val_auc, on_step=False, on_epoch=True)
            self.log("val/average_precision", self.val_apr, on_step=False, on_epoch=True)
        
        return outputs.loss

    def test_step(self, batch, batch_idx):
        """Test step for the model."""
        outputs = self.forward(batch)
        
        # Log losses
        self.log("test/loss", outputs.loss, on_step=False, on_epoch=True)
        self.log("test/contrastive_loss", outputs.contrastive_loss, on_step=False, on_epoch=True)
        
        # Update and log metrics if not in representation mode
        if not self.cfg.get_representations:
            preds = torch.sigmoid(outputs.logits.squeeze())
            targets = batch[self.task_name].float()
            
            self.test_acc(preds, targets)
            self.test_auc(preds, targets)
            self.test_apr(preds, targets)
            
            self.log("test/accuracy", self.test_acc, on_step=False, on_epoch=True)
            self.log("test/auc", self.test_auc, on_step=False, on_epoch=True)
            self.log("test/average_precision", self.test_apr, on_step=False, on_epoch=True)
        
        return outputs.loss

    def predict_step(self, batch, batch_idx):
        """Prediction step for the model."""
        outputs = self.forward(batch)
        
        # Return predictions and embeddings
        predictions = {
            'embeddings': outputs.embeddings,
            'logits': outputs.logits,
        }
        
        if not self.cfg.get_representations:
            predictions['probabilities'] = torch.sigmoid(outputs.logits)
            
        return predictions