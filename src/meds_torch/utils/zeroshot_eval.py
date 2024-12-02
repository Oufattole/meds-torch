import torch
from torch import Tensor
from torchmetrics import Metric
import torch.nn.functional as F
from torchmetrics.functional import auroc

class CLIPZeroShotAccuracy(Metric):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.add_state("correct", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, image_embeddings: Tensor, text_embeddings: Tensor, target: Tensor) -> None:
        """
        Update the metric state with a batch of predictions.

        Args:
            image_embeddings (Tensor): Image embeddings of shape (batch_size, embedding_dim)
            text_embeddings (Tensor): Text embeddings of shape (num_classes, embedding_dim)
            target (Tensor): Ground truth labels of shape (batch_size,)
        """ 
        # Normalize embeddings to compute cosine similarity
        image_embeddings = F.normalize(image_embeddings, dim=-1)
        text_embeddings = F.normalize(text_embeddings, dim=-1)

        # Compute cosine similarity between image embeddings and text embeddings
        similarity = torch.matmul(image_embeddings, text_embeddings.T)  # Shape: (batch_size, num_classes)

        # Predicted class is the one with the highest similarity
        preds = similarity.argmax(dim=-1)

        # Update correct and total
        self.correct += torch.sum(preds == target)
        self.total += target.numel()

    def compute(self) -> Tensor:
        """Compute the final accuracy."""
        return self.correct.float() / self.total


class CLIPZeroShotAUROC(Metric):
    def __init__(self, num_classes: int, **kwargs):
        super().__init__(**kwargs)
        self.num_classes = num_classes
        self.add_state("preds", default=torch.tensor([], dtype=torch.float32), dist_reduce_fx="cat")
        self.add_state("targets", default=torch.tensor([], dtype=torch.int), dist_reduce_fx="cat")

    def update(
        self, 
        meas_norm_embeddings: Tensor, 
        text_norm_embeddings: Tensor, 
        target: Tensor, 
        tau: float = 1.0
    ) -> None:
        """
        Update the metric state with predictions and ground truth labels.

        Args:
            meas_norm_embeddings (Tensor): Normalized measurement embeddings of shape (batch_size, embedding_dim).
            text_norm_embeddings (Tensor): Normalized text embeddings of shape (num_classes, embedding_dim).
            target (Tensor): Ground truth labels of shape (batch_size,).
            tau (float): Temperature scaling factor.
        """
        # Compute similarity scores
        similarity = torch.matmul(meas_norm_embeddings, text_norm_embeddings.T) * tau  # Shape: (batch_size, num_classes)

        # Collect predictions (similarity scores) for each class
        self.preds = torch.cat([self.preds, similarity[:, 0]])  # Binary task: Class scores for positive class
        self.targets = torch.cat([self.targets, (target == 1).int()])  # Binary targets for the positive class

    def compute(self) -> Tensor:
        """
        Compute the AUROC for each class and return the average.

        Returns:
            Tensor: AUROC score.
        """
        return auroc(self.preds, self.targets, task="binary")

		