from omegaconf import DictConfig
from torch import nn

from meds_torch.input_encoder import INPUT_ENCODER_MASK_KEY, INPUT_ENCODER_TOKENS_KEY
from meds_torch.utils.module_class import Module


class EicEncoder(nn.Module, Module):
    """Placeholder encoder, does nothing as the x-transformer model handles the embedding of tokens."""

    def __init__(self, cfg: DictConfig):
        super().__init__()
        self.cfg = cfg

    def forward(self, batch):
        batch[INPUT_ENCODER_MASK_KEY] = batch["mask"]
        batch[INPUT_ENCODER_TOKENS_KEY] = batch["code"]
        return batch
