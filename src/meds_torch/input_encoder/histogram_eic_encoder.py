import torch
from omegaconf import DictConfig
from torch import nn

from meds_torch.input_encoder import INPUT_ENCODER_MASK_KEY, INPUT_ENCODER_TOKENS_KEY
from meds_torch.utils.module_class import Module


def get_dummy_batch_and_cfg(num_samples: int = 3):
    class DummyConfig:
        vocab_size = 7
        token_dim = 3

    cfg = DummyConfig()
    histogram_values = [
        [0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
        [0.0, 1.0, 1.0, 0.0, 1.0, 1.0, 0.0],
        [0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
        [1.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0],
        [1.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
        [0.0, 1.0, 1.0, 0.0, 1.0, 1.0, 0.0],
        [0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
        [0.0, 1.0, 0.0, 1.0, 1.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
    ]
    histogram_tensor = torch.stack([torch.tensor(histogram_values)] * num_samples)
    # torch.zeros((num_samples, 21, 7))

    data_dict = {
        "code": torch.tensor([[4, 5, 1, 2, 4, 5, 3, 0, 4, 5, 1, 2, 4, 5, 1, 3, 4, 5, 2, 3, 4]] * num_samples),
        "mask": torch.ones((num_samples, 21), dtype=torch.int64),
        "histogram": histogram_tensor,
        "subject_id": torch.arange(num_samples),
    }
    return data_dict, cfg


# Set the histogram values for both batches


class EicEncoder(nn.Module, Module):
    """Embeds integer codes and combines them with histogram embeddings.

    Example:
        >>> batch, cfg = get_dummy_batch_and_cfg()
        >>> encoder = EicEncoder(cfg)
        >>> output = encoder(batch)
        >>> output[INPUT_ENCODER_TOKENS_KEY].shape
        torch.Size([3, 21, 3])
        >>> output[INPUT_ENCODER_MASK_KEY].shape
        torch.Size([3, 21])
    """

    def __init__(self, cfg: DictConfig):
        super().__init__()
        self.cfg = cfg
        self.code_embedder = nn.Embedding(cfg.vocab_size, cfg.token_dim)
        self.histogram_embedder = torch.nn.Linear(cfg.vocab_size, cfg.token_dim)
        self.projector = nn.Linear(cfg.token_dim * 2, cfg.token_dim)

    def forward(self, batch):
        batch[INPUT_ENCODER_MASK_KEY] = batch["mask"]
        embedded_codes = self.code_embedder(batch["code"])
        embedded_histograms = self.histogram_embedder(batch["histogram"])
        embeddings = self.projector(torch.cat([embedded_codes, embedded_histograms], dim=-1))
        batch[INPUT_ENCODER_TOKENS_KEY] = embeddings
        return batch
