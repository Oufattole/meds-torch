import torch
from loguru import logger
from omegaconf import DictConfig

try:
    from mamba_ssm.models.config_mamba import MambaConfig
    from mamba_ssm.models.mixer_seq_simple import MambaLMHeadModel
except ImportError:
    logger.warning("Please install mamba-ssm to use this model.")

from meds_torch.model.architectures.utils import get_last_token


class MambaModel(torch.nn.Module):
    """Wrapper of MambaLMHeadModel for use in MEDS with triplet token embeddings."""

    def __init__(self, cfg: DictConfig):
        super().__init__()
        self.cfg = cfg
        model = MambaLMHeadModel(
            config=MambaConfig(
                d_model=cfg.model.embedder.token_dim,
                vocab_size=2,
                n_layer=cfg.model.params.n_layers,
            )
        )
        model.backbone.embedding = torch.nn.Identity()
        model.lm_head = torch.nn.Identity()
        self.model = model
        self.get_last_token = cfg.model.get_last_token

    def forward(self, batch, mask):
        output = self.model(batch.transpose(1, 2))
        if self.get_last_token:
            return get_last_token(output.logits, mask)
        else:
            return output
