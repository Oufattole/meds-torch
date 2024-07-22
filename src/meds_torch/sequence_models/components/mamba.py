import torch
from loguru import logger
from omegaconf import DictConfig

try:
    from mamba_ssm.models.config_mamba import MambaConfig
    from mamba_ssm.models.mixer_seq_simple import MambaLMHeadModel
except ImportError:
    logger.warning("Please install mamba-ssm to use this model.")

from meds_torch.sequence_models.components.utils import get_last_token
from meds_torch.utils.module_class import Module


class MambaModel(torch.nn.Module, Module):
    """Wrapper of MambaLMHeadModel for use in MEDS with triplet token embeddings."""

    def __init__(self, cfg: DictConfig):
        super().__init__()
        self.cfg = cfg
        model = MambaLMHeadModel(
            config=MambaConfig(
                d_model=cfg.token_dim,
                vocab_size=2,
                n_layer=cfg.n_layers,
            )
        )
        model.backbone.embedding = torch.nn.Identity()
        model.lm_head = torch.nn.Identity()
        self.model = model
        self.get_last_token = cfg.get_last_token

    def forward(self, batch, mask):
        output = self.model(batch.transpose(1, 2))
        if self.get_last_token:
            return get_last_token(output.logits, mask)
        else:
            return output
