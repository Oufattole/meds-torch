import torch
from loguru import logger
from omegaconf import DictConfig

try:
    from mamba_ssm.models.config_mamba import MambaConfig
    from mamba_ssm.models.mixer_seq_simple import MambaLMHeadModel
except ImportError:
    logger.warning("Please install mamba-ssm to use this model.")

from meds_torch.input_encoder import INPUT_ENCODER_MASK_KEY, INPUT_ENCODER_TOKENS_KEY
from meds_torch.models import BACKBONE_EMBEDDINGS_KEY, BACKBONE_TOKENS_KEY
from meds_torch.models.components.utils import get_last_token
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
        if cfg.pos_encoding != "absolute_sinusoidal" and cfg.pos_encoding is not None:
            raise ValueError(f"Unknown positional encoding: {cfg.pos_encoding}")
        model.backbone.embedding = torch.nn.Identity()
        model.lm_head = torch.nn.Identity()
        self.model = model
        self.get_last_token = cfg.get_last_token

    def forward(self, batch):
        input_data, mask = batch[INPUT_ENCODER_TOKENS_KEY], batch[INPUT_ENCODER_MASK_KEY]
        output = self.model(input_data.transpose(1, 2))
        batch[BACKBONE_TOKENS_KEY] = output
        batch[BACKBONE_EMBEDDINGS_KEY] = get_last_token(output, mask)
        return batch
