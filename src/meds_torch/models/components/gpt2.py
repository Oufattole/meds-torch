import torch
from omegaconf import DictConfig
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    GPTNeoXConfig,
    GPTNeoXForCausalLM,
)

from meds_torch.input_encoder import INPUT_ENCODER_MASK_KEY, INPUT_ENCODER_TOKENS_KEY
from meds_torch.models import BACKBONE_EMBEDDINGS_KEY, BACKBONE_TOKENS_KEY
from meds_torch.models.components.utils import get_last_token
from meds_torch.utils.module_class import Module


class GPTLanguageModel(torch.nn.Module, Module):
    """
    GPT with a Language Model head.
    """

    def __init__(self, config: DictConfig) -> None:
        super().__init__()
        self.config = config

        # Enable flash attention
        # if torch.cuda.get_device_capability("cuda")[0] >= 8:
        kwargs = {
            "attn_implementation": "flash_attention_2",
            "torch_dtype": torch.float16,
        }
        # else:
        #     kwargs = {}

        # Model specs
        model_config: GPTNeoXConfig = AutoConfig.from_pretrained("EleutherAI/gpt-neox-20b", **kwargs)
        for key, val in config.gpt_config_kwargs.items():
            assert hasattr(model_config, key), f"Config for HF model gpt-neox does not have attribute {key}"
            setattr(model_config, key, val)
        model_config.intermediate_size = 4 * model_config.hidden_size

        self.model = AutoModelForCausalLM.from_config(model_config, **kwargs)


class GPT2Wrapper(torch.nn.Module, Module):
    def __init__(self, cfg: DictConfig):
        super().__init__()
        self.cfg = cfg
        self.model = self.cfg.model

    def forward(self, batch, do_get_last_token=None):
        input_data, mask = batch[INPUT_ENCODER_TOKENS_KEY], batch[INPUT_ENCODER_MASK_KEY]
        gpt2_model: GPTNeoXForCausalLM = self.model.model
        output = gpt2_model(input_ids=input_data, attention_mask=mask.float(), return_dict=True)
        logits = output.logits
        if do_get_last_token is None and self.cfg.get_last_token:
            embeddings = get_last_token(logits, ~(mask.to(torch.bool)))
        elif do_get_last_token:
            embeddings = get_last_token(logits, ~(mask.to(torch.bool)))
        batch[BACKBONE_TOKENS_KEY] = logits
        batch[BACKBONE_EMBEDDINGS_KEY] = embeddings
        return batch
