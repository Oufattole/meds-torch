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

    PRECISION_TO_MODEL_WEIGHTS_DTYPE = {
        "32-true": torch.float32,
        "16-true": torch.float16,
        "16-mixed": torch.float32,
        "bf16-true": torch.bfloat16,
        "bf16-mixed": torch.float32,
        "transformer-engine": torch.bfloat16,
    }

    def __init__(self, config: DictConfig) -> None:
        super().__init__()
        self.config = config

        # Enable flash attention
        # if torch.cuda.get_device_capability("cuda")[0] >= 8:
        kwargs = {
            "attn_implementation": "flash_attention_2",
            # "torch_dtype": self.PRECISION_TO_MODEL_WEIGHTS_DTYPE.get("16-mixed"),
        }
        # else:
        #     kwargs = {}

        # Model specs
        model_config: GPTNeoXConfig = AutoConfig.from_pretrained("EleutherAI/gpt-neox-20b", **kwargs)
        for key, val in config.gpt_config_kwargs.items():
            assert hasattr(model_config, key), f"Config for HF model gpt-neox does not have attribute {key}"
            setattr(model_config, key, val)
        model_config.intermediate_size = 4 * model_config.hidden_size
        model_config.max_position_embeddings += self.config.token_bin_size + 1

        self.model = AutoModelForCausalLM.from_config(model_config, **kwargs)


class GPT2Wrapper(torch.nn.Module, Module):
    def __init__(self, cfg: DictConfig):
        super().__init__()
        self.cfg = cfg
        self.model = self.cfg.model
        if self.cfg.get("use_lora", False):
            from peft import LoraConfig, get_peft_model

            # 1) Define a LoRA config. r=8 or 16 is a good starting point.
            lora_config = LoraConfig(
                task_type="CAUSAL_LM",
                inference_mode=False,
                r=8,
                lora_alpha=16,
                target_modules=["query_key_value"],  # GPT-NeoX QKV
                lora_dropout=0.05,
                bias="none",
            )
            self.model.model = get_peft_model(self.model.model, lora_config)

    def forward(self, batch, do_get_last_token=None):
        input_data, mask = batch[INPUT_ENCODER_TOKENS_KEY], batch[INPUT_ENCODER_MASK_KEY]
        gpt2_model: GPTNeoXForCausalLM = self.model.model
        if len(input_data.shape) == 2:
            kwargs = dict(input_ids=input_data)
        elif len(input_data.shape) == 3:
            kwargs = dict(inputs_embeds=input_data)
        else:
            raise ValueError(f"Invalid input_data shape: {input_data.shape}")
        output = gpt2_model(
            **kwargs, attention_mask=mask.float(), return_dict=True, output_hidden_states=True
        )
        last_hidden_state = output.hidden_states[-1]
        logits = output.logits
        if do_get_last_token is None and self.cfg.get_last_token:
            embeddings = get_last_token(last_hidden_state, ~(mask.to(torch.bool)))
        elif do_get_last_token:
            embeddings = get_last_token(last_hidden_state, ~(mask.to(torch.bool)))
        else:
            embeddings = last_hidden_state
        batch[BACKBONE_TOKENS_KEY] = logits
        batch[BACKBONE_EMBEDDINGS_KEY] = embeddings
        return batch
