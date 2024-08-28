from collections.abc import Callable

import torch
import torch.nn.functional as F
from omegaconf import DictConfig
from torch import nn
from x_transformers import AutoregressiveWrapper
from x_transformers.autoregressive_wrapper import (
    cast_tuple,
    contrastive_decode_fn,
    eval_decorator,
    exists,
    identity,
    top_k,
)

from meds_torch.input_encoder import INPUT_ENCODER_MASK_KEY, INPUT_ENCODER_TOKENS_KEY
from meds_torch.input_encoder.triplet_encoder import TripletEncoder
from meds_torch.input_encoder.triplet_prompt_encoder import TripletPromptEncoder
from meds_torch.models import BACKBONE_TOKENS_KEY, MODEL_LOSS_KEY
from meds_torch.models.base_model import BaseModule
from meds_torch.models.components import AUTOREGRESSIVE_MODELS

# from meds_torch.model.architectures.mamba import MambaModel


NUMERICAL_VALUE_LOGITS = "MODEL//NUMERICAL_VALUE_LOGITS"
CODE_LOGITS = "MODEL//CODE_LOGITS"
TIME_LOGITS = "MODEL//TIME_LOGITS"

NUMERICAL_VALUE_LOSS = "MODEL//NUMERICAL_VALUE_LOSS"
CODE_LOSS = "MODEL//CODE_LOSS"
TIME_LOSS = "MODEL//TIME_LOSS"


def top_k_logits(logits, k):
    if k == 0:
        return logits
    values, _ = torch.topk(logits, k)
    min_values = values[:, -1]
    return torch.where(logits < min_values, torch.ones_like(logits, dtype=logits.dtype) * -1e10, logits)


def align_right(t, lens, pad_id=0):
    batch, seq_len, _, device = *t.shape, t.device

    assert lens.ndim == 1 and lens.shape[0] == batch
    assert lens.amax() <= seq_len

    pad_lens = seq_len - lens
    max_pad_len = pad_lens.amax()

    batch_arange = torch.arange(batch, device=device, dtype=torch.long)[..., None]
    prompt_len_arange = torch.arange(seq_len, device=device, dtype=torch.long)

    t = F.pad(t, (0, 0, max_pad_len, 0, 0, 0), value=0)
    offset = max_pad_len - pad_lens

    # TODO: you may need to mask the padding out, x_transformers might take care of this double check
    aligned = t[batch_arange, prompt_len_arange + offset[..., None], :]
    return aligned


def select_values_from_logits(logits, target_indices):
    """Selects values from a 3D logits tensor based on indices specified for the last dimension.

    :param logits: A tensor of shape [batch_size, seq_length, num_classes]
    :param target_indices: A tensor of indices with shape [batch_size, seq_length] where each index is valid
        within the range of the last dimension of logits
    :return: A tensor of selected values with shape [batch_size, seq_length]
    """
    batch_size, seq_length, _ = logits.shape

    # Create batch and sequence indices
    batch_indices = torch.arange(batch_size).unsqueeze(1).expand(-1, seq_length).reshape(-1)
    seq_indices = torch.arange(seq_length).repeat(batch_size)

    # Flatten target_indices to match the expanded batch and sequence indices
    flat_target_indices = target_indices.reshape(-1)

    # Use advanced indexing to select the appropriate elements from logits
    selected_values = logits[batch_indices, seq_indices, flat_target_indices].reshape(batch_size, seq_length)

    return selected_values


class TokenForecastingModule(BaseModule):
    """Triplet based GPT Forecasting Model."""

    def __init__(self, cfg: DictConfig):
        super().__init__(cfg)
        if not isinstance(self.model, AUTOREGRESSIVE_MODELS):
            raise ValueError(
                f"Unsupported model type: {type(self.model)}, choose one from {AUTOREGRESSIVE_MODELS}"
            )
        self.setup_heads()

    def setup_heads(self):
        if isinstance(self.input_encoder, TripletEncoder):
            self.numerical_value_head = nn.Linear(
                self.cfg.token_dim,
                self.cfg.vocab_size,
                bias=False,
            )
            self.code_head = nn.Linear(
                self.cfg.token_dim,
                self.cfg.vocab_size,
                bias=False,
            )
            self.time_head = nn.Linear(self.cfg.token_dim, 1, bias=False)
        elif isinstance(self.input_encoder, TripletPromptEncoder):
            self.numerical_value_head = nn.Linear(
                self.cfg.token_dim,
                1,
                bias=False,
            )
            # TODO add vocab size + 2 offset to config
            self.code_head = nn.Linear(
                self.cfg.token_dim,
                self.cfg.vocab_size + 2,
                bias=False,
            )
        else:
            raise NotImplementedError(f"Unsupported input encoder type: {type(self.input_encoder)}")

    def process_numerical_values(self, numerical_value_logits, code_target):
        if isinstance(self.input_encoder, TripletEncoder):
            return select_values_from_logits(numerical_value_logits, code_target)
        elif isinstance(self.input_encoder, TripletPromptEncoder):
            return numerical_value_logits.squeeze(dim=-1)
        else:
            raise NotImplementedError(f"Unsupported input encoder type: {type(self.input_encoder)}")

    def get_time_loss(self, time_logits, time_delta_days_target, dynamic_mask):
        if isinstance(self.input_encoder, TripletEncoder):
            # Time Loss
            time_loss = F.mse_loss(time_logits, time_delta_days_target.unsqueeze(-1), reduction="none")
            time_loss = (time_loss.squeeze(dim=-1) * dynamic_mask).sum() / dynamic_mask.sum()
            # Summing all losses
            return time_loss
        return 0

    def get_loss(
        self,
        batch,
    ):
        code_logits = batch[CODE_LOGITS]
        numerical_value_logits = batch[NUMERICAL_VALUE_LOGITS]
        time_logits = batch[TIME_LOGITS]
        # Code Mask
        dynamic_mask = ~batch["static_mask"]
        code_target = batch["code"]
        # Load data
        numerical_value_target = batch["numeric_value"]
        time_delta_days_target = batch["time_delta_days"]
        numerical_value_mask = batch["numerical_value_mask"]
        # Code Loss
        code_loss = F.cross_entropy(
            code_logits.view(-1, code_logits.size(-1)),
            code_target.view(-1).to(dtype=torch.long),
            reduction="mean",
        )
        # Numerical Value Loss
        numerical_value_preds = self.process_numerical_values(numerical_value_logits, code_target)
        numerical_value_loss = F.mse_loss(numerical_value_preds, numerical_value_target, reduction="none")
        numerical_value_loss = (
            numerical_value_loss * numerical_value_mask
        ).sum() / numerical_value_mask.sum()
        # Time Loss
        time_loss = self.get_time_loss(time_logits, time_delta_days_target, dynamic_mask)

        total_loss = code_loss + numerical_value_loss + time_loss

        batch[MODEL_LOSS_KEY] = total_loss
        batch[NUMERICAL_VALUE_LOSS] = numerical_value_loss
        batch[CODE_LOSS] = code_loss
        batch[TIME_LOSS] = time_loss
        return batch

    def get_forecast_logits(self, model_output):
        if isinstance(model_output, torch.Tensor):
            all_token_embeddings = model_output
        else:
            all_token_embeddings = model_output[BACKBONE_TOKENS_KEY]
        numerical_value_logits = self.numerical_value_head(all_token_embeddings)
        code_logits = self.code_head(all_token_embeddings)
        if isinstance(self.input_encoder, TripletEncoder):
            time_logits = self.time_head(all_token_embeddings)
        else:
            time_logits = None
        return {
            NUMERICAL_VALUE_LOGITS: numerical_value_logits,
            CODE_LOGITS: code_logits,
            TIME_LOGITS: time_logits,
        }

    def forward(self, batch):
        batch = self.input_encoder(batch)
        model_output = self.model(batch)

        forecast = self.get_forecast_logits(model_output)
        batch[NUMERICAL_VALUE_LOGITS] = forecast[NUMERICAL_VALUE_LOGITS]
        batch[CODE_LOGITS] = forecast[CODE_LOGITS]
        batch[TIME_LOGITS] = forecast[TIME_LOGITS]

        batch = self.get_loss(batch)
        return batch

    def _log(self, batch, split):
        self.log(split + "/code_loss", batch[CODE_LOSS])
        self.log(split + "/numerical_value_loss", batch[NUMERICAL_VALUE_LOSS])
        self.log(split + "/time_loss", batch[TIME_LOSS])
        self.log(split + "/loss", batch[MODEL_LOSS_KEY])

    def training_step(self, batch):
        batch = self(batch)
        assert not torch.isnan(batch[MODEL_LOSS_KEY]), "Loss is NaN"
        self._log(batch, "train")
        return batch[MODEL_LOSS_KEY]

    def validation_step(self, batch):
        batch = self(batch)
        assert not torch.isnan(batch[MODEL_LOSS_KEY]), "Loss is NaN"
        self._log(batch, "val")
        return batch[MODEL_LOSS_KEY]

    def test_step(self, batch):
        batch = self(batch)
        assert not torch.isnan(batch[MODEL_LOSS_KEY]), "Loss is NaN"
        self._log(batch, "test")
        return batch[MODEL_LOSS_KEY]

    @torch.no_grad()
    @eval_decorator
    def generate(
        self,
        batch,
        seq_len,
        eos_token=None,
        temperature=1.0,
        prompt_lens: torch.Tensor | None = None,
        filter_logits_fn: Callable = top_k,
        restrict_to_max_seq_len=True,
        amateur_model: torch.nn.Module | tuple[torch.nn.Module] | None = None,
        filter_kwargs: dict = dict(),
        contrastive_decode_kwargs: dict | tuple[dict] = dict(beta=0.5, alpha=0.1),
        cache_kv=True,
        **kwargs,
    ):
        """Modified from https://github.com/lucidrains/x-
        transformers/blob/02b0190aa21ceb7688baa4bd40e6a4a3b9880446/x_transformers/autoregressive_wrapper.py#L1
        32."""
        model = AutoregressiveWrapper(self.model.model)
        batch = self.input_encoder(batch)
        prompts, mask = batch[INPUT_ENCODER_TOKENS_KEY].transpose(1, 2), batch[INPUT_ENCODER_MASK_KEY]
        max_seq_len, greedy = model.max_seq_len, temperature == 0.0

        t = prompts.shape[1]
        prompt_lens = mask.sum(axis=1)

        # handle variable lengthed prompts (prefixes)

        seq_start_pos = None
        if exists(prompt_lens):
            prompts = align_right(prompts, prompt_lens, pad_id=model.pad_value)
            seq_start_pos = t - prompt_lens

        # output from which sampled tokens appended to

        out = prompts

        # kv caches

        cache = None

        # if doing contrastive decoding, turn off filter automatically

        if exists(amateur_model):
            amateur_model = cast_tuple(amateur_model)
            contrastive_decode_kwargs = cast_tuple(contrastive_decode_kwargs)

            assert len(amateur_model) == len(contrastive_decode_kwargs)

            amateur_caches = [None] * len(amateur_model)
            filter_logits_fn = identity

            for i, module in enumerate(amateur_model):
                if isinstance(module, AutoregressiveWrapper):
                    amateur_model[i] = module.net

                module.eval()

        # sampling up to seq_len

        for _ in range(seq_len):
            if restrict_to_max_seq_len:
                max_len_exceeded = out.shape[-1] > max_seq_len

                assert not (
                    cache_kv and max_len_exceeded and not model.net.can_cache_kv_outside_max_seq_len
                ), "the network cannot use cached key values when decoding outside the max sequence length. most likely because you are using absolute positional embedding. you can switch to rotary embeddings to resolve this issue"  # noqa E501

                x = out[:, -max_seq_len:]

                if exists(cache):
                    for inter in cache.attn_intermediates:
                        inter.cached_kv = [t[..., -(max_seq_len - 1) :, :] for t in inter.cached_kv]

            logits, new_cache = model.net(
                x, return_intermediates=True, cache=cache, seq_start_pos=seq_start_pos, **kwargs
            )

            if cache_kv and model.net.can_cache_kv:
                cache = new_cache

            logits = logits[:, -1]

            # handle contrastive decoding, Li et al.
            # https://arxiv.org/abs/2210.15097

            if exists(amateur_model):
                for i, (amateur, amateur_cache, amateur_contrastive_decode_kwargs) in enumerate(
                    zip(amateur_model, amateur_caches, contrastive_decode_kwargs)
                ):
                    amateur_logits, next_amateur_cache = amateur(
                        x,
                        return_intermediates=True,
                        cache=amateur_cache,
                        seq_start_pos=seq_start_pos,
                        **kwargs,
                    )

                    amateur_logits = amateur_logits[:, -1]

                    assert (
                        amateur_logits.shape == logits.shape
                    ), "logits dimension are not the same between amateur and expert model"
                    logits = contrastive_decode_fn(
                        logits, amateur_logits, **amateur_contrastive_decode_kwargs
                    )

                    if cache_kv and amateur.can_cache_kv:
                        amateur_caches[i] = next_amateur_cache

            # filter by top_k, top_p (nucleus), top_a, or custom

            # sample
            forecast = self.get_forecast_logits(logits.unsqueeze(dim=1))

            if greedy:
                code_sample = forecast[CODE_LOGITS].argmax(dim=-1, keepdim=True).squeeze(dim=-1)
            else:
                filtered_logits = filter_logits_fn(forecast[CODE_LOGITS].squeeze(dim=1), **filter_kwargs)
                probs = F.softmax(filtered_logits / temperature, dim=-1)
                code_sample = torch.multinomial(probs, 1)

            numerical_value_sample = self.process_numerical_values(
                forecast[NUMERICAL_VALUE_LOGITS], code_sample
            )
            if isinstance(self.input_encoder, TripletEncoder):
                time_sample = forecast[TIME_LOGITS].squeeze(dim=-1)
            else:
                time_sample = torch.zeros_like(numerical_value_sample) * torch.nan

            # encode sample
            sample_batch = dict()
            sample_batch["static_mask"] = torch.zeros_like(code_sample).bool()
            sample_batch["code"] = code_sample
            sample_batch["numeric_value"] = numerical_value_sample
            sample_batch["time_delta_days"] = time_sample
            sample_batch["numerical_value_mask"] = torch.zeros_like(numerical_value_sample).bool()
            sample = self.input_encoder.get_embedding(sample_batch)

            # concat sample
            out = torch.cat((out, sample.transpose(1, 2)), dim=1)

            if not exists(eos_token):
                continue

            is_eos_tokens = out == eos_token

            if is_eos_tokens.any(dim=-1).all():
                break

        if exists(eos_token):
            # mask out everything after the eos tokens
            shifted_is_eos_tokens = F.pad(is_eos_tokens, (1, -1))
            mask = shifted_is_eos_tokens.float().cumsum(dim=-1) >= 1
            out = out.masked_fill(mask, model.pad_value)

        out = out[:, t:]

        return out
