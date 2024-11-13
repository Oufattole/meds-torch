from collections.abc import Callable
from functools import wraps

import torch
from loguru import logger
from omegaconf import DictConfig
from torch.functional import F
from x_transformers import TransformerWrapper
from x_transformers.autoregressive_wrapper import (
    FILTER_LOGITS_FN,
    align_right,
    exists,
    identity,
    join,
    pack,
    unpack,
)

from meds_torch.input_encoder import INPUT_ENCODER_MASK_KEY, INPUT_ENCODER_TOKENS_KEY
from meds_torch.models import BACKBONE_EMBEDDINGS_KEY, BACKBONE_TOKENS_KEY
from meds_torch.models.components.utils import get_last_token
from meds_torch.utils.module_class import Module


def eval_decorator(fn):
    @wraps(fn)
    def inner(model, *args, **kwargs):
        was_training = model.training
        model.eval()
        out = fn(model, *args, **kwargs)
        model.train(was_training)
        return out

    return inner


class GenerationBudget:
    """A class to handle generation budgets with mutually exclusive sequence length or time length.

    This class ensures that only one of max_seq_len or min_time_len can be set.

    Examples:
        >>> # Create from sequence length
        >>> budget_seq = GenerationBudget.from_seq_len(100)
        >>> budget_seq.max_seq_len
        100
        >>> budget_seq.min_time_len is None
        True

        >>> # Create from time length
        >>> budget_time = GenerationBudget.from_time_len(60)
        >>> budget_time.min_time_len
        60
        >>> budget_time.max_seq_len is None
        True

        >>> # Values remain consistent
        >>> budget_seq = GenerationBudget.from_seq_len(500)
        >>> (budget_seq.max_seq_len, budget_seq.min_time_len)
        (500, None)

        >>> # Direct initialization isn't intended for use
        >>> budget = GenerationBudget(100)
    """

    def __init__(self, value: int):
        self._value = value

    @classmethod
    def from_seq_len(cls, value: int) -> "GenerationBudget":
        """Create a GenerationBudget from a maximum sequence length.

        Args:
            value: The maximum sequence length

        Returns:
            A GenerationBudget instance with max_seq_len set
        """
        instance = cls(value)
        instance._is_seq_len = True
        return instance

    @classmethod
    def from_time_len(cls, value: int) -> "GenerationBudget":
        """Create a GenerationBudget from a minimum time length.

        Args:
            value: The minimum time length

        Returns:
            A GenerationBudget instance with min_time_len set
        """
        instance = cls(value)
        return instance

    @property
    def max_seq_len(self) -> int | None:
        """The maximum sequence length, if this budget was created with from_seq_len."""
        return self._value if hasattr(self, "_is_seq_len") else None

    @property
    def min_time_len(self) -> int | None:
        """The minimum time length, if this budget was created with from_time_len."""
        return self._value if not hasattr(self, "_is_seq_len") else None


def update_generation_budget(
    cumulative_time: torch.Tensor,
    current_sample: torch.Tensor,
    num_generated_tokens: int,
    get_next_token_time: Callable | None,
    budget: GenerationBudget,
) -> tuple[torch.Tensor, bool]:
    """Update generation budget tracking and check stopping conditions.

    Args:
        cumulative_time: Current cumulative time for each sequence
        current_output: Current sampled tokens
        num_generated_tokens: Number of generated tokens
        get_next_token_time: Function to predict time for next token
        mask: Optional attention mask
        is_time_budget: Whether using time-based budget
        min_time_len: Minimum time length for generation
        max_seq_len: Maximum sequence length to generate

    Returns:
        Tuple containing:
        - Updated cumulative time tensor
        - Boolean indicating whether to continue generation

    Examples:
        >>> import torch
        >>> B = 2  # batch_size
        >>> device = 'cpu'
        >>> # Test time budget
        >>> cumulative = torch.zeros(B, device=device)
        >>> # Create a single token mask
        >>> current_sample = torch.randint(0, 5, (5,), device=device)
        >>> get_time = lambda x: torch.ones(B, device=device)
        >>> time_budget = GenerationBudget.from_time_len(1.5)  # Generate until cumulative time > 1.5
        >>> new_time, continue_gen = update_generation_budget(
        ...     cumulative_time=cumulative,
        ...     current_sample=current_sample,
        ...     num_generated_tokens=2,
        ...     get_next_token_time=get_time,
        ...     budget=time_budget,
        ... )
        >>> new_time  # Should be 1
        tensor([1., 1.])
        >>> continue_gen  # Should continue since < min_time_len
        True
        >>> new_time, continue_gen = update_generation_budget(
        ...     cumulative_time=new_time,
        ...     current_sample=current_sample,
        ...     num_generated_tokens=3,
        ...     get_next_token_time=get_time,
        ...     budget=time_budget,
        ... )
        >>> continue_gen
        False
        >>> new_time # Should be incremented by 1
        tensor([2., 2.])

        >>> seq_len_budget = GenerationBudget.from_seq_len(10)  # Generate until max_seq_len >= 10
        >>> # Test sequence length budget
        >>> _, continue_gen = update_generation_budget(
        ...     cumulative_time=cumulative,
        ...     current_sample=current_sample,
        ...     num_generated_tokens=5,
        ...     get_next_token_time=None,
        ...     budget=seq_len_budget,
        ... )
        >>> continue_gen  # Should continue
        True
        >>> seq_len_budget = GenerationBudget.from_seq_len(5)  # Generate until max_seq_len >= 10
        >>> # Test sequence length budget
        >>> _, continue_gen = update_generation_budget(
        ...     cumulative_time=cumulative,
        ...     current_sample=current_sample,
        ...     num_generated_tokens=5,
        ...     get_next_token_time=None,
        ...     budget=seq_len_budget,
        ... )
        >>> continue_gen # Should stop since > max_seq_len
        False
    """
    # Get budget constraints
    min_time_len = budget.min_time_len
    is_time_budget = exists(min_time_len)
    if is_time_budget:
        # Get time prediction for the new token
        with torch.no_grad():
            if get_next_token_time is None:
                raise ValueError("`get_next_token_time` is required when generating using the time budget")
            pred_time = get_next_token_time(current_sample)
            cumulative_time = cumulative_time + pred_time.squeeze(-1)

        continue_generation = (cumulative_time < min_time_len).any().item()
    else:
        continue_generation = num_generated_tokens < budget.max_seq_len

    return cumulative_time, continue_generation


class TransformerDecoderModel(torch.nn.Module, Module):
    """Wrapper of Decoder Transformer for use in MEDS with triplet token embeddings.

    This model handles both forward passes and generation with different budget types.

    Examples:
        >>> # Setup mock configuration
        >>> import torch
        >>> from omegaconf import OmegaConf, open_dict
        >>> from x_transformers import TransformerWrapper, Decoder
        >>> from hydra.utils import instantiate
        >>> B, S, L = 2, 5, 8 # [batch_size, input_sequence_length, token_dim]
        >>> max_seq_len = 7
        >>> vocab_size = 4
        >>> cfg = instantiate({
        ...     'token_dim': L,
        ...     'vocab_size': vocab_size,
        ...     'max_seq_len': max_seq_len,
        ...     'get_last_token': True,
        ...     'temperature': 1.0,
        ...     'token_emb': None,
        ...     'generation_budget': {
        ...         '_target_': ("meds_torch.models.components.transformer_decoder."
        ...                      "GenerationBudget.from_seq_len"),
        ...         "value": max_seq_len,
        ...     },
        ...     'model': {
        ...         '_target_': 'x_transformers.TransformerWrapper',
        ...         'num_tokens': vocab_size,
        ...         'max_seq_len': max_seq_len,
        ...         'use_abs_pos_emb': False,
        ...         'attn_layers': {
        ...             '_target_': 'x_transformers.Decoder',
        ...             'dim': L,
        ...             'depth': 2,
        ...             'heads': 2,
        ...             'rotary_pos_emb': True,
        ...         }
        ...     }
        ... })
        >>> # Initialize model
        >>> model = TransformerDecoderModel(cfg)

        >>> # Test forward pass
        >>> mask = torch.ones(B, S, dtype=torch.bool)
        >>> mask[0,-1] = 0
        >>> batch = {  # Both have shapes: [batch_size, input_sequence_length]
        ...     INPUT_ENCODER_TOKENS_KEY: torch.randint(B, vocab_size, (B, S)),
        ...     INPUT_ENCODER_MASK_KEY: torch.ones(B, S, dtype=torch.bool)
        ... }
        >>> output = model(batch)
        >>> assert BACKBONE_TOKENS_KEY in output
        >>> assert BACKBONE_EMBEDDINGS_KEY in output
        >>> output[BACKBONE_TOKENS_KEY].shape
        torch.Size([2, 5, 4])
        >>> # Test generation with sequence length budget
        >>> prompts = torch.randint(0, vocab_size, (B, S))  # [batch_size, prompt_len]
        >>> gen_output = model.generate(
        ...     prompts=prompts,
        ...     mask=mask,
        ... )
        >>> # Since we started with 5 prompt tokens we can generate 2 more to get to a max_seq_length of 7:
        >>> gen_output.shape
        torch.Size([2, 7])
        >>> # Test generation with time budget and sliding window
        >>> # The timing function (get_next_token_time) maps token embeddings to predicted duration
        >>> # Here we use a simple function that returns 1.0 for each token
        >>> time_budget = GenerationBudget.from_time_len(10)  # Generate until cumulative time >= 10
        >>> cfg.generation_budget = time_budget # modify cfg to use the time budget
        >>> gen_output = model.generate(
        ...     prompts=prompts,
        ...     mask=mask,
        ...     temperature=0.7,
        ...     get_next_token_time=lambda code: torch.ones(B).float(),
        ... )
        >>> # Should have generated 10 tokens (since each token takes 1.0 time)
        >>> gen_output.shape
        torch.Size([2, 10])
    """

    def __init__(self, cfg: DictConfig):
        super().__init__()
        self.cfg = cfg
        self.model = cfg.model

        if cfg.token_emb:
            self.model.token_emb = cfg.token_emb

    def forward(self, batch):
        input_data, mask = batch[INPUT_ENCODER_TOKENS_KEY], batch[INPUT_ENCODER_MASK_KEY]
        output, embeddings = self.model(input_data, mask=mask, return_logits_and_embeddings=True)
        if self.cfg.get_last_token:
            embeddings = get_last_token(embeddings, ~mask)
        batch[BACKBONE_TOKENS_KEY] = output
        batch[BACKBONE_EMBEDDINGS_KEY] = embeddings
        return batch

    def generate(
        self,
        prompts: torch.Tensor,
        mask: torch.Tensor | None = None,
        get_next_token_time: Callable[[torch.Tensor, torch.Tensor], torch.Tensor] | None = None,
        eos_token=None,
        temperature=1.0,
        filter_logits_fn: str | Callable = identity,
        restrict_to_max_seq_len: bool = True,
        filter_kwargs: dict = dict(),
        cache_kv=True,
        pad_value: int = 0,
        **kwargs,
    ) -> torch.Tensor:
        """Generate tokens with either sequence length or time budget constraints.

        Args:
            prompts: Input token sequences [batch_size, seq_len]
            budget: GenerationBudget specifying either max_seq_len or min_time_len
            mask: Optional attention mask for prompts
            temperature: Sampling temperature
            filter_logits_fn: Name of logit filtering function ('top_k', 'top_p', etc.)
            **kwargs: Additional arguments passed to the model

        Returns:
            torch.Tensor: Generated token sequences [batch_size, generated_len]
        """
        if mask is None:
            prompt_lengths = torch.tensor([prompts.shape[1]] * prompts.shape[0])
        else:
            prompt_lengths = mask.sum(dim=-1)
            right_pad_mask = torch.arange(mask.size(1), device=mask.device).unsqueeze(
                0
            ) < prompt_lengths.unsqueeze(1)
            if not torch.equal(right_pad_mask, mask):
                raise ValueError("Mask must correspond to right padding")

        return generate(
            self.model,
            prompts=prompts,
            budget=self.cfg.generation_budget,
            get_next_token_time=get_next_token_time,
            eos_token=eos_token,
            temperature=temperature,
            prompt_lens=prompt_lengths,
            filter_logits_fn=filter_logits_fn,
            restrict_to_max_seq_len=restrict_to_max_seq_len,
            filter_kwargs=filter_kwargs,
            cache_kv=cache_kv,
            pad_value=pad_value,
            **kwargs,
        )


@torch.no_grad()
@eval_decorator
def generate(
    model: TransformerWrapper,
    prompts: torch.Tensor,
    budget: GenerationBudget,
    get_next_token_time: Callable[[torch.Tensor, torch.Tensor], torch.Tensor] | None = None,
    eos_token=None,
    temperature=1.0,
    prompt_lens: torch.Tensor | None = None,
    filter_logits_fn: str | Callable = identity,
    restrict_to_max_seq_len: bool = True,
    filter_kwargs: dict = dict(),
    cache_kv=True,
    pad_value: int = 0,
    **kwargs,
):
    """Generate tokens autoregressively from given prompts.

    Args:
        prompts: Input token sequence [batch_size, seq_len]
        seq_len: Number of tokens to generate
        eos_token: Optional end of sequence token to stop generation
        temperature: Sampling temperature (0 for greedy)
        prompt_lens: Optional lengths for variable length prompts
        filter_logits_fn: Optional filtering function for logits
        restrict_to_max_seq_len: Whether to restrict to model's max sequence length
        filter_kwargs: Additional arguments for filter_logits_fn
        cache_kv: Whether to use key-value caching
        **kwargs: Additional arguments passed to model forward

    Returns:
        Generated token sequence [batch_size, seq_len]

    Examples:
    >>> import torch
    >>> from x_transformers import TransformerWrapper, Decoder
    >>> # Create mock transformer
    >>> B, S, L = 2, 4, 8  # batch_size, seq_len, dim
    >>> vocab_size = 5
    >>> model = TransformerWrapper(
    ...     use_abs_pos_emb=False,
    ...     num_tokens=vocab_size,
    ...     max_seq_len=10,
    ...     attn_layers=Decoder(dim=L, depth=1, heads=2, rotary_pos_emb=True)
    ... )

    >>> # Test basic generation
    >>> budget = GenerationBudget.from_seq_len(2)
    >>> prompts = torch.randint(0, vocab_size, (B, S))
    >>> gen_tokens = generate(model=model,
    ...     prompts=prompts,
    ...     budget=budget,
    ...     temperature=1.0,
    ...     filter_logits_fn='top_k',
    ...     filter_kwargs={'k': 10},
    ... )
    >>> gen_tokens.shape  # Should be [batch_size, seq_len]
    torch.Size([2, 2])

    >>> # Test with EOS token
    >>> eos_token = vocab_size - 1
    >>> budget = GenerationBudget.from_seq_len(2)
    >>> gen_tokens = generate(model=model,
    ...     prompts=prompts,
    ...     budget=budget,
    ...     eos_token=eos_token,
    ...     temperature=0.0  # greedy
    ... )
    >>> gen_tokens.shape[0]  # Batch size preserved
    2

    >>> # Test with variable length prompts
    >>> prompt_lens = torch.tensor([2, 3])
    >>> gen_tokens = generate(model=model,
    ...     prompts=prompts,
    ...     budget=budget,
    ...     prompt_lens=prompt_lens,
    ...     temperature=1.0
    ... )
    >>> gen_tokens.shape  # Should be [batch_size, seq_len]
    torch.Size([2, 2])

    >>> # Test with KV caching
    >>> prompt_lens = torch.tensor([S-2] + [S]*(B-1))  # Mask out last two tokens of first sequence
    >>> budget = GenerationBudget.from_seq_len(3)
    >>> gen_tokens = generate(model=model,
    ...     prompts=prompts,
    ...     prompt_lens=prompt_lens,
    ...     budget=budget,
    ...     temperature=1.0,
    ...     cache_kv=True
    ... )
    >>> gen_tokens.shape  # Should be [batch_size, seq_len]
    torch.Size([2, 3])

    >>> # Test with all-True mask
    >>> prompt_lens = torch.tensor([S]*B, dtype=torch.int)  # Variable sequence lengths
    >>> gen_tokens = generate(model=model,
    ...     prompts=prompts,
    ...     prompt_lens=prompt_lens,
    ...     budget=budget,
    ...     temperature=1.0,
    ...     cache_kv=True
    ... )
    >>> gen_tokens.shape  # Should be [batch_size, seq_len]
    torch.Size([2, 3])

    >>> # Test with max sequence length restriction
    >>> long_prompts = torch.randint(0, vocab_size, (B, 8))  # Longer sequence
    >>> prompt_lens = torch.tensor([5] + [8]*(B-1))  # Variable sequence lengths
    >>> gen_tokens = generate(model=model,
    ...     prompts=long_prompts,
    ...     prompt_lens=prompt_lens,
    ...     budget=budget,
    ...     temperature=1.0,
    ...     cache_kv=True,
    ... )
    >>> gen_tokens.shape  # Should be [batch_size, seq_len]
    torch.Size([2, 3])
    """
    max_seq_len, greedy = model.max_seq_len, temperature == 0.0

    prompts, ps = pack([prompts], "* n")

    b, t = prompts.shape

    # handle filter logits fn given as string

    if isinstance(filter_logits_fn, str):
        assert filter_logits_fn in FILTER_LOGITS_FN, f"only {join(FILTER_LOGITS_FN.keys())} are available"

        filter_logits_fn = FILTER_LOGITS_FN[filter_logits_fn]

    # handle variable lengthed prompts (prefixes)

    seq_start_pos = None
    if exists(prompt_lens):
        prompts = align_right(prompts, prompt_lens, pad_id=pad_value)
        seq_start_pos = t - prompt_lens

    # output from which sampled tokens appended to
    out = prompts
    # kv caches
    cache = None
    # Initialize tracking tensors
    cumulative_time = torch.zeros(b, device=prompts.device)

    # sampling up to budget
    out = prompts
    cache = None

    continue_generation = True

    num_generated_tokens = 0
    while continue_generation:
        if restrict_to_max_seq_len:
            max_len_exceeded = out.shape[-1] > max_seq_len

            if cache_kv and max_len_exceeded and not model.can_cache_kv_outside_max_seq_len:
                raise ValueError(
                    "the network cannot use cached key values when decoding outside the "
                    "max sequence length. most likely because you are using absolute "
                    "positional embedding. you can switch to rotary embeddings to resolve "
                    "this issue"
                )

            x = out[:, -max_seq_len:]

            if exists(cache):
                for inter in cache.attn_intermediates:
                    if inter.layer_type == "a":
                        inter.cached_kv = [t[..., -(max_seq_len - 1) :, :] for t in inter.cached_kv]

        logits, new_cache = model(
            x, return_intermediates=True, cache=cache, seq_start_pos=seq_start_pos, **kwargs
        )

        if cache_kv and model.can_cache_kv:
            cache = new_cache

        logits = logits[:, -1]

        # filter by top_k, top_p (nucleus), top_a, or custom

        if greedy:
            sample = logits.argmax(dim=-1, keepdim=True)
        else:
            filtered_logits = filter_logits_fn(logits, **filter_kwargs)
            probs = F.softmax(filtered_logits / temperature, dim=-1)
            sample = torch.multinomial(probs, 1)

        num_generated_tokens += 1
        if num_generated_tokens % max_seq_len == 0 and exists(budget.min_time_len):
            mean_percent_time = round(cumulative_time.mean().item() / budget.min_time_len * 100, 2)
            min_percent_time = round(cumulative_time.min().item() / budget.min_time_len * 100, 2)
            logger.warning(
                "GENERATION TIME BUDGET WARNING: Generated trajectory is long\n"
                f"Generated {num_generated_tokens} tokens\n"
                f"mean time cutoff percent: {mean_percent_time}\n"
                f"min time cutoff percent: {min_percent_time}\n"
            )
        cumulative_time, continue_generation = update_generation_budget(
            cumulative_time, sample, num_generated_tokens, get_next_token_time, budget
        )

        # concat sample
        out = torch.cat((out, sample), dim=-1)

        if not exists(eos_token):
            continue

        is_eos_tokens = out == eos_token

        if is_eos_tokens.any(dim=-1).all():
            break

    if exists(eos_token):
        # mask out everything after the eos tokens
        shifted_is_eos_tokens = F.pad(is_eos_tokens, (1, -1))
        mask = shifted_is_eos_tokens.float().cumsum(dim=-1) >= 1
        out = out.masked_fill(mask, pad_value)

    out = out[:, t:]

    (out,) = unpack(out, ps, "* n")

    return out
