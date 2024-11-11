from collections.abc import Callable
from typing import Any

import torch
from omegaconf import DictConfig
from torch.functional import F
from x_transformers.autoregressive_wrapper import (
    FILTER_LOGITS_FN,
    eval_decorator,
    exists,
)

from meds_torch.input_encoder import INPUT_ENCODER_MASK_KEY, INPUT_ENCODER_TOKENS_KEY
from meds_torch.models import BACKBONE_EMBEDDINGS_KEY, BACKBONE_TOKENS_KEY
from meds_torch.models.components.utils import get_last_token
from meds_torch.utils.module_class import Module


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


def generate_next_token(
    model: Any,
    current_output: torch.Tensor,
    sliding_window_size: int | None,
    cache: Any | None,
    mask: torch.Tensor | None,
    temperature: float,
    filter_logits_fn: str | Callable = torch.nn.Identity(),
    **kwargs,
) -> tuple[torch.Tensor, torch.Tensor, Any | None, torch.Tensor | None]:
    """Generate the next token using sliding window attention and sampling.

    Args:
        model: The transformer model to use for generation
        current_output: Current sequence of tokens [batch_size, seq_len]
        sliding_window_size: Size of sliding window for attention context
        cache: Optional KV cache from previous generation steps
        mask: Optional attention mask [batch_size, seq_len]
        temperature: Sampling temperature (0 for greedy)
        filter_logits_fn: Name of logit filtering function or callable
        **kwargs: Additional arguments passed to model forward

    Returns:
        Tuple containing:
        - Generated token tensor [batch_size, 1]
        - Updated sequence with new token appended
        - Updated KV cache if model supports caching
        - Updated attention mask if input mask was provided

    Examples:
        >>> import torch
        >>> from x_transformers import TransformerWrapper, Decoder
        >>> B, S, L = 2, 5, 8  # batch_size, seq_len, dim
        >>> vocab_size = 100
        >>> # Create mock transformer
        >>> model = TransformerWrapper(
        ...     num_tokens=vocab_size,
        ...     max_seq_len=10,
        ...     attn_layers=Decoder(dim=L, depth=2, heads=2)
        ... )
        >>> # Test with sliding window and mask
        >>> current_output = torch.randint(0, vocab_size, (B, S))
        >>> orig_mask = torch.ones((B, S), dtype=torch.bool)  # All tokens attended to
        >>> orig_mask[0,-1] = 0 # Mask a single token
        >>> sliding_window = 3
        >>> next_token, new_out, new_cache, new_mask = generate_next_token(
        ...     model=model,
        ...     current_output=current_output,
        ...     sliding_window_size=sliding_window,
        ...     cache=None,
        ...     mask=orig_mask,
        ...     temperature=1.0,
        ... )
        >>> next_token.shape  # Single token per sequence
        torch.Size([2, 1])
        >>> new_out.shape  # Original sequence + 1 new token
        torch.Size([2, 6])
        >>> new_mask.shape  # Mask should match new sequence length
        torch.Size([2, 6])
        >>> # Verify mask was properly extended
        >>> (new_mask[:, :-1] == orig_mask).all()  # Old mask values preserved
        tensor(True)
        >>> new_mask[:, -1].all()  # New token is attended to
        tensor(True)

        >>> # Test with sliding window mask shift
        >>> long_output = torch.randint(0, vocab_size, (B, 8))  # Longer than window
        >>> long_mask = torch.ones((B, 8), dtype=torch.bool)
        >>> next_token, new_out, new_cache, new_mask = generate_next_token(
        ...     model=model,
        ...     current_output=long_output,
        ...     sliding_window_size=sliding_window,
        ...     cache=None,
        ...     mask=long_mask,
        ...     temperature=1.0,
        ... )
        >>> # Check if mask was properly sliced for sliding window
        >>> truncated_mask = new_mask[:, -sliding_window:]
        >>> truncated_mask.shape  # Should match sliding window size
        torch.Size([2, 3])
        >>> truncated_mask.all()  # All tokens in window should be attended to
        tensor(True)

        >>> # Test without mask (should still work)
        >>> next_token, new_out, _, output_mask = generate_next_token(
        ...     model=model,
        ...     current_output=current_output,
        ...     sliding_window_size=None,
        ...     cache=None,
        ...     mask=None,
        ...     temperature=1.0,
        ...     filter_logits_fn='top_k'
        ... )
        >>> output_mask is None  # Should remain None if no input mask
        True
    """
    # Handle sliding window for long sequences
    if exists(sliding_window_size) and current_output.shape[1] > sliding_window_size:
        window_start = max(0, current_output.shape[1] - sliding_window_size)
        x = current_output[:, window_start:]

        # Shift mask for sliding window if it exists
        window_mask = mask[:, window_start:] if exists(mask) else None

        # Adjust cache for sliding window
        if exists(cache):
            for inter in cache.attn_intermediates:
                inter.cached_kv = [t[..., -(sliding_window_size - 1) :, :] for t in inter.cached_kv]
    else:
        x = current_output
        window_mask = mask

    # Generate next token logits
    logits, new_cache = model(x, mask=window_mask, return_intermediates=True, cache=cache, **kwargs)
    # model(x, mask=window_mask, return_intermediates=True, cache=None, **kwargs)[0]

    if model.can_cache_kv:
        cache = new_cache

    # Sample next token
    logits = logits[:, -1]
    if temperature == 0:
        sample = logits.argmax(dim=-1, keepdim=True)
    else:
        if isinstance(filter_logits_fn, str):
            filter_fn = FILTER_LOGITS_FN[filter_logits_fn]
            filtered_logits = filter_fn(logits)
        else:
            filtered_logits = filter_logits_fn(logits)
        probs = F.softmax(filtered_logits / temperature, dim=-1)
        sample = torch.multinomial(probs, 1)

    # Update output
    new_output = torch.cat((current_output, sample), dim=-1)

    # Update mask if it exists
    new_mask = None
    if exists(mask):
        new_mask = F.pad(mask, (0, 1), value=True)  # Extend mask for new token

    return sample, new_output, cache, new_mask


def update_generation_budget(
    cumulative_time: torch.Tensor,
    current_output: torch.Tensor,
    get_next_token_time: Callable | None,
    mask: torch.Tensor | None,
    budget: GenerationBudget,
) -> tuple[torch.Tensor, bool]:
    """Update generation budget tracking and check stopping conditions.

    Args:
        cumulative_time: Current cumulative time for each sequence
        current_output: Current sequence of tokens
        get_next_token_time: Function to predict time for next token
        mask: Optional attention mask
        is_time_budget: Whether using time-based budget
        min_time_len: Minimum time length for generation
        max_seq_len: Maximum sequence length

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
        >>> mask = torch.ones(B, 5)
        >>> mask[0,-1] = 0
        >>> current_out = torch.randint(0, 5, (B, 5), device=device)
        >>> get_time = lambda x, m: torch.ones(B, device=device)
        >>> time_budget = GenerationBudget.from_time_len(1.5)  # Generate until cumulative time > 1.5
        >>> new_time, continue_gen = update_generation_budget(
        ...     cumulative_time=cumulative,
        ...     current_output=current_out,
        ...     get_next_token_time=get_time,
        ...     mask=mask,
        ...     budget=time_budget,
        ... )
        >>> new_time  # Should be 1
        tensor([1., 1.])
        >>> continue_gen  # Should continue since < min_time_len
        True
        >>> new_time, continue_gen = update_generation_budget(
        ...     cumulative_time=new_time,
        ...     current_output=current_out,
        ...     get_next_token_time=get_time,
        ...     mask=mask,
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
        ...     current_output=current_out,
        ...     get_next_token_time=None,
        ...     mask=mask,
        ...     budget=seq_len_budget,
        ... )
        >>> continue_gen  # Should continue
        True
        >>> seq_len_budget = GenerationBudget.from_seq_len(5)  # Generate until max_seq_len >= 10
        >>> # Test sequence length budget
        >>> _, continue_gen = update_generation_budget(
        ...     cumulative_time=cumulative,
        ...     current_output=current_out,
        ...     get_next_token_time=None,
        ...     mask=mask,
        ...     budget=seq_len_budget,
        ... )
        >>> continue_gen # Should stop since > max_seq_len
        False
    """
    # Get budget constraints
    max_seq_len = budget.max_seq_len
    min_time_len = budget.min_time_len
    is_time_budget = exists(min_time_len)
    if is_time_budget:
        # Get time prediction for the new token
        with torch.no_grad():
            if get_next_token_time is None:
                raise ValueError("`get_next_token_time` is required when generating using the time budget")
            pred_time = get_next_token_time(current_output, mask)
            cumulative_time = cumulative_time + pred_time.squeeze(-1)

        continue_generation = (cumulative_time < min_time_len).any().item()
    else:
        continue_generation = current_output.shape[1] < max_seq_len

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
        ...         'attn_layers': {
        ...             '_target_': 'x_transformers.Decoder',
        ...             'dim': L,
        ...             'depth': 2,
        ...             'heads': 2,
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
        torch.Size([2, 2])
        >>> # Test generation with time budget and sliding window
        >>> # The timing function (get_next_token_time) maps token embeddings to predicted duration
        >>> # Here we use a simple function that returns 1.0 for each token
        >>> time_budget = GenerationBudget.from_time_len(10)  # Generate until cumulative time >= 10
        >>> cfg.generation_budget = time_budget # modify cfg to use the time budget
        >>> sliding_window_size = 7  # Small window to force sliding
        >>> gen_output = model.generate(
        ...     prompts=prompts,
        ...     mask=mask,
        ...     sliding_window_size=sliding_window_size,
        ...     temperature=0.7,
        ...     get_next_token_time=lambda code, next_token_mask: torch.ones(B).float(),
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

    @torch.no_grad()
    @eval_decorator
    def generate(
        self,
        prompts: torch.Tensor,
        mask: torch.Tensor | None = None,
        get_next_token_time: Callable[[torch.Tensor, torch.Tensor], torch.Tensor] | None = None,
        temperature: float = 1.0,
        filter_logits_fn: str | Callable = torch.nn.Identity(),
        sliding_window_size: int | None = None,
        **kwargs,
    ) -> torch.Tensor:
        """Generate tokens with either sequence length or time budget constraints.

        Args:
            prompts: Input token sequences [batch_size, seq_len]
            budget: GenerationBudget specifying either max_seq_len or min_time_len
            mask: Optional attention mask for prompts
            temperature: Sampling temperature
            filter_logits_fn: Name of logit filtering function ('top_k', 'top_p', etc.)
            sliding_window_size: Size of sliding window for attention context
            **kwargs: Additional arguments passed to the model

        Returns:
            torch.Tensor: Generated token sequences [batch_size, generated_len]
        """
        budget = self.cfg.generation_budget
        device = prompts.device
        batch_size, prompt_len = prompts.shape
        sliding_window_size = sliding_window_size or self.cfg.max_seq_len

        # Initialize tracking tensors
        cumulative_time = torch.zeros(batch_size, device=device)
        out = prompts
        cache = None

        continue_generation = True

        while continue_generation:
            _, out, cache, mask = generate_next_token(
                self.model, out, sliding_window_size, cache, mask, temperature, filter_logits_fn, **kwargs
            )
            if mask is not None:
                cache = None  # Caching doesn't work if `mask` is used in x-transformers, ses #128
            cumulative_time, continue_generation = update_generation_budget(
                cumulative_time, out, get_next_token_time, mask, budget
            )
            from loguru import logger

            logger.info(f"Time: {cumulative_time}")
            logger.info(f"Out: {out}")
            logger.info(temperature)

        # Return only the generated part (excluding prompt)
        return out[:, prompt_len:]
