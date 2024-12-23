from abc import ABC, abstractmethod
from collections.abc import Callable
from functools import wraps
from types import SimpleNamespace

import torch
from clinical_zeroshot_labeler.labeler import SequenceLabeler, WindowStatus
from loguru import logger
from omegaconf import DictConfig
from torch.functional import F
from x_transformers import Decoder, TransformerWrapper
from x_transformers.autoregressive_wrapper import (
    FILTER_LOGITS_FN,
    align_right,
    exists,
    identity,
    join,
    pack,
    unpack,
)


def eval_decorator(fn):
    @wraps(fn)
    def inner(generative_model, *args, **kwargs):
        transformer_decoder = generative_model.model.model
        was_training = transformer_decoder.training
        transformer_decoder.eval()
        out = fn(generative_model, *args, **kwargs)
        transformer_decoder.train(was_training)
        return out

    return inner


def get_last_token(output: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Get the last non-masked token from the output tensor.

    Args: output (torch.Tensor): The output tensor of shape (batch_size, seq_len, hidden_dim) mask
    (torch.Tensor): The mask tensor of shape (batch_size, seq_len) where True indicates masked positions

    Returns: torch.Tensor: The last non-masked token for each sequence in the batch
    """
    # Find the last non-masked position
    last_non_masked = (~mask).float().cumsum(dim=1).argmax(dim=1)

    # Handle cases where all positions are masked
    all_masked = mask.all(dim=1)
    last_non_masked[all_masked] = 0

    if all_masked.any():
        raise ValueError(
            f"{all_masked.sum().item()} sequences have all positions masked. Mask should likely be negated"
        )

    # Create indices for gathering
    batch_size, _, hidden_dim = output.shape
    indices = last_non_masked.view(batch_size, 1, 1).expand(-1, 1, hidden_dim)

    # Gather the last non-masked tokens
    last_token = torch.gather(output, dim=1, index=indices).squeeze(1)

    return last_token


class BaseGenerativeModel(ABC):
    """Base class for generative models with custom token processing.

    This class defines the interface for models that need to generate sequences
    with custom token processing logic. Each subclass can implement its own
    strategy for processing tokens and updating generation state.
    """

    @abstractmethod
    def update_generation_state(
        self,
        tokens: torch.Tensor,
        cumulative_time: torch.Tensor,
        trajectory_labeler: SequenceLabeler | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, bool, torch.Tensor]:
        """Update generation state based on new tokens.

        Args:
            tokens: Generated tokens of shape (batch_size, num_tokens)
            cumulative_time: Current cumulative time for each sequence
            trajectory_labeler: Optional sequence labeler for monitoring conditions

        Returns:
            Tuple containing:
                - Updated cumulative time
                - Status from trajectory labeler (if used)
                - Whether generation should finish
                - Boolean mask indicating which sequences have ended
        """

    def _check_valid_mask(self, mask, prompt_lens):
        """Verifies mask contains only contiguous 1s followed by all 0s."""
        right_pad_mask = torch.arange(mask.size(1), device=mask.device).unsqueeze(0) < prompt_lens.unsqueeze(
            1
        )
        if not torch.equal(right_pad_mask, mask):
            raise ValueError("Mask must correspond to right padding")

    def _track_sliding_window_generation(
        self, out, max_seq_len, cache_kv, cache, transformer_decoder, seq_start_pos
    ):
        max_len_exceeded = out.shape[-1] > max_seq_len

        if cache_kv and max_len_exceeded and not transformer_decoder.can_cache_kv_outside_max_seq_len:
            raise ValueError(
                "the network cannot use cached key values when decoding outside the "
                "max sequence length. most likely because you are using absolute "
                "positional embedding. you can switch to rotary embeddings to resolve "
                "this issue"
            )

        x = out[:, -max_seq_len:]

        num_clipped_tokens = out.shape[1] - max_seq_len
        current_start_pos = seq_start_pos
        if num_clipped_tokens > 0:
            current_start_pos = (current_start_pos - num_clipped_tokens).clip(min=0)

        if exists(cache):
            for inter in cache.attn_intermediates:
                if inter.layer_type == "a":
                    inter.cached_kv = [t[..., -(max_seq_len - 1) :, :] for t in inter.cached_kv]
        return x, cache, current_start_pos

    @torch.no_grad()
    @eval_decorator
    def generate(
        self,
        prompts: torch.Tensor,
        mask: torch.Tensor | None,
        trajectory_labeler: SequenceLabeler | None = None,
        time_offset_years: torch.Tensor | None = None,
        eos_tokens: list[int] | None = None,
        temperature: float = 1.0,
        filter_logits_fn: str | Callable = identity,
        filter_kwargs: dict = dict(),
        cache_kv: bool = True,
        pad_value: int = 0,
        **kwargs,
    ) -> tuple[torch.Tensor, torch.Tensor, dict | None]:
        """Generate token sequences with model-specific processing.

        This implements the core generation loop while delegating token
        processing to subclass implementations.

        Args:
            prompts: Input token sequences [batch_size, seq_len]
            mask: Optional attention mask for prompts
            trajectory_labeler: Optional labeler for monitoring conditions
            time_offset_years: Optional time offsets per sequence
            eos_tokens: Optional tokens that should end generation
            temperature: Sampling temperature
            filter_logits_fn: Optional logits filtering function
            filter_kwargs: Additional args for filtering
            cache_kv: Whether to use KV caching
            pad_value: Value to use for padding
            **kwargs: Additional arguments passed to model

        Returns:
            Tuple containing:
                - Generated token sequences
                - Sequence lengths
                - Optional metadata dict

        Examples:
            >>> # Basic generation test
            >>> model = TestModel()
            >>> prompts = torch.randint(0, 5, (2, 3))  # batch_size=2, seq_len=3
            >>> mask = torch.ones((2, 3), dtype=torch.bool)

            >>> tokens, lengths, meta = model.generate(prompts, mask, temperature=1.0)
            >>> assert tokens.shape[1] <= model.cfg.max_tokens_budget  # Respects budget
            >>> assert lengths.shape == (2,)  # Batch size preserved

            >>> # Test with trajectory labeler
            >>> labeler = DummyLabeler()
            >>> tokens, lengths, meta = model.generate(
            ...     prompts,
            ...     mask,
            ...     trajectory_labeler=labeler,
            ...     temperature=1.0
            ... )
            >>> assert meta is not None  # Metadata returned with labeler
            >>> assert "labels" in meta
            >>> assert "status" in meta

            >>> # Test with EOS token
            >>> tokens, lengths, meta = model.generate(
            ...     prompts,
            ...     mask,
            ...     eos_tokens=[4],  # Use token 4 as EOS
            ...     temperature=0.0  # Greedy sampling
            ... )
            >>> assert tokens.shape[1] <= model.cfg.max_tokens_budget

            # >>> # Test with time offset
            # >>> time_offset = torch.tensor([1.0, 2.0])
            # >>> tokens, lengths, meta = model.generate(
            # ...     prompts,
            # ...     mask,
            # ...     time_offset_years=time_offset,
            # ...     temperature=1.0
            # ... )
            # >>> assert tokens.shape[1] <= model.cfg.max_tokens_budget
        """
        transformer_decoder = self.model.model
        max_seq_len = transformer_decoder.max_seq_len

        prompts, ps = pack([prompts], "* n")
        b, t = prompts.shape

        # Handle filter logits fn given as string
        if isinstance(filter_logits_fn, str):
            assert filter_logits_fn in FILTER_LOGITS_FN, f"only {join(FILTER_LOGITS_FN.keys())} are available"
            filter_logits_fn = FILTER_LOGITS_FN[filter_logits_fn]

        # Align prompts
        prompt_lens = mask.sum(dim=-1)
        self._check_valid_mask(mask, prompt_lens)
        prompts = align_right(prompts, prompt_lens, pad_id=pad_value)

        seq_start_pos = t - prompt_lens

        if exists(eos_tokens):
            eos_tokens = torch.tensor(eos_tokens)

        # Initialize state
        out = prompts
        cache = None
        cumulative_time = (
            time_offset_years if time_offset_years is not None else torch.zeros(b, device=prompts.device)
        )
        ended_sequences = torch.zeros(b, dtype=torch.bool)
        is_finished = False
        num_generated_tokens = 0
        out_lengths = torch.zeros(b, dtype=torch.int32)
        metadata = None
        status = None

        while not is_finished:
            x, cache, current_start_pos = self._track_sliding_window_generation(
                out, max_seq_len, cache_kv, cache, transformer_decoder, seq_start_pos
            )

            # Get next token predictions
            logits, new_cache = transformer_decoder(
                x, return_intermediates=True, cache=cache, seq_start_pos=current_start_pos, **kwargs
            )

            if cache_kv and transformer_decoder.can_cache_kv:
                cache = new_cache

            logits = logits[:, -1]

            # Sample next tokens
            if temperature == 0.0:  # greedy sampling
                sample = logits.argmax(dim=-1, keepdim=True)
            else:
                filtered_logits = filter_logits_fn(logits, **filter_kwargs)
                probs = F.softmax(filtered_logits / temperature, dim=-1)
                sample = torch.multinomial(probs, 1)

            # Update generation state
            num_generated_tokens += 1
            if num_generated_tokens % max_seq_len == 0:
                logger.warning(f"Generated {num_generated_tokens} tokens")

            # Append new tokens
            out = torch.cat((out, sample), dim=-1)

            # Update cumulative time and check status
            cumulative_time, status, is_finished, new_ended_sequences = self.update_generation_state(
                out,
                cumulative_time,
                trajectory_labeler,
            )

            # Update sequence end flags
            new_ended_sequences |= ended_sequences
            if exists(eos_tokens):
                new_ended_sequences |= sample.flatten().cpu() == eos_tokens
            out_lengths[new_ended_sequences != ended_sequences] = num_generated_tokens
            ended_sequences = new_ended_sequences
            # Check max token budget condition
            if self.cfg.max_tokens_budget is not None and num_generated_tokens >= self.cfg.max_tokens_budget:
                is_finished = True
                out_lengths[~ended_sequences] = num_generated_tokens

            if ended_sequences.all():
                is_finished = True

        # Get final metadata if using labeler
        if status is not None:
            metadata = dict(labels=trajectory_labeler.get_labels(), status=status)

        # Process final sequences
        out = out[:, t:]
        (out,) = unpack(out, ps, "* n")

        return out, out_lengths, metadata


class DummyLabeler:
    """Simple labeler for testing that completes after 3 steps."""

    def __init__(self):
        self.step = 0

    def process_step(self, *args):
        self.step += 1
        return torch.tensor([WindowStatus.ACTIVE.value])

    def is_finished(self):
        return self.step >= 3

    def get_labels(self):
        return torch.tensor([1.0])


class TestModel(BaseGenerativeModel):
    """Simple test implementation of BaseGenerativeModel."""

    def __init__(self):
        self.model = SimpleNamespace()
        self.model.model = TransformerWrapper(
            num_tokens=5,
            max_seq_len=10,
            attn_layers=Decoder(dim=8, depth=1, heads=2, rotary_pos_emb=True),
            use_abs_pos_emb=False,
        )
        self.cfg = DictConfig(dict(max_tokens_budget=5))

    def update_generation_state(self, tokens, time, labeler=None):
        # Simple implementation that ends after 3 tokens
        is_done = time.sum() >= 3
        if labeler is not None:
            if is_done:
                status = torch.tensor([WindowStatus.ACTIVE.value] * tokens.shape[0])
            else:
                status = torch.tensor([WindowStatus.SATISFIED.value] * tokens.shape[0])
        else:
            status = None
        return time + 1, status, is_done, torch.tensor([is_done] * tokens.shape[0])

    def process_generated_tokens(self, tokens, mask, metadata=None):
        return {"tokens": tokens, "mask": mask}
