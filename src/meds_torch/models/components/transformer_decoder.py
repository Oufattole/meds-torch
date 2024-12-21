from collections.abc import Callable
from functools import wraps

import torch
from clinical_zeroshot_labeler.labeler import SequenceLabeler, WindowStatus
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


class DummyTrajectoryLabeler:
    def __init__(self, B):
        self.counter = 0
        self.status = [
            torch.tensor([WindowStatus.UNDETERMINED.value] * B),
            torch.tensor([WindowStatus.ACTIVE.value] * B),
            torch.tensor([WindowStatus.SATISFIED.value] * B),
        ]
        self.labels = torch.zeros((B,), dtype=torch.bool)

    def process_step(self, tokens, times, values):
        status = self.status[min(self.counter, len(self.status) - 1)]
        self.counter += 1
        return status

    def is_finished(self):
        return self.counter >= len(self.status)

    def get_labels(self):
        return self.labels


def update_state(
    cumulative_time: torch.Tensor,
    current_sample: torch.Tensor,
    get_next_token_time: Callable | None,
    get_next_token_value: Callable | None,
    trajectory_labeler: SequenceLabeler | None,
) -> tuple[torch.Tensor, torch.Tensor, bool]:
    """Updates trajectory_labeler state, and returns state information.

    Examples:
        >>> import torch
        >>> from clinical_zeroshot_labeler.labeler import WindowStatus
        >>> B = 2  # batch_size
        >>> device = 'cpu'

        >>> # Setup basic test case
        >>> cumulative = torch.tensor([0.0, 0.0], device=device)
        >>> current_sample = torch.randint(0, 5, (B,), device=device)
        >>> get_time = lambda x: torch.ones(B, device=device)
        >>> get_value = lambda x: torch.ones(B, device=device)

        >>> # Test trajectory labeler progression
        >>> labeler = DummyTrajectoryLabeler(B)
        >>> time, status, is_finished, ended = update_state(
        ...     cumulative_time=cumulative,
        ...     current_sample=current_sample,
        ...     get_next_token_time=get_time,
        ...     get_next_token_value=get_value,
        ...     trajectory_labeler=labeler,
        ... )
        >>> assert time.shape == (B,)
        >>> assert status.shape == (B,)
        >>> assert not is_finished
        >>> assert not ended.any()

        >>> # Test second step shows active status
        >>> time, status, is_finished, ended = update_state(
        ...     cumulative_time=time,
        ...     current_sample=current_sample,
        ...     get_next_token_time=get_time,
        ...     get_next_token_value=get_value,
        ...     trajectory_labeler=labeler,
        ... )
        >>> assert (status == WindowStatus.ACTIVE.value).all()
        >>> assert not is_finished
        >>> assert not ended.any()

        >>> # Test third step shows satisfied status and finished
        >>> time, status, is_finished, ended = update_state(
        ...     cumulative_time=time,
        ...     current_sample=current_sample,
        ...     get_next_token_time=get_time,
        ...     get_next_token_value=get_value,
        ...     trajectory_labeler=labeler,
        ... )
        >>> assert (status == WindowStatus.SATISFIED.value).all()
        >>> assert is_finished
        >>> assert ended.all()

        >>> # Test without trajectory labeler
        >>> time, status, is_finished, ended = update_state(
        ...     cumulative_time=cumulative,
        ...     current_sample=current_sample,
        ...     get_next_token_time=get_time,
        ...     get_next_token_value=get_value,
        ...     trajectory_labeler=None,
        ... )
        >>> assert time.shape == (B,)
        >>> assert status is None
        >>> assert not is_finished
        >>> assert not ended.any()
    """
    if get_next_token_time is None:
        raise ValueError("`get_next_token_time` is required when generating using the time budget")
    pred_time = get_next_token_time(current_sample)
    cumulative_time = cumulative_time + pred_time.squeeze(-1)
    if get_next_token_time is None:
        raise ValueError(
            "`get_next_token_value` is required when generating using the sequence length budget"
        )
    current_value = get_next_token_value(current_sample)
    if trajectory_labeler is not None:
        status = trajectory_labeler.process_step(current_sample, cumulative_time, current_value)
        is_finished = trajectory_labeler.is_finished()
        ended_sequences = torch.logical_or(
            status == WindowStatus.SATISFIED.value, status == WindowStatus.IMPOSSIBLE.value
        )
    else:
        status = None
        is_finished = False
        ended_sequences = torch.zeros((current_sample.shape[0]), dtype=torch.bool)
    return cumulative_time, status, is_finished, ended_sequences


class TransformerDecoderModel(torch.nn.Module, Module):
    """Wrapper of Decoder Transformer for use in MEDS with triplet token embeddings.

    This model handles both forward passes and generation with different budget types.

    Examples:
        >>> # Setup mock configuration and model components
        >>> import torch
        >>> from omegaconf import OmegaConf
        >>> from x_transformers import TransformerWrapper, Decoder
        >>> from hydra.utils import instantiate
        >>> from enum import Enum
        >>> from clinical_zeroshot_labeler.labeler import WindowStatus

        >>> # Mock configuration
        >>> B, S, L = 2, 5, 8  # batch_size, seq_len, dim
        >>> max_seq_len = 7
        >>> vocab_size = 4
        >>> cfg = instantiate({
        ...     'token_dim': L,
        ...     'vocab_size': vocab_size,
        ...     'max_seq_len': max_seq_len,
        ...     'get_last_token': True,
        ...     'temperature': 1.0,
        ...     'token_emb': None,
        ...     'max_tokens_budget': 10,
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
        >>> # Test basic generation with trajectory labeler
        >>> prompts = torch.randint(0, vocab_size, (B, S))
        >>> mask = torch.ones(B, S, dtype=torch.bool)
        >>> labeler = DummyTrajectoryLabeler(B)
        >>> get_time = lambda x: torch.ones(B, dtype=torch.float)
        >>> get_value = lambda x: torch.ones(B, dtype=torch.float)
        >>> gen_output, lengths, metadata = model.generate(
        ...     prompts=prompts,
        ...     mask=mask,
        ...     get_next_token_time=get_time,
        ...     get_next_token_value=get_value,
        ...     trajectory_labeler=labeler,
        ...     temperature=1.0
        ... )
        >>> assert 'labels' in metadata
        >>> assert 'status' in metadata
        >>> assert metadata['labels'].shape == (B,)
        >>> assert metadata['status'].shape == (B,)

        >>> # Test generation with max tokens budget
        >>> gen_output, lengths, metadata = model.generate(
        ...     prompts=prompts,
        ...     mask=mask,
        ...     get_next_token_time=get_time,
        ...     get_next_token_value=get_value,
        ...     temperature=1.0
        ... )
        >>> assert gen_output.shape[1] <= cfg.max_tokens_budget

        >>> # Test generation with variable length prompts
        >>> prompt_lens = torch.tensor([2, 3])
        >>> gen_output, lengths, metadata = model.generate(
        ...     prompts=prompts,
        ...     get_next_token_time=get_time,
        ...     get_next_token_value=get_value,
        ...     temperature=1.0
        ... )
        >>> assert gen_output.shape[1] <= cfg.max_tokens_budget

        >>> # Test with KV caching
        >>> prompt_lens = torch.tensor([S-2] + [S]*(B-1))
        >>> gen_output, lengths, metadata = model.generate(
        ...     prompts=prompts,
        ...     get_next_token_time=get_time,
        ...     get_next_token_value=get_value,
        ...     temperature=1.0,
        ...     cache_kv=True
        ... )
        >>> assert gen_output.shape[1] <= cfg.max_tokens_budget
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
        get_next_token_value: Callable[[torch.Tensor, torch.Tensor], torch.Tensor] | None = None,
        trajectory_labeler: Callable | None = None,
        time_offset_years: torch.Tensor | None = None,
        eos_tokens: list[int] | None = None,
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
            max_tokens_budget=self.cfg.max_tokens_budget,
            time_offset_years=time_offset_years,
            get_next_token_time=get_next_token_time,
            get_next_token_value=get_next_token_value,
            trajectory_labeler=trajectory_labeler,
            eos_tokens=eos_tokens,
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
    max_tokens_budget: int | None = None,
    time_offset_years: torch.Tensor | None = None,
    get_next_token_time: Callable[[torch.Tensor, torch.Tensor], torch.Tensor] | None = None,
    get_next_token_value: Callable[[torch.Tensor, torch.Tensor], torch.Tensor] | None = None,
    trajectory_labeler: Callable | None = None,
    eos_tokens: list[int] | None = None,
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
        >>> from clinical_zeroshot_labeler.labeler import WindowStatus

        >>> # Create mock transformer
        >>> B, S, L = 2, 4, 8  # batch_size, seq_len, dim
        >>> vocab_size = 5
        >>> model = TransformerWrapper(
        ...     use_abs_pos_emb=False,
        ...     num_tokens=vocab_size,
        ...     max_seq_len=10,
        ...     attn_layers=Decoder(dim=L, depth=1, heads=2, rotary_pos_emb=True)
        ... )

        >>> # Basic test setup
        >>> prompts = torch.randint(0, vocab_size, (B, S))
        >>> get_time = lambda x: torch.ones(B, dtype=torch.float)
        >>> get_value = lambda x: torch.ones(B, dtype=torch.float)
        >>> labeler = DummyTrajectoryLabeler(B)

        >>> # Test basic generation with trajectory labeler
        >>> max_tokens = 5
        >>> gen_tokens, lengths, metadata = generate(
        ...     model=model,
        ...     prompts=prompts,
        ...     max_tokens_budget=max_tokens,
        ...     get_next_token_time=get_time,
        ...     get_next_token_value=get_value,
        ...     trajectory_labeler=labeler,
        ...     temperature=1.0
        ... )
        >>> assert gen_tokens.shape[1] <= max_tokens  # Check max tokens budget
        >>> assert isinstance(metadata, dict)
        >>> assert 'labels' in metadata
        >>> assert 'status' in metadata
        >>> assert metadata['labels'].shape == (B,)
        >>> assert metadata['status'].shape == (B,)

        >>> # Test without trajectory labeler (max tokens only)
        >>> gen_tokens, lengths, metadata = generate(
        ...     model=model,
        ...     prompts=prompts,
        ...     max_tokens_budget=max_tokens,
        ...     get_next_token_time=get_time,
        ...     get_next_token_value=get_value,
        ...     temperature=1.0
        ... )
        >>> assert gen_tokens.shape[1] <= max_tokens
        >>> assert metadata is None  # No metadata without labeler

        >>> # Test with EOS tokens
        >>> eos_tokens = [vocab_size - 1]
        >>> gen_tokens, lengths, metadata = generate(
        ...     model=model,
        ...     prompts=prompts,
        ...     max_tokens_budget=max_tokens,
        ...     eos_tokens=eos_tokens,
        ...     get_next_token_time=get_time,
        ...     get_next_token_value=get_value,
        ...     temperature=0.0  # greedy
        ... )
        >>> assert gen_tokens.shape[0] == B  # Batch size preserved
        >>> assert gen_tokens.shape[1] <= max_tokens

        >>> # Test with variable length prompts
        >>> prompt_lens = torch.tensor([2, 3])
        >>> gen_tokens, lengths, metadata = generate(
        ...     model=model,
        ...     prompts=prompts,
        ...     prompt_lens=prompt_lens,
        ...     max_tokens_budget=max_tokens,
        ...     get_next_token_time=get_time,
        ...     get_next_token_value=get_value,
        ...     temperature=1.0
        ... )
        >>> assert gen_tokens.shape[1] <= max_tokens

        >>> # Test with KV caching
        >>> prompt_lens = torch.tensor([S-2] + [S]*(B-1))  # Mask out last two tokens of first sequence
        >>> gen_tokens, lengths, metadata = generate(
        ...     model=model,
        ...     prompts=prompts,
        ...     prompt_lens=prompt_lens,
        ...     max_tokens_budget=max_tokens,
        ...     get_next_token_time=get_time,
        ...     get_next_token_value=get_value,
        ...     temperature=1.0,
        ...     cache_kv=True
        ... )
        >>> assert gen_tokens.shape[1] <= max_tokens

        >>> # Test with trajectory labeler and time offset
        >>> time_offset = torch.tensor([1.0, 2.0])
        >>> labeler = DummyTrajectoryLabeler(B)
        >>> gen_tokens, lengths, metadata = generate(
        ...     model=model,
        ...     prompts=prompts,
        ...     max_tokens_budget=max_tokens,
        ...     time_offset_years=time_offset,
        ...     get_next_token_time=get_time,
        ...     get_next_token_value=get_value,
        ...     trajectory_labeler=labeler,
        ...     temperature=1.0
        ... )
        >>> assert gen_tokens.shape[1] <= max_tokens
        >>> assert 'labels' in metadata
        >>> assert 'status' in metadata
        >>> assert metadata['labels'].shape == (B,)
        >>> assert metadata['status'].shape == (B,)

        >>> # Test with max sequence length restriction
        >>> long_prompts = torch.randint(0, vocab_size, (B, 8))  # Longer sequence
        >>> prompt_lens = torch.tensor([5] + [8]*(B-1))  # Variable sequence lengths
        >>> gen_tokens, lengths, metadata = generate(
        ...     model=model,
        ...     prompts=long_prompts,
        ...     prompt_lens=prompt_lens,
        ...     max_tokens_budget=max_tokens,
        ...     get_next_token_time=get_time,
        ...     get_next_token_value=get_value,
        ...     temperature=1.0,
        ...     cache_kv=True,
        ...     restrict_to_max_seq_len=True
        ... )
        >>> assert gen_tokens.shape[1] <= max_tokens
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
    if exists(eos_tokens):
        eos_tokens = torch.tensor(eos_tokens, device=prompts.device)

    # output from which sampled tokens appended to
    out = prompts
    # kv caches
    cache = None
    # Initialize tracking tensors
    cumulative_time = (
        time_offset_years if time_offset_years is not None else torch.zeros(b, device=prompts.device)
    )

    # sampling up to budget
    out = prompts
    cache = None
    ended_sequences = torch.zeros(b, device=prompts.device, dtype=torch.bool)

    is_finished = False

    num_generated_tokens = 0
    out_lengths = torch.zeros(b, device=prompts.device, dtype=torch.int32)
    metadata = None
    status = None

    while not is_finished:
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

        # Update generation state
        num_generated_tokens += 1
        if num_generated_tokens % max_seq_len == 0:
            logger.warning(
                "GENERATION TIME BUDGET WARNING: Generated trajectory is long\n"
                f"Generated {num_generated_tokens} tokens\n"
            )

        # Update budget tracking and check stopping conditions
        cumulative_time, status, is_finished, new_ended_sequences = update_state(
            cumulative_time=cumulative_time,
            current_sample=sample.squeeze(-1),
            get_next_token_time=get_next_token_time,
            get_next_token_value=get_next_token_value,
            trajectory_labeler=trajectory_labeler,
        )
        # Update the output lengths for trajectories that have ended generation
        out_lengths[new_ended_sequences != ended_sequences] = num_generated_tokens
        ended_sequences = new_ended_sequences

        # Append new token and check for stopping
        out = torch.cat((out, sample), dim=-1)

        if max_tokens_budget and num_generated_tokens >= max_tokens_budget:
            is_finished = True

        if is_finished:
            break

    if status is not None:
        metadata = dict(labels=trajectory_labeler.get_labels(), status=status)

    out = out[:, t:]

    (out,) = unpack(out, ps, "* n")

    return out, out_lengths, metadata
