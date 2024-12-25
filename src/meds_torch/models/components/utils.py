from abc import ABC, abstractmethod
from collections.abc import Callable
from contextlib import nullcontext
from functools import wraps
from types import SimpleNamespace

import torch
from clinical_zeroshot_labeler.labeler import SequenceLabeler, WindowStatus
from omegaconf import DictConfig
from rich.progress import (
    BarColumn,
    Progress,
    ProgressColumn,
    Task,
    TaskProgressColumn,
    TextColumn,
    TimeRemainingColumn,
    filesize,
)
from rich.text import Text
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


class RateColumn(ProgressColumn):  # pragma: no cover
    """Renders human readable processing rate."""

    def render(self, task: "Task") -> Text:
        """Render the speed in iterations per second."""
        speed = task.finished_speed or task.speed
        if speed is None:
            return Text("", style="progress.percentage")
        unit, suffix = filesize.pick_unit_and_suffix(
            int(speed),
            ["", "×10³", "×10⁶", "×10⁹", "×10¹²"],
            1000,
        )
        data_speed = speed / unit
        return Text(f"{data_speed:.1f}{suffix} it/s", style="progress.percentage")


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


def slice_cache(cache, active_indices):
    """Slice a transformer KV cache to only active indices.

    Args:
        cache: Cache object with attn_intermediates containing cached_kv tensors
        active_indices: Indices of active (non-completed) sequences

    Returns:
        Modified cache with updated cached_kv tensors

    Examples:
        >>> import torch
        >>> from types import SimpleNamespace
        >>> # Create mock cache with 4 sequences
        >>> mock_cache = SimpleNamespace()
        >>> mock_cache.attn_intermediates = []
        >>> inter = SimpleNamespace()
        >>> inter.layer_type = "a"
        >>> # Create mock KV tensors [batch=4, heads=2, seq_len=3, head_dim=8]
        >>> k = torch.arange(192).float().reshape(4, 2, 3, 8)
        >>> v = torch.arange(192).float().reshape(4, 2, 3, 8) + 100
        >>> inter.cached_kv = [k, v]
        >>> mock_cache.attn_intermediates.append(inter)

        >>> # Test initial slicing - keep sequences 0 and 2
        >>> active = torch.tensor([0, 2])
        >>> sliced = slice_cache(mock_cache, active)
        >>> sliced.attn_intermediates[0].cached_kv[0].shape
        torch.Size([2, 2, 3, 8])
        >>> # Verify we kept the right sequences
        >>> torch.allclose(sliced.attn_intermediates[0].cached_kv[0][0], k[0])
        True
        >>> torch.allclose(sliced.attn_intermediates[0].cached_kv[0][1], k[2])
        True

        >>> # Test iterative slicing - now only second sequence
        >>> active2 = torch.tensor([1])  # Index 1 in already sliced cache
        >>> sliced2 = slice_cache(sliced, active2)
        >>> sliced2.attn_intermediates[0].cached_kv[0].shape
        torch.Size([1, 2, 3, 8])
        >>> # Verify we kept original sequence 2
        >>> torch.allclose(sliced2.attn_intermediates[0].cached_kv[0][0], k[2])
        True

        >>> # Test empty cache passes through
        >>> assert slice_cache(None, active) is None
    """
    if cache is None:
        return None

    for inter in cache.attn_intermediates:
        if inter.layer_type == "a":
            device = inter.cached_kv[0].device
            active_indices = active_indices.to(device)
            inter.cached_kv = [t[active_indices] for t in inter.cached_kv]

    return cache


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
        log_progress: bool = False,
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

            >>> # Test with time offset
            >>> time_offset = torch.tensor([1.0, 2.0])
            >>> tokens, lengths, meta = model.generate(
            ...     prompts,
            ...     mask,
            ...     time_offset_years=time_offset,
            ...     temperature=1.0
            ... )
            >>> assert tokens.shape[1] <= model.cfg.max_tokens_budget
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

        progress = (
            Progress(
                TextColumn("[progress.description]{task.description} {task.completed}"),
                BarColumn(),
                TaskProgressColumn(),
                RateColumn(),  # Shows speed of updates
                TimeRemainingColumn(),
                transient=True,
            )
            if log_progress
            else nullcontext()
        )

        with progress:
            if log_progress:
                tokens_task = progress.add_task(
                    "[cyan]Tokens Generated...",  # Static description
                    total=self.cfg.max_tokens_budget if self.cfg.max_tokens_budget is not None else None,
                )
                sequences_task = progress.add_task(
                    f"[green]Trajectories ({b} total)...",  # Static description with total
                    total=b,
                )

            while not is_finished:
                # Get indices relative to original batch
                orig_indices = (~ended_sequences).nonzero().squeeze(-1)
                if len(orig_indices.shape) == 0:  # Handle single active sequence
                    orig_indices = orig_indices.unsqueeze(0)

                # Track which indices are active in our currently sliced tensors
                active_indices = torch.arange(len(orig_indices), device=orig_indices.device)

                # Track sliding window for full batch
                x_full, cache_full, current_start_pos_full = self._track_sliding_window_generation(
                    out, max_seq_len, cache_kv, cache, transformer_decoder, seq_start_pos
                )

                # Select active sequences and their cache
                x = x_full[orig_indices]  # Use orig_indices for first slice from full batch
                current_start_pos = (
                    current_start_pos_full[orig_indices] if current_start_pos_full is not None else None
                )
                # Use active_indices for cache since it's already sliced
                cache = slice_cache(cache_full, active_indices) if cache_full is not None else None

                # Get next token predictions for active sequences only
                logits, new_cache = transformer_decoder(
                    x, return_intermediates=True, cache=cache, seq_start_pos=current_start_pos, **kwargs
                )

                # Map logits back to full batch size
                full_logits = torch.zeros(
                    (b, logits.shape[1], logits.shape[2]), device=logits.device, dtype=logits.dtype
                )
                full_logits[active_indices] = logits
                logits = full_logits

                # Update cache with pruned version
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
                if log_progress:
                    progress.update(
                        tokens_task,
                        advance=1,
                    )

                    completed_sequences = ended_sequences.int().sum().item()
                    progress.update(
                        sequences_task,
                        completed=completed_sequences,
                    )

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
                if (
                    self.cfg.max_tokens_budget is not None
                    and num_generated_tokens >= self.cfg.max_tokens_budget
                ):
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
            attn_layers=Decoder(dim=9, depth=1, heads=5, rotary_pos_emb=True),
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


from dataclasses import dataclass
from datetime import datetime

import numpy as np
import polars as pl
import torch


@dataclass
class TrajectoryBatch:
    """
    Initialize a batch of trajectories.

    Args:
        time (torch.Tensor): Tensor of shape (batch_size, sequence_length) containing days after prediction
            time. Values must be monotonically increasing within each sequence.
        code (torch.Tensor): Tensor of shape (batch_size, sequence_length) containing event code vocabulary
            indices
        mask (torch.Tensor): Tensor of shape (batch_size, sequence_length) indicates valid measurements/codes
        numeric_value (torch.Tensor): Tensor of shape (batch_size, sequence_length) containing numeric values
        numeric_value_mask (torch.Tensor): Tensor of shape (batch_size, sequence_length) indicating valid
            numeric values
        metadata_df (pl.DataFrame): DataFrame containing code vocabulary mapping with 'code' and
            'code/vocab_index' columns
        time_scale: scale of the time, by default it is 'Y' (years). Any numpy datetime units can be used,
            see https://numpy.org/doc/2.1/reference/arrays.datetime.html#datetime-units
    """

    time: torch.Tensor
    code: torch.Tensor
    mask: torch.Tensor
    numeric_value: torch.Tensor
    numeric_value_mask: torch.Tensor
    metadata_df: pl.DataFrame
    time_scale: str = "Y"

    def to_meds(self, prediction_time: list[datetime], subject_id: list[str | int]) -> pl.DataFrame:
        """Convert the trajectory batch to MEDS format.

        Args:
            prediction_time: List of prediction times for each trajectory in the batch
            subject_id: List of subject IDs for each trajectory in the batch

        Returns:
            pl.DataFrame: MEDS format DataFrame with columns:
                - time: Absolute timestamp of the event
                - code: The medical code string
                - numeric_value: The numeric value associated with the code (if any)
                - subject_id: ID of the subject
                - prediction_time: The prediction time for this trajectory

        Example:
        >>> metadata_df = pl.DataFrame({
        ...     'code': ['A1', 'A2', 'A3', 'A4', 'A5', 'A6'],
        ...     'code/vocab_index': [1, 2, 3, 4, 5, 6]
        ... })
        >>> batch = TrajectoryBatch(
        ...     time=torch.tensor([[0, .5, 2], [0, 3, 5]]),
        ...     code=torch.tensor([[1, 2, 3], [4, 5, 6]]),
        ...     mask=torch.tensor([[1, 1, 1], [1, 1, 0]]),
        ...     numeric_value=torch.tensor([[0.5, 1.0, 1.5], [2.0, 2.5, 3.0]]),
        ...     numeric_value_mask=torch.tensor([[1, 1, 0], [1, 0, 0]]),
        ...     metadata_df=metadata_df
        ... )
        >>> prediction_times = [datetime(2024, 1, 1), datetime(2024, 1, 1)]
        >>> subject_ids = [1, 2]
        >>> df = batch.to_meds(prediction_times, subject_ids)
        >>> df.sort("subject_id", "code")
        shape: (5, 6)
        ┌────────────┬─────────────────────┬─────────────────────┬──────┬──────────────────┬───────────────┐
        │ subject_id ┆ prediction_time     ┆ time                ┆ code ┆ code/vocab_index ┆ numeric_value │
        │ ---        ┆ ---                 ┆ ---                 ┆ ---  ┆ ---              ┆ ---           │
        │ i32        ┆ datetime[ns]        ┆ datetime[ns]        ┆ str  ┆ i64              ┆ f32           │
        ╞════════════╪═════════════════════╪═════════════════════╪══════╪══════════════════╪═══════════════╡
        │ 1          ┆ 2024-01-01 00:00:00 ┆ 2024-01-01 00:00:00 ┆ A1   ┆ 1                ┆ 0.5           │
        │ 1          ┆ 2024-01-01 00:00:00 ┆ 2024-07-01 14:54:36 ┆ A2   ┆ 2                ┆ 1.0           │
        │ 1          ┆ 2024-01-01 00:00:00 ┆ 2025-12-31 11:38:24 ┆ A3   ┆ 3                ┆ NaN           │
        │ 2          ┆ 2024-01-01 00:00:00 ┆ 2024-01-01 00:00:00 ┆ A4   ┆ 4                ┆ 2.0           │
        │ 2          ┆ 2024-01-01 00:00:00 ┆ 2026-12-31 17:27:36 ┆ A5   ┆ 5                ┆ NaN           │
        └────────────┴─────────────────────┴─────────────────────┴──────┴──────────────────┴───────────────┘
        """
        if len(prediction_time) != len(subject_id) or len(prediction_time) != self.time.shape[0]:
            raise ValueError("Number of prediction times and subject IDs must match batch size")
        schema = {
            "time": pl.Datetime,
            "code": self.metadata_df.schema["code/vocab_index"],
            "numeric_value": pl.Float32,
            "subject_id": pl.Int32,
            "prediction_time": pl.Datetime,
        }

        # Pre-filter masked data using torch operations for efficiency
        batch_indices, seq_indices = torch.where(self.mask)
        if len(batch_indices) == 0:
            return pl.DataFrame(schema=schema)

        # Gather only the valid data points using index tensors
        time_values = self.time[batch_indices, seq_indices]
        code_values = self.code[batch_indices, seq_indices]
        numeric_values = self.numeric_value[batch_indices, seq_indices]
        numeric_value_masks = self.numeric_value_mask[batch_indices, seq_indices]

        # Convert to numpy for faster processing
        time_array = time_values.numpy()
        code_array = code_values.numpy().astype(np.int32)
        numeric_value_array = numeric_values.numpy()
        numeric_value_mask_array = numeric_value_masks.numpy()
        batch_indices = batch_indices.numpy()

        # Create arrays for prediction times and subject IDs
        pred_times = np.array(prediction_time, dtype="datetime64[ns]")[batch_indices]
        subject_ids = np.array(subject_id)[batch_indices].astype(np.int32)

        # Parallel processing using numpy vectorization
        time_deltas = time_array * np.timedelta64(1, self.time_scale).astype("timedelta64[ns]")
        timestamps = pred_times + time_deltas

        # Create the final dictionary with only valid data
        data_dict = {
            "time": timestamps,
            "code": code_array,
            "numeric_value": np.where(
                numeric_value_mask_array, numeric_value_array.astype(np.float32), np.nan
            ),
            "subject_id": subject_ids,
            "prediction_time": pred_times,
        }

        # Create DataFrame directly from the efficient dictionary
        df = pl.from_dict(data_dict, schema=schema)

        # Convert code vocab indexes to strings using the metadata mapping
        df = df.join(
            self.metadata_df.select("code", "code/vocab_index"), left_on="code", right_on="code/vocab_index"
        ).rename({"code_right": "code", "code": "code/vocab_index"})

        return df["subject_id", "prediction_time", "time", "code", "code/vocab_index", "numeric_value"]


def get_time_days_delta(
    pred_times: list[datetime], input_end_times: list[datetime], device: torch.device = torch.device("cpu")
) -> torch.Tensor:
    """
    Convert lists of prediction times and generation start times to the continuous number
    of days (including fractional days) between each pair.

    Args:
        pred_times (List[datetime]): List of prediction times
        input_end_times (List[datetime]): List of generation start times, one for each prediction time
        device (torch.device, optional): Device to place the output tensor on. Defaults to CPU.

    Returns:
        torch.Tensor: Tensor of shape (len(pred_times),) containing the number of days (including
            fractional parts) after each generation start time for each prediction time.

    Examples:
        >>> from datetime import datetime
        >>> import torch
        >>> pred_times = [
        ...     datetime(2022, 1, 1, 12, 0),  # Noon on Jan 1
        ...     datetime(2022, 1, 2, 0, 0)    # Midnight on Jan 2
        ... ]
        >>> end_times = [
        ...     datetime(2022, 1, 1, 0, 0),   # Midnight on Jan 1
        ...     datetime(2022, 1, 1, 0, 0)    # Midnight on Jan 1
        ... ]
        >>> get_time_days_delta(pred_times, end_times)
        tensor([0.5000, 1.0000])

        >>> pred_times = [
        ...     datetime(2022, 1, 1, 6, 0),   # 6 AM on Jan 1
        ...     datetime(2022, 1, 1, 18, 0)   # 6 PM on Jan 1
        ... ]
        >>> end_times = [
        ...     datetime(2022, 1, 1, 0, 0),   # Midnight on Jan 1
        ...     datetime(2022, 1, 1, 12, 0)   # Noon on Jan 1
        ... ]
        >>> get_time_days_delta(pred_times, end_times)
        tensor([0.2500, 0.2500])
    """
    if len(pred_times) != len(input_end_times):
        raise ValueError(
            f"Length mismatch: pred_times has {len(pred_times)} elements "
            f"but input_end_times has {len(input_end_times)} elements"
        )

    # Convert datetime lists to numpy arrays of timestamps
    pred_timestamps = np.array([dt.timestamp() for dt in pred_times])
    end_timestamps = np.array([dt.timestamp() for dt in input_end_times])

    # Calculate time differences in seconds
    time_deltas_seconds = pred_timestamps - end_timestamps

    # Convert to days (86400 seconds per day)
    days_delta = time_deltas_seconds / 86400

    return torch.tensor(days_delta, device=device, dtype=torch.float32)
