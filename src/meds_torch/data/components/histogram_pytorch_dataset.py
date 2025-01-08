from datetime import datetime
from enum import StrEnum

import numpy as np
import torch
from mixins import SeedableMixin
from omegaconf import DictConfig, OmegaConf, open_dict

from meds_torch.data.components.pytorch_dataset import DummyConfig, PytorchDataset


class TokenInsertionStrategy(StrEnum):
    TOKEN_COUNT = "token_count"
    TIME_BINS = "time_bins"


def topk(x: torch.Tensor, k: torch.Tensor) -> torch.Tensor:
    """
    Creates a mask where values are 0 for the top-k elements in each batch and 1 elsewhere.

    Args:
        x : tensor of shape [B, L] containing values to find top-k elements
        k : tensor of shape [B, 1] containing the number of top elements to find for each batch

    Returns:
        final_mask: tensor of shape [B,L] where final_mask[b,i] = 0 if x[b,i] is in
                   the k[b] biggest values of x[b,:], else final_mask[b,i] = 1
    """
    B, L = x.shape  # batchsize, list size

    # Get indices sorted in descending order
    _, indices_des = torch.sort(x, dim=-1, descending=True)

    # Create range mask [1, L] and repeat it B times
    mask = torch.arange(L, device=x.device).unsqueeze(0).expand(B, -1)
    k_expanded = k.expand(-1, L)
    mask = mask < k_expanded

    # Create one-hot encoding and apply mask
    one_hot = torch.nn.functional.one_hot(indices_des, num_classes=L).float()
    one_hot = one_hot * mask.unsqueeze(-1)

    # Sum along the appropriate dimension to get final mask
    final_mask = one_hot.sum(dim=1)

    # Flip the mask (0 for top-k, 1 for others)
    return final_mask


def get_time_bin_indices(time_deltas, time_bin_size) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Gets the indices of the

    Args:
        time_deltas (np.ndarray): Shape [L] array of time deltas
        time_bin_size (float): The size of the time bin to use for inserting H and O tokens

    Returns:
        _type_: _description_

    Examples:
    >>> import numpy as np
    >>> # Test case 1: Multiple codes with small time delta
    >>> time_deltas = [0,1,0,0,0]
    >>> left_indices, right_indices, empty_indices, intervals = get_time_bin_indices(time_deltas, 1.0)
    >>> np.array_equal(left_indices, [1,0])
    True
    >>> np.array_equal(right_indices, [5,1])
    True
    >>> np.array_equal(empty_indices, [False,False])
    True
    >>> np.array_equal(intervals, [1., 0., -1.])
    True

    >>> # Test case 2: Single code
    >>> time_deltas = [0]
    >>> left_indices, right_indices, empty_indices, intervals = get_time_bin_indices(time_deltas, 1.0)
    >>> np.array_equal(left_indices, [0])
    True
    >>> np.array_equal(right_indices, [1])
    True
    >>> np.array_equal(empty_indices, [False])
    True
    >>> np.array_equal(intervals, [0., -1.])
    True

    >>> # Test case 3: Larger time gap with empty bins
    >>> time_deltas = [0,10,0]
    >>> left_indices, right_indices, empty_indices, intervals = get_time_bin_indices(time_deltas, 1.0)
    >>> np.array_equal(left_indices, [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0])
    True
    >>> np.array_equal(right_indices, [3, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1])
    True
    >>> np.array_equal(empty_indices, [False, True, True, True, True, True, True, True, True, True, False])
    True
    >>> np.array_equal(intervals, [10., 9., 8., 7., 6., 5., 4., 3., 2., 1., 0., -1.])
    True
    """
    cum_time = np.cumsum(time_deltas)
    # get intervals that constitute time bins (with the last window being at cum_time[-1])
    intervals = np.arange(cum_time[-1], -2 * time_bin_size, -time_bin_size)
    # get the indices of the intervals that are within the time bins
    left_indices = np.searchsorted(cum_time, intervals[1:], side="right")
    right_indices = np.searchsorted(cum_time, intervals[:-1], side="right")
    empty_indices = left_indices == right_indices
    return left_indices, right_indices, empty_indices, intervals


def drop_empty_bins(left_indices, right_indices, empty_indices):
    cum_empty_count = np.cumsum(empty_indices)[~empty_indices]
    return left_indices[~empty_indices], right_indices[~empty_indices], cum_empty_count


def insert_h_o_tokens_with_time_bins(
    codes: np.ndarray, time_deltas: np.ndarray, time_bin_size: float, h_token: int, o_token: int
) -> tuple[np.ndarray, np.ndarray]:
    """Inserts H and O tokens into the codes array at the appropriate positions based on the time bins.

    Args:
        codes (np.ndarray): The original codes array.
        time_deltas (np.ndarray): The time deltas array.
        time_bin_size (float): The size of the time bins.
        h_token (int): The token to insert for H.
        o_token (int): The token to insert for O.

    Returns:
        tuple[np.ndarray, np.ndarray]: The resulting array with H and O tokens inserted and the
        number of empty bins.

    Examples:
        >>> import numpy as np
        >>> H_TOKEN = 3
        >>> O_TOKEN = 4
        >>> # Test case 1: Multiple codes with small time delta
        >>> codes = np.array([1,2,1,1,1])
        >>> time_deltas = [0,1,0,0,0]
        >>> result, empty_count = insert_h_o_tokens_with_time_bins(codes, time_deltas, 1.0, H_TOKEN, O_TOKEN)
        >>> np.array_equal(result, [3,4,1,3,4,2,1,1,1,3])
        True
        >>> np.array_equal(empty_count, [0,0,0,0,0,0,0,0,0,0])
        True

        >>> # Test case 2: Single code
        >>> codes = np.array([1])
        >>> time_deltas = [0]
        >>> result, empty_count = insert_h_o_tokens_with_time_bins(codes, time_deltas, 1.0, H_TOKEN, O_TOKEN)
        >>> np.array_equal(result, [3,4,1,3])
        True
        >>> np.array_equal(empty_count, [0,0,0,0])
        True

        >>> # Test case 3: Larger time gap
        >>> codes = np.array([1,2,2])
        >>> time_deltas = [0,10,0]
        >>> result, empty_count = insert_h_o_tokens_with_time_bins(codes, time_deltas, 1.0, H_TOKEN, O_TOKEN)
        >>> np.array_equal(result, [3,4,1,3,4,2,2,3])
        True
        >>> np.array_equal(empty_count, [0,0,0,0,9,9,9,9])
        True
    """
    left_indices, right_indices, empty_indices, _ = get_time_bin_indices(time_deltas, time_bin_size)
    left_indices, right_indices, empty_count = drop_empty_bins(left_indices, right_indices, empty_indices)

    # Calculate output length based on token_bin_size
    num_prepended_h_o_tokens = len(left_indices)  # compute all non-empty bins
    output_length = len(codes) + 2 * num_prepended_h_o_tokens + 1  # +1 for the last h_token

    # Create output array with zeros
    result = np.zeros(output_length, dtype=codes.dtype)

    # Calculate positions for H and O tokens
    token_positions = left_indices[::-1] + np.arange(0, len(left_indices)) * 2
    h_positions = np.hstack([token_positions, output_length - 1])
    o_positions = token_positions + 1

    # Insert tokens
    result[h_positions] = h_token
    result[o_positions] = o_token

    # Insert original codes in remaining positions
    data_positions = np.ones(output_length, dtype=bool)
    data_positions[h_positions] = False
    data_positions[o_positions] = False
    result[data_positions] = codes

    num_tokens_for_each_interval = h_positions[1:] - h_positions[:-1]
    num_tokens_for_each_interval[0] += 1
    inserted_empty_count = np.repeat(empty_count, num_tokens_for_each_interval)

    return result, inserted_empty_count


def compute_cumulative_count(codes: np.ndarray, vocab_size: int) -> np.ndarray:
    """
    Compute the cumulative count of the codes.

    Args:
        codes (np.ndarray): Shape [L] array of codes to compute the cumulative count for.
        vocab_size (int): The size of the vocabulary.

    Returns:
        np.ndarray: Shape [vocab_size+1, L] array of the cumulative count of the codes.

    Examples:
        >>> import numpy as np
        >>> # Test case 1: Multiple different tokens
        >>> codes = [1,2,1]
        >>> result = compute_cumulative_count(codes, vocab_size=2)
        >>> expected = np.array([[0,1,0], [0,1,1], [0,2,1]])
        >>> np.array_equal(result, expected)
        True

        >>> # Test case 2: Single token
        >>> codes = [1]
        >>> result = compute_cumulative_count(codes, vocab_size=2)
        >>> expected = np.array([[0,1,0]])
        >>> np.array_equal(result, expected)
        True

        >>> # Test case 3: Repeated tokens
        >>> codes = [1,1]
        >>> result = compute_cumulative_count(codes, vocab_size=2)
        >>> expected = np.array([[0,1,0], [0,2,0]])
        >>> np.array_equal(result, expected)
        True
    """
    one_hot = np.eye(vocab_size + 1)[codes]
    return np.cumsum(one_hot, axis=0)


def insert_h_o_tokens(codes: np.ndarray, token_bin_size: float, h_token: int, o_token: int) -> np.ndarray:
    """Insert H and O tokens into a sequence to mark bin boundaries.

    Args:
        codes (np.ndarray): Shape [L] array of codes to insert H and O tokens into.
        token_bin_size (float): The size of the time bin to use for inserting H and O tokens.
        h_token (int): The token to insert for H.
        o_token (int): The token to insert for O.

    Returns:
        np.ndarray: Shape [L + num_prepended_h_o_tokens + 1] array of codes with H and O tokens inserted.

    Examples:
        >>> import numpy as np
        >>> H_TOKEN = 3
        >>> O_TOKEN = 4
        >>> # Test case 1: Multiple tokens
        >>> codes = [1,2,1,1,1]
        >>> result = insert_h_o_tokens(codes, 2, H_TOKEN, O_TOKEN)
        >>> expected = [3,4,1,2,3,4,1,1,3,4,1,3]
        >>> np.array_equal(result, expected)
        True

        >>> # Test case 2: Single token
        >>> codes = [1]
        >>> result = insert_h_o_tokens(codes, 2, H_TOKEN, O_TOKEN)
        >>> expected = [3,4,1,3]
        >>> np.array_equal(result, expected)
        True
    """
    codes = np.array(codes)
    # Calculate output length based on token_bin_size
    num_prepended_h_o_tokens = ((len(codes) + token_bin_size - 1) // token_bin_size) * 2
    output_length = len(codes) + num_prepended_h_o_tokens + 1  # +1 for the last h_token

    # Create output array with zeros
    result = np.zeros(output_length, dtype=codes.dtype)

    # Calculate positions for H and O tokens
    token_positions = np.arange(0, output_length - 1, token_bin_size + 2)
    h_positions = np.hstack([token_positions, output_length - 1])
    o_positions = token_positions + 1

    # Insert tokens
    result[h_positions] = h_token
    result[o_positions] = o_token

    # Insert original codes in remaining positions
    data_positions = np.ones(output_length, dtype=bool)
    data_positions[h_positions] = False
    data_positions[o_positions] = False
    result[data_positions] = codes

    return result


def get_segment_indices(inserted_codes: np.ndarray, o_token: int) -> tuple[np.ndarray, np.ndarray]:
    """Generate indices for computing histograms between O tokens in a sequence.

    For a sequence containing special O tokens that mark segment boundaries,
    generates two arrays of indices that can be used to efficiently compute
    histograms between each position and its next O token.

    Args:
        inserted_codes (np.ndarray): Sequence containing regular tokens and O tokens,
            where O tokens mark segment boundaries

    Returns:
        Tuple[np.ndarray, np.ndarray]: Two arrays (i_indices, end_indices) where:
            - i_indices: Array containing all positions i in the sequence
            - end_indices: Array of same length as i_indices, containing for each
              position i the position of its next O token

    Examples:
        >>> H_TOKEN = 3
        >>> O_TOKEN = 4
        >>> # Test case 1: Multiple tokens
        >>> codes = [1,2,1]
        >>> inserted_codes = insert_h_o_tokens(codes, 2, H_TOKEN, O_TOKEN)
        >>> i_indices, end_indices = get_segment_indices(inserted_codes, O_TOKEN)
        >>> np.array_equal(i_indices, [0, 1, 2, 3, 4, 5, 6])
        True
        >>> np.array_equal(end_indices, [1, 5, 5, 5, 5, 7, 7])
        True

        >>> # Test case 2: Single token
        >>> codes = [1]
        >>> inserted_codes = insert_h_o_tokens(codes, 2, H_TOKEN, O_TOKEN)
        >>> i_indices, end_indices = get_segment_indices(inserted_codes, O_TOKEN)
        >>> np.array_equal(i_indices, [0, 1, 2])
        True
        >>> np.array_equal(end_indices, [1, 3, 3])
        True

        >>> # Test case 3: Two identical tokens
        >>> codes = [1,1]
        >>> inserted_codes = insert_h_o_tokens(codes, 2, H_TOKEN, O_TOKEN)
        >>> i_indices, end_indices = get_segment_indices(inserted_codes, O_TOKEN)
        >>> np.array_equal(i_indices, [0, 1, 2, 3])
        True
        >>> np.array_equal(end_indices, [1, 4, 4, 4])
        True
    """
    # Get o token locations
    o_token_indices = np.where(inserted_codes == o_token)[0]
    o_token_indices = np.concatenate([[0], o_token_indices, [len(inserted_codes) - 1]])
    starts = o_token_indices[:-1]
    ends = o_token_indices[1:]

    # Create arrays for vectorized difference computation
    # Compute lengths of each segment
    segment_lengths = ends - starts
    # Create array containing the number of indices we need for each start-end pair
    i_offsets = np.arange(max(segment_lengths))

    # Create a mask for valid indices
    mask = i_offsets < segment_lengths[:, None]

    # Create the final i_indices and end_indices arrays
    i_indices = starts[:, None] + i_offsets
    i_indices = i_indices[mask]
    end_indices = np.repeat(ends, segment_lengths)

    return i_indices, end_indices


def compute_count_histogram(inserted_codes: np.ndarray, vocab_size: int, o_token: int) -> np.ndarray:
    """Compute count histograms for a sequence with special H and O tokens marking bin boundaries.

    For each position i in the sequence, computes a histogram of all tokens between i and
    the next O token. Uses vectorized operations for efficiency.

    Args:
        inserted_codes (np.ndarray): Input sequence of token indices to process
        vocab_size (int): Size of vocabulary for the original tokens
        o_token (int): The token to insert for O.

    Returns:
        np.ndarray: Array of shape [L, vocab_size+1] where L is the length of the sequence
            with inserted H,O tokens. Each row contains the count histogram of tokens
            between that position and the next O token. The O_TOKEN column is always 1.

    Examples:
        >>> import numpy as np
        >>> H_TOKEN = 3
        >>> O_TOKEN = 4
        >>> vocab_size = 4  # 2 for original vocab, 2 for H and O tokens

        >>> # Test case 1: Multiple different tokens
        >>> codes = [1,2,1]
        >>> token_bin_size = 2
        >>> inserted_codes = insert_h_o_tokens(codes, token_bin_size, H_TOKEN, O_TOKEN)
        >>> result = compute_count_histogram(inserted_codes, vocab_size, O_TOKEN)
        >>> expected = np.array([
        ...     [0,0,0,0,1], # at h token
        ...     [0,1,1,1,1], # at first token
        ...     [0,0,1,1,1], # at second token
        ...     [0,0,0,1,1], # at third token
        ...     [0,0,0,0,1], # at h token
        ...     [0,1,0,1,1], # at o token
        ...     [0,0,0,1,1], # at last token
        ...     [0,0,0,0,1]  # at h token
        ... ])
        >>> np.array_equal(result, expected)
        True

        >>> # Test case 2: Single token
        >>> codes = [1]
        >>> inserted_codes = insert_h_o_tokens(codes, token_bin_size, H_TOKEN, O_TOKEN)
        >>> result = compute_count_histogram(inserted_codes, vocab_size, O_TOKEN)
        >>> expected = np.array([
        ...     [0,0,0,0,1], # at h token
        ...     [0,1,0,1,1], # at o token
        ...     [0,0,0,1,1], # at token
        ...     [0,0,0,0,1]  # at h token
        ... ])
        >>> np.array_equal(result, expected)
        True

        >>> # Test case 3: Two identical tokens
        >>> codes = [1,1]
        >>> inserted_codes = insert_h_o_tokens(codes, token_bin_size, H_TOKEN, O_TOKEN)
        >>> result = compute_count_histogram(inserted_codes, vocab_size, O_TOKEN)
        >>> expected = np.array([
        ...     [0,0,0,0,1], # at h token
        ...     [0,2,0,1,1], # at o token - full histogram
        ...     [0,1,0,1,1], # at first token
        ...     [0,0,0,1,1], # at second token
        ...     [0,0,0,0,1]  # at h token
        ... ])
        >>> np.array_equal(result, expected)
        True
    """

    # Compute cumulative count
    cumulative_count = compute_cumulative_count(inserted_codes, vocab_size)

    # Get o token locations
    i_indices, end_indices = get_segment_indices(inserted_codes, o_token)

    # Compute all differences in parallel
    histograms = np.zeros_like(cumulative_count, dtype=int)
    histograms[i_indices] = cumulative_count[end_indices] - cumulative_count[i_indices]

    # Set O_TOKEN column to 1
    histograms[:, o_token] = 1

    return histograms


def fill_dummy_config(cfg: DummyConfig):
    cfg = OmegaConf.structured(cfg)
    with open_dict(cfg):
        cfg.vocab_size = 6
        cfg.token_bin_size = 2
        cfg.H_TOKEN = cfg.vocab_size - 2
        cfg.O_TOKEN = cfg.vocab_size - 1
        cfg.token_insertion_strategy = "token_count"
    return cfg


class HistogramPytorchDataset(PytorchDataset):
    """A PyTorch Dataset class that computes histograms over future time intervals or token counts.

    Examples:
    First let's look at what a batch looks like from this dataset
        >>> import tempfile
        >>> from pathlib import Path
        >>> from meds_torch.data.components.pytorch_dataset import create_dummy_dataset
        >>> # Test initialization without task
        >>> tmp_dir_obj = tempfile.TemporaryDirectory()
        >>> tmp_dir = tmp_dir_obj.name
        >>> config = create_dummy_dataset(tmp_dir)
        >>> config = fill_dummy_config(config)
        >>> config.task_label_path = None
        >>> config.task_name = None
        >>> config.do_include_prediction_time = False
        >>> config.postpend_eos_token = False
        >>> dataset = HistogramPytorchDataset(config, split='train')
        >>> print(f"Dataset size: {len(dataset)}")
        Dataset size: 3
        >>> print(f"Has task: {dataset.has_task}")
        Has task: False
        >>> # Test data loading
        >>> sample = dataset[0]
        >>> # Let's see what keys are in a batch dictionary from the dataset
        >>> for key in sorted(list(sample.keys())): print(f"{key}")
        cum_sum
        dynamic
        end_idx
        end_time
        start_idx
        start_time
        static_indices
        static_values
        subject_id
        >>> batch = dataset.collate([sample, dataset[1]])
        >>> print(list(batch.keys()))
        ['code', 'mask', 'histogram', 'start_idx', 'end_idx', 'start_time', 'end_time', 'subject_id']
        >>> print(batch['code'].shape)
        torch.Size([2, 21])
        >>> print(batch['mask'].shape)
        torch.Size([2, 21])
        >>> print(batch['histogram'].shape)
        torch.Size([2, 21, 7])


    Now let's test initialization with task
        # >>> # Test initialization with task
        # >>> tmp_dir_obj = tempfile.TemporaryDirectory()
        # >>> tmp_dir = tmp_dir_obj.name
        # >>> config = create_dummy_dataset(tmp_dir)
        # >>> dataset = HistogramPytorchDataset(config, split='train')
        # >>> print(f"Dataset size: {len(dataset)}")
        # Dataset size: 3
        # >>> print(f"Has task: {dataset.has_task}")
        # Has task: True
        # >>> print(f"First subject label: {dataset.labels[0]}")  # Subject IDs start at 1
        # First subject label: 0
    """

    def __init__(self, cfg: DictConfig, split: str):
        super().__init__(cfg, split)
        self.cfg = cfg
        if self.cfg.postpend_eos_token:
            raise NotImplementedError("EOS token not supported for HistogramPytorchDataset")

    @SeedableMixin.WithSeed
    def _seeded_getitem(self, idx: int) -> dict:
        """Get a randomly windowed item from the dataset.

        Args:
            idx (int): Index of the item to retrieve.

        Returns:
            dict: A dictionary containing randomly generated windows of the sequence.
        """
        out = super()._seeded_getitem(idx)
        codes = out["dynamic"].tensors["dim0/code"]
        time_deltas = out["dynamic"].tensors["dim0/time_delta_days"]
        if self.cfg.token_insertion_strategy == TokenInsertionStrategy.TOKEN_COUNT:
            inserted_codes = insert_h_o_tokens(
                codes, self.cfg.token_bin_size, self.cfg.H_TOKEN, self.cfg.O_TOKEN
            )
        elif self.cfg.token_insertion_strategy == TokenInsertionStrategy.TIME_BINS:
            inserted_codes = insert_h_o_tokens_with_time_bins(
                codes, time_deltas, self.cfg.time_bin_size, self.cfg.H_TOKEN, self.cfg.O_TOKEN
            )
        else:
            raise ValueError(
                f"Invalid token insertion strategy: {self.cfg.token_insertion_strategy}, "
                f"should be one of {TokenInsertionStrategy}"
            )
        histogram = compute_count_histogram(inserted_codes, self.cfg.vocab_size, self.cfg.O_TOKEN)

        out["cum_sum"] = dict(
            codes=inserted_codes,
            histogram=histogram,
        )

        return out

    def collate(self, batch: list[dict]) -> dict:
        """Combines a batch of data points into a single, tensorized batch.

        The collated output is a fully tensorized and padded dictionary, ready for input into an
        `input_encoder`. This method uses the JointNestedRaggedTensorDict API to collate and pad the data.

        Args:
            batch (list[dict]): A list of dictionaries, each representing a single sample as
                returned by the __getitem__ method.

        Returns:
            dict: A dictionary containing the collated batch data.
        """
        codes = [torch.as_tensor(item["cum_sum"]["codes"], dtype=torch.long) for item in batch]
        masks = [torch.ones_like(code) for code in codes]
        histograms = [torch.as_tensor(item["cum_sum"]["histogram"], dtype=torch.float32) for item in batch]
        tensorized = {}
        tensorized["code"] = torch.nn.utils.rnn.pad_sequence(codes, batch_first=True)
        tensorized["mask"] = torch.nn.utils.rnn.pad_sequence(masks, batch_first=True)
        tensorized["histogram"] = torch.nn.utils.rnn.pad_sequence(histograms, batch_first=True)
        # Add task labels to batch
        for k in batch[0].keys():
            if k not in ("dynamic", "static_values", "static_indices", "static_mask", "cum_sum"):
                if isinstance(batch[0][k], datetime):
                    tensorized[k] = [item[k] for item in batch]
                else:
                    if k == "boolean_value" and batch[0][k] is None:
                        tensorized[k] = torch.Tensor([False] * len(batch))
                    else:
                        tensorized[k] = torch.Tensor([item[k] for item in batch])
        return tensorized
