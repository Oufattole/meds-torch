from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from mixins import SeedableMixin, TimeableMixin
from nested_ragged_tensors.ragged_numpy import JointNestedRaggedTensorDict

from meds_torch.data.components.pytorch_dataset import create_dummy_dataset  # noqa
from meds_torch.data.components.pytorch_dataset import (
    PytorchDataset,
    SubsequenceSamplingStrategy,
)


def get_nonnan_indices(arr: torch.Tensor):
    """Returns (row, col) indices of non-NaN values in row-major order

    >>> x = torch.tensor([[1., float('nan')], [float('nan'), 2.]])
    >>> get_nonnan_indices(x)
    (tensor([0, 1]), tensor([0, 1]))
    """
    rows, cols = torch.nonzero(~torch.isnan(arr), as_tuple=True)
    order = torch.argsort(rows * arr.shape[1] + cols)
    return rows[order], cols[order]


def subsample_subject_data(
    subject_data: JointNestedRaggedTensorDict,
    max_seq_len: int,
    sampling_strategy: SubsequenceSamplingStrategy,
    do_flatten_tensors: bool = True,
    global_st: int = 0,
) -> tuple[JointNestedRaggedTensorDict, int, int]:
    """Subsample subject data based on maximum sequence length and sampling strategy.

    This function handles subsampling for both flattened and nested tensor structures.

    Args:
        subject_data: Input tensor dictionary containing the sequence data
        max_seq_len: Maximum allowed sequence length
        sampling_strategy: Strategy for selecting subsequence (RANDOM, TO_END, FROM_START)
        do_flatten_tensors: Whether to flatten tensors before subsampling
        global_st: Starting index offset for maintaining global indexing

    Returns:
        tuple containing:
        - Subsampled tensor dictionary
        - New global start index
        - New global end index

    Examples:
        >>> import numpy as np
        >>> np.random.seed(42)
        >>> # Create sample nested data
        >>> tensors = {
        ...     "code": [[1,2],[3,4],[5,6],[7,8,9,10],[11,12]],
        ...     "time": [0,1,2,3,4],
        ... }
        >>> data = JointNestedRaggedTensorDict(raw_tensors=tensors)
        >>> # Test FROM_START strategy without flattening
        >>> subsampled, st, end, flat_st, flat_end = subsample_subject_data(
        ...     data, max_seq_len=2,
        ...     sampling_strategy=SubsequenceSamplingStrategy.FROM_START,
        ...     do_flatten_tensors=False
        ... )
        >>> subsampled.tensors["dim1/code"].tolist()
        [1, 2, 3, 4]
        >>> subsampled.tensors["dim0/time"].tolist()
        [0, 1]
        >>> st, end
        (0, 2)
        >>> flat_st, flat_end
        (None, None)

        >>> # Test TO_END strategy with flattening
        >>> data = JointNestedRaggedTensorDict(raw_tensors=tensors)
        >>> subsampled, st, end, flat_st, flat_end = subsample_subject_data(
        ...     data, max_seq_len=4,
        ...     sampling_strategy=SubsequenceSamplingStrategy.TO_END,
        ...     do_flatten_tensors=True
        ... )
        >>> subsampled.tensors["dim0/code"].tolist()
        [9, 10, 11, 12]
        >>> subsampled.tensors["dim0/time"].tolist()
        [0, 0, 4, 0]
        >>> st, end
        (3, 5)
        >>> flat_st, flat_end
        (8, 12)

        >>> # Test TO_END strategy
        >>> data = JointNestedRaggedTensorDict(raw_tensors=tensors)
        >>> subsampled, st, end, flat_st, flat_end = subsample_subject_data(
        ...     data, max_seq_len=2,
        ...     sampling_strategy=SubsequenceSamplingStrategy.TO_END,
        ...     do_flatten_tensors=False,
        ... )
        >>> st, end
        (3, 5)
        >>> flat_st, flat_end
        (None, None)

        >>> # Test RANDOM strategy
        >>> data = JointNestedRaggedTensorDict(raw_tensors=tensors)
        >>> subsampled, st, end, flat_st, flat_end = subsample_subject_data(
        ...     data, max_seq_len=2,
        ...     sampling_strategy=SubsequenceSamplingStrategy.RANDOM,
        ...     do_flatten_tensors=True,
        ... )
        >>> len(subsampled.tensors["dim0/code"]) == 2
        True
        >>> flat_st, flat_end
        (6, 8)
    """
    seq_len = len(subject_data)
    flatten_start_offset = None
    flatten_end_offset = None

    if do_flatten_tensors:
        # Store original lengths for each time step before flattening
        cum_lens = subject_data.tensors["dim1/bounds"]

        subject_data = subject_data.flatten()
        seq_len = len(subject_data)
        if seq_len > max_seq_len:
            match sampling_strategy:
                case SubsequenceSamplingStrategy.RANDOM:
                    start_offset = np.random.choice(seq_len - max_seq_len)
                case SubsequenceSamplingStrategy.TO_END:
                    start_offset = seq_len - max_seq_len
                case SubsequenceSamplingStrategy.FROM_START:
                    start_offset = 0
                case _:
                    raise ValueError(f"Invalid subsequence sampling strategy {sampling_strategy}!")
        else:
            start_offset = 0
        end = min(seq_len, start_offset + max_seq_len)
        subject_data = subject_data[start_offset:end]

        # Map flattened indices back to original time indices
        new_global_st = global_st + np.searchsorted(cum_lens, start_offset, side="right").item()
        new_global_end = global_st + np.searchsorted(cum_lens, end, side="right").item()
        flatten_start_offset = start_offset
        flatten_end_offset = end
    else:
        if seq_len <= max_seq_len:
            return subject_data, global_st, global_st + seq_len
        match sampling_strategy:
            case SubsequenceSamplingStrategy.RANDOM:
                start_offset = np.random.choice(seq_len - max_seq_len)
            case SubsequenceSamplingStrategy.TO_END:
                start_offset = seq_len - max_seq_len
            case SubsequenceSamplingStrategy.FROM_START:
                start_offset = 0
            case _:
                raise ValueError(f"Invalid subsequence sampling strategy {sampling_strategy}!")

        end = min(seq_len, start_offset + max_seq_len)
        subject_data = subject_data[start_offset:end]

        new_global_st = global_st + start_offset
        new_global_end = new_global_st + len(subject_data)

    return subject_data, new_global_st, new_global_end, flatten_start_offset, flatten_end_offset


class CumSumPytorchDataset(PytorchDataset):
    """A PyTorch Dataset class that additionally serves the cumulative sum of code counts for each time

    Args:
        cfg (DictConfig): Configuration options for the dataset.
        split (str): The data split to use (e.g., 'train', 'validation', 'test').
        min_window_size (int): Minimum size of generated windows.
        max_window_size (int): Maximum size of generated windows.
        n_windows (int): Number of windows to generate for each sample.
    """

    @TimeableMixin.TimeAs
    def load_subject_dynamic_data(self, idx: int):
        """Loads and returns the dynamic data slice for a given subject index, with subject ID and time range.

        Args:
            idx (int): Index of the subject in the dataset index

        Returns:
            tuple: (subject_dynamic_data, subject_id, st, end) where:
                - subject_dynamic_data is a JointNestedRaggedTensorDict containing the dynamic data
                - subject_id is the ID of the subject
                - st is the start time index
                - end is the end time index

        Examples:
        >>> import tempfile
        >>> from pathlib import Path
        >>> import polars as pl
        >>> from nested_ragged_tensors.ragged_numpy import JointNestedRaggedTensorDict

        >>> # Create a test dataset and initialize the PytorchDataset
        >>> with tempfile.TemporaryDirectory() as tmp_dir:
        ...     config = create_dummy_dataset(tmp_dir)
        ...     dataset = CumSumPytorchDataset(config, split='train')
        ...
        ...     # Test loading dynamic data for first subject
        ...     dynamic_data, subject_id, st, end = dataset.load_subject_dynamic_data(0)
        ...     print(f"Subject ID: {subject_id}")
        ...     print(f"Time range: {st} to {end}")
        ...     print(f"Dynamic data keys: {sorted(dynamic_data.tensors.keys())}")
        Subject ID: 0
        Time range: 0 to 4
        Dynamic data keys: ['dim0/time_delta_days', 'dim1/bounds', 'dim1/code', 'dim1/numeric_value']

        >>> # Test loading dynamic data for second subject
        >>> with tempfile.TemporaryDirectory() as tmp_dir:
        ...     config = create_dummy_dataset(tmp_dir)
        ...     dataset = CumSumPytorchDataset(config, split='train')
        ...
        ...     # Load second subject
        ...     dynamic_data, subject_id, st, end = dataset.load_subject_dynamic_data(1)
        ...     print(f"Subject ID: {subject_id}")
        ...     print(f"Time range: {st} to {end}")
        ...     # Verify data structure
        ...     print(f"Has numeric values: {'dim1/numeric_value' in dynamic_data.tensors}")
        ...     print(f"Has time deltas: {'dim0/time_delta_days' in dynamic_data.tensors}")
        Subject ID: 1
        Time range: 0 to 4
        Has numeric values: True
        Has time deltas: True

        >>> # Test error case with invalid index
        >>> with tempfile.TemporaryDirectory() as tmp_dir:
        ...     config = create_dummy_dataset(tmp_dir)
        ...     dataset = CumSumPytorchDataset(config, split='train')
        ...     try:
        ...         dynamic_data = dataset.load_subject_dynamic_data(999)  # Invalid index
        ...     except IndexError as e:
        ...         print("Caught expected IndexError")
        Caught expected IndexError
        """
        subject_id, st, end = self.index[idx]
        shard = self.subj_map[subject_id]
        subject_idx = self.subj_indices[subject_id]

        dynamic_data_fp = Path(self.config.data_dir) / "data" / f"{shard}.nrt"

        subject_dynamic_data = JointNestedRaggedTensorDict(tensors_fp=dynamic_data_fp)[subject_idx, st:end]

        return subject_dynamic_data, subject_id, st, end

    @SeedableMixin.WithSeed
    @TimeableMixin.TimeAs
    def load_subject(
        self, subject_dynamic_data, subject_id: int, global_st: int, global_end: int
    ) -> dict[str, list[float]]:
        """Load and process data for a single subject.

        Args:
            subject_dynamic_data: The dynamic data for the subject.
            subject_id (int): The ID of the subject to load.
            global_st (int): The start index of the sequence to load.
            global_end (int): The end index of the sequence to load.

        Returns:
            dict: A dictionary containing the processed data for the subject.

        Examples:
        >>> import tempfile
        >>> from pathlib import Path
        >>> import polars as pl
        >>> import numpy as np
        >>> from nested_ragged_tensors.ragged_numpy import JointNestedRaggedTensorDict

        >>> # Test basic subject loading
        >>> with tempfile.TemporaryDirectory() as tmp_dir:
        ...     config = create_dummy_dataset(tmp_dir)
        ...     dataset = CumSumPytorchDataset(config, split='train')
        ...
        ...     # First get the dynamic data using load_subject_dynamic_data
        ...     dynamic_data, subject_id, st, end = dataset.load_subject_dynamic_data(0)
        ...
        ...     # Then load the complete subject data
        ...     subject_data = dataset.load_subject(dynamic_data, subject_id, st, end)
        ...
        ...     # Verify the returned data structure
        ...     print("Keys in subject data:")
        ...     for key in sorted(subject_data.keys()): print(f"{key}")
        ...     print()
        ...     print(f"Has static indices: {len(subject_data['static_indices']) > 0}")
        ...     print(f"Has dynamic data: {isinstance(subject_data['dynamic'], JointNestedRaggedTensorDict)}")
        ...     print(f"Has prediction time: {'prediction_time' in subject_data}")
        Keys in subject data:
        dynamic
        end_idx
        prediction_time
        start_idx
        start_time
        static_indices
        static_values
        <BLANKLINE>
        Has static indices: True
        Has dynamic data: True
        Has prediction time: True

        >>> # Test with different configuration settings
        >>> with tempfile.TemporaryDirectory() as tmp_dir:
        ...     # Create config with modified settings
        ...     config = create_dummy_dataset(tmp_dir)
        ...     config.do_prepend_static_data = False
        ...     config.postpend_eos_token = False
        ...     config.do_include_start_time_min = False
        ...
        ...     dataset = CumSumPytorchDataset(config, split='train')
        ...     dynamic_data, subject_id, st, end = dataset.load_subject_dynamic_data(0)
        ...     subject_data = dataset.load_subject(dynamic_data, subject_id, st, end)
        ...
        ...     # Verify the modified behavior
        ...     print(f"Contains start time: {'start_time' in subject_data}")
        ...     dynamic_tensors = subject_data['dynamic'].tensors
        ...     has_eos = np.any(dynamic_tensors['dim0/code'] == config.EOS_TOKEN_ID)
        ...     print(f"Contains EOS token: {has_eos}")
        Contains start time: False
        Contains EOS token: False

        >>> # Test with maximum sequence length constraint
        >>> with tempfile.TemporaryDirectory() as tmp_dir:
        ...     config = create_dummy_dataset(tmp_dir)
        ...     config.max_seq_len = 5  # Set small max sequence length
        ...
        ...     dataset = CumSumPytorchDataset(config, split='train')
        ...     dynamic_data, subject_id, st, end = dataset.load_subject_dynamic_data(0)
        ...     subject_data = dataset.load_subject(dynamic_data, subject_id, st, end)
        ...
        ...     # Verify sequence length constraints
        ...     dynamic_len = len(subject_data['dynamic'].tensors['dim0/code'])
        ...     print(f"Dynamic sequence length: {dynamic_len}")
        ...     print(f"Respects max length: {dynamic_len <= config.max_seq_len}")
        Dynamic sequence length: 5
        Respects max length: True
        """
        shard = self.subj_map[subject_id]
        subject_idx = self.subj_indices[subject_id]
        static_row = self.static_dfs[shard][subject_idx].to_dict()

        max_seq_len = self.config.max_seq_len

        out = {
            "static_indices": static_row["static_indices"].item().to_list(),
            "static_values": static_row["static_values"].item().to_list(),
        }

        if self.config.do_prepend_static_data:
            n_static = len(out["static_indices"])
            if n_static >= max_seq_len:
                raise ValueError(
                    f"Static data length {n_static} matches or exceeds "
                    f"max_seq_len {max_seq_len} for subject {subject_id}!"
                )

            max_seq_len -= n_static
        if self.config.postpend_eos_token:
            max_seq_len -= 1

        subject_dynamic_data, global_st, global_end, flat_start, flat_end = subsample_subject_data(
            subject_dynamic_data,
            max_seq_len,
            self.config.subsequence_sampling_strategy,
            self.config.do_flatten_tensors,
            global_st,
        )

        if self.config.do_include_subsequence_indices:
            out["start_idx"] = global_st
            out["end_idx"] = global_end

        tensors = subject_dynamic_data.tensors

        if self.config.do_prepend_static_data:
            tensors["dim0/time_delta_days"] = np.concatenate(
                [np.zeros(len(out["static_indices"])), tensors["dim0/time_delta_days"]]
            )
            tensors["dim0/static_mask"] = np.concatenate(
                [
                    np.ones(len(out["static_indices"]), dtype=bool),
                    np.zeros(len(tensors["dim0/code"]), dtype=bool),
                ]
            )
            tensors["dim0/code"] = np.concatenate([out["static_indices"], tensors["dim0/code"]])
            tensors["dim0/numeric_value"] = np.concatenate(
                [out["static_values"], tensors["dim0/numeric_value"]]
            )
        else:
            tensors["dim0/static_mask"] = np.zeros(len(tensors["dim0/code"]), dtype=bool)

        if self.config.postpend_eos_token:
            tensors["dim0/code"] = np.append(tensors["dim0/code"], [self.config.EOS_TOKEN_ID])
            tensors["dim0/static_mask"] = np.append(tensors["dim0/static_mask"], [False])
            tensors["dim0/numeric_value"] = np.append(tensors["dim0/numeric_value"], [0])
            tensors["dim0/time_delta_days"] = np.append(tensors["dim0/time_delta_days"], [0])

        subject_dynamic_data = JointNestedRaggedTensorDict(processed_tensors=tensors)

        out["dynamic"] = subject_dynamic_data

        if self.config.do_include_start_time_min:
            out["start_time"] = static_row["time"].item().to_list()[global_st]

        if self.config.do_include_prediction_time:
            out["prediction_time"] = static_row["time"].item().to_list()[global_end - 1]

        return out

    @TimeableMixin.TimeAs
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

        data = JointNestedRaggedTensorDict.vstack([item["dynamic"] for item in batch]).to_dense()
        tensorized = {k: torch.as_tensor(v) for k, v in data.items()}
        tensorized["code"] = tensorized["code"].long()
        tensorized["mask"] = tensorized.pop("dim1/mask")
        tensorized["numeric_value_mask"] = ~torch.isnan(tensorized["numeric_value"])
        tensorized["time_delta_days"] = torch.nan_to_num(tensorized["time_delta_days"], nan=0).float()
        tensorized["numeric_value"] = torch.nan_to_num(tensorized["numeric_value"], nan=0).float()
        tensorized["cum_sum"] = tensorized["cum_sum"].long()

        # Add task labels to batch
        for k in batch[0].keys():
            if k not in ("dynamic", "static_values", "static_indices", "static_mask"):
                if isinstance(batch[0][k], datetime):
                    tensorized[k] = [item[k] for item in batch]
                else:
                    tensorized[k] = torch.Tensor([item[k] for item in batch])
        return tensorized
