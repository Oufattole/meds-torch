import datetime
from enum import StrEnum
from pathlib import Path

import numpy as np
import polars as pl
import torch
from loguru import logger
from mixins import SeedableMixin, TimeableMixin
from nested_ragged_tensors.ragged_numpy import JointNestedRaggedTensorDict
from omegaconf import DictConfig

BINARY_LABEL_COL = "boolean_value"


class SubsequenceSamplingStrategy(StrEnum):
    """An enumeration of the possible subsequence sampling strategies for the dataset.

    Attributes:
        RANDOM: Randomly sample a subsequence from the full sequence.
        TO_END: Sample a subsequence from the end of the full sequence.
            Note this starts at the last element and moves back.
        FROM_START: Sample a subsequence from the start of the full sequence.
    """

    RANDOM = "random"
    TO_END = "to_end"
    FROM_START = "from_start"


class SeqPaddingSide(StrEnum):
    """An enumeration of the possible sequence padding sides for the dataset."""

    LEFT = "left"
    RIGHT = "right"


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
        >>> subsampled, st, end = subsample_subject_data(
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

        >>> # Test TO_END strategy with flattening
        >>> data = JointNestedRaggedTensorDict(raw_tensors=tensors)
        >>> subsampled, st, end = subsample_subject_data(
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

        >>> # Test TO_END strategy
        >>> data = JointNestedRaggedTensorDict(raw_tensors=tensors)
        >>> subsampled, st, end = subsample_subject_data(
        ...     data, max_seq_len=2,
        ...     sampling_strategy=SubsequenceSamplingStrategy.TO_END,
        ...     do_flatten_tensors=False,
        ... )
        >>> st, end
        (3, 5)

        >>> # Test RANDOM strategy
        >>> data = JointNestedRaggedTensorDict(raw_tensors=tensors)
        >>> subsampled, st, end = subsample_subject_data(
        ...     data, max_seq_len=2,
        ...     sampling_strategy=SubsequenceSamplingStrategy.RANDOM,
        ...     do_flatten_tensors=True,
        ... )
        >>> len(subsampled.tensors["dim0/code"]) == 2
        True
    """
    seq_len = len(subject_data)

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

    return subject_data, new_global_st, new_global_end


def get_task_indices_and_labels(
    task_df: pl.DataFrame, static_dfs: dict[str, pl.DataFrame]
) -> tuple[list[tuple[int, int, int]], dict[str, list]]:
    """Processes the joint DataFrame to determine the index range for each subject's task.

    For each row in task_df_joint, it is assumed that `time` is a sorted column and the function
    computes the index of the last event at `prediction_time`.

    Parameters:
    - task_df_joint (DataFrame): A DataFrame resulting from the merge_task_with_static function.

    Returns:
    - list: list of index tuples of format (subject_id, start_idx, end_idx).
    - dict: dictionary of task names to lists of labels in the same order as the indexes.
    """

    static_df = pl.concat(static_dfs.values()).select("subject_id", "time")

    end_idx_expr = (
        (pl.col("time").search_sorted(pl.col("prediction_time"), side="right")).last().alias("end_idx")
    )

    label_df = (
        task_df.join(static_df, on="subject_id", how="inner")
        .with_row_index("_row_index")
        .explode("time")
        .group_by("_row_index", "subject_id", "prediction_time", "boolean_value", maintain_order=True)
        .agg(end_idx_expr)
    )

    indexes = list(zip(label_df["subject_id"], label_df["end_idx"]))
    labels = label_df[BINARY_LABEL_COL].to_list()

    return indexes, labels


class PytorchDataset(SeedableMixin, torch.utils.data.Dataset, TimeableMixin):
    """A PyTorch Dataset class for handling complex, multi-modal medical data.

    This dataset is designed to work with data from the MEDS (Medical Event Data Set) format, supporting
    various types of medical events, static patient information, and task-specific labels. It provides
    functionality for loading, processing, and collating data for use in PyTorch models.

    Key Features:
    - Supports task-specific data handling for binary classification
    - Implements custom sampling strategies and sequence length constraints

    Args:
        cfg (DictConfig): Configuration options for the dataset.
        split (str): The data split to use (e.g., 'train', 'validation', 'test').

    Attributes:
        config (DictConfig): The dataset configuration.
        split (str): The current data split. code_metadata (pl.LazyFrame): Metadata for event codes.
        static_dfs (dict): Dictionary of static DataFrames for each data shard.
        subj_indices (dict): Mapping of subject IDs to their indices in the dataset.
        subj_seq_bounds (dict): Sequence bounds (start, end) for each subject.
        index (list): List of (subject_id, start, end) tuples for data access.
        labels (dict): Task-specific labels for each data point.
        tasks (list): List of task names.

    Methods:
    __len__(): Returns the number of items in the dataset.
    __getitem__(idx): Retrieves a single data point.
    collate(batch): Collates a batch of data points based on the specified collation strategy.
    """

    def __init__(self, cfg: DictConfig, split: str):
        super().__init__()

        self.config = cfg
        self.split = split

        logger.info("Scanning code metadata")
        self.code_metadata = pl.scan_parquet(self.config.code_metadata_fp)

        logger.info("Reading subject schema and static data")
        self.read_subject_descriptors()

    def read_subject_descriptors(self):
        """Read subject schemas and static data from the dataset.

        This method processes the Parquet files for each shard in the dataset, extracting static data and
        creating various mappings and indices for efficient data access.

        The method populates the following instance attributes:
        - self.static_dfs: Dictionary of static DataFrames for each shard.
        - self.subj_indices: Mapping of subject IDs to their indices.
        - self.subj_seq_bounds: Dictionary of sequence bounds for each subject.
        - self.index: List of (subject_id, start, end) tuples for data access.
        - self.labels: Dictionary of task labels (if tasks are specified).

        If a task is specified in the configuration, this method also processes the task labels and
        integrates them with the static data.

        Raises:
            ValueError: If duplicate subjects are found across shards.
            FileNotFoundError: If specified task files are not found.
        """

        schema_root = Path(self.config.schema_files_root)

        self.static_dfs = {}
        self.subj_indices = {}
        self.subj_seq_bounds = {}
        self.subj_map = {}

        schema_files = list(schema_root.glob(f"{self.split}/*.parquet"))
        if not schema_files:
            raise FileNotFoundError(
                f"No schema files found in {schema_root}! If your data is not sharded by split, this error "
                "may occur because this codebase does not handle non-split sharded data. See Issue #79 for "
                "tracking this issue."
            )

        for schema_fp in schema_files:
            shard = str(schema_fp.relative_to(schema_root).with_suffix(""))

            df = (
                pl.read_parquet(
                    schema_fp,
                    columns=[
                        "subject_id",
                        "start_time",
                        "code",
                        "numeric_value",
                        "time",
                    ],
                    use_pyarrow=True,
                )
                .rename({"code": "static_indices", "numeric_value": "static_values"})
                .with_columns(
                    pl.col("static_values").list.eval(pl.element().fill_null(0)),
                    pl.col("static_indices").list.eval(pl.element().fill_null(0)),
                )
            )

            self.static_dfs[shard] = df
            subject_ids = df["subject_id"]
            self.subj_map.update({subj: shard for subj in subject_ids})

            n_events = df.select(pl.col("time").list.len().alias("n_events")).get_column("n_events")
            for i, (subj, n_events_count) in enumerate(zip(subject_ids, n_events)):
                if subj in self.subj_indices or subj in self.subj_seq_bounds:
                    raise ValueError(f"Duplicate subject {subj} in {shard}!")

                self.subj_indices[subj] = i
                self.subj_seq_bounds[subj] = (0, n_events_count)

        if self.has_task:
            task_df_fp = Path(self.config.task_label_path)
            if not task_df_fp.is_file():
                logger.info(f"If the task file is not found at {task_df_fp}")
                task_df_fp = task_df_fp.with_suffix("") / "**/*.parquet"
                logger.info(f"Searching for task parquets over the glob {task_df_fp}")

            logger.info(f"Reading task constraints for {self.config.task_name} from {task_df_fp}")
            task_df = pl.read_parquet(task_df_fp)

            subjs_and_ends, self.labels = get_task_indices_and_labels(task_df, self.static_dfs)
            self.index = [(subj, 0, end) for subj, end in subjs_and_ends]
        else:
            self.index = [(subj, *bounds) for subj, bounds in self.subj_seq_bounds.items()]
            self.labels = None

    @property
    def subject_ids(self) -> list[int]:
        return [x[0] for x in self.index]

    def __len__(self):
        return len(self.index)

    @property
    def has_task(self) -> bool:
        return self.config.task_name is not None

    @property
    def max_seq_len(self) -> int:
        return self.config.max_seq_len

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        """Retrieve a single data point from the dataset.

        This method returns a dictionary corresponding to a single subject's data at the specified index. The
        data is not tensorized in this method, as that work is typically done in the collate function.

        Args:
            idx (int): The index of the data point to retrieve.

        Returns:
            dict: A dictionary containing the data for the specified index. The structure typically includes:
                - 'time_delta_days': List of time deltas between events.
                - 'dynamic_indices': List of categorical metadata elements.
                - 'dynamic_values': List of numerical metadata elements.
                - 'static_indices': List of static categorical metadata elements.
                Additional keys may be present based on the dataset configuration.

        Notes:
            This method uses the SeedableMixin to ensure reproducibility in data loading.
        """
        return self._seeded_getitem(idx)

    @TimeableMixin.TimeAs
    def load_subject_dynamic_data(self, idx: int):
        """Loads and returns the dynamic data slice for a given subject index, with subject ID and time
        range."""
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

        This method retrieves and processes the data for a specific subject, applying various transformations
        and filters based on the dataset configuration.

        Args:
            subject_dynamic_data: The dynamic data for the subject.
            subject_id (int): The ID of the subject to load.
            st (int): The start index of the sequence to load.
            end (int): The end index of the sequence to load.

        Returns:
            dict: A dictionary containing the processed data for the subject. The exact contents
                depend on the dataset configuration but typically include:
                - Static data (indices and values)
                - Dynamic data (time series data, event codes, etc.)
                - Sequence information (start and end indices, if configured)

        Raises:
            ValueError: If the sequence length is invalid or if there are inconsistencies in the data shapes.

        Notes:
            - This method applies sequence length constraints and sampling strategies as
                specified in the dataset configuration.
            - It handles different collate types (event_stream, triplet, etc.) differently.
            - The method is decorated with @SeedableMixin.WithSeed to ensure reproducibility.
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

        subject_dynamic_data, global_st, global_end = subsample_subject_data(
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
    def _seeded_getitem(self, idx: int) -> dict[str, list[float]]:
        """Returns a Returns a dictionary corresponding to a single subject's data.

        This function is a seedable version of `__getitem__`.
        """

        subject_dynamic_data, subject_id, st, end = self.load_subject_dynamic_data(idx)

        out = self.load_subject(subject_dynamic_data, subject_id, st, end)

        if self.config.do_include_subject_id:
            out["subject_id"] = subject_id

        if self.labels is not None:
            out[BINARY_LABEL_COL] = self.labels[idx]

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
            dict: A dictionary containing the collated batch data. The exact structure depends on the
                collation strategy used.
        """

        data = JointNestedRaggedTensorDict.vstack([item["dynamic"] for item in batch]).to_dense()
        tensorized = {k: torch.as_tensor(v) for k, v in data.items()}
        tensorized["code"] = tensorized["code"].long()
        tensorized["mask"] = tensorized.pop("dim1/mask")
        tensorized["numeric_value_mask"] = ~torch.isnan(tensorized["numeric_value"])
        tensorized["time_delta_days"] = torch.nan_to_num(tensorized["time_delta_days"], nan=0).float()
        tensorized["numeric_value"] = torch.nan_to_num(tensorized["numeric_value"], nan=0).float()

        # Add task labels to batch
        for k in batch[0].keys():
            if k not in ("dynamic", "static_values", "static_indices"):
                if isinstance(batch[0][k], datetime.datetime):
                    tensorized[k] = [item[k] for item in batch]
                else:
                    tensorized[k] = torch.Tensor([item[k] for item in batch])
        return tensorized
