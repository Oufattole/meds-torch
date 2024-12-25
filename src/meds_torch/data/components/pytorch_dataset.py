from dataclasses import dataclass
from datetime import datetime
from enum import StrEnum
from pathlib import Path

import numpy as np
import polars as pl
import torch
from dateutil.relativedelta import relativedelta
from loguru import logger
from mixins import SeedableMixin, TimeableMixin
from nested_ragged_tensors.ragged_numpy import JointNestedRaggedTensorDict
from omegaconf import DictConfig


@dataclass
class DummyConfig:
    """Dummy configuration for testing MEDS dataset"""

    schema_files_root: str
    task_label_path: str
    data_dir: str
    task_name: str = "dummy_task"
    max_seq_len: int = 10
    do_prepend_static_data: bool = True
    postpend_eos_token: bool = True
    do_flatten_tensors: bool = True
    EOS_TOKEN_ID: int = 5
    do_include_subject_id: bool = True
    do_include_subsequence_indices: bool = True
    do_include_start_time_min: bool = True
    do_include_end_time: bool = True
    do_include_prediction_time: bool = True
    subsequence_sampling_strategy: str = "from_start"


def create_dummy_dataset(
    base_dir: str | Path, n_subjects: int = 3, split: str = "train", seed: int | None = 42
) -> DummyConfig:
    """Creates a dummy MEDS dataset for testing purposes.

    Args:
        base_dir: Directory where the dummy dataset will be created
        n_subjects: Number of test subjects to generate
        split: Dataset split to create ('train', 'validation', or 'test')
        seed: Random seed for reproducible data generation

    Returns:
        DummyConfig object with paths to the created dataset files

    Examples:
        >>> from pprint import pprint
        >>> import tempfile
        >>> with tempfile.TemporaryDirectory() as tmp_dir:
        ...     config = create_dummy_dataset(tmp_dir)
        ...     # Verify directory structure
        ...     data_dir = Path(tmp_dir)
        ...     print(sorted(str(p.relative_to(tmp_dir))
        ...           for p in data_dir.glob("**/*")
        ...           if p.is_file()))
        ['data/train/shard_0.nrt', 'schema/train/shard_0.parquet', 'task_labels.parquet']

        >>> # Test creating dataset with different parameters
        >>> with tempfile.TemporaryDirectory() as tmp_dir:
        ...     config = create_dummy_dataset(
        ...         tmp_dir, n_subjects=2, split='validation', seed=123
        ...     )
        ...     # Verify static data
        ...     static_df = pl.read_parquet(
        ...         Path(config.schema_files_root) / "validation/shard_0.parquet"
        ...     )
        ...     print(f"Number of subjects: {len(static_df)}")
        ...     print(f"Columns: {static_df.columns}")
        Number of subjects: 2
        Columns: ['subject_id', 'start_time', 'time', 'code', 'numeric_value']

        >>> # Test loading dynamic data
        >>> with tempfile.TemporaryDirectory() as tmp_dir:
        ...     config = create_dummy_dataset(tmp_dir)
        ...     dynamic_data = JointNestedRaggedTensorDict(
        ...         tensors_fp=Path(tmp_dir) / "data/train/shard_0.nrt")
        ...     print(f"Dynamic data length: {len(dynamic_data)}")
        ...     print("Available features:")
        ...     for feature in sorted(dynamic_data.tensors.keys()): print(f"\t{feature}")
        Dynamic data length: 3
        Available features:
            dim1/bounds
            dim1/time_delta_days
            dim2/bounds
            dim2/code
            dim2/numeric_value
        >>> # Test loading static data and task labels
        >>> # Notice that the first index in the dynamic data corresponds to the first row in the static data,
        >>> # and that the second index in the dynamic data corresponds to the second row in the static data
        >>> # and so on.
        >>> with tempfile.TemporaryDirectory() as tmp_dir:
        ...     config = create_dummy_dataset(tmp_dir)
        ...     print(pl.read_parquet(Path(tmp_dir) / "schema/train/shard_0.parquet"))
        ...     print(pl.read_parquet(Path(tmp_dir) / "task_labels.parquet"))
        ...     pprint(config)
        shape: (3, 5)
        ┌────────────┬─────────────────────┬─────────────────────────────────┬───────────┬─────────────────┐
        │ subject_id ┆ start_time          ┆ time                            ┆ code      ┆ numeric_value   │
        │ ---        ┆ ---                 ┆ ---                             ┆ ---       ┆ ---             │
        │ i64        ┆ datetime[μs]        ┆ list[datetime[μs]]              ┆ list[i64] ┆ list[f64]       │
        ╞════════════╪═════════════════════╪═════════════════════════════════╪═══════════╪═════════════════╡
        │ 0          ┆ 1995-01-01 00:00:00 ┆ [1995-01-01 00:00:00, 1996-01-… ┆ [1, 2, 3] ┆ [0.1, 0.2, 0.3] │
        │ 1          ┆ 1995-01-01 00:00:00 ┆ [1995-01-01 00:00:00, 1996-01-… ┆ [1, 2, 3] ┆ [0.1, 0.2, 0.3] │
        │ 2          ┆ 1995-01-01 00:00:00 ┆ [1995-01-01 00:00:00, 1996-01-… ┆ [1, 2, 3] ┆ [0.1, 0.2, 0.3] │
        └────────────┴─────────────────────┴─────────────────────────────────┴───────────┴─────────────────┘
        shape: (3, 3)
        ┌────────────┬─────────────────────┬───────────────┐
        │ subject_id ┆ prediction_time     ┆ boolean_value │
        │ ---        ┆ ---                 ┆ ---           │
        │ i64        ┆ datetime[μs]        ┆ i64           │
        ╞════════════╪═════════════════════╪═══════════════╡
        │ 0          ┆ 1998-01-01 00:00:00 ┆ 0             │
        │ 1          ┆ 1998-01-01 00:00:00 ┆ 1             │
        │ 2          ┆ 1998-01-01 00:00:00 ┆ 0             │
        └────────────┴─────────────────────┴───────────────┘
        DummyConfig(schema_files_root='.../schema',
                    task_label_path='.../task_labels.parquet',
                    data_dir='...',
                    task_name='dummy_task',
                    max_seq_len=10,
                    do_prepend_static_data=True,
                    postpend_eos_token=True,
                    do_flatten_tensors=True,
                    EOS_TOKEN_ID=5,
                    do_include_subject_id=True,
                    do_include_subsequence_indices=True,
                    do_include_start_time_min=True,
                    do_include_end_time=True,
                    do_include_prediction_time=True,
                    subsequence_sampling_strategy='from_start')
    """
    if seed is not None:
        np.random.seed(seed)

    base_dir = Path(base_dir)

    # Create directories
    schema_dir = base_dir / "schema" / split
    schema_dir.mkdir(parents=True, exist_ok=True)
    base_dir.joinpath("data").mkdir(exist_ok=True)

    # Create static data
    base_datetime = datetime(1995, 1, 1)
    static_data = []
    for subject_id in range(n_subjects):
        static_data.append(
            {
                "subject_id": subject_id,
                "start_time": base_datetime,
                "time": [
                    base_datetime,
                    base_datetime + relativedelta(years=1),
                    base_datetime + relativedelta(years=2),
                    base_datetime + relativedelta(years=3),
                    base_datetime + relativedelta(years=4),
                ],
                "code": [1, 2, 3],
                "numeric_value": [0.1, 0.2, 0.3],
            }
        )
    static_df = pl.DataFrame(static_data)
    static_df.write_parquet(schema_dir / "shard_0.parquet", use_pyarrow=True)

    # Create dynamic data with consistent sequence lengths
    subject_dynamic_data = []
    for subject_id in range(n_subjects):
        dynamic_data = JointNestedRaggedTensorDict(
            raw_tensors={
                "code": [[0, 1, 2], [1, 3], [2, 3, 4], [1], [4]],
                "numeric_value": [[1.0, np.nan, 3.0], [np.nan, 5.0], [6.0, np.nan, 8.0], [np.nan], [10.0]],
                "time_delta_days": [0, 1, 2, 3, 4],
            }
        )
        subject_dynamic_data.append(dynamic_data)
    dynamic_data = JointNestedRaggedTensorDict.vstack(subject_dynamic_data)

    nrt_output_dir = base_dir / "data" / split
    nrt_output_dir.mkdir(parents=True, exist_ok=True)
    dynamic_data.save(nrt_output_dir / "shard_0.nrt")

    # Create task labels
    task_df = pl.DataFrame(
        {
            "subject_id": list(range(n_subjects)),
            "prediction_time": [base_datetime + relativedelta(years=3)] * n_subjects,
            "boolean_value": [i % 2 for i in range(n_subjects)],
        }
    )

    task_fp = base_dir / "task_labels.parquet"
    task_df.write_parquet(task_fp, use_pyarrow=True)

    return DummyConfig(
        schema_files_root=str(base_dir / "schema"),
        task_label_path=str(task_fp),
        data_dir=str(base_dir),
    )


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

    Examples:
    >>> import tempfile
    >>> with tempfile.TemporaryDirectory() as tmp_dir:
    ...     config = create_dummy_dataset(tmp_dir)
    ...     shard = "train/shard_0"
    ...     task_df = pl.read_parquet(Path(config.data_dir) / "task_labels.parquet")
    ...     static_dfs = {"shard_0": pl.read_parquet(Path(config.data_dir) / "schema/train/shard_0.parquet")}
    >>> task_df
    shape: (3, 3)
    ┌────────────┬─────────────────────┬───────────────┐
    │ subject_id ┆ prediction_time     ┆ boolean_value │
    │ ---        ┆ ---                 ┆ ---           │
    │ i64        ┆ datetime[μs]        ┆ i64           │
    ╞════════════╪═════════════════════╪═══════════════╡
    │ 0          ┆ 1998-01-01 00:00:00 ┆ 0             │
    │ 1          ┆ 1998-01-01 00:00:00 ┆ 1             │
    │ 2          ┆ 1998-01-01 00:00:00 ┆ 0             │
    └────────────┴─────────────────────┴───────────────┘
    >>> static_dfs["shard_0"]
    shape: (3, 5)
    ┌────────────┬─────────────────────┬─────────────────────────────────┬───────────┬─────────────────┐
    │ subject_id ┆ start_time          ┆ time                            ┆ code      ┆ numeric_value   │
    │ ---        ┆ ---                 ┆ ---                             ┆ ---       ┆ ---             │
    │ i64        ┆ datetime[μs]        ┆ list[datetime[μs]]              ┆ list[i64] ┆ list[f64]       │
    ╞════════════╪═════════════════════╪═════════════════════════════════╪═══════════╪═════════════════╡
    │ 0          ┆ 1995-01-01 00:00:00 ┆ [1995-01-01 00:00:00, 1996-01-… ┆ [1, 2, 3] ┆ [0.1, 0.2, 0.3] │
    │ 1          ┆ 1995-01-01 00:00:00 ┆ [1995-01-01 00:00:00, 1996-01-… ┆ [1, 2, 3] ┆ [0.1, 0.2, 0.3] │
    │ 2          ┆ 1995-01-01 00:00:00 ┆ [1995-01-01 00:00:00, 1996-01-… ┆ [1, 2, 3] ┆ [0.1, 0.2, 0.3] │
    └────────────┴─────────────────────┴─────────────────────────────────┴───────────┴─────────────────┘
    >>>
    >>> # Run the function
    >>> BINARY_LABEL_COL = "boolean_value"  # Define the constant used in the function
    >>> indices, labels, pred_times = get_task_indices_and_labels(task_df, static_dfs)
    >>>
    >>> # Check the results
    >>> print(indices)  # Only subjects 1 and 2 should be present (inner join)
    [(0, 4), (1, 4), (2, 4)]
    >>> print(labels)  # Labels for subjects 1 and 2
    [0, 1, 0]
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
    prediction_times = label_df["prediction_time"].to_list()

    return indexes, labels, prediction_times


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
        split (str): The current data split.
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

    Examples:
        >>> import tempfile
        >>> from pathlib import Path
        >>>
        >>> # Test initialization without task
        >>> with tempfile.TemporaryDirectory() as tmp_dir:
        ...     config = create_dummy_dataset(tmp_dir)
        ...     # Remove task path to test taskless initialization
        ...     config.task_label_path = None
        ...     config.task_name = None
        ...     config.do_include_prediction_time = False
        ...     dataset = PytorchDataset(config, split='train')
        ...     print(f"Dataset size: {len(dataset)}")
        ...     print(f"Has task: {dataset.has_task}")
        ...     # Test data loading
        ...     sample = dataset[0]
        ...     print("Sample keys:")
        ...     for key in sorted(list(sample.keys())): print(f"\t{key}")
        Dataset size: 3
        Has task: False
        Sample keys:
                dynamic
                end_idx
                end_time
                start_idx
                start_time
                static_indices
                static_values
                subject_id
        >>> # Test initialization with task
        >>> with tempfile.TemporaryDirectory() as tmp_dir:
        ...     config = create_dummy_dataset(tmp_dir)
        ...     dataset = PytorchDataset(config, split='train')
        ...     print(f"Dataset size: {len(dataset)}")
        ...     print(f"Has task: {dataset.has_task}")
        ...     print(f"First subject label: {dataset.labels[0]}")  # Subject IDs start at 1
        Dataset size: 3
        Has task: True
        First subject label: 0
    """

    def __init__(self, cfg: DictConfig, split: str):
        super().__init__()

        self.config = cfg
        self.split = split

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

            subjs_and_ends, self.labels, self.prediction_times = get_task_indices_and_labels(
                task_df, self.static_dfs
            )
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
                - code: List of categorical metadata elements.
                - mask: Mask of valid elements in the sequence, False means it is a padded element.
                - numeric_value: List of dynamic numeric values.
                - numeric_value_mask: Mask of numeric values (False means no numeric value was recorded)
                - time_delta_days: List of dynamic time deltas between observations.
                - static_indices(Optional): List of static MEDS codes.
                - static_values(Optional): List of static MEDS numeric values.
                - static_mask(Optional): List of static masks (True means the value is static).

        Notes:
            This method uses the SeedableMixin to ensure reproducibility in data loading.
        """
        return self._seeded_getitem(idx)

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
        ...     dataset = PytorchDataset(config, split='train')
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
        ...     dataset = PytorchDataset(config, split='train')
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
        ...     dataset = PytorchDataset(config, split='train')
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
        ...     dataset = PytorchDataset(config, split='train')
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
        ...     print(f"Has end time: {'end_time' in subject_data}")
        Keys in subject data:
        dynamic
        end_idx
        end_time
        start_idx
        start_time
        static_indices
        static_values
        <BLANKLINE>
        Has static indices: True
        Has dynamic data: True
        Has end time: True

        >>> # Test with different configuration settings
        >>> with tempfile.TemporaryDirectory() as tmp_dir:
        ...     # Create config with modified settings
        ...     config = create_dummy_dataset(tmp_dir)
        ...     config.do_prepend_static_data = False
        ...     config.postpend_eos_token = False
        ...     config.do_include_start_time_min = False
        ...
        ...     dataset = PytorchDataset(config, split='train')
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
        ...     dataset = PytorchDataset(config, split='train')
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
        if self.config.do_include_end_time:
            out["end_time"] = static_row["time"].item().to_list()[global_end - 1]

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
        if self.config.do_include_prediction_time:
            if not self.has_task:
                if not self.config.do_include_end_time:
                    raise ValueError(
                        "Cannot include prediction_time without a task specified " "or do_include_end_time!"
                    )
                else:
                    out["prediction_time"] = out["end_time"]
            else:
                out["prediction_time"] = self.prediction_times[idx]

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
            dict: A dictionary containing the collated batch data.
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
            if k not in ("dynamic", "static_values", "static_indices", "static_mask"):
                if isinstance(batch[0][k], datetime):
                    tensorized[k] = [item[k] for item in batch]
                else:
                    tensorized[k] = torch.Tensor([item[k] for item in batch])
        return tensorized
