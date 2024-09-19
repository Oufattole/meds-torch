import json
import os
from collections import defaultdict
from datetime import datetime, timedelta
from enum import StrEnum
from pathlib import Path

import numpy as np
import polars as pl
import torch
from loguru import logger
from mixins import SeedableMixin, TimeableMixin
from nested_ragged_tensors.ragged_numpy import (
    NP_FLOAT_TYPES,
    NP_INT_TYPES,
    NP_UINT_TYPES,
    JointNestedRaggedTensorDict,
)
from omegaconf import DictConfig
from transformers import AutoTokenizer

IDX_COL = "_row_index"


class CollateType(StrEnum):
    event_stream = "event_stream"
    triplet = "triplet"
    text_code = "text_code"
    triplet_prompt = "triplet_prompt"
    eic = "eic"


def generate_subject_split_dict(meds_dir):
    """Generate a dictionary mapping split names to lists of subject IDs.

    This function scans through the directory structure of a MEDS dataset, reading Parquet files to extract
    unique subject IDs for each split.

    Args:     meds_dir (str): Path to the root directory of the MEDS dataset.

    Returns:     dict: A dictionary where keys are split names (e.g., 'train/shard0')           and values are
    lists of subject IDs belonging to that split.

    Raises:     FileNotFoundError: If the specified directory does not exist.

    Notes:     - The function expects a directory structure where split directories       contain Parquet
    files with subject data.     - It logs a warning if a specified path is not a directory.
    """
    subject_split_dict = {}

    for split_dir in os.listdir(meds_dir):
        split_path = Path(meds_dir) / split_dir
        if split_path.is_dir():
            for shard_file in split_path.glob("*.parquet"):
                split_name = f"{split_dir}/{shard_file.stem}"
                df = pl.read_parquet(shard_file)
                subject_ids = df["subject_id"].unique().to_list()
                subject_split_dict[split_name] = subject_ids
        else:
            logger.warning(f"Directory {split_path} does not exist or is not a directory.")

    return subject_split_dict


def subpad_vectors(a: np.ndarray, b: np.ndarray):
    """Create a new array by placing elements of 'a' at indices specified by 'b'.

    This function creates an array of zeros with length equal to the maximum value in 'b',
    then places the values from 'a' at the indices specified by 'b'.

    Args:
        a (numpy.ndarray): The source array containing values to be placed.
        b (numpy.ndarray): The array specifying the indices where values from 'a' should be placed.

    Returns:
        numpy.ndarray: A new array with values from 'a' placed at indices specified by 'b',
                       and zeros elsewhere.

    Example:
    >>> a = np.array([2, 4, 5])
    >>> b = np.array([3, 5, 10])
    >>> subpad_vectors(a, b)
    array([2, 0, 0, 4, 0, 5, 0, 0, 0, 0])
    """
    total_length = b[-1]
    result = np.zeros(total_length, dtype=a.dtype)
    result[[0] + list(b[:-1])] = a
    return result


def count_or_proportion(N: int | pl.Expr | None, cnt_or_prop: int | float) -> int:
    """Returns `cnt_or_prop` if it is an integer or `int(N*cnt_or_prop)` if it is a float.

    Resolves cutoff variables that can either be passed as integer counts or fractions of a whole. E.g., the
    vocabulary should contain only elements that occur with count or proportion at least X, where X might be
    20 times, or 1%.

    Arguments:
        N: The total number of elements in the whole. Only used if `cnt_or_prop` is a proportion (float).
        cnt_or_prop: The cutoff value, either as an integer count or a proportion of the whole.

    Returns:
        The cutoff value as an integer count of the whole.

    Raises:
        TypeError: If `cnt_or_prop` is not an integer or a float or if `N` is needed and is not an integer or
            a polars Expression.
        ValueError: If `cnt_or_prop` is not a positive integer or a float between 0 and 1.

    Examples:
        >>> count_or_proportion(100, 0.1)
        10
        >>> count_or_proportion(None, 11)
        11
        >>> count_or_proportion(100, 0.116)
        12
        >>> count_or_proportion(None, 0)
        Traceback (most recent call last):
            ...
        ValueError: 0 must be positive if it is an integer
        >>> count_or_proportion(None, 1.3)
        Traceback (most recent call last):
            ...
        ValueError: 1.3 must be between 0 and 1 if it is a float
        >>> count_or_proportion(None, "a")
        Traceback (most recent call last):
            ...
        TypeError: a must be a positive integer or a float between 0 or 1
        >>> count_or_proportion("a", 0.2)
        Traceback (most recent call last):
            ...
        TypeError: a must be an integer or a polars.Expr when cnt_or_prop is a float!
    """

    match cnt_or_prop:
        case int() if 0 < cnt_or_prop:
            return cnt_or_prop
        case int():
            raise ValueError(f"{cnt_or_prop} must be positive if it is an integer")
        case float() if 0 < cnt_or_prop < 1:
            pass
        case float():
            raise ValueError(f"{cnt_or_prop} must be between 0 and 1 if it is a float")
        case _:
            raise TypeError(f"{cnt_or_prop} must be a positive integer or a float between 0 or 1")

    match N:
        case int():
            return int(round(cnt_or_prop * N))
        case pl.Expr():
            return (N * cnt_or_prop).round(0).cast(int)
        case _:
            raise TypeError(f"{N} must be an integer or a polars.Expr when cnt_or_prop is a float!")


class SubsequenceSamplingStrategy(StrEnum):
    """An enumeration of the possible subsequence sampling strategies for the dataset.

    Attributes:     RANDOM: Randomly sample a subsequence from the full sequence.     TO_END: Sample a
    subsequence from the end of the full sequence.         Note this starts at the last element and moves
    back.     FROM_START: Sample a subsequence from the start of the full sequence.
    """

    RANDOM = "random"
    TO_END = "to_end"
    FROM_START = "from_start"


class SeqPaddingSide(StrEnum):
    """An enumeration of the possible sequence padding sides for the dataset."""

    LEFT = "left"
    RIGHT = "right"


def to_int_index(col: pl.Expr) -> pl.Expr:
    """Returns an integer index of the unique elements seen in this column.

    The returned index is into a vocabulary sorted lexographically.

    Args:
        col: The column containing the data to be converted into integer indices.

    Examples:
        >>> import polars as pl
        >>> X = pl.DataFrame({
        ...     'c': ['foo', 'bar', 'foo', 'bar', 'baz', None, 'bar', 'aba'],
        ...     'd': [1, 2, 3, 4, 5, 6, 7, 8]
        ... })
        >>> X.with_columns(to_int_index(pl.col('c')).alias("c_index"))
        shape: (8, 3)
        ┌──────┬─────┬─────────┐
        │ c    ┆ d   ┆ c_index │
        │ ---  ┆ --- ┆ ---     │
        │ str  ┆ i64 ┆ u32     │
        ╞══════╪═════╪═════════╡
        │ foo  ┆ 1   ┆ 3       │
        │ bar  ┆ 2   ┆ 1       │
        │ foo  ┆ 3   ┆ 3       │
        │ bar  ┆ 4   ┆ 1       │
        │ baz  ┆ 5   ┆ 2       │
        │ null ┆ 6   ┆ null    │
        │ bar  ┆ 7   ┆ 1       │
        │ aba  ┆ 8   ┆ 0       │
        └──────┴─────┴─────────┘
    """

    indices = col.drop_nulls().unique().sort().search_sorted(col, side="left")
    return pl.when(col.is_null()).then(pl.lit(None)).otherwise(indices).alias(col.meta.output_name())


def merge_task_with_static(task_df: pl.DataFrame, static_dfs: dict[str, pl.DataFrame], tasks: list[str]):
    """Merges a DataFrame containing task information with multiple static DataFrames on the 'subject_id'
    column. The function performs a sequence of operations to merge these dataframes based on subject
    identifiers and respective timestamps.

    Parameters:
    - task_df (DataFrame): A DataFrame with columns 'subject_id', 'start_time', 'end_time', and 'label'.
    - static_dfs (dict of DataFrames): A dictionary of DataFrames indexed by their source names,
      each containing 'subject_id', 'start_time', 'static_indices', 'static_values', and "time".
    - tasks (list[str]): A list of task names to be merged with the static DataFrames.

    Returns:
    - DataFrame: The merged DataFrame containing data from task_df and all static_dfs.

    Example:
    >>> from datetime import datetime
    >>> import polars as pl
    >>> task_df = pl.DataFrame({
    ...     "subject_id": [1, 2],
    ...     "start_time": [datetime(2020, 1, 1), datetime(2020, 1, 2)],
    ...     "end_time": [datetime(2020, 1, 2), datetime(2020, 1, 3)],
    ...     "label": [0, 1]
    ... })
    >>> static_dfs = {
    ...     'train/0': pl.DataFrame({
    ...         "subject_id": [1, 2],
    ...         "start_time": [datetime(2020, 1, 1), datetime(2020, 1, 2)],
    ...         "time": [[datetime(2020, 1, 1, 1), datetime(2020, 1, 1, 3)],
    ...                       [datetime(2020, 1, 2), datetime(2020, 1, 1, 2, 3)]]
    ...     })
    ... }
    >>> tasks = ["label"]
    >>> result = merge_task_with_static(task_df, static_dfs, tasks)
    >>> result.select(['subject_id', 'end_time', 'label', 'time'])
    shape: (2, 4)
    ┌────────────┬───────────────────────┬───────────┬─────────────────────────────────┐
    │ subject_id ┆ end_time              ┆ label     ┆ time                            │
    │ ---        ┆ ---                   ┆ ---       ┆ ---                             │
    │ i64        ┆ list[datetime[μs]]    ┆ list[i64] ┆ list[datetime[μs]]              │
    ╞════════════╪═══════════════════════╪═══════════╪═════════════════════════════════╡
    │ 1          ┆ [2020-01-02 00:00:00] ┆ [0]       ┆ [2020-01-01 01:00:00, 2020-01-… │
    │ 2          ┆ [2020-01-03 00:00:00] ┆ [1]       ┆ [2020-01-02 00:00:00, 2020-01-… │
    └────────────┴───────────────────────┴───────────┴─────────────────────────────────┘
    """
    task_df_joint = (
        task_df.select("subject_id", "start_time", "end_time", *tasks)
        .with_row_index(IDX_COL)
        .group_by(IDX_COL, "subject_id", maintain_order=True)
        .agg("start_time", "end_time", *tasks)
        .join(
            pl.concat(static_dfs.values()).select(
                "subject_id", pl.col("start_time").alias("start_time_global"), "time"
            ),
            on="subject_id",
            how="left",
        )
        .with_columns(pl.col("time"), pl.col(tasks))
    )
    return task_df_joint


def get_task_indices_and_labels(
    task_df_joint: pl.DataFrame, tasks: list[str]
) -> tuple[list[tuple[int, int, int]], dict[str, list]]:
    """Processes the joint DataFrame to determine the index range for each subject's tasks.

    For each row in task_df_joint, it is assumed that `time` is a sorted column and the function
    computes the start index and end index of the span of time values in between `start_time` and `end_time`.

    Parameters:
    - task_df_joint (DataFrame): A DataFrame resulting from the merge_task_with_static function.
    - tasks (list[str]): A list of task names that are columns in task_df_joint.

    Returns:
    - list: list of index tuples of format (subject_id, start_idx, end_idx).
    - dict: dictionary of task names to lists of labels in the same order as the indexes.

    Example:
    >>> from datetime import datetime
    >>> df = pl.DataFrame({
    ...     IDX_COL: [i for i in range(5)],
    ...     "subject_id": [i for i in range(5)],
    ...     "start_time": [
    ...         [datetime(2021, 1, 1)],
    ...         [datetime(2021, 1, 1)],
    ...         [datetime(2021, 1, 1)],
    ...         [datetime(2021, 1, 2)],
    ...         [datetime(2021, 1, 3)]
    ...     ],
    ...     "end_time": [
    ...         [datetime(2021, 1, 2)],
    ...         [datetime(2021, 1, 2)],
    ...         [datetime(2021, 1, 3)],
    ...         [datetime(2021, 1, 4)],
    ...         [datetime(2021, 1, 4)]
    ...     ],
    ...     "time": [
    ...         pl.date_range(datetime(2021, 1, 1), datetime(2021, 1, 5), "1d", eager=True)
    ...     ]*5,
    ...     "label": [[0], [0], [0], [1], [1]],
    ... })
    >>> tasks = ["label"]
    >>> indexes, labels = get_task_indices_and_labels(df, tasks)
    >>> indexes
    [(0, 0, 1), (1, 0, 1), (2, 0, 2), (3, 1, 3), (4, 2, 3)]
    >>> labels
    {'label': [0, 0, 0, 1, 1]}
    """
    start_idx_expr = (
        (pl.col("time").search_sorted(pl.col("start_time"), side="left")).first().alias("start_idx")
    )
    end_idx_expr = (pl.col("time").search_sorted(pl.col("end_time"), side="left")).last().alias("end_idx")
    task_index_df = (
        task_df_joint.explode("start_time", "end_time", *tasks)
        .explode("time")
        .group_by(IDX_COL, "subject_id", "start_time", "end_time", maintain_order=True)
        .agg(start_idx_expr, end_idx_expr)
    )

    label_df = task_index_df.join(task_df_joint[IDX_COL, *tasks], how="left", on=IDX_COL).sort(IDX_COL)
    label_df = label_df.explode(*tasks)
    if not label_df.shape[0] == task_index_df.shape[0]:
        raise ValueError(
            "There are multiple labels for a single task index!"
            f"There are {label_df.shape[0]} labels for {task_index_df.shape[0]} task indexes."
        )

    subject_ids = label_df["subject_id"]
    start_indices = label_df["start_idx"]
    end_indices = label_df["end_idx"]
    labels = {task: label_df[task].to_list() for task in tasks}

    indexes = list(zip(subject_ids, start_indices, end_indices))

    return indexes, labels


class PytorchDataset(SeedableMixin, torch.utils.data.Dataset, TimeableMixin):
    """A PyTorch Dataset class for handling complex, multi-modal medical data.

    This dataset is designed to work with data from the MEDS (Medical Event Data Set) format, supporting
    various types of medical events, static patient information, and task-specific labels. It provides
    functionality for loading, processing, and collating data for use in PyTorch models.

    Key Features: - Handles different collation strategies (event stream, triplet, text-code, etc.) - Supports
    task-specific data handling for binary classification - Implements custom sampling strategies and sequence
    length constraints

    Args:     cfg (DictConfig): Configuration options for the dataset.     split (str): The data split to use
    (e.g., 'train', 'validation', 'test').

    Attributes:     config (DictConfig): The dataset configuration.     split (str): The current data split.
    code_metadata (pl.LazyFrame): Metadata for event codes.     static_dfs (dict): Dictionary of static
    DataFrames for each data shard.     subj_indices (dict): Mapping of subject IDs to their indices in the
    dataset.     subj_seq_bounds (dict): Sequence bounds (start, end) for each subject.     index (list): List
    of (subject_id, start, end) tuples for data access.     labels (dict): Task-specific labels for each data
    point.     tasks (list): List of task names.     task_types (dict): Mapping of task names to their types
    (classification, regression, etc.).     task_vocabs (dict): Vocabularies for classification tasks.
    tokenized_codes (dict): Tokenized representations of event codes (for text-code collation).

    Methods:     __len__(): Returns the number of items in the dataset.     __getitem__(idx): Retrieves a
    single data point.     collate(batch): Collates a batch of data points based on the specified collation
    strategy.
    """

    TYPE_CHECKERS = {
        "multi_class_classification": [
            (
                {pl.UInt8, pl.UInt16, pl.UInt32, pl.UInt64, pl.Int8, pl.Int16, pl.Int32, pl.Int64},
                None,
            ),
            ({pl.Categorical(ordering="physical"), pl.Categorical(ordering="lexical")}, to_int_index),
            ({pl.Utf8}, to_int_index),
        ],
        "binary_classification": [({pl.Boolean}, lambda Y: Y.cast(pl.Float32))],
        "regression": [({pl.Float32, pl.Float64}, None)],
    }
    """Type checker and conversion parameters for labeled datasets."""

    @classmethod
    def normalize_task(cls, col: pl.Expr, dtype: pl.DataType) -> tuple[str, pl.Expr]:
        """Normalize task labels to a common format based on their data type.

        This method determines the appropriate task type (e.g., multi-class classification, binary
        classification, regression) based on the data type of the label column and applies any necessary
        transformations to normalize the data.

        Args:     col (pl.Expr): The polars Expression containing the task labels.     dtype (pl.DataType):
        The polars data type of the task labels.

        Returns:     tuple: A tuple containing two elements:         - str: The determined task type (e.g.,
        'multi_class_classification', 'binary_classification', 'regression').         - pl.Expr: The
        normalized column expression.

        Raises:     TypeError: If the task labels are not of a supported type.
        """
        for task_type, checkers in cls.TYPE_CHECKERS.items():
            for valid_dtypes, normalize_fn in checkers:
                if dtype in valid_dtypes:
                    return task_type, (col if normalize_fn is None else normalize_fn(col))
        raise TypeError(f"Can't process label of {dtype} type!")

    def __init__(self, cfg: DictConfig, split: str):
        super().__init__()

        self.config = cfg
        self.split = split

        logger.info("Scanning code metadata")
        self.code_metadata = pl.scan_parquet(self.config.code_metadata_fp)

        logger.info("Reading splits & subject shards")
        self.read_shards()

        logger.info("Reading subject descriptors")
        self.read_subject_descriptors()

        if self.config.min_seq_len is not None and self.config.min_seq_len > 1:
            logger.info(f"Restricting to subjects with at least {self.config.min_seq_len} events")
            self.filter_to_min_seq_len()

        if self.config.train_subset_size not in (None, "FULL") and self.split == "train":
            logger.info(f"Filtering training subset size to {self.config.train_subset_size}")
            self.filter_to_subset()

        self.set_inter_event_time_stats()

        # Initialize tokenizer here
        self.init_tokenizer()

    def init_tokenizer(self):
        if self.config.collate_type == CollateType.text_code:
            if not hasattr(self, "tokenized_codes"):
                # Disable parallelism for tokenization as it will cause issues when num_workers > 0 in the
                # pytorch dataloader
                os.environ["TOKENIZERS_PARALLELISM"] = "false"
                tokenizer = AutoTokenizer.from_pretrained(
                    self.config.tokenizer, model_max_length=self.config.text_max_seq_len
                )
                self.tokenized_codes = self.tokenize_metadata(
                    tokenizer, self.code_metadata, special_tokens={self.config.EOS_TOKEN_ID: "[CLS]"}
                )

    def read_shards(self):
        """Reads the split-specific subject shards from the MEDS dataset.

        This method scans the specified MEDS cohort directory for Parquet files, organizes them by split, and
        creates mappings between subjects and their respective shards.
        """
        all_shards = generate_subject_split_dict(Path(self.config.meds_cohort_dir) / "data")
        self.shards = {sp: subjs for sp, subjs in all_shards.items() if sp.startswith(f"{self.split}")}
        self.subj_map = {subj: sp for sp, subjs in self.shards.items() for subj in subjs}
        if not self.shards:
            logger.warning(
                f"No shards found for split {self.split}. Check the directory structure and file names."
            )

    def read_subject_descriptors(self):
        """Read subject schemas and static data from the dataset.

        This method processes the Parquet files for each shard in the dataset, extracting static data and
        creating various mappings and indices for efficient data access.

        The method populates the following instance attributes: - self.static_dfs: Dictionary of static
        DataFrames for each shard. - self.subj_indices: Mapping of subject IDs to their indices. -
        self.subj_seq_bounds: Dictionary of sequence bounds for each subject. - self.index: List of
        (subject_id, start, end) tuples for data access. - self.labels: Dictionary of task labels (if tasks
        are specified). - self.tasks: List of task names. - self.task_types: Dictionary of task types. -
        self.task_vocabs: Dictionary of task vocabularies.

        If tasks are specified in the configuration, this method also processes the task labels and integrates
        them with the static data.

        Raises:     ValueError: If duplicate subjects are found across shards or if                 required
        task information is missing.     FileNotFoundError: If specified task files are not found.
        """
        self.static_dfs = {}
        self.subj_indices = {}
        self.subj_seq_bounds = {}

        for shard in self.shards.keys():
            static_fp = Path(self.config.schema_files_root) / f"{shard}.parquet"
            df = (
                pl.read_parquet(
                    static_fp,
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
            n_events = df.select(pl.col("time").list.len().alias("n_events")).get_column("n_events")
            for i, (subj, n_events_count) in enumerate(zip(subject_ids, n_events)):
                if subj in self.subj_indices or subj in self.subj_seq_bounds:
                    raise ValueError(f"Duplicate subject {subj} in {shard}!")

                self.subj_indices[subj] = i
                self.subj_seq_bounds[subj] = (0, n_events_count)

        if self.has_task:
            if self.config.task_root_dir is None:
                raise ValueError("`task_root_dir` must be provided if task is specified!")
            task_df_fp = Path(self.config.task_label_path)
            if not task_df_fp.is_file():
                logger.info(f"If the task file is not found at {task_df_fp}")
                task_df_fp = task_df_fp.with_suffix("") / "**/*.parquet"
                logger.info(f"Searching for task parquets over the glob {task_df_fp}")

            task_info_fp = Path(self.config.task_info_path)

            logger.info(f"Reading task constraints for {self.config.task_name} from {task_df_fp}")
            task_df = pl.read_parquet(task_df_fp)
            if "prediction_time" in task_df.columns:
                task_df = task_df.with_columns(end_time=pl.col("prediction_time"))
            elif "end_time" not in task_df.columns:
                raise ValueError("Task dataframe must contain either 'prediction_time' or 'end_time' column.")

            if "start_time" not in task_df.columns:
                task_df = task_df.with_columns(start_time=pl.lit(datetime(1900, 1, 1)))

            if "boolean_value" in task_df.columns:
                task_df = task_df.with_columns(pl.col("boolean_value").alias(self.config.task_name))
                self.tasks = [self.config.task_name]
            else:
                self.tasks = sorted(
                    [c for c in task_df.columns if c not in ["subject_id", "start_time", "end_time"]]
                )

            task_info = self.get_task_info(task_df)

            if task_info_fp.is_file():
                loaded_task_info = json.loads(task_info_fp.read_text())
                if loaded_task_info != task_info:
                    raise ValueError(
                        f"Task info differs from on disk!\nDisk:\n{loaded_task_info}\n"
                        f"Local:\n{task_info}\nSplit: {self.split}"
                    )
                logger.info(f"Re-built existing {task_info_fp} and it matches.")
            else:
                task_info_fp.parent.mkdir(exist_ok=True, parents=True)
                task_info_fp.write_text(json.dumps(task_info))

            idx_col = "_row_index"
            while idx_col in task_df.columns:
                idx_col = f"_{idx_col}"

            task_df_joint = merge_task_with_static(task_df, self.static_dfs, self.tasks)
            # Filter out subjects that are not in the split
            split_subjects = set(
                pl.concat(self.static_dfs.values())
                .select(pl.col("subject_id").unique())
                .to_series()
                .to_list()
            )
            task_df_joint = task_df_joint.filter(pl.col("subject_id").is_in(split_subjects))
            # Convert dates to indexes in the nested ragged tensor, (for fast indexing of data)
            self.index, self.labels = get_task_indices_and_labels(task_df_joint, self.tasks)
        else:
            self.index = [(subj, *bounds) for subj, bounds in self.subj_seq_bounds.items()]
            self.labels = {}
            self.tasks = None
            self.task_types = None
            self.task_vocabs = None

    def get_task_info(self, task_df: pl.DataFrame):
        """Extract and process task information from the task DataFrame.

        This method analyzes the task DataFrame to determine the type of each task (e.g., binary
        classification, multi-class classification) and creates appropriate vocabularies for classification
        tasks.

        Args:     task_df (pl.DataFrame): DataFrame containing task labels and related information.

        Returns:     dict: A dictionary containing processed task information:         - 'tasks': List of task
        names.         - 'vocabs': Dictionary mapping task names to their vocabularies.         - 'types':
        Dictionary mapping task names to their types.

        Raises:     NotImplementedError: If an unsupported task type is encountered.
        """
        self.task_types = {}
        self.task_vocabs = {}

        normalized_cols = []
        for t in self.tasks:
            task_type, normalized_vals = self.normalize_task(col=pl.col(t), dtype=task_df.schema[t])
            self.task_types[t] = task_type
            normalized_cols.append(normalized_vals.alias(t))

        task_df = task_df.with_columns(normalized_cols)

        for t in self.tasks:
            match self.task_types[t]:
                case "binary_classification":
                    self.task_vocabs[t] = [False, True]
                case "multi_class_classification":
                    self.task_vocabs[t] = list(range(task_df.select(pl.col(t).max()).item() + 1))
                case _:
                    raise NotImplementedError(f"Task type {self.task_types[t]} not implemented!")

        return {"tasks": sorted(self.tasks), "vocabs": self.task_vocabs, "types": self.task_types}

    def filter_to_min_seq_len(self):
        """Filter the dataset to include only subjects with at least the minimum sequence length.

        This method removes data points where the sequence length is less than the specified minimum sequence
        length (self.config.min_seq_len). It updates the dataset's index and labels accordingly.

        Notes:     - This method modifies the self.index and self.labels attributes in-place.     - It logs
        information about the number of data points and subjects before       and after filtering.     - If
        tasks are specified, it warns that filtering may affect model comparability.

        Raises:     AttributeError: If self.config.min_seq_len is not set.

        Side effects:     - Reduces the size of self.index and self.labels.     - Logs information about the
        filtering process.
        """
        if self.has_task:
            logger.warning(
                f"Filtering task {self.config.task_name} to min_seq_len {self.config.min_seq_len}. "
                "This may result in incomparable model results against runs with different constraints!"
            )

        orig_len = len(self)
        orig_n_subjects = len(set(self.subject_ids))
        valid_indices = [
            i for i, (subj, start, end) in enumerate(self.index) if end - start >= self.config.min_seq_len
        ]
        self.index = [self.index[i] for i in valid_indices]
        self.labels = {t: [t_labels[i] for i in valid_indices] for t, t_labels in self.labels.items()}
        new_len = len(self)
        new_n_subjects = len(set(self.subject_ids))
        logger.info(
            f"Filtered data due to sequence length constraint (>= {self.config.min_seq_len}) from "
            f"{orig_len} to {new_len} rows and {orig_n_subjects} to {new_n_subjects} subjects."
        )

    def filter_to_subset(self):
        """Filter the dataset to include only a subset of subjects for the training split.

        This method randomly selects a subset of subjects based on the self.config.train_subset_size
        parameter. It's typically used to create smaller training sets for experimentation or debugging
        purposes.

        The method only applies to the training split ('train') and uses a random number generator seeded with
        self.config.train_subset_seed for reproducibility.

        Notes:     - This method modifies the self.index and self.labels attributes in-place.     - It logs
        information about the number of data points and subjects before       and after filtering.     - The
        subset size can be specified as an integer (exact number of subjects)       or a float (proportion of
        total subjects).

        Raises:     ValueError: If self.config.train_subset_size is not properly set.

        Side effects:     - Reduces the size of self.index and self.labels for the training split.     - Logs
        information about the subset selection process.
        """

        orig_len = len(self)
        orig_n_subjects = len(set(self.subject_ids))
        rng = np.random.default_rng(self.config.train_subset_seed)
        subset_subjects = rng.choice(
            list(set(self.subject_ids)),
            size=count_or_proportion(orig_n_subjects, self.config.train_subset_size),
            replace=False,
        )
        valid_indices = [i for i, (subj, start, end) in enumerate(self.index) if subj in subset_subjects]
        self.index = [self.index[i] for i in valid_indices]
        self.labels = {t: [t_labels[i] for i in valid_indices] for t, t_labels in self.labels.items()}
        new_len = len(self)
        new_n_subjects = len(set(self.subject_ids))
        logger.info(
            f"Filtered data to subset of {self.config.train_subset_size} subjects from "
            f"{orig_len} to {new_len} rows and {orig_n_subjects} to {new_n_subjects} subjects."
        )

    def set_inter_event_time_stats(self):
        """Calculate and set inter-event time statistics for the dataset.

        This method computes statistics related to the time differences between consecutive events in the
        dataset. It calculates the minimum, mean (log), and standard deviation (log) of inter-event times.

        The computed statistics are stored as instance attributes: - self.mean_log_inter_event_time_min: Mean
        of log inter-event times - self.std_log_inter_event_time_min: Standard deviation of log inter-event
        times

        Raises:     ValueError: If the dataset is empty (no static DataFrames).

        Side effects:     - Sets the above-mentioned instance attributes.     - Logs warnings if any non-
        positive inter-event times are found.     - Removes subjects with invalid (non-positive) inter-event
        times from the dataset.

        TODO: allow use of inter-event time statistics for normalizing time-deltas
        """
        if len(self.static_dfs) == 0:
            raise ValueError(
                f"The {self.split} dataset is empty, there should be at least one static dataframe."
            )
        data_for_stats = pl.concat([x.lazy() for x in self.static_dfs.values()])
        stats = (
            data_for_stats.select(
                pl.col("time").list.diff().explode().drop_nulls().drop_nans().alias("inter_event_time")
            )
            .select(
                pl.col("inter_event_time").min().alias("min"),
                pl.col("inter_event_time").log().mean().alias("mean_log"),
                pl.col("inter_event_time").log().std().alias("std_log"),
            )
            .collect()
        )

        if stats["min"].item() <= timedelta(0):
            bad_inter_event_times = data_for_stats.filter(
                pl.col("time").list.diff().list.min() <= 0
            ).collect()
            bad_subject_ids = set(bad_inter_event_times["subject_id"].to_list())
            warning_strs = [
                f"Observed inter-event times <= 0 for {len(bad_inter_event_times)} subjects!",
                f"Bad Subject IDs: {', '.join(str(x) for x in bad_subject_ids)}",
                f"Global min: {stats['min'].item()}",
            ]
            if self.config.meds_dir is not None:
                fp = Path(self.config.meds_dir) / f"malformed_data_{self.split}.parquet"
                bad_inter_event_times.write_parquet(fp)
                warning_strs.append(f"Wrote malformed data records to {fp}")
            warning_strs.append("Removing malformed subjects")

            logger.warning("\n".join(warning_strs))

            self.index = [x for x in self.index if x[0] not in bad_subject_ids]

        self.mean_log_inter_event_time_min = stats["mean_log"].item()
        self.std_log_inter_event_time_min = stats["std_log"].item()

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

        Args:     idx (int): The index of the data point to retrieve.

        Returns:     dict: A dictionary containing the data for the specified index. The structure typically
        includes:         - 'time_delta_days': List of time deltas between events.         -
        'dynamic_indices': List of categorical metadata elements.         - 'dynamic_values': List of
        numerical metadata elements.         - 'static_indices': List of static categorical metadata elements.
        Additional keys may be present based on the dataset configuration.

        Notes:     - The exact structure of the output dictionary depends on the dataset configuration and the
        collate type specified.     - This method uses the SeedableMixin to ensure reproducibility in data
        loading.     - The returned data is typically in a 'raw' format and may require further processing or
        tensorization in the collate function.
        """
        return self._seeded_getitem(idx)

    @SeedableMixin.WithSeed
    @TimeableMixin.TimeAs
    def load_subject(
        self, subject_dynamic_data, subject_id: int, st: int, end: int
    ) -> dict[str, list[float]]:
        """Load and process data for a single subject.

        This method retrieves and processes the data for a specific subject, applying various transformations
        and filters based on the dataset configuration.

        Args:     subject_dynamic_data: The dynamic data for the subject.     subject_id (int): The ID of the
        subject to load.     st (int): The start index of the sequence to load.     end (int): The end index
        of the sequence to load.

        Returns:     dict: A dictionary containing the processed data for the subject. The exact contents
        depend on the dataset configuration but typically include:           - Static data (indices and
        values)           - Dynamic data (time series data, event codes, etc.)           - Sequence
        information (start and end indices, if configured)

        Raises:     ValueError: If the sequence length is invalid or if there are inconsistencies in the data
        shapes.

        Notes:     - This method applies sequence length constraints and sampling strategies       as
        specified in the dataset configuration.     - It handles different collate types (event_stream,
        triplet, etc.) differently.     - The method is decorated with @SeedableMixin.WithSeed to ensure
        reproducibility.
        """
        shard = self.subj_map[subject_id]
        subject_idx = self.subj_indices[subject_id]
        static_row = self.static_dfs[shard][subject_idx].to_dict()

        out = {
            "static_indices": static_row["static_indices"].item().to_list(),
            "static_values": static_row["static_values"].item().to_list(),
        }

        # TODO: remove this and handle flattening in the NRT class
        if self.config.collate_type == CollateType.event_stream:
            seq_len = end - st
        if self.config.collate_type != CollateType.event_stream:
            event_seq_len = end - st
            tensors = subject_dynamic_data.tensors
            seq_len = sum([array.size for array in tensors["dim1/code"][st:end]])
            if not seq_len >= event_seq_len:
                raise ValueError(
                    f"Measurement sequence length {seq_len} is less than event sequence length"
                    f" {event_seq_len}!"
                )
            tensors["dim1/numeric_value"] = np.concatenate(tensors["dim1/numeric_value"][st:end], axis=0)
            tensors["dim1/code"] = np.concatenate(tensors["dim1/code"][st:end], axis=0)
            seq_len = tensors["dim1/code"].shape[0]
            tensors["dim0/time_delta_days"] = subpad_vectors(
                tensors["dim0/time_delta_days"][st:end], tensors["dim1/bounds"][st:end]
            )
            st = 0
            end = st + seq_len

        if seq_len > self.config.max_seq_len:
            match self.config.subsequence_sampling_strategy:
                case SubsequenceSamplingStrategy.RANDOM:
                    start_offset = np.random.choice(seq_len - self.config.max_seq_len)
                case SubsequenceSamplingStrategy.TO_END:
                    start_offset = seq_len - self.config.max_seq_len
                case SubsequenceSamplingStrategy.FROM_START:
                    start_offset = 0
                case _:
                    raise ValueError(
                        f"Invalid subsequence sampling strategy {self.config.subsequence_sampling_strategy}!"
                    )

            st += start_offset
            end = min(end, st + self.config.max_seq_len)

        if self.config.do_include_subsequence_indices:
            out["start_idx"] = st
            out["end_idx"] = end

        if self.config.collate_type == CollateType.event_stream:
            out["dynamic"] = subject_dynamic_data[st:end]
        else:
            tensors["dim1/code"] = tensors["dim1/code"][st:end]
            tensors["dim1/numeric_value"] = tensors["dim1/numeric_value"][st:end]
            tensors["dim0/time_delta_days"] = tensors["dim0/time_delta_days"][st:end]
            out["dynamic"] = tensors

        if self.config.do_include_start_time_min:
            out["start_time"] = static_row["time"].item().to_list()[st]

        if end - st > self.config.max_seq_len:
            raise ValueError(f"Sequence length {end - st} exceeds max_seq_len {self.config.max_seq_len}!")
        if self.config.min_seq_len and (end - st < self.config.min_seq_len):
            raise ValueError(
                f"Sequence length {end - st} is less than min_seq_len {self.config.min_seq_len}!"
            )

        if end == st:
            raise ValueError(f"Sequence length {end - st} is 0!")

        if self.config.postpend_eos_token:
            if self.config.collate_type == CollateType.event_stream:
                # Append EOS token to the end of the sequence
                eos_token = np.array([self.config.EOS_TOKEN_ID], dtype=out["dynamic"]["dim1/code"].dtype)
                out["dynamic"]["dim1/code"] = np.append(out["dynamic"]["dim1/code"], eos_token)

                # Extend other relevant arrays
                numeric_dtype = out["dynamic"]["dim1/numeric_value"].dtype
                time_dtype = out["dynamic"]["dim0/time_delta_days"].dtype
                out["dynamic"]["dim1/numeric_value"] = np.append(
                    out["dynamic"]["dim1/numeric_value"], np.array([0], dtype=numeric_dtype)
                )
                out["dynamic"]["dim0/time_delta_days"] = np.append(
                    out["dynamic"]["dim0/time_delta_days"], np.array([0], dtype=time_dtype)
                )

            else:
                # For other collate types
                eos_token = np.array([self.config.EOS_TOKEN_ID], dtype=out["dynamic"]["dim1/code"].dtype)
                out["dynamic"]["dim1/code"] = np.append(out["dynamic"]["dim1/code"], eos_token)

                numeric_dtype = out["dynamic"]["dim1/numeric_value"].dtype
                time_dtype = out["dynamic"]["dim0/time_delta_days"].dtype
                out["dynamic"]["dim1/numeric_value"] = np.append(
                    out["dynamic"]["dim1/numeric_value"], np.array([0], dtype=numeric_dtype)
                )
                out["dynamic"]["dim0/time_delta_days"] = np.append(
                    out["dynamic"]["dim0/time_delta_days"], np.array([0], dtype=time_dtype)
                )

        # Update end_idx if it's included
        if self.config.do_include_subsequence_indices:
            out["end_idx"] = end

        if self.config.collate_type != CollateType.event_stream and not (
            len(out["dynamic"]["dim1/code"])
            == len(out["dynamic"]["dim1/numeric_value"])
            == len(out["dynamic"]["dim0/time_delta_days"])
        ):
            code_shape = out["dynamic"]["dim1/code"].shape
            numeric_shape = out["dynamic"]["dim1/numeric_value"].shape
            time_shape = out["dynamic"]["dim0/time_delta_days"].shape
            raise ValueError(f"Shape mismatch: {code_shape} vs {numeric_shape} vs {time_shape}")

        return out

    @SeedableMixin.WithSeed
    @TimeableMixin.TimeAs
    def _seeded_getitem(self, idx: int) -> dict[str, list[float]]:
        """Returns a Returns a dictionary corresponding to a single subject's data.

        This function is a seedable version of `__getitem__`.
        """

        subject_id, st, end = self.index[idx]

        shard = self.subj_map[subject_id]
        subject_idx = self.subj_indices[subject_id]

        subject_dynamic_data = JointNestedRaggedTensorDict.load_slice(
            Path(self.config.data_dir) / "data" / f"{shard}.nrt", subject_idx
        )
        out = self.load_subject(subject_dynamic_data, subject_id, st, end)

        if self.config.do_include_subject_id:
            out["subject_id"] = subject_id

        for t, t_labels in self.labels.items():
            out[t] = t_labels[idx]

        assert "dynamic" in out, f"Failed to load dynamic data for subject {subject_id} in {shard}!"
        return out

    @TimeableMixin.TimeAs
    def __dynamic_only_collate(self, batch: list[dict[str, list[float]]]) -> dict:
        """An internal collate function for only dynamic data."""
        keys = batch[0].keys()
        dense_keys = {k for k in keys if k not in ("dynamic", "static_indices", "static_values")}

        if dense_keys:
            dense_collated = torch.utils.data.default_collate([{k: x[k] for k in dense_keys} for x in batch])
        else:
            dense_collated = {}

        dynamic = JointNestedRaggedTensorDict.vstack([x["dynamic"] for x in batch]).to_dense(
            padding_side=self.config.seq_padding_side
        )
        dynamic["event_mask"] = dynamic.pop("dim1/mask")
        dynamic["dynamic_values"] = dynamic.pop("numeric_value")
        dynamic["dynamic_indices"] = dynamic.pop("code")
        dynamic["dynamic_values_mask"] = dynamic.pop("dim2/mask") & ~np.isnan(dynamic["dynamic_values"])

        dynamic_collated = {}
        for k, v in dynamic.items():
            if k.endswith("mask"):
                dynamic_collated[k] = torch.from_numpy(v)
            elif v.dtype in NP_UINT_TYPES + NP_INT_TYPES:
                dynamic_collated[k] = torch.from_numpy(v.astype(int)).long()
            elif v.dtype in NP_FLOAT_TYPES:
                dynamic_collated[k] = torch.from_numpy(v.astype(float)).float()
            else:
                raise TypeError(f"Don't know how to tensorify {k} of type {v.dtype}!")

        collated = {**dense_collated, **dynamic_collated}

        out_batch = {}
        out_batch["event_mask"] = collated["event_mask"]
        out_batch["dynamic_values_mask"] = collated["dynamic_values_mask"]
        out_batch["time_delta_days"] = torch.nan_to_num(collated["time_delta_days"].float(), nan=0)
        out_batch["dynamic_indices"] = collated["dynamic_indices"].long()
        out_batch["dynamic_values"] = torch.nan_to_num(collated["dynamic_values"].float(), nan=0)

        if self.config.do_include_start_time_min:
            out_batch["start_time"] = collated["start_time"].float()
        if self.config.do_include_subsequence_indices:
            out_batch["start_idx"] = collated["start_idx"].long()
            out_batch["end_idx"] = collated["end_idx"].long()
        if self.config.do_include_subject_id:
            out_batch["subject_id"] = collated["subject_id"].long()

        if not self.has_task:
            return out_batch

        out_labels = {}
        for task in self.tasks:
            match self.task_types[task]:
                case "multi_class_classification":
                    out_labels[task] = collated[task].long()
                case "binary_classification":
                    out_labels[task] = collated[task].float()
                case "regression":
                    out_labels[task] = collated[task].float()
                case _:
                    raise TypeError(f"Don't know how to tensorify task of type {self.task_types[task]}!")

        # add task labels
        for k in batch[0].keys():
            if k not in ("dynamic", "static_values", "static_indices"):
                out_batch[k] = torch.Tensor([item[k] for item in batch])

        return out_batch

    def collate_event_stream(self, batch: list[dict]) -> dict:
        """Combines the ragged dictionaries produced by `__getitem__` into a tensorized batch.

        This function handles conversion of arrays to tensors and padding of elements within the batch across
        static data elements, sequence events, and dynamic data elements.

        Args:     batch: A list of `__getitem__` format output dictionaries.

        Returns:     A fully collated, tensorized, and padded batch.
        """

        out_batch = self.__dynamic_only_collate(batch)

        max_n_static = max(len(x["static_indices"]) for x in batch)
        static_padded_fields = defaultdict(list)
        for e in batch:
            n_static = len(e["static_indices"])
            static_delta = max_n_static - n_static
            for k in ("static_indices", "static_values"):
                if static_delta > 0:
                    static_padded_fields[k].append(
                        torch.nn.functional.pad(
                            torch.tensor(e[k], dtype=torch.long), (0, static_delta), value=0
                        )
                    )
                else:
                    static_padded_fields[k].append(torch.tensor(e[k], dtype=torch.long))

        for k, v in static_padded_fields.items():
            out_batch[k] = torch.cat([T.unsqueeze(0) for T in v], dim=0)

        return out_batch

    @classmethod
    def process_triplet(cls, item: dict, do_prepend_static_data=True) -> dict:
        """Process a single triplet of dynamic and static data.

        This method takes a dictionary containing dynamic and static data for a single
        data point and processes it into a unified representation suitable for model input.

        Args:
            item (dict): A dictionary containing 'dynamic' and 'static' data.
            do_prepend_static_data (bool, optional): Whether to prepend static data
                                                     to the dynamic data. Defaults to True.

        Returns:
            dict: A processed dictionary containing:
                - mask: Boolean mask indicating valid data points.
                - static_mask: Boolean mask indicating static data points.
                - code: Concatenated static and dynamic codes.
                - numeric_value: Concatenated static and dynamic numerical values.
                - time_delta_days: Time deltas between events.
                - numeric_value_mask: Boolean mask for valid numeric values.

        Notes:
            - This method handles the integration of static and dynamic data.
            - It applies appropriate type conversions and handles missing values.
            - The resulting dictionary is suitable for further tensorization or batching.

        Examples:
            >>> import numpy as np
            >>> import tempfile, json, os
            >>> from omegaconf import DictConfig
            >>> item =  {
            ...         'dynamic': {
            ...                 'dim1/code': np.array([5, 6, 1, 2]),
            ...                 'dim1/numeric_value': np.array([50.0, 60.0, np.nan, np.nan]),
            ...                 'dim0/time_delta_days': np.array([0, 0, 12, 0])
            ...         },
            ...         'static_values': [70.0],
            ...         'static_indices': [2]
            ...     }
            >>> triplet_item = PytorchDataset.process_triplet(item)
            >>> for each in sorted(list(triplet_item.keys())): print(each)
            code
            mask
            numeric_value
            numeric_value_mask
            static_mask
            time_delta_days
            >>> for key, value in triplet_item.items(): print(key, value);
            mask [ True  True  True  True  True]
            static_mask [ True False False False False]
            code tensor([2, 5, 6, 1, 2])
            numeric_value [70. 50. 60.  0.  0.]
            time_delta_days [ 0  0  0 12  0]
            numeric_value_mask [ True  True  True False False]
        """
        dynamic_data = item["dynamic"]
        code = dynamic_data["dim1/code"]
        numeric_value = dynamic_data["dim1/numeric_value"]
        time_delta_days = dynamic_data["dim0/time_delta_days"]

        static_mask = np.zeros(len(code), dtype=bool)
        if do_prepend_static_data:
            static_values = np.asarray(item["static_values"], dtype=np.float32)
            static_indices = np.asarray(item["static_indices"], dtype=np.int32)
            code = np.concatenate([static_indices, code], dtype=np.int32, casting="unsafe")
            numeric_value = np.concatenate([static_values, numeric_value])
            static_mask = np.zeros(len(code), dtype=bool)
            static_mask[: len(static_values)] = True
            time_delta_days = np.concatenate(
                [np.zeros(len(static_values), dtype=time_delta_days.dtype), time_delta_days]
            )

        numeric_value_mask = ~np.isnan(numeric_value)
        # Replace NaNs with 0s
        np.nan_to_num(numeric_value, nan=0, copy=False)
        np.nan_to_num(time_delta_days, nan=0, copy=False)

        mask = np.ones(len(code), dtype=bool)
        if not len(mask) == len(code) == len(numeric_value) == len(time_delta_days):
            raise ValueError(
                f"Shape mismatch: {code.shape} vs {mask.shape} vs "
                f"{numeric_value.shape} vs {time_delta_days.shape}"
            )

        output = dict(
            mask=mask,
            static_mask=static_mask,
            code=torch.as_tensor(code, dtype=torch.int64),
            numeric_value=numeric_value,
            time_delta_days=time_delta_days,
            numeric_value_mask=numeric_value_mask,
        )
        return output

    @classmethod
    def collate_triplet(cls, batch: list[dict], do_prepend_static_data=True) -> dict:
        """Collate a batch of triplet format data into a unified batch dictionary.

        This method combines multiple data points in triplet format (times, codes, values)
        into a single batch, applying necessary padding and tensorization.

        Args:
            batch (list[dict]): A list of dictionaries, each representing a single data point.
            do_prepend_static_data (bool, optional): Whether to prepend static data to the
                                                     dynamic data. Defaults to True.

        Returns:
            dict: A dictionary containing the collated batch data, including:
                - mask: Tensor indicating valid data points across the batch.
                - static_mask: Tensor indicating static data points.
                - code: Tensor of concatenated static and dynamic codes.
                - numeric_value: Tensor of concatenated static and dynamic numerical values.
                - time_delta_days: Tensor of time deltas between events.
                - Additional task-specific labels if present in the input data.

        Notes:
            - This method handles padding to ensure uniform sequence lengths within the batch.
            - It converts all data to PyTorch tensors.
            - The method is flexible to handle additional task-specific data present in the input.

        Examples:
            >>> import torch
            >>> import numpy as np
            >>> batch = [
            ...     {
            ...         'dynamic': {
            ...              'dim1/code': np.array([1]),
            ...              'dim1/numeric_value': np.array([10.0]),
            ...              'dim0/time_delta_days': np.array([0])},
            ...         'static_values': [20.0],
            ...         'static_indices': [0],
            ...         'label': 0,
            ...     },
            ...     {
            ...         'dynamic':{
            ...              'dim1/code': np.array([5, 6, 1, 2]),
            ...              'dim1/numeric_value': np.array([50.0, 60.0, 0, 0]),
            ...              'dim0/time_delta_days': np.array([0, 0, 12, 0])},
            ...         'static_values': [70.0],
            ...         'static_indices': [2],
            ...         'label': 1,
            ...         },
            ... ]
            >>> collated_batch = PytorchDataset.collate_triplet(batch)
            >>> from pprint import pprint
            >>> pprint(collated_batch)
            {'code': tensor([[0, 1, 0, 0, 0],
                    [2, 5, 6, 1, 2]]),
             'label': tensor([0., 1.]),
             'mask': tensor([[ True,  True, False, False, False],
                    [ True,  True,  True,  True,  True]]),
             'numeric_value': tensor([[20., 10.,  0.,  0.,  0.],
                    [70., 50., 60.,  0.,  0.]], dtype=torch.float64),
             'numeric_value_mask': tensor([[ True,  True, False, False, False],
                    [ True,  True,  True,  True,  True]]),
             'static_mask': tensor([[ True, False, False, False, False],
                    [ True, False, False, False, False]]),
             'time_delta_days': tensor([[ 0,  0,  0,  0,  0],
                    [ 0,  0,  0, 12,  0]])}
        """
        processed_batch = [cls.process_triplet(item, do_prepend_static_data) for item in batch]
        tensorized_batch = {
            k: torch.nn.utils.rnn.pad_sequence(
                [torch.as_tensor(x[k]) for x in processed_batch],
                batch_first=True,
                padding_value=0,
            )
            for k in processed_batch[0].keys()
        }

        # Add task labels to batch
        for k in batch[0].keys():
            if k not in ("dynamic", "static_values", "static_indices"):
                tensorized_batch[k] = torch.Tensor([item[k] for item in batch])
        return tensorized_batch

    @classmethod
    def process_text_code(cls, item: dict, tokenized_codes: dict, do_prepend_static_data=True) -> dict:
        """Process a single data point for text-code format.

        This method takes a dictionary containing dynamic and static data and processes
        it into a format suitable for text-code based models, including tokenization of codes.

        Args:
            item (dict): A dictionary containing 'dynamic' and 'static' data.
            tokenized_codes (dict): A dictionary mapping codes to their tokenized representations.
            do_prepend_static_data (bool, optional): Whether to prepend static data
                                                     to the dynamic data. Defaults to True.

        Returns:
            dict: A processed dictionary containing:
                - mask: Boolean mask indicating valid data points.
                - static_mask: Boolean mask indicating static data points.
                - code: Original codes (not tokenized).
                - code_tokens: Tokenized representations of codes.
                - code_mask: Mask for tokenized code sequences.
                - numeric_value: Concatenated static and dynamic numerical values.
                - time_delta_days: Time deltas between events.
                - numeric_value_mask: Boolean mask for valid numeric values.

        Notes:
            - This method is specifically designed for models that use text representations of codes.
            - It integrates tokenization information with the original data structure.
            - The resulting dictionary is suitable for further tensorization or batching in text-code models.

        Examples:
            >>> import numpy as np
            >>> import tempfile, json, os
            >>> from torch import Tensor as tensor
            >>> from omegaconf import DictConfig
            >>> item =  {
            ...         'dynamic': {
            ...                 'dim1/code': np.array([5, 6, 1, 2]),
            ...                 'dim1/numeric_value': np.array([50.0, 60.0, np.nan, np.nan]),
            ...                 'dim0/time_delta_days': np.array([0, 0, 12, 0])
            ...         },
            ...         'static_values': [70.0],
            ...         'static_indices': [2]
            ...     }
            >>> tokenized_metadata = {
            ...     2: (tensor([1037, 2518,    0,    0]), tensor([1, 1, 0, 0])),
            ...     5: (tensor([2138,    0,    0,    0]), tensor([1, 0, 0, 0])),
            ...     6: (tensor([1039,    0,    0,    0]), tensor([1, 0, 0, 0])),
            ...     1: (tensor([2093, 1999, 1037, 5216]), tensor([1, 1, 1, 1]))}
            >>> text_code_item = PytorchDataset.process_text_code(item, tokenized_metadata)
            >>> for each in sorted(list(text_code_item.keys())): print(each)
            code
            code_mask
            code_tokens
            mask
            numeric_value
            numeric_value_mask
            static_mask
            time_delta_days
            >>> for key, value in text_code_item.items(): print(key, value);
            mask [ True  True  True  True  True]
            static_mask [ True False False False False]
            code [2 5 6 1 2]
            code_tokens [[1037. 2518.    0.    0.]
             [2138.    0.    0.    0.]
             [1039.    0.    0.    0.]
             [2093. 1999. 1037. 5216.]
             [1037. 2518.    0.    0.]]
            code_mask [[1. 1. 0. 0.]
             [1. 0. 0. 0.]
             [1. 0. 0. 0.]
             [1. 1. 1. 1.]
             [1. 1. 0. 0.]]
            numeric_value [70. 50. 60.  0.  0.]
            time_delta_days [ 0  0  0 12  0]
            numeric_value_mask [ True  True  True False False]
        """
        dynamic_data = item["dynamic"]
        code = dynamic_data["dim1/code"]
        numeric_value = dynamic_data["dim1/numeric_value"]
        time_delta_days = dynamic_data["dim0/time_delta_days"]

        static_mask = np.zeros(len(code), dtype=bool)
        if do_prepend_static_data:
            static_values = np.asarray(item["static_values"], dtype=np.float32)
            static_indices = np.asarray(item["static_indices"], dtype=np.int32)
            code = np.concatenate([static_indices, code], dtype=np.int32, casting="unsafe")
            numeric_value = np.concatenate([static_values, numeric_value])
            static_mask = np.zeros(len(code), dtype=bool)
            static_mask[: len(static_values)] = True
            time_delta_days = np.concatenate(
                [np.zeros(len(static_values), dtype=time_delta_days.dtype), time_delta_days]
            )

        numeric_value_mask = ~np.isnan(numeric_value)
        # Replace NaNs with 0s
        np.nan_to_num(numeric_value, nan=0, copy=False)
        np.nan_to_num(time_delta_days, nan=0, copy=False)

        mask = np.ones(len(time_delta_days), dtype=bool)

        tokens = [tokenized_codes[c] for c in code]
        code_tokens, code_mask = zip(*tokens)
        code_tokens = np.array(code_tokens)
        code_mask = np.array(code_mask)

        if not mask.shape == code.shape:
            raise ValueError(f"Code and mask shape mismatch: {code.shape} vs {mask.shape}")

        output = dict(
            mask=mask,
            static_mask=static_mask,
            code=code,
            code_tokens=code_tokens,
            code_mask=code_mask,
            numeric_value=numeric_value,
            time_delta_days=time_delta_days,
            numeric_value_mask=numeric_value_mask,
        )
        return output

    @classmethod
    def collate_text_code(cls, tokenized_codes: dict, batch: list[dict], prepend_static_data) -> dict:
        """Collate a batch of text-code format data into a unified batch dictionary.

        This method combines multiple data points in text-code format into a single batch, applying necessary
        padding, tensorization, and handling of tokenized code representations.

        Args:     tokenized_codes (dict): A dictionary mapping codes to their tokenized representations. batch
        (list[dict]): A list of dictionaries, each representing a single data point. prepend_static_data
        (bool): Whether to prepend static data to the dynamic data.

        Returns:     dict: A dictionary containing the collated batch data, including:         - mask: Tensor
        indicating valid data points across the batch.         - static_mask: Tensor indicating static data
        points.         - code: Tensor of original codes.         - code_tokens: Tensor of tokenized code
        representations.         - code_mask: Tensor mask for tokenized code sequences.         -
        numeric_value: Tensor of numerical values.         - time_delta_days: Tensor of time deltas between
        events.         - Additional task-specific labels if present in the input data.

        Notes:     - This method is specifically designed for models that use text representations of codes. -
        It handles padding to ensure uniform sequence lengths within the batch.     - All data is converted to
        PyTorch tensors.     - The method is flexible to handle additional task-specific data present in the
        input.
        """
        processed_batch = [
            cls.process_text_code(item, tokenized_codes, prepend_static_data) for item in batch
        ]
        tensorized_batch = {
            k: torch.nn.utils.rnn.pad_sequence(
                [torch.as_tensor(x[k]) for x in processed_batch],
                batch_first=True,
                padding_value=0,
            )
            for k in processed_batch[0].keys()
        }
        for k in batch[0].keys():
            if k not in ("dynamic", "static_values", "static_indices"):
                tensorized_batch[k] = torch.Tensor([item[k] for item in batch])

        return tensorized_batch

    @classmethod
    def tokenize_metadata(cls, tokenizer, code_metadata, padding=True, special_tokens={}) -> dict:
        """Tokenize metadata using the provided tokenizer.

        This class method applies tokenization to the metadata, typically used for
        processing code descriptions or other textual metadata in the dataset.

        Args:
            tokenizer: The tokenizer object to use for tokenization.
            code_metadata (pl.LazyFrame): A LazyFrame containing the metadata to tokenize.
            padding (bool, optional): Whether to apply padding to the tokenized outputs.
                                      Defaults to True.
            special_tokens (dict, optional): A dictionary of special tokens to include
                                             in the tokenization process. Defaults to an empty dict.

        Returns:
            dict: A dictionary mapping vocabulary indices to tuples of (tokenized_output, attention_mask).

        Notes:
            - The method preprocesses the metadata by replacing '//' and '_' with spaces.
            - Special tokens, if provided, are tokenized along with the metadata.
            - The output dictionary uses the 'code/vocab_index' as keys.
            - This method is crucial for preparing textual data for use in models that require
                tokenized input.

        Examples:
        >>> from transformers import AutoTokenizer
        >>> import polars as pl
        >>> tokenizer = AutoTokenizer.from_pretrained("emilyalsentzer/Bio_ClinicalBERT")
        >>> code_metadata = pl.LazyFrame({
        ...     "code": ["A//thing", "Because", "C", "three//in_a//row"],
        ...     "code/vocab_index": [2, 5, 6, 1]
        ...     })
        >>> tokenized_metadata = PytorchDataset.tokenize_metadata(tokenizer, code_metadata)
        >>> for each in tokenized_metadata.items(): print(each)
        (2, ([170, 1645, 0, 0], [1, 1, 0, 0]))
        (5, ([1272, 0, 0, 0], [1, 0, 0, 0]))
        (6, ([172, 0, 0, 0], [1, 0, 0, 0]))
        (1, ([1210, 1107, 170, 5105], [1, 1, 1, 1]))
        """
        code_metadata = code_metadata.with_columns(
            pl.col("code").fill_null("").str.replace_all("//", " ").str.replace_all("_", " ")
        )
        special_token_keys = list(special_tokens.keys())
        special_token_values = [special_tokens[k] for k in special_token_keys]
        tokens = tokenizer(
            code_metadata.select("code").collect().to_series().to_list() + special_token_values,
            padding=padding,
            return_token_type_ids=False,
            add_special_tokens=False,
            truncation=True,
        )
        token_key = list(tokens.keys())[0]
        mask_key = list(tokens.keys())[1]

        token_dict = dict(
            zip(
                code_metadata.select("code/vocab_index").collect().to_series().to_list() + special_token_keys,
                zip(tokens[token_key], tokens[mask_key]),
            )
        )
        return token_dict

    @TimeableMixin.TimeAs
    def collate(self, batch: list[dict]) -> dict:
        """Combine a batch of data points into a single, tensorized batch.

        This method serves as the main collation function, handling different collation strategies based on
        the dataset configuration. It delegates to specific collation methods depending on the collate_type
        specified in the configuration.

        Args:     batch (list[dict]): A list of dictionaries, each representing a single data point as
        returned by the __getitem__ method.

        Returns:     dict: A dictionary containing the collated batch data. The exact structure depends on the
        collation strategy used.

        Raises:     NotImplementedError: If an unsupported collate type is specified in the configuration.

        Notes:     - The method supports various collation strategies including event_stream,       triplet,
        triplet_prompt, eic, and text_code.     - Each collation strategy is optimized for different model
        architectures and data representations.     - The collated output is fully tensorized and padded,
        ready for input into a PyTorch model.
        """
        collate_type = self.config.collate_type
        if collate_type == CollateType.event_stream:
            return self.collate_event_stream(batch)
        elif collate_type == CollateType.triplet_prompt:
            return self.collate_triplet(batch)
        elif collate_type == CollateType.eic:
            return self.collate_triplet(batch, self.config.do_prepend_static_data)
        elif collate_type == CollateType.triplet:
            return self.collate_triplet(batch, self.config.do_prepend_static_data)
        elif collate_type == CollateType.text_code:
            return self.collate_text_code(self.tokenized_codes, batch, self.config.do_prepend_static_data)
        else:
            raise NotImplementedError(f"Unsupported collate type {collate_type}!")
