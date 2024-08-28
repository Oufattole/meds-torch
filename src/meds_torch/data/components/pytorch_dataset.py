import json
import os
from collections import defaultdict
from datetime import timedelta
from enum import StrEnum
from pathlib import Path

import numpy as np

MAX_POLARS_THREADS = 1
import polars as pl
import torch
from loguru import logger
from mixins import SeedableMixin
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
    text_observation = "text_observation"
    all_text = "all_text"
    triplet_prompt = "triplet_prompt"
    eic = "eic"


def generate_patient_split_dict(meds_dir):
    patient_split_dict = {}

    for split_dir in os.listdir(meds_dir):
        split_path = Path(meds_dir) / split_dir
        if split_path.is_dir():
            for shard_file in split_path.glob("*.parquet"):
                split_name = f"{split_dir}/{shard_file.stem}"
                df = pl.read_parquet(shard_file)
                patient_ids = df["patient_id"].unique().to_list()
                patient_split_dict[split_name] = patient_ids
        else:
            logger.warning(f"Directory {split_path} does not exist or is not a directory.")

    return patient_split_dict


def debug_fn(data_dict, max_seq_len=None):
    for key, value in data_dict.items():
        if isinstance(value, torch.Tensor):
            if torch.isnan(value).any() or torch.isinf(value).any():
                raise ValueError(f"Found NaN or infinite values in key: {key}")
            if max_seq_len and value.shape[-1] > max_seq_len:
                raise ValueError(f"Found sequence length {value.shape[-1]} in key: {key}")
        elif isinstance(value, np.ndarray):
            if max_seq_len and np.isnan(value).any() or np.isinf(value).any():
                raise ValueError(f"Found NaN or infinite values in key: {key}")
            if value.shape[-1] > max_seq_len:
                raise ValueError(f"Found sequence length {value.shape[-1]} in key: {key}")
        else:
            raise TypeError(f"Unsupported data type for key: {key}")


def subpad_vectors(a, b):
    """Calculate the total length of the output array Create an array of zeros with the total length Place the
    values from 'a' at the indices specified by 'b'.

    Example:
    >>> a = np.array([2, 4, 5])
    >>> b = np.array([3, 5, 10])
    >>> subpad_vectors(a, b)
    array([0, 0, 0, 2, 0, 4, 0, 0, 0, 5, 0])
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
    """An enumeration of the possible subsequence sampling strategies for the dataset."""

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


def merge_task_with_static(task_df, static_dfs):
    """Merges a DataFrame containing task information with multiple static DataFrames on the 'patient_id'
    column. The function performs a sequence of operations to merge these dataframes based on patient
    identifiers and respective timestamps.

    Parameters:
    - task_df (DataFrame): A DataFrame with columns 'patient_id', 'start_time', 'end_time', and 'label'.
    - static_dfs (dict of DataFrames): A dictionary of DataFrames indexed by their source names,
      each containing 'patient_id', 'start_time', 'static_indices', 'static_values', and "time".

    Returns:
    - DataFrame: The merged DataFrame containing data from task_df and all static_dfs.

    # Example:
    # >>> from datetime import datetime
    # >>> import polars as pl
    # >>> task_df = pl.DataFrame({
    # ...     "patient_id": [1, 2],
    # ...     "start_time": [datetime(2020, 1, 1), datetime(2020, 1, 2)],
    # ...     "end_time": [datetime(2020, 1, 2), datetime(2020, 1, 3)],
    # ...     "label": [0, 1]
    # ... })
    # >>> static_dfs = {
    # ...     'train/0': pl.DataFrame({
    # ...         "patient_id": [1, 2],
    # ...         "start_time": [datetime(2020, 1, 1), datetime(2020, 1, 2)],
    # ...         "time": [[datetime(2020, 1, 1, 1), datetime(2020, 1, 1, 3)],
    # ...                       [datetime(2020, 1, 2), datetime(2020, 1, 1, 2, 3)]]
    # ...     })
    # ... }
    # >>> merge_task_with_static(task_df, static_dfs)
    # shape: (2, 6)
    # ┌────────────┬────────────┬─────────────┬────────────┬────────────┬────────────┐
    # │ _row_index ┆ patient_id ┆ start_time  ┆ end_time   ┆ start_time ┆ timestamp  │
    # │ ---        ┆ ---        ┆ ---         ┆ ---        ┆ _global    ┆ ---        │
    # │ u32        ┆ i64        ┆ list[dateti ┆ list[datet ┆ ---        ┆ list[datet │
    # │            ┆            ┆ me[μs]]     ┆ ime[μs]]   ┆ datetime[μ ┆ ime[μs]]   │
    # │            ┆            ┆             ┆            ┆ s]         ┆            │
    # ╞════════════╪════════════╪═════════════╪════════════╪════════════╪════════════╡
    # │ 0          ┆ 1          ┆ [2020-01-01 ┆ [2020-01-0 ┆ 2020-01-01 ┆ [2020-01-0 │
    # │            ┆            ┆ 00:00:00]   ┆ 2          ┆ 00:00:00   ┆ 1          │
    # │            ┆            ┆             ┆ 00:00:00]  ┆            ┆ 01:00:00,  │
    # │            ┆            ┆             ┆            ┆            ┆ 2020-01-…  │
    # │ 1          ┆ 2          ┆ [2020-01-02 ┆ [2020-01-0 ┆ 2020-01-02 ┆ [2020-01-0 │
    # │            ┆            ┆ 00:00:00]   ┆ 3          ┆ 00:00:00   ┆ 2          │
    # │            ┆            ┆             ┆ 00:00:00]  ┆            ┆ 00:00:00,  │
    # │            ┆            ┆             ┆            ┆            ┆ 2020-01-…  │
    # └────────────┴────────────┴─────────────┴────────────┴────────────┴────────────┘
    """
    task_df_joint = (
        task_df.select("patient_id", "start_time", "end_time")
        .with_row_index(IDX_COL)
        .group_by(IDX_COL, "patient_id", maintain_order=True)
        .agg("start_time", "end_time")
        .join(
            pl.concat(static_dfs.values()).select(
                "patient_id", pl.col("start_time").alias("start_time_global"), "time"
            ),
            on="patient_id",
            how="left",
        )
        .with_columns(pl.col("time"))
    )
    return task_df_joint


def get_task_indexes(task_df_joint) -> list[tuple[int, int, int]]:
    """Processes the joint DataFrame to determine the index range for each patient's tasks.

    For each row in task_df_joint, it is assumed that `timestamp` is a sorted column
    and the start index and end index of the span of timestamps in between `start_time` and `end_time`
    are computed.

    Parameters:
    - task_df_joint (DataFrame): A DataFrame resulting from the merge_task_with_static function.

    Returns:
    - list: list of tuples (patient_id, start_idx, end_idx).

    Example:
    >>> from datetime import datetime
    >>> df = pl.DataFrame({
    ...     IDX_COL: [i for i in range(5)],
    ...     "patient_id": [i for i in range(5)],
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
    ...     ]*5
    ... })
    >>> get_task_indexes(df)
    [(0, 0, 1), (1, 0, 1), (2, 0, 2), (3, 1, 3), (4, 2, 3)]
    """
    start_idx_expr = (
        (pl.col("time").search_sorted(pl.col("start_time"), side="left")).first().alias("start_idx")
    )
    end_idx_expr = (pl.col("time").search_sorted(pl.col("end_time"), side="left")).last().alias("end_idx")
    task_df_joint = (
        task_df_joint.explode("start_time", "end_time")
        .explode("time")
        .group_by(IDX_COL, "patient_id", "start_time", "end_time", maintain_order=True)
        .agg(start_idx_expr, end_idx_expr)
    )

    patient_ids = task_df_joint["patient_id"]
    start_indices = task_df_joint["start_idx"]
    end_indices = task_df_joint["end_idx"]

    indexes = list(zip(patient_ids, start_indices, end_indices))

    return indexes


class PytorchDataset(SeedableMixin, torch.utils.data.Dataset):
    """A PyTorch Dataset class.

    Args:     config: Configuration options for the dataset, in an `omegaconf.DictConfig` object.     split:
    The split of data which should be used in this dataset (e.g., ``'train'``, ``'tuning'``, ``'held_out'``).
    This will dictate where the system looks for files.
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
        """Normalizes the task labels in `col` of dtype `dtype` to a common format.

        Args:     col: The column containing the task labels, in polars expression format.     dtype: The
        polars data type of the task labels.

        Returns:     The task type (a string key into the `TYPE_CHECKERS` dictionary) and the normalized
        column     expression.

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

        logger.info("Reading splits & patient shards")
        self.read_shards()

        logger.info("Reading patient descriptors")
        self.read_patient_descriptors()

        if self.config.min_seq_len is not None and self.config.min_seq_len > 1:
            logger.info(f"Restricting to subjects with at least {self.config.min_seq_len} events")
            self.filter_to_min_seq_len()

        if self.config.train_subset_size not in (None, "FULL") and self.split == "train":
            logger.info(f"Filtering training subset size to {self.config.train_subset_size}")
            self.filter_to_subset()

        self.set_inter_event_time_stats()

    def read_shards(self):
        """Reads the split-specific patient shards from the ESGPT or MEDS dataset."""
        all_shards = generate_patient_split_dict(Path(self.config.meds_cohort_dir) / "data")
        self.shards = {sp: subjs for sp, subjs in all_shards.items() if sp.startswith(f"{self.split}")}
        self.subj_map = {subj: sp for sp, subjs in self.shards.items() for subj in subjs}
        if not self.shards:
            logger.warning(
                f"No shards found for split {self.split}. Check the directory structure and file names."
            )

    def read_patient_descriptors(self):
        """Reads the patient schemas and static data."""
        self.static_dfs = {}
        self.subj_indices = {}
        self.subj_seq_bounds = {}

        for shard in self.shards.keys():
            static_fp = Path(self.config.schema_files_root) / f"{shard}.parquet"
            df = (
                pl.read_parquet(
                    static_fp,
                    columns=[
                        "patient_id",
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
            patient_ids = df["patient_id"]
            n_events = df.select(pl.col("time").list.len().alias("n_events")).get_column("n_events")
            for i, (subj, n_events_count) in enumerate(zip(patient_ids, n_events)):
                # TODO fix bug where n_events_count is not the same as the number of events in the
                # tensorized data, seems to be shifting up by 1 multiple times
                # if not n_events_count == JointNestedRaggedTensorDict.load_slice(
                #     Path(self.config.tensorized_root)
                #     / f"{shard}.nrt", i).tensors["dim0/time_delta_days"].shape[0]:
                #     logger.info(f"Event count mismatch for {subj} in {shard}!")
                if subj in self.subj_indices or subj in self.subj_seq_bounds:
                    raise ValueError(f"Duplicate subject {subj} in {shard}!")

                self.subj_indices[subj] = i
                self.subj_seq_bounds[subj] = (0, n_events_count)

        if self.has_task:
            if self.config.task_root_dir is None:
                raise ValueError("`task_root_dir` must be provided if task is specified!")
            task_df_fp = Path(self.config.task_label_path)
            task_info_fp = Path(self.config.task_info_path)

            logger.info(f"Reading task constraints for {self.config.task_name} from {task_df_fp}")
            task_df = pl.read_parquet(task_df_fp, use_pyarrow=True)

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

            task_df_joint = merge_task_with_static(task_df, self.static_dfs)
            # Filter out patients that are not in the split
            split_patients = set(
                pl.concat(self.static_dfs.values())
                .select(pl.col("patient_id").unique())
                .to_series()
                .to_list()
            )
            task_df_joint = task_df_joint.filter(pl.col("patient_id").is_in(split_patients))
            # Convert dates to indexes in the nested ragged tensor, (for fast indexing of data)
            self.index = get_task_indexes(task_df_joint)
            self.labels = {t: task_df.get_column(t).to_list() for t in self.tasks}
        else:
            self.index = [(subj, *bounds) for subj, bounds in self.subj_seq_bounds.items()]
            self.labels = {}
            self.tasks = None
            self.task_types = None
            self.task_vocabs = None

    def get_task_info(self, task_df: pl.DataFrame):
        """Gets the task information from the task dataframe."""
        self.tasks = sorted([c for c in task_df.columns if c not in ["patient_id", "start_time", "end_time"]])

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
        """Filters the dataset to only include subjects with at least `self.config.min_seq_len` events."""
        if self.has_task:
            logger.warning(
                f"Filtering task {self.config.task_name} to min_seq_len {self.config.min_seq_len}. "
                "This may result in incomparable model results against runs with different constraints!"
            )

        orig_len = len(self)
        orig_n_subjects = len(set(self.patient_ids))
        valid_indices = [
            i for i, (subj, start, end) in enumerate(self.index) if end - start >= self.config.min_seq_len
        ]
        self.index = [self.index[i] for i in valid_indices]
        self.labels = {t: [t_labels[i] for i in valid_indices] for t, t_labels in self.labels.items()}
        new_len = len(self)
        new_n_subjects = len(set(self.patient_ids))
        logger.info(
            f"Filtered data due to sequence length constraint (>= {self.config.min_seq_len}) from "
            f"{orig_len} to {new_len} rows and {orig_n_subjects} to {new_n_subjects} subjects."
        )

    def filter_to_subset(self):
        """Filters the dataset to only include a subset of subjects."""

        orig_len = len(self)
        orig_n_subjects = len(set(self.patient_ids))
        rng = np.random.default_rng(self.config.train_subset_seed)
        subset_subjects = rng.choice(
            list(set(self.patient_ids)),
            size=count_or_proportion(orig_n_subjects, self.config.train_subset_size),
            replace=False,
        )
        valid_indices = [i for i, (subj, start, end) in enumerate(self.index) if subj in subset_subjects]
        self.index = [self.index[i] for i in valid_indices]
        self.labels = {t: [t_labels[i] for i in valid_indices] for t, t_labels in self.labels.items()}
        new_len = len(self)
        new_n_subjects = len(set(self.patient_ids))
        logger.info(
            f"Filtered data to subset of {self.config.train_subset_size} subjects from "
            f"{orig_len} to {new_len} rows and {orig_n_subjects} to {new_n_subjects} subjects."
        )

    @classmethod
    def tokenize_batch(cls, tokenizer, batch: list[str], padding=False) -> dict:
        """Tokenizes the batch using the provided tokenizer.

        Args:     tokenizer: The tokenizer to use.     batch: The batch to tokenize.

        Returns:     A dictionary containing the tokenized batch.
        """
        output = tokenizer(
            batch,
            padding=padding,
            return_attention_mask=padding,
            # return_tensors="pt",
            return_token_type_ids=False,
            add_special_tokens=False,
        )
        # if len(output.keys()) == 1:
        #     return output[list(output.keys())[0]]
        return output

    def set_inter_event_time_stats(self):
        """Sets the inter-event time statistics for the dataset."""
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
            bad_patient_ids = set(bad_inter_event_times["patient_id"].to_list())
            warning_strs = [
                f"Observed inter-event times <= 0 for {len(bad_inter_event_times)} subjects!",
                f"Bad Subject IDs: {', '.join(str(x) for x in bad_patient_ids)}",
                f"Global min: {stats['min'].item()}",
            ]
            if self.config.meds_dir is not None:
                fp = Path(self.config.meds_dir) / f"malformed_data_{self.split}.parquet"
                bad_inter_event_times.write_parquet(fp)
                warning_strs.append(f"Wrote malformed data records to {fp}")
            warning_strs.append("Removing malformed subjects")

            logger.warning("\n".join(warning_strs))

            self.index = [x for x in self.index if x[0] not in bad_patient_ids]

        self.mean_log_inter_event_time_min = stats["mean_log"].item()
        self.std_log_inter_event_time_min = stats["std_log"].item()

    @property
    def patient_ids(self) -> list[int]:
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
        """Returns a Returns a dictionary corresponding to a single subject's data.

        The output of this will not be tensorized as that work will need to be re-done in the collate function
        regardless. The output will have structure: `` {     'time_delta_days': [seq_len], 'dynamic_indices':
        [seq_len, n_data_per_event] (ragged),     'dynamic_values': [seq_len, n_data_per_event] (ragged),
        'static_indices': [seq_len, n_data_per_event] (ragged), } ``

        1. ``time_delta_days`` captures the time between each event and the subsequent event in days. 2.
        ``dynamic_indices`` captures the categorical metadata elements listed in `self.data_cols` in a unified
        vocabulary space spanning all metadata vocabularies. 3. ``dynamic_values`` captures the numerical
        metadata elements listed in `self.data_cols`. If no    numerical elements are listed in
        `self.data_cols` for a given categorical column, the according    index in this output will be
        `np.NaN`. 5. ``static_indices`` captures the categorical metadata elements listed in
        `self.static_cols` in a    unified vocabulary.
        """
        return self._seeded_getitem(idx)

    @SeedableMixin.WithSeed
    def load_patient(
        self, patient_dynamic_data, patient_id: int, st: int, end: int
    ) -> dict[str, list[float]]:
        """Returns a Returns a dictionary corresponding to a single subject's data.

        This function is a seedable version of `__getitem__`.
        """
        shard = self.subj_map[patient_id]
        patient_idx = self.subj_indices[patient_id]
        static_row = self.static_dfs[shard][patient_idx].to_dict()

        out = {
            "static_indices": static_row["static_indices"].item().to_list(),
            "static_values": static_row["static_values"].item().to_list(),
        }

        # TODO: remove this check after fixing the end bug, sometimes it is for the wrong patient
        # TODO check the dataset ground truth length where n_event and end differ!
        end = min(patient_dynamic_data.tensors["dim0/time_delta_days"].shape[0], end)
        # TODO: remove this and handle flattening in the NRT class
        # if self.config.collate_type == CollateType.triplet:
        #     end = sum(len(array) for array in patient_dynamic_data.tensors["dim1/numeric_value"])
        if self.config.collate_type == CollateType.event_stream:
            seq_len = end - st
        if self.config.collate_type != CollateType.event_stream:
            tensors = patient_dynamic_data.tensors
            tensors["dim1/numeric_value"] = np.concatenate(tensors["dim1/numeric_value"], axis=0)
            tensors["dim1/code"] = np.concatenate(tensors["dim1/code"], axis=0)
            seq_len = tensors["dim1/code"].shape[0]
            st = 0
            if self.config.collate_type != CollateType.eic:
                # TODO: pad times
                tensors["dim0/time_delta_days"] = subpad_vectors(
                    tensors["dim0/time_delta_days"], tensors["dim1/bounds"]
                )

        if seq_len > self.max_seq_len:
            match self.config.subsequence_sampling_strategy:
                case SubsequenceSamplingStrategy.RANDOM:
                    start_offset = np.random.choice(seq_len - self.max_seq_len)
                case SubsequenceSamplingStrategy.TO_END:
                    start_offset = seq_len - self.max_seq_len
                case SubsequenceSamplingStrategy.FROM_START:
                    start_offset = 0
                case _:
                    raise ValueError(
                        f"Invalid subsequence sampling strategy {self.config.subsequence_sampling_strategy}!"
                    )

            st += start_offset
            end = min(end, st + self.max_seq_len)

        if self.config.do_include_subsequence_indices:
            out["start_idx"] = st
            out["end_idx"] = end

        if self.config.collate_type == CollateType.event_stream:
            out["dynamic"] = patient_dynamic_data[st:end]
        else:
            tensors["dim1/code"] = tensors["dim1/code"][st:end]
            if self.config.collate_type != CollateType.eic:
                tensors["dim1/numeric_value"] = tensors["dim1/numeric_value"][st:end]
                tensors["dim0/time_delta_days"] = tensors["dim0/time_delta_days"][st:end]
            out["dynamic"] = tensors

        if self.config.do_include_start_time_min:
            out["start_time"] = static_row["time"].item().to_list()[st]

        if end - st > self.config.max_seq_len:
            raise ValueError(f"Sequence length {end - st} exceeds max_seq_len {self.config.max_seq_len}!")

        if end == st:
            raise ValueError(f"Sequence length {end - st} is 0!")

        return out

    @SeedableMixin.WithSeed
    def _seeded_getitem(self, idx: int) -> dict[str, list[float]]:
        """Returns a Returns a dictionary corresponding to a single subject's data.

        This function is a seedable version of `__getitem__`.
        """

        patient_id, st, end = self.index[idx]

        shard = self.subj_map[patient_id]
        patient_idx = self.subj_indices[patient_id]

        patient_dynamic_data = JointNestedRaggedTensorDict.load_slice(
            Path(self.config.data_dir) / "data" / f"{shard}.nrt", patient_idx
        )
        out = self.load_patient(patient_dynamic_data, patient_id, st, end)

        if self.config.do_include_patient_id:
            out["patient_id"] = patient_id

        for t, t_labels in self.labels.items():
            out[t] = t_labels[idx]

        assert "dynamic" in out, f"Failed to load dynamic data for patient {patient_id} in {shard}!"
        return out

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
        if self.config.do_include_patient_id:
            out_batch["patient_id"] = collated["patient_id"].long()

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
        """Processes a single triplet of dynamic and static data.

        This function takes a dictionary containing dynamic and static data,
        processes the tensors, and concatenates them appropriately to create
        a unified representation of the data.

        Args:
            item: A dictionary containing 'dynamic' and 'static' data.

        Returns:
            A dictionary with the processed data including:
                - mask: A mask indicating valid data points.
                - static_mask: A mask indicating static data points.
                - code: Concatenated static and dynamic codes.
                - numerical_value: Concatenated static and dynamic numerical values.
                - time_delta_days: Concatenated static and dynamic time deltas.

        Examples:
            >>> import numpy as np
            >>> import tempfile, json, os
            >>> from omegaconf import DictConfig
            >>> item =  {
            ...         'dynamic': JointNestedRaggedTensorDict({
            ...                 'code': [np.array([5, 6]), np.array([1, 2])],
            ...                 'numeric_value': [np.array([50.0, 60.0]), np.array([np.nan, np.nan])],
            ...                 'time_delta_days': np.array([np.nan, 12])
            ...         }),
            ...         'static_values': [70.0],
            ...         'static_indices': [2]
            ...     }
            >>> triplet_item = PytorchDataset.process_triplet(item)
            >>> for each in sorted(list(triplet_item.keys())): print(each)
            code
            mask
            numerical_value
            numerical_value_mask
            static_mask
            time_delta_days
            >>> for key, value in triplet_item.items(): print(key, value);
            mask [ True  True  True  True  True]
            static_mask [ True False False False False]
            code [2 5 6 1 2]
            numerical_value [70. 50. 60.  0.  0.]
            time_delta_days [ 0.  0.  0. 12. 12.]
            numerical_value_mask [ True  True  True False False]
        """
        dynamic_data = item["dynamic"]
        code = dynamic_data["dim1/code"]
        numerical_value = dynamic_data["dim1/numeric_value"]
        time_delta_days = dynamic_data["dim0/time_delta_days"]

        static_mask = np.zeros(len(code), dtype=bool)
        if do_prepend_static_data:
            static_values = np.asarray(item["static_values"], dtype=np.float32)
            static_indices = np.asarray(item["static_indices"], dtype=np.int32)
            code = np.concatenate([static_indices, code], dtype=np.int32, casting="unsafe")
            numerical_value = np.concatenate([static_values, numerical_value])
            static_mask = np.zeros(len(code), dtype=bool)
            static_mask[: len(static_values)] = True

        numerical_value_mask = ~np.isnan(numerical_value)
        # Replace NaNs with 0s
        np.nan_to_num(numerical_value, nan=0, copy=False)
        np.nan_to_num(time_delta_days, nan=0, copy=False)

        mask = np.ones(len(time_delta_days), dtype=bool)

        output = dict(
            mask=mask,
            static_mask=static_mask,
            code=torch.as_tensor(code, dtype=torch.int64),
            numeric_value=numerical_value,
            time_delta_days=time_delta_days,
            numerical_value_mask=numerical_value_mask,
        )
        # debug_fn(output)
        return output

    @classmethod
    def collate_triplet(cls, batch: list[dict], do_prepend_static_data=True) -> dict:
        """Combines the ragged dictionaries  into a triplet format (times, codes, values) batch.

        This function handles conversion of arrays to tensors and padding of elements within the
        batch across static data elements, sequence events, and dynamic data elements. It ensures
        that each batch has uniform shape by padding shorter sequences with zeros.

        Args:
            batch: A list of dictionaries with dynamic and static data from `__getitem__` method outputs.

        Returns:
            A dictionary containing tensorized and padded data for each key. The keys include 'mask',
            'static_mask', 'code', 'numeric_value', 'numerical_value_mask', and 'time_delta_days'.

        Examples:
            >>> import torch
            >>> import numpy as np
            >>> batch = [
            ...     {
            ...         'dynamic': JointNestedRaggedTensorDict(
            ...             {'code': [np.array([1])],
            ...              'numeric_value': [np.array([10.0])],
            ...              'time_delta_days': np.array([0])}),
            ...         'static_values': [20.0],
            ...         'static_indices': [0],
            ...         'label': 0,
            ...     },
            ...     {
            ...         'dynamic': JointNestedRaggedTensorDict(
            ...             {'code': [np.array([5, 6]), np.array([1, 2])],
            ...              'numeric_value': [np.array([50.0, 60.0]), np.array([np.nan, np.nan])],
            ...              'time_delta_days': np.array([np.nan, 12])}),
            ...         'static_values': [70.0],
            ...         'static_indices': [2],
            ...         'label': 1,
            ...         },
            ... ]
            >>> collated_batch = PytorchDataset.collate_triplet(batch)
            >>> from pprint import pprint
            >>> pprint(collated_batch)
            {'code': tensor([[0, 1, 0, 0, 0],
                    [2, 5, 6, 1, 2]], dtype=torch.int32),
             'label': tensor([0., 1.]),
             'mask': tensor([[ True,  True, False, False, False],
                    [ True,  True,  True,  True,  True]]),
             'numeric_value': tensor([[20., 10.,  0.,  0.,  0.],
                    [70., 50., 60.,  0.,  0.]]),
             'numerical_value_mask': tensor([[ True,  True, False, False, False],
                    [ True,  True,  True, False, False]]),
             'static_mask': tensor([[ True, False, False, False, False],
                    [ True, False, False, False, False]]),
             'time_delta_days': tensor([[ 0,  0,  0,  0,  0],
                    [ 0,  0,  0, 12, 12]], dtype=torch.uint8)}
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
        # debug_fn(tensorized_batch)
        return tensorized_batch

    @classmethod
    def collate_triplet_prompt(cls, batch: list[dict]) -> dict:
        """Combines the ragged dictionaries  into a triplet format (times, codes, values) batch.

        This function handles conversion of arrays to tensors and padding of elements within the
        batch across static data elements, sequence events, and dynamic data elements. It ensures
        that each batch has uniform shape by padding shorter sequences with zeros.

        Args:
            batch: A list of dictionaries with dynamic and static data from `__getitem__` method outputs.

        Returns:
            A dictionary containing tensorized and padded data for each key. The keys include 'mask',
            'static_mask', 'code', 'numeric_value', 'numerical_value_mask', and 'time_delta_days'.

        Examples:
            >>> import torch
            >>> import numpy as np
            >>> batch = [
            ...     {
            ...         'dynamic': JointNestedRaggedTensorDict(
            ...             {'code': [np.array([1])],
            ...              'numeric_value': [np.array([10.0])],
            ...              'time_delta_days': np.array([0])}),
            ...         'static_values': [20.0],
            ...         'static_indices': [0],
            ...         'label': 0,
            ...     },
            ...     {
            ...         'dynamic': JointNestedRaggedTensorDict(
            ...             {'code': [np.array([5, 6]), np.array([1, 2])],
            ...              'numeric_value': [np.array([50.0, 60.0]), np.array([np.nan, np.nan])],
            ...              'time_delta_days': np.array([np.nan, 12])}),
            ...         'static_values': [70.0],
            ...         'static_indices': [2],
            ...         'label': 1,
            ...         },
            ... ]
            >>> collated_batch = PytorchDataset.collate_triplet(batch)
            >>> from pprint import pprint
            >>> pprint(collated_batch)
            {'code': tensor([[0, 1, 0, 0, 0],
                    [2, 5, 6, 1, 2]], dtype=torch.int32),
             'label': tensor([0., 1.]),
             'mask': tensor([[ True,  True, False, False, False],
                    [ True,  True,  True,  True,  True]]),
             'numeric_value': tensor([[20., 10.,  0.,  0.,  0.],
                    [70., 50., 60.,  0.,  0.]]),
             'numerical_value_mask': tensor([[ True,  True, False, False, False],
                    [ True,  True,  True, False, False]]),
             'static_mask': tensor([[ True, False, False, False, False],
                    [ True, False, False, False, False]]),
             'time_delta_days': tensor([[ 0,  0,  0,  0,  0],
                    [ 0,  0,  0, 12, 12]], dtype=torch.uint8)}
        """
        processed_batch = [cls.process_triplet(item) for item in batch]
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
    def process_text_code(cls, item: dict, tokenized_codes: dict) -> dict:
        """Processes a single triplet of dynamic and static data.

        This function takes a dictionary containing dynamic and static data,
        processes the tensors, and concatenates them appropriately to create
        a unified representation of the data.

        Args:
            item: A dictionary containing 'dynamic' and 'static' data.
            tokenized_codes: A dictionary containing the tokenized codes.

        Returns:
            A dictionary with the processed data including:
                - mask: A mask indicating valid data points.
                - static_mask: A mask indicating static data points.
                - code: Concatenated static and dynamic codes.
                - numerical_value: Concatenated static and dynamic numerical values.
                - time_delta_days: Concatenated static and dynamic time deltas.

        Examples:
            >>> import numpy as np
            >>> import tempfile, json, os
            >>> from torch import Tensor as tensor
            >>> from omegaconf import DictConfig
            >>> item =  {
            ...         'dynamic': JointNestedRaggedTensorDict({
            ...                 'code': [np.array([5, 6]), np.array([1, 2])],
            ...                 'numeric_value': [np.array([50.0, 60.0]), np.array([np.nan, np.nan])],
            ...                 'time_delta_days': np.array([np.nan, 12])
            ...         }),
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
            code_mask
            code_tokens
            mask
            numerical_value
            numerical_value_mask
            static_mask
            time_delta_days
            >>> for key, value in text_code_item.items(): print(key, value);
            mask [ True  True  True  True  True]
            static_mask [ True False False False False]
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
            numerical_value [70. 50. 60.  0.  0.]
            time_delta_days [ 0.  0.  0. 12. 12.]
            numerical_value_mask [ True  True  True False False]
        """
        dynamic_data = item["dynamic"]
        raw_codes = dynamic_data["dim1/code"]
        raw_values = dynamic_data["dim1/numeric_value"]
        raw_times = dynamic_data["dim0/time_delta_days"]

        static_values = np.asarray(item["static_values"], dtype=raw_values[0].dtype)
        static_indices = np.asarray(item["static_indices"], dtype=raw_codes[0].dtype)
        code = np.concatenate([static_indices, raw_codes], dtype=np.int32, casting="unsafe")
        tokens = [tokenized_codes[c] for c in code]
        code_tokens, code_mask = zip(*tokens)
        code_tokens = np.array(code_tokens)
        code_mask = np.array(code_mask)
        numerical_value = np.concatenate([static_values, raw_values])
        numerical_value_mask = ~np.isnan(numerical_value)
        # Replace NaNs with 0s
        np.nan_to_num(numerical_value, nan=0, copy=False)
        np.nan_to_num(raw_times, nan=0, copy=False)

        static_mask = np.zeros(len(code), dtype=bool)
        static_mask[: len(static_values)] = True

        lengths = np.concatenate([[len(static_values)], dynamic_data["dim1/lengths"]])
        time_delta_days = np.repeat(
            np.concatenate([np.array([0], dtype=raw_times.dtype), raw_times]), lengths
        )
        mask = np.ones(len(time_delta_days), dtype=bool)

        return dict(
            mask=mask,
            static_mask=static_mask,
            code_tokens=code_tokens,
            code_mask=code_mask,
            numeric_value=numerical_value,
            time_delta_days=time_delta_days,
            numerical_value_mask=numerical_value_mask,
        )

    @classmethod
    def process_text_observation(
        cls, item: dict, tokenized_codes: dict, tokenized_sentence, tokenizer, whole_observation=False
    ) -> dict:
        """Processes a single triplet of dynamic and static data.

        This function takes a dictionary containing dynamic and static data,
        processes the tensors, and concatenates them appropriately to create
        a unified representation of the data.

        Args:
            item: A dictionary containing 'dynamic' and 'static' data.
            tokenized_codes: A dictionary containing the tokenized codes.

        Returns:
            A dictionary with the processed data including:
                - mask: A mask indicating valid data points.
                - static_mask: A mask indicating static data points.
                - code: Concatenated static and dynamic codes.
                - numerical_value: Concatenated static and dynamic numerical values.
                - time_delta_days: Concatenated static and dynamic time deltas.

        Examples:
            >>> import numpy as np
            >>> import tempfile, json, os
            >>> from torch import Tensor as tensor
            >>> from omegaconf import DictConfig
            >>> item =  {
            ...         'dynamic': JointNestedRaggedTensorDict({
            ...                 'code': [np.array([5, 6]), np.array([1, 2])],
            ...                 'numeric_value': [np.array([50.0, 60.0]), np.array([np.nan, np.nan])],
            ...                 'time_delta_days': np.array([np.nan, 12])
            ...         }),
            ...         'static_values': [70.0],
            ...         'static_indices': [2]
            ...     }
            >>> tokenized_metadata = {
            ...     2: (tensor([1037, 2518,    0,    0]), tensor([1, 1, 0, 0])),
            ...     5: (tensor([2138,    0,    0,    0]), tensor([1, 0, 0, 0])),
            ...     6: (tensor([1039,    0,    0,    0]), tensor([1, 0, 0, 0])),
            ...     1: (tensor([2093, 1999, 1037, 5216]), tensor([1, 1, 1, 1]))}
            >>> tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
            >>> tokenized_sentence = [[3642], [2038, 3643], [7594], [2044, 1996, 3025, 8089], [1012], [10]]
            >>> text_code_item = PytorchDataset.process_text_observation(item,
            ...     tokenized_metadata, tokenized_sentence, tokenizer)
            >>> for each in sorted(list(text_code_item.keys())): print(each)
            mask
            observation_mask
            observation_tokens
            >>> for key, value in text_code_item.items(): print(key, value);
            mask tensor([True, True, True, True, True])
            observation_tokens tensor([[3642., 1037., 2518.,    0.,    0., 2038., 3643., 3963., 1012., 1014.,
                     7594., 1014., 1012., 1014., 2044., 1996., 3025., 8089., 1012.],
                    [3642., 2138.,    0.,    0.,    0., 2038., 3643., 2753., 1012., 1014.,
                     1012.,    0.,    0.,    0.,    0.,    0.,    0.,    0.,    0.],
                    [3642., 1039.,    0.,    0.,    0., 2038., 3643., 3438., 1012., 1014.,
                     1012.,    0.,    0.,    0.,    0.,    0.,    0.,    0.,    0.],
                    [3642., 2093., 1999., 1037., 5216., 1012.,    0.,    0.,    0.,    0.,
                        0.,    0.,    0.,    0.,    0.,    0.,    0.,    0.,    0.],
                    [3642., 1037., 2518.,    0.,    0., 1012.,    0.,    0.,    0.,    0.,
                        0.,    0.,    0.,    0.,    0.,    0.,    0.,    0.,    0.]])
            observation_mask tensor([[1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
                     1.],
                    [1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 0., 0., 0., 0., 0., 0., 0.,
                     0.],
                    [1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 0., 0., 0., 0., 0., 0., 0.,
                     0.],
                    [1., 1., 1., 1., 1., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
                     0.],
                    [1., 1., 1., 1., 1., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
                     0.]])
            >>> all_text_items = PytorchDataset.process_text_observation(item,
            ...         tokenized_metadata, tokenized_sentence, tokenizer, whole_observation=True)
            >>> for each in sorted(list(all_text_items.keys())): print(each)
            mask
            observation_mask
            observation_tokens
            >>> for key, value in all_text_items.items(): print(key, value);
            mask tensor([True])
            observation_tokens tensor([[3642., 1037., 2518.,    0.,    0., 2038., 3643., 3963., 1012., 1014.,
                     7594., 1014., 1012., 1014., 2044., 1996., 3025., 8089., 1012.,   10.,
                     3642., 2138.,    0.,    0.,    0., 2038., 3643., 2753., 1012., 1014.,
                     1012.,   10., 3642., 1039.,    0.,    0.,    0., 2038., 3643., 3438.,
                     1012., 1014., 1012.,   10., 3642., 2093., 1999., 1037., 5216., 1012.,
                       10., 3642., 1037., 2518.,    0.,    0., 1012.,   10.]])
            observation_mask tensor([[1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
                     1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
                     1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
                     1., 1., 1., 1.]])
        """
        dynamic_data = item["dynamic"]
        raw_codes = dynamic_data["dim1/code"]
        raw_values = dynamic_data["dim1/numeric_value"]
        raw_times = dynamic_data["dim0/time_delta_days"]

        static_values = np.asarray(item["static_values"], dtype=raw_values[0].dtype)
        static_indices = np.asarray(item["static_indices"], dtype=raw_codes[0].dtype)
        code = np.concatenate([static_indices, raw_codes], dtype=np.int32, casting="unsafe")
        tokens = [tokenized_codes[c] for c in code]
        code_tokens, code_mask = zip(*tokens)

        numerical_value = np.concatenate([static_values, raw_values])
        numerical_value_mask = ~np.isnan(numerical_value)
        # # Replace NaNs with 0s
        np.nan_to_num(numerical_value, nan=0, copy=False)
        np.nan_to_num(raw_times, nan=0, copy=False)

        static_mask = np.zeros(len(code), dtype=bool)
        static_mask[: len(static_values)] = True

        lengths = np.concatenate([[len(static_values)], dynamic_data["dim1/lengths"]])
        time_delta_days = np.repeat(
            np.concatenate([np.array([0], dtype=raw_times.dtype), raw_times]), lengths
        )
        tokenized_values = cls.tokenize_batch(tokenizer, numerical_value.astype(str).tolist(), padding=False)
        tokenized_values = tokenized_values[list(tokenized_values.keys())[0]]
        tokenized_time = cls.tokenize_batch(tokenizer, time_delta_days.astype(str).tolist(), padding=False)
        tokenized_time = tokenized_time[list(tokenized_time.keys())[0]]

        tokenized_observations = []
        for i, code_token in enumerate(code_tokens):
            observation = []
            observation.extend(tokenized_sentence[0] + list(code_token))
            if numerical_value_mask[i]:
                observation.extend(tokenized_sentence[1] + tokenized_values[i])
            if static_mask[i]:
                observation.extend(tokenized_sentence[2] + tokenized_time[i] + tokenized_sentence[3])
            observation.extend(tokenized_sentence[4])
            tokenized_observations.append(torch.tensor(observation))

        if whole_observation:
            suffix_tensor = torch.tensor(tokenized_sentence[5])
            tokenized_observations = [
                torch.cat([torch.cat([observation, suffix_tensor]) for observation in tokenized_observations])
            ]

        tokenized_observations_padded = torch.nn.utils.rnn.pad_sequence(
            tokenized_observations,
            batch_first=True,
            padding_value=0,
        )
        observation_mask = torch.nn.utils.rnn.pad_sequence(
            [torch.ones(len(observation)) for observation in tokenized_observations],
            batch_first=True,
            padding_value=0,
        )

        return dict(
            mask=observation_mask[:, 0].bool(),
            # static_mask=static_mask,
            observation_tokens=tokenized_observations_padded,
            observation_mask=observation_mask,
            # numeric_value=numerical_value,
            # time_delta_days=time_delta_days,
            # numerical_value_mask=numerical_value_mask,
        )

    @classmethod
    def collate_text_code(cls, tokenized_codes: dict, batch: list[dict]) -> dict:
        """Combines the ragged dictionaries produced by `__getitem__` into a tensorized batch.

        This function handles conversion of arrays to tensors and padding of elements within the batch across
        static data elements, sequence observations, and dynamic data elements.

        Args:     code_metadata: A DataFrame containing the code metadata.     batch: A list of dictionaries
        with dynamic and static data from `__getitem__` method outputs.

        Returns:     A dictionary containing tensorized and padded data for each key. The keys include 'mask',
        'static_mask', 'code_text', 'code_text_mask', 'numeric_value', 'numerical_value_mask',     and
        'time_delta_days'.
        """

        processed_batch = [cls.process_text_code(item, tokenized_codes) for item in batch]
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
    def tokenize_metadata(cls, tokenizer, code_metadata, padding=True) -> dict:
        """Tokenizes the metadata using the provided tokenizer.

        Args:
            tokenizer: The tokenizer to use.
            metadata: The metadata to tokenize.

        Returns:
            A list of tokenized metadata.

        Examples:
        >>> from transformers import AutoTokenizer
        >>> import polars as pl
        >>> tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
        >>> code_metadata = pl.LazyFrame({
        ...     "code": ["A//thing", "Because", "C", "three//in_a//row"],
        ...     "code/vocab_index": [2, 5, 6, 1]
        ...     })
        >>> tokenized_metadata = PytorchDataset.tokenize_metadata(tokenizer, code_metadata)
        >>> for each in tokenized_metadata.items(): print(each)
        (2, ([1037, 2518, 0, 0], [1, 1, 0, 0]))
        (5, ([2138, 0, 0, 0], [1, 0, 0, 0]))
        (6, ([1039, 0, 0, 0], [1, 0, 0, 0]))
        (1, ([2093, 1999, 1037, 5216], [1, 1, 1, 1]))
        """
        # dict where code/vocab_index is key and value is code
        # check if this is too slow --> should we move it somewhere else so we do it once and not per batch?
        # change values so any // or _ is replaced with a space, check if value is not None
        # make sure this is actually what we want
        code_metadata = code_metadata.with_columns(
            pl.col("code").fill_null("").str.replace_all("//", " ").str.replace_all("_", " ")
        )
        tokens = tokenizer(
            code_metadata.select("code").collect().to_series().to_list(),
            padding=padding,
            # return_tensors="pt",
            return_token_type_ids=False,
            add_special_tokens=False,
        )
        # TODO(@oufattole) generalize to other hf models
        token_key = list(tokens.keys())[0]
        mask_key = list(tokens.keys())[1]
        return dict(
            zip(
                code_metadata.select("code/vocab_index").collect().to_series().to_list(),
                zip(tokens[token_key], tokens[mask_key]),
            )
        )

    @classmethod
    def collate_text_observation(
        cls, batch: list[dict], tokenized_codes, tokenized_sentence, tokenizer, whole_observation=False
    ) -> dict:
        """Combines the ragged dictionaries produced by `__getitem__` into a tensorized batch.

        This function handles conversion of arrays to tensors and padding of elements within the batch across
        static data elements, sequence observations, and dynamic data elements.

        Args:     batch: A list of dictionaries with dynamic and static data from `__getitem__` method
        outputs.

        Returns:     A dictionary containing tensorized and padded data for each key. The keys include 'mask',
        'static_mask', 'code_text', 'code_text_mask', 'numeric_value', 'numerical_value_mask',     and
        'time_delta_days'.
        """
        processed_batch = [
            cls.process_text_observation(
                item, tokenized_codes, tokenized_sentence, tokenizer, whole_observation=whole_observation
            )
            for item in batch
        ]

        max_sizes = {}
        for k in processed_batch[0].keys():
            max_size = max([x[k].shape[1] for x in processed_batch if x[k].ndim > 1] + [0])
            if max_size > 0:
                max_sizes[k] = max_size
        for k in max_sizes.keys():
            # pad the tensors for everything with multiple dims to the max size
            for x in processed_batch:
                if x[k].ndim == 1:
                    # add dimension
                    x[k] = x[k].unsqueeze(0)
                x[k] = torch.nn.functional.pad(x[k], (0, max_sizes[k] - x[k].shape[1]), value=0)

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

    def collate(self, batch: list[dict]) -> dict:
        """Combines the ragged dictionaries produced by `__getitem__` into a tensorized batch.

        This function handles conversion of arrays to tensors and padding of elements within the batch across
        static data elements, sequence observations, and dynamic data elements.

        Args:     batch: A list of `__getitem__` format output dictionaries.

        Returns:     A fully collated, tensorized, and padded batch.
        """
        collate_type = self.config.collate_type
        if collate_type == CollateType.event_stream:
            return self.collate_event_stream(batch)
        elif collate_type == CollateType.triplet_prompt:
            return self.collate_triplet_prompt(batch)
        elif collate_type == CollateType.eic:
            return self.collate_triplet(batch)
        elif collate_type == CollateType.triplet:
            return self.collate_triplet(batch, self.config.do_prepend_static_data)
        elif collate_type == CollateType.text_code:
            # check this
            if not hasattr(self, "tokenized_codes"):
                tokenizer = AutoTokenizer.from_pretrained(self.config.tokenizer)
                self.tokenized_codes = self.tokenize_metadata(tokenizer, self.code_metadata)
            return self.collate_text_code(self.tokenized_codes, batch)
        elif collate_type == CollateType.text_observation:
            if not hasattr(self, "tokenized_codes"):
                self.tokenizer = AutoTokenizer.from_pretrained(self.config.tokenizer)
                self.tokenized_codes = self.tokenize_metadata(
                    self.tokenizer, self.code_metadata, padding=False
                )
                self.tokenized_sentence = self.tokenize_batch(
                    self.tokenizer,
                    ["Code", "has value", "measured", "after the previous observation", ".", "\n"],
                )
                # TODO(@oufattole) generalize to other hf models
                self.tokenized_sentence = self.tokenized_sentence[list(self.tokenized_sentence.keys())[0]]
            return self.collate_text_observation(
                batch, self.tokenized_codes, self.tokenized_sentence, self.tokenizer
            )
        elif collate_type == CollateType.all_text:
            if not hasattr(self, "tokenized_codes"):
                self.tokenizer = AutoTokenizer.from_pretrained(self.config.tokenizer)
                self.tokenized_codes = self.tokenize_metadata(
                    self.tokenizer, self.code_metadata, padding=False
                )
                self.tokenized_sentence = self.tokenize_batch(
                    self.tokenizer,
                    ["Code", "has value", "measured", "after the previous observation", ".", "\n"],
                )
                # TODO(@oufattole) generalize to other hf models
                self.tokenized_sentence = self.tokenized_sentence[list(self.tokenized_sentence.keys())[0]]
            return self.collate_text_observation(
                batch, self.tokenized_codes, self.tokenized_sentence, self.tokenizer, whole_observation=True
            )
        else:
            raise NotImplementedError(f"Unsupported collate type {collate_type}!")
