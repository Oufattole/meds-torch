import json
from collections import defaultdict
from datetime import timedelta
from enum import Enum, StrEnum
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

IDX_COL = "_row_index"


class CollateType(Enum):
    event_stream = "event_stream"
    triplet = "triplet"


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
      each containing 'patient_id', 'start_time', 'static_indices', 'static_values', and 'timestamp'.

    Returns:
    - DataFrame: The merged DataFrame containing data from task_df and all static_dfs.

    Example:
    >>> from datetime import datetime
    >>> import polars as pl
    >>> task_df = pl.DataFrame({
    ...     "patient_id": [1, 2],
    ...     "start_time": [datetime(2020, 1, 1), datetime(2020, 1, 2)],
    ...     "end_time": [datetime(2020, 1, 2), datetime(2020, 1, 3)],
    ...     "label": [0, 1]
    ... })
    >>> static_dfs = {
    ...     'train/0': pl.DataFrame({
    ...         "patient_id": [1, 2],
    ...         "start_time": [datetime(2020, 1, 1), datetime(2020, 1, 2)],
    ...         "timestamp": [[datetime(2020, 1, 1, 1), datetime(2020, 1, 1, 3)],
    ...                       [datetime(2020, 1, 2), datetime(2020, 1, 1, 2, 3)]]
    ...     })
    ... }
    >>> merge_task_with_static(task_df, static_dfs)
    shape: (2, 6)
    ┌────────────┬────────────┬─────────────┬────────────┬────────────┬────────────┐
    │ _row_index ┆ patient_id ┆ start_time  ┆ end_time   ┆ start_time ┆ timestamp  │
    │ ---        ┆ ---        ┆ ---         ┆ ---        ┆ _global    ┆ ---        │
    │ u32        ┆ i64        ┆ list[dateti ┆ list[datet ┆ ---        ┆ list[datet │
    │            ┆            ┆ me[μs]]     ┆ ime[μs]]   ┆ datetime[μ ┆ ime[μs]]   │
    │            ┆            ┆             ┆            ┆ s]         ┆            │
    ╞════════════╪════════════╪═════════════╪════════════╪════════════╪════════════╡
    │ 0          ┆ 1          ┆ [2020-01-01 ┆ [2020-01-0 ┆ 2020-01-01 ┆ [2020-01-0 │
    │            ┆            ┆ 00:00:00]   ┆ 2          ┆ 00:00:00   ┆ 1          │
    │            ┆            ┆             ┆ 00:00:00]  ┆            ┆ 01:00:00,  │
    │            ┆            ┆             ┆            ┆            ┆ 2020-01-…  │
    │ 1          ┆ 2          ┆ [2020-01-02 ┆ [2020-01-0 ┆ 2020-01-02 ┆ [2020-01-0 │
    │            ┆            ┆ 00:00:00]   ┆ 3          ┆ 00:00:00   ┆ 2          │
    │            ┆            ┆             ┆ 00:00:00]  ┆            ┆ 00:00:00,  │
    │            ┆            ┆             ┆            ┆            ┆ 2020-01-…  │
    └────────────┴────────────┴─────────────┴────────────┴────────────┴────────────┘
    """
    task_df_joint = (
        task_df.select("patient_id", "start_time", "end_time")
        .with_row_index(IDX_COL)
        .group_by(IDX_COL, "patient_id", maintain_order=True)
        .agg("start_time", "end_time")
        .join(
            pl.concat(static_dfs.values()).select(
                "patient_id", pl.col("start_time").alias("start_time_global"), "timestamp"
            ),
            on="patient_id",
            how="left",
        )
        .with_columns(pl.col("timestamp"))
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
    ...     "timestamp": [
    ...         pl.date_range(datetime(2021, 1, 1), datetime(2021, 1, 5), "1d", eager=True)
    ...     ]*5
    ... })
    >>> get_task_indexes(df)
    [(0, 0, 1), (1, 0, 1), (2, 0, 2), (3, 1, 3), (4, 2, 3)]
    """
    start_idx_expr = (
        (pl.col("timestamp").search_sorted(pl.col("start_time"), side="left")).first().alias("start_idx")
    )
    end_idx_expr = (
        (pl.col("timestamp").search_sorted(pl.col("end_time"), side="left")).last().alias("end_idx")
    )
    task_df_joint.explode("start_time", "end_time")
    task_df_joint = (
        task_df_joint.explode("start_time", "end_time")
        .explode("timestamp")
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
        all_shards = json.loads(Path(self.config.split_shards_fp).read_text())
        self.shards = {sp: subjs for sp, subjs in all_shards.items() if sp.startswith(f"{self.split}/")}
        self.subj_map = {subj: sp for sp, subjs in self.shards.items() for subj in subjs}

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
                        "numerical_value",
                        "timestamp",
                    ],
                    use_pyarrow=True,
                )
                .rename({"code": "static_indices", "numerical_value": "static_values"})
                .with_columns(
                    pl.col("static_values").list.eval(pl.element().fill_null(0)),
                    pl.col("static_indices").list.eval(pl.element().fill_null(0)),
                )
            )

            self.static_dfs[shard] = df
            patient_ids = df["patient_id"]
            n_events = df.select(pl.col("timestamp").list.len().alias("n_events")).get_column("n_events")
            for i, (subj, n_events) in enumerate(zip(patient_ids, n_events)):
                if subj in self.subj_indices or subj in self.subj_seq_bounds:
                    raise ValueError(f"Duplicate subject {subj} in {shard}!")

                self.subj_indices[subj] = i
                self.subj_seq_bounds[subj] = (0, n_events)

        if self.has_task:
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

    def set_inter_event_time_stats(self):
        """Sets the inter-event time statistics for the dataset."""
        if len(self.static_dfs) == 0:
            raise ValueError(
                f"The {self.split} dataset is empty, there should be at least one static dataframe."
            )
        data_for_stats = pl.concat([x.lazy() for x in self.static_dfs.values()])
        stats = (
            data_for_stats.select(
                pl.col("timestamp").list.diff().explode().drop_nulls().drop_nans().alias("inter_event_time")
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
                pl.col("timestamp").list.diff().list.min() <= 0
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

        seq_len = end - st
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

        out["dynamic"] = patient_dynamic_data[st:end]

        if self.config.do_include_start_time_min:
            out["start_time"] = static_row["timestamp"].item().to_list()[st]

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
            Path(self.config.tensorized_root) / f"{shard}.nrt", patient_idx
        )
        out = self.load_patient(patient_dynamic_data, patient_id, st, end)

        if self.config.do_include_patient_id:
            out["patient_id"] = patient_id

        for t, t_labels in self.labels.items():
            out[t] = t_labels[idx]

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
        dynamic["dynamic_values"] = dynamic.pop("numerical_value")
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
    def process_triplet(cls, item: dict) -> dict:
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
            ...                 'numerical_value': [np.array([50.0, 60.0]), np.array([np.nan, np.nan])],
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
        raw_codes = dynamic_data.tensors["dim1/code"]
        raw_values = dynamic_data.tensors["dim1/numerical_value"]
        raw_times = dynamic_data.tensors["dim0/time_delta_days"]

        static_values = np.asarray(item["static_values"], dtype=np.float32)
        static_indices = np.asarray(item["static_indices"], dtype=np.int32)
        code = np.concatenate([np.array(static_indices)] + raw_codes, dtype=np.int32, casting="unsafe")
        numerical_value = np.concatenate([np.array(static_values)] + raw_values)
        numerical_value_mask = ~np.isnan(numerical_value)
        # Replace NaNs with 0s
        np.nan_to_num(numerical_value, nan=0, copy=False)
        np.nan_to_num(raw_times, nan=0, copy=False)

        static_mask = np.zeros(len(code), dtype=bool)
        static_mask[: len(static_values)] = True

        lengths = np.concatenate([[len(static_values)], dynamic_data.tensors["dim1/lengths"]])
        time_delta_days = np.repeat(
            np.concatenate([np.array([0], dtype=raw_times.dtype), raw_times]), lengths
        )
        mask = np.ones(len(time_delta_days), dtype=bool)

        return dict(
            mask=mask,
            static_mask=static_mask,
            code=code,
            numerical_value=numerical_value,
            time_delta_days=time_delta_days,
            numerical_value_mask=numerical_value_mask,
        )

    @classmethod
    def collate_triplet(cls, batch: list[dict]) -> dict:
        """Combines the ragged dictionaries  into a triplet format (times, codes, values) batch.

        This function handles conversion of arrays to tensors and padding of elements within the
        batch across static data elements, sequence events, and dynamic data elements. It ensures
        that each batch has uniform shape by padding shorter sequences with zeros.

        Args:
            batch: A list of dictionaries with dynamic and static data from `__getitem__` method outputs.

        Returns:
            A dictionary containing tensorized and padded data for each key. The keys include 'mask',
            'static_mask', 'code', 'numerical_value', 'numerical_value_mask', and 'time_delta_days'.

        Examples:
        Examples:
            >>> import torch
            >>> import numpy as np
            >>> batch = [
            ...     {
            ...         'dynamic': JointNestedRaggedTensorDict(
            ...             {'code': [np.array([1])],
            ...              'numerical_value': [np.array([10.0])],
            ...              'time_delta_days': np.array([0])}),
            ...         'static_values': [20.0],
            ...         'static_indices': [0],
            ...         'label': 0,
            ...     },
            ...     {
            ...         'dynamic': JointNestedRaggedTensorDict(
            ...             {'code': [np.array([5, 6]), np.array([1, 2])],
            ...              'numerical_value': [np.array([50.0, 60.0]), np.array([np.nan, np.nan])],
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
             'numerical_value': tensor([[20., 10.,  0.,  0.,  0.],
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

    def collate(self, batch: list[dict]) -> dict:
        """Combines the ragged dictionaries produced by `__getitem__` into a tensorized batch.

        This function handles conversion of arrays to tensors and padding of elements within the batch across
        static data elements, sequence events, and dynamic data elements.

        Args:     batch: A list of `__getitem__` format output dictionaries.

        Returns:     A fully collated, tensorized, and padded batch.
        """
        collate_type = CollateType[self.config.collate_type]
        if collate_type == CollateType.event_stream:
            return self.collate_event_stream(batch)
        elif collate_type == CollateType.triplet:
            return self.collate_triplet(batch)
        else:
            raise NotImplementedError(f"Unsupported collate type {collate_type}!")
