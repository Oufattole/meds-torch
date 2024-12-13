import datetime
from dataclasses import asdict, dataclass
from enum import StrEnum
from pathlib import Path

import numpy as np
import polars as pl
import torch
from filelock import FileLock
from loguru import logger
from mixins import SeedableMixin
from nested_ragged_tensors.ragged_numpy import JointNestedRaggedTensorDict
from omegaconf import DictConfig

from meds_torch.data.components.pytorch_dataset import (
    DummyConfig,
    PytorchDataset,
    create_dummy_dataset,
)


def fuse_window_data(windows_data: dict, windows_to_fuse: list[str], fused_window_name: str) -> dict:
    """Fuse multiple windows into a single window, tracking the lengths of original windows.

    Warning:
        - This function assumes that the static data is not prepended to the dynamic data
        - We also assume that static data is the same for all time windows, as it is invariant
            to time windows.

    Args:
        windows_data (dict): Dictionary containing data from multiple windows to be fused
        windows_to_fuse (list[str]): List of window names to fuse in specified order
        fused_window_name (str): Name for the resulting fused window

    Returns:
        dict: Fused window data with length tracking information

    Raises:
        ValueError: If fusion configuration is invalid or data type is unsupported

    Example:
    >>> import torch
    >>> # Create mock data with different types
    >>> windows_data = {
    ...     "pre": {
    ...         # 1D tensor
    ...         "static_values": torch.tensor([1.0, 2.0]),
    ...         # 2D tensor (batch_size=2, seq_len=3, features=2)
    ...         "embeddings": torch.ones(2, 3, 2),
    ...         # List
    ...         "codes": ["A", "B", "C"],
    ...     },
    ...     "post": {
    ...         "static_values": torch.tensor([3.0, 4.0]),
    ...         "embeddings": torch.ones(2, 2, 2) * 2,
    ...         "codes": ["D", "E"],
    ...     }
    ... }
    >>> # Fuse windows
    >>> fused = fuse_window_data(
    ...     windows_data,
    ...     windows_to_fuse=["pre", "post"],
    ...     fused_window_name="fused"
    ... )
    Traceback (most recent call last):
    ...
    ValueError: Unsupported data type <class 'list'> for key codes
    >>> # Remove the list keys
    >>> del windows_data['pre']['codes']
    >>> del windows_data['post']['codes']
    >>> fused = fuse_window_data(
    ...     windows_data,
    ...     windows_to_fuse=["pre", "post"],
    ...     fused_window_name="fused"
    ... )
    >>> # Check lengths tracking
    >>> fused["LENGTHS//static_values"]  # Two windows should select just the first one
    [2, 2]
    >>> fused["LENGTHS//embeddings"]  # Two windows with sequence lengths 3 and 2
    [3, 2]
    >>> # Check only first window's static values (as static data is the same across all windows)
    >>> torch.equal(fused["static_values"], torch.tensor([1.0, 2.0]))
    True
    >>> # Check 2D tensor concatenation along sequence dimension
    >>> torch.equal(fused["embeddings"],
    ...            torch.cat([torch.ones(2, 3, 2),
    ...                      torch.ones(2, 2, 2) * 2], dim=1))
    True
    >>> # Test error handling
    >>> fuse_window_data(windows_data, ["nonexistent"], "fused")
    Traceback (most recent call last):
    ...
    ValueError: Window nonexistent specified in windows_to_fuse not found in data
    >>> # Test error when no fused_window_name
    >>> fuse_window_data(windows_data, ["pre", "post"], None)
    Traceback (most recent call last):
    ...
    ValueError: fused_window_name must not be empty
    """
    if not fused_window_name:
        raise ValueError("fused_window_name must not be empty")

    fused_window = {}
    lengths_tracking = {}

    # Process each window in the specified order
    for window in windows_to_fuse:
        if window not in windows_data:
            raise ValueError(f"Window {window} specified in windows_to_fuse not found in data")

        window_data = windows_data[window]

        # Process each key in the window data
        for key, data in window_data.items():
            # Initialize containers in fused window
            if key not in fused_window:
                lengths_tracking[f"LENGTHS//{key}"] = []

            if not isinstance(data, torch.Tensor):
                raise ValueError(f"Unsupported data type {type(data)} for key {key}")
            # TODO(Oufattole): perform a single concatenation here for efficiency
            logger.warning(f"key: {key} of type: {type(data)} handled as tensor")
            concat_dim = int(len(data.shape) > 1)
            lengths_tracking[f"LENGTHS//{key}"].append(data.shape[concat_dim])
            if key not in fused_window:
                fused_window[key] = data
            else:  # 2D or higher tensor
                if not key.startswith("static_"):
                    fused_window[key] = torch.cat([fused_window[key], data], dim=concat_dim)

    # Convert lists to tensors where appropriate
    for key, value in fused_window.items():
        if isinstance(value, list) and value and isinstance(value[0], torch.Tensor):
            fused_window[key] = torch.cat(value)

    # Add lengths tracking to the fused window
    fused_window.update(lengths_tracking)

    return fused_window


@dataclass
class DummyMultiWindowConfig(DummyConfig):
    """Configuration for MultiWindow dataset"""

    raw_windows_fp: str = None
    cache_dir: str = None
    window_size: int = 30  # Size of each window in days
    window_stride: int = 7  # Stride between windows in days
    min_window_events: int = 1  # Minimum number of events required in a window
    max_windows_per_subject: int | None = None  # Maximum number of windows per subject
    subject_level_sampling: bool = True  # Whether to sample windows at subject level
    default_window_name: str = None
    early_fusion_windows: bool = False  # Whether to fuse windows


def create_dummy_multiwindow_dataset(
    base_dir: str | Path, n_subjects: int = 3, split: str = "train", seed: int | None = 42
) -> DummyMultiWindowConfig:
    """Creates a dummy MultiWindow dataset.

    Args:
        base_dir (str | Path): directory to store the dataset in.
        n_subjects (int, optional): Number of subjects to generate.
        split (str, optional): Is the dataset split. Defaults to "train".
        seed (int | None, optional): Seed used for rng when making the dataset. Defaults to 42.

    Returns:
        DummyMultiWindowConfig: dataset config that can be used to create a MultiWindow dataset.

    Example:
    >>> import tempfile
    >>> _ = pl.Config.set_tbl_width_chars(106)
    >>> with tempfile.TemporaryDirectory() as tmp_dir:
    ...     config = create_dummy_multiwindow_dataset(tmp_dir)
    ...     task_df = pl.read_parquet(Path(config.data_dir) / "task_labels.parquet")
    ...     print(task_df)
    ...     raw_windows_df = pl.read_parquet(Path(config.data_dir) / "raw_windows.parquet")
    ...     print(raw_windows_df.sort("subject_id"))
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
    shape: (3, 4)
    ┌────────────┬─────────────────────┬─────────────────────────────────┬─────────────────────────────────┐
    │ subject_id ┆ trigger             ┆ window_1_summary                ┆ window_2_summary                │
    │ ---        ┆ ---                 ┆ ---                             ┆ ---                             │
    │ i64        ┆ datetime[μs]        ┆ struct[4]                       ┆ struct[4]                       │
    ╞════════════╪═════════════════════╪═════════════════════════════════╪═════════════════════════════════╡
    │ 0          ┆ 1998-01-01 00:00:00 ┆ {0,1998-01-01 00:00:00,1995-01… ┆ {0,1998-01-01 00:00:00,1998-01… │
    │ 1          ┆ 1998-01-01 00:00:00 ┆ {1,1998-01-01 00:00:00,1995-01… ┆ {1,1998-01-01 00:00:00,1998-01… │
    │ 2          ┆ 1998-01-01 00:00:00 ┆ {2,1998-01-01 00:00:00,1995-01… ┆ {2,1998-01-01 00:00:00,1998-01… │
    └────────────┴─────────────────────┴─────────────────────────────────┴─────────────────────────────────┘
    """
    config = create_dummy_dataset(base_dir, n_subjects, split, seed)
    raw_windows_fp = Path(base_dir) / "raw_windows.parquet"
    cache_dir = Path(base_dir) / "cache"
    task_df = pl.read_parquet(Path(config.data_dir) / "task_labels.parquet")
    static_df = pl.read_parquet(Path(config.data_dir) / "schema/train/shard_0.parquet")
    window_1 = task_df.hstack(static_df.drop("subject_id")).select(
        "subject_id",
        "prediction_time",
        pl.col("start_time").alias("timestamp_at_start"),
        pl.col("prediction_time").alias("timestamp_at_end"),
    )
    window_2 = task_df.hstack(static_df.drop("subject_id")).select(
        "subject_id",
        "prediction_time",
        pl.col("prediction_time").alias("timestamp_at_start"),
        pl.col("time").list.max().alias("timestamp_at_end"),
    )
    raw_windows_df = task_df.select("subject_id", pl.col("prediction_time").alias("trigger")).with_columns(
        window_1.to_struct().alias("window_1_summary"), window_2.to_struct().alias("window_2_summary")
    )
    raw_windows_df.write_parquet(raw_windows_fp)

    return DummyMultiWindowConfig(raw_windows_fp=raw_windows_fp, cache_dir=cache_dir, **asdict(config))


def get_window_indexes(static_df: pl.DataFrame, windows_df: pl.DataFrame) -> pl.DataFrame:
    """Computes the start and end indexes of time windows for each entry in the provided DataFrame. This
    function assumes that the "time" in `timestamps_series` is sorted. It finds the index of timestamps that
    fall between 'start' and 'end' times specified in `windows_df`.

    Parameters:
    - static_df (pl.Series): A Polars dataframe containing sorted datetime values for each subject.
    - windows_df (pl.DataFrame): A DataFrame with columns 'name', 'start', and 'end' specifying the time
        windows.

    Returns:
    - pl.DataFrame: A DataFrame with the original columns of `windows_df` plus for each <COL_NAME>.start
        and <COL_NAME>.end in windows_df, the column '<COL_NAME>.start_idx' and '<COL_NAME>.end_idx'
        indicating the inclusive index range of timestamps within each window is added.

    Example:
    >>> import pprint
    >>> _ = pl.Config.set_tbl_width_chars(106)
    >>> timeseries_df = pl.DataFrame({
    ...     "subject_id": [1, 2],
    ...     "time": [
    ...         pl.Series(["1978-03-09 00:00:00", "2010-05-26 02:30:56", "2010-05-26 04:51:52"]
    ...             ).str.strptime(pl.Datetime),
    ...         pl.Series(["1970-03-09 00:00:00", "1972-05-26 02:30:56", "1975-05-26 04:51:52"]
    ...             ).str.strptime(pl.Datetime)
    ...     ]
    ... })
    >>> windows_df = pl.DataFrame({
    ...     "subject_id": [1, 2],
    ...     "pre.start": [["1978-03-09 00:00:00", "2010-05-26 02:30:56"], ["1969-05-26 02:30:56"]],
    ...     "pre.end": [["2010-05-26 02:30:56", "2010-05-26 04:51:52"], ["1971-05-26 04:51:52"]],
    ...     "post.start": [["1978-03-09 00:00:00", "2010-05-26 02:30:56"], ["1971-05-26 02:30:56"]],
    ...     "post.end": [["2010-05-26 02:30:56", "2010-05-26 04:51:52"], ["1980-05-26 04:51:52"]],
    ... }).with_columns([
    ...     pl.col("pre.start").list.eval(pl.element().str.to_datetime()),
    ...     pl.col("pre.end").list.eval(pl.element().str.to_datetime()),
    ...     pl.col("post.start").list.eval(pl.element().str.to_datetime()),
    ...     pl.col("post.end").list.eval(pl.element().str.to_datetime()),
    ... ])
    >>> timeseries_df
    shape: (2, 2)
    ┌────────────┬─────────────────────────────────┐
    │ subject_id ┆ time                            │
    │ ---        ┆ ---                             │
    │ i64        ┆ list[datetime[μs]]              │
    ╞════════════╪═════════════════════════════════╡
    │ 1          ┆ [1978-03-09 00:00:00, 2010-05-… │
    │ 2          ┆ [1970-03-09 00:00:00, 1972-05-… │
    └────────────┴─────────────────────────────────┘
    >>> windows_df.shape
    (2, 5)
    >>> get_window_indexes(
    ...     timeseries_df, windows_df).select("subject_id", pl.col("^.*_idx$")).sort("subject_id")
    shape: (2, 5)
    ┌────────────┬───────────────┬─────────────┬────────────────┬──────────────┐
    │ subject_id ┆ pre.start_idx ┆ pre.end_idx ┆ post.start_idx ┆ post.end_idx │
    │ ---        ┆ ---           ┆ ---         ┆ ---            ┆ ---          │
    │ i64        ┆ list[u32]     ┆ list[u32]   ┆ list[u32]      ┆ list[u32]    │
    ╞════════════╪═══════════════╪═════════════╪════════════════╪══════════════╡
    │ 1          ┆ [0, 1]        ┆ [1, 2]      ┆ [0, 1]         ┆ [1, 2]       │
    │ 2          ┆ [0]           ┆ [1]         ┆ [1]            ┆ [3]          │
    └────────────┴───────────────┴─────────────┴────────────────┴──────────────┘
    >>> single_event_per_subject_window_df = (windows_df
    ...                                       .explode(pl.exclude("subject_id"))
    ...                                       .group_by("subject_id").first()
    ...                                       .group_by("subject_id").agg(pl.all()))
    >>> pprint.pprint({k: v.to_list() for k,v in single_event_per_subject_window_df
    ...                  .sort("subject_id").to_dict().items()})
    {'post.end': [[datetime.datetime(2010, 5, 26, 2, 30, 56)],
                  [datetime.datetime(1980, 5, 26, 4, 51, 52)]],
     'post.start': [[datetime.datetime(1978, 3, 9, 0, 0)],
                    [datetime.datetime(1971, 5, 26, 2, 30, 56)]],
     'pre.end': [[datetime.datetime(2010, 5, 26, 2, 30, 56)],
                 [datetime.datetime(1971, 5, 26, 4, 51, 52)]],
     'pre.start': [[datetime.datetime(1978, 3, 9, 0, 0)],
                   [datetime.datetime(1969, 5, 26, 2, 30, 56)]],
     'subject_id': [1, 2]}
    >>> get_window_indexes(
    ...     timeseries_df, single_event_per_subject_window_df
    ... ).select("subject_id", pl.col("^.*_idx$")).sort("subject_id")
    shape: (2, 5)
    ┌────────────┬───────────────┬─────────────┬────────────────┬──────────────┐
    │ subject_id ┆ pre.start_idx ┆ pre.end_idx ┆ post.start_idx ┆ post.end_idx │
    │ ---        ┆ ---           ┆ ---         ┆ ---            ┆ ---          │
    │ i64        ┆ list[u32]     ┆ list[u32]   ┆ list[u32]      ┆ list[u32]    │
    ╞════════════╪═══════════════╪═════════════╪════════════════╪══════════════╡
    │ 1          ┆ [0]           ┆ [1]         ┆ [0]            ┆ [1]          │
    │ 2          ┆ [0]           ┆ [1]         ┆ [1]            ┆ [3]          │
    └────────────┴───────────────┴─────────────┴────────────────┴──────────────┘
    """
    datetime_cols = [col for col in windows_df.columns if col.endswith(".start") or col.endswith(".end")]
    output_cols = [f"{col}_idx" for col in datetime_cols]
    windows_df = windows_df.join(how="inner", other=static_df, on="subject_id")
    expr = [
        (pl.col("time").explode().search_sorted(pl.col(col).explode()).alias(f"{col}_idx"))
        for col in datetime_cols
    ]
    windows_df = windows_df.group_by(pl.col("subject_id")).agg(expr)
    if output_cols and not isinstance(windows_df.schema[output_cols[0]], pl.List):
        windows_df = windows_df.group_by("subject_id").agg(pl.all())
    return windows_df


def cache_window_indexes(cfg: DictConfig, split: str, static_dfs) -> pl.DataFrame:
    """Caches window indexes for the given split of the dataset.

    Args:
        cfg (DictConfig): _description_
        split (str): _description_
        static_dfs (_type_): _description_

    Returns:
        pl.DataFrame: _description_

    Example:
    >>> import tempfile
    >>> import pprint
    >>> with tempfile.TemporaryDirectory() as tmpdir:
    ...     config = create_dummy_multiwindow_dataset(tmpdir)
    ...     static_df = pl.read_parquet(Path(config.data_dir) / "schema/train/shard_0.parquet")
    ...     raw_windows_df = pl.read_parquet(Path(config.data_dir) / "raw_windows.parquet")
    ...     cache_window_indexes(config, "train", {"shard_0": static_df})
    ...     cached_window_df = pl.read_parquet(Path(config.cache_dir) / "train.parquet").sort("subject_id")
    >>> pprint.pprint({k: v.to_list() for k,v in cached_window_df.to_dict().items()})
    {'subject_id': [0, 1, 2],
     'window_1_summary.end_idx': [[3], [3], [3]],
     'window_1_summary.start_idx': [[0], [0], [0]],
     'window_2_summary.end_idx': [[4], [4], [4]],
     'window_2_summary.start_idx': [[3], [3], [3]]}
    """
    window_df = pl.read_parquet(cfg.raw_windows_fp)
    window_cols = [col for col in window_df.columns if col.endswith("_summary")]
    col = "pre.start_summary"
    exprs = [pl.col("subject_id")]
    for col in window_cols:
        for side in ("start", "end"):
            parsed_col = col.split(".")[0] + f".{side}"
            exprs.append(pl.col(col).struct.field(f"timestamp_at_{side}").alias(parsed_col))
    window_df = window_df.select(exprs)
    window_df = window_df.group_by("subject_id").agg(pl.all())
    static_df = pl.concat(static_dfs.values()).select("subject_id", "time")
    cached_window_df = get_window_indexes(static_df, window_df)
    cache_window_fp = Path(cfg.cache_dir) / f"{split}.parquet"
    cache_window_fp.parent.mkdir(parents=True, exist_ok=True)
    cached_window_df.write_parquet(cache_window_fp)


class MultiWindowSamplingStrategy(StrEnum):
    """Enumeration of sampling strategies for multi-window datasets.

    Attributes:
        RANDOM: Randomly sample and window from the dataset, partitioning it into separate
            subwindows.
        PREDEFINED: Use predefined windows around events to sample from the dataset.
    """

    RANDOM = "random"
    PREDEFINED = "predefined"


class MultiWindowPytorchDataset(SeedableMixin, torch.utils.data.Dataset):
    """A Multi-Window PyTorch Dataset class for contrastive learning pretraining.

    This class extends the functionality of a standard PyTorch Dataset to support multiple time windows for
    each subject. It's designed to work with medical event data, where each subject may have multiple relevant
    time windows for analysis.

    The dataset can be configured to sample at the subject level or at the window level, allowing for flexible
    data loading strategies in contrastive learning scenarios.

    Args:
        cfg (DictConfig): Configuration options for the dataset.
        split (str): The data split to use (e.g., 'train', 'validation', 'test').

    Attributes:
        config (DictConfig): The configuration object for the dataset.
        split (str): The current data split being used.
        pytorch_dataset (PytorchDataset): The underlying PyTorch dataset.
        window_cols (list): List of window column names.
        index (list): List of dictionaries, each representing a subject or a window,
            depending on the sampling strategy.

    Example:
    >>> import tempfile
    >>> from pathlib import Path
    >>> import torch
    >>> import polars as pl
    >>> from omegaconf import OmegaConf
    >>>
    >>> # Create dummy dataset in a temporary directory
    >>> with tempfile.TemporaryDirectory() as tmp_dir:
    ...     # Generate dummy config with sample data
    ...     config = create_dummy_multiwindow_dataset(tmp_dir)
    ...     cfg = config
    ...
    ...     # Initialize the dataset
    ...     dataset = MultiWindowPytorchDataset(cfg, split="train")
    ...
    ...     # Check the available windows
    ...     print(f"Window columns: {dataset.window_cols}")
    ...
    ...     # Get a sample item
    ...     sample = dataset[0]
    ...
    ...     # Examine the structure of the first window
    ...     window_name = dataset.window_cols[0]
    ...     print(f"\\nStructure of {window_name} window:")
    ...     for key, value in sample[window_name].items():
    ...         if isinstance(value, dict):
    ...             print(f"{key}:")
    ...             for subkey, subvalue in value.items():
    ...                 print(f"  {subkey}: {type(subvalue)}")
    ...         else:
    ...             print(f"{key}: {type(value)}")
    ...
    ...     # Create a dataloader and get a batch
    ...     dataloader = torch.utils.data.DataLoader(
    ...         dataset,
    ...         batch_size=2,
    ...         collate_fn=dataset.collate
    ...     )
    ...     batch = next(iter(dataloader))
    Window columns: ['window_1_summary', 'window_2_summary']
    <BLANKLINE>
    Structure of window_1_summary window:
    static_indices: <class 'list'>
    static_values: <class 'list'>
    start_idx: <class 'int'>
    end_idx: <class 'int'>
    dynamic: <class 'nested_ragged_tensors.ragged_numpy.JointNestedRaggedTensorDict'>
    start_time: <class 'datetime.datetime'>
    end_time: <class 'datetime.datetime'>
    """

    def __init__(self, cfg: DictConfig, split: str):
        """Initialize the MultiWindowPytorchDataset.

        This method sets up the dataset by loading or creating cached window indexes, initializing the
        underlying PytorchDataset, and preparing the index based on the specified sampling strategy.

        Args:
            cfg (DictConfig): Configuration options for the dataset.
            split (str): The data split to use (e.g., 'train', 'validation', 'test').
        """
        super().__init__()

        self.config = cfg
        self.split = split
        self.pytorch_dataset = PytorchDataset(cfg, split)
        cached_windows_fp = Path(cfg.cache_dir) / f"{split}.parquet"
        with FileLock(f"{cached_windows_fp}.lock"):
            if cached_windows_fp.exists():
                window_df = pl.read_parquet(cached_windows_fp)
            else:
                logger.info("No cached windows found. Caching now.")
                cache_window_indexes(cfg, split, self.pytorch_dataset.static_dfs)
                window_df = pl.read_parquet(cached_windows_fp)
        self.window_cols = sorted(
            list({col.split(".")[0] for col in window_df.columns if col.endswith("_idx")})
        )
        window_df = self.filter_invalid_window(window_df)
        if self.config.subject_level_sampling:
            # index by subject_id
            self.index = window_df.to_dicts()
        else:
            # index by windows
            self.index = window_df.explode(
                [col for col in window_df.columns if col.endswith("_idx")]
            ).to_dicts()

    def filter_invalid_window(cls, window_df: pl.DataFrame) -> pl.DataFrame:
        """Filter out invalid windows if st index >= end index."""
        window_df = window_df.explode(pl.exclude("subject_id"))
        num_rows = window_df.shape[0]
        window_cols = {col[:-10] for col in window_df.columns if col.endswith("start_idx")}
        window_exprs = [
            pl.col(f"{window_col}.end_idx") > pl.col(f"{window_col}.start_idx") for window_col in window_cols
        ]
        window_df = window_df.filter(window_exprs)
        if window_df.shape[0] != num_rows:
            logger.warning(f"Filtered out {num_rows - window_df.shape[0]} invalid windows.")
        window_df = window_df.group_by("subject_id").agg(pl.all())
        return window_df

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

    def collate(self, batch: dict[str : np.array]) -> dict[str : torch.Tensor]:
        """Collate a batch of data samples into a single batch.

        This method is responsible for combining multiple data samples into a single batch that can be
        processed by a PyTorch model. It handles both window-specific data and any additional data that might
        be present.

        Args:
            batch (dict[str, np.array]): A dictionary of arrays, each representing a batch of data for a
                specific feature.

        Returns:
            dict[str, torch.Tensor]: A dictionary of tensors, representing the collated batch data.
        """
        out = {}
        for col in self.window_cols:
            out[col] = self.pytorch_dataset.collate([x[col] for x in batch])
        for key in batch[0].keys():
            if key not in self.window_cols:
                if isinstance(batch[0][key], datetime.datetime):
                    out[key] = [item[key] for item in batch]
                else:
                    out[key] = torch.Tensor([item[key] for item in batch])
        # Fuse windows
        if self.config.early_fusion_windows:
            if self.config.do_prepend_static_data:
                raise ValueError("early fusion of windows with do_prepend_static_data is not supported.")
            out[self.config.early_fusion_window_name] = fuse_window_data(
                out, self.config.early_fusion_windows, self.config.early_fusion_window_name
            )
        if self.config.default_window_name:
            for key in out[self.config.default_window_name].keys():
                out[key] = out[self.config.default_window_name][key]

        return out

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        """Get a single item from the dataset.

        This method retrieves data for a single subject, returning a dictionary where each key represents a
        window, and the corresponding value is the output of PytorchDataset.__getitem__ for that window.

        Args:     idx (int): The index of the item to retrieve.

        Returns:
            dict[str, dict]: A dictionary where keys are window names (e.g., 'pre', 'post') and
                values are dictionaries containing the data for each window.
                The structure of each window's data typically includes:
                    - 'static_indices': List of static categorical metadata elements.
                    - 'static_values': List of static numerical metadata elements.
                    - 'dynamic':
                        Dictionary containing:
                        - 'time_delta_days': List of time deltas between events.
                        - 'dim1/code': List of dynamic categorical metadata elements.
                        - 'dim1/numeric_value': List of dynamic numerical metadata elements.
        """
        return self._seeded_getitem(idx)

    @SeedableMixin.WithSeed
    def _seeded_getitem(self, idx: int) -> dict[str, list[float]]:
        """Seedable version of __getitem__.

        This method is a seedable version of __getitem__, allowing for reproducible data retrieval. It handles
        both subject-level and window-level sampling, returning a dictionary of window data for the selected
        subject.

        Args:
            idx (int): The index of the item to retrieve.

        Returns:
            dict[str, dict]: A dictionary where keys are window names (e.g., 'pre', 'post') and
                values are dictionaries containing the data for each window,
                as returned by PytorchDataset.load_subject().
        """
        if self.config.subject_level_sampling:
            # index by subject_id
            subject_data = self.index[idx]
            num_windows = len(subject_data[f"{self.window_cols[0]}.start_idx"])
            selected_window_idx = np.random.choice(num_windows)
            windows = {
                col: [
                    subject_data[f"{col}.start_idx"][selected_window_idx],
                    subject_data[f"{col}.end_idx"][selected_window_idx],
                ]
                for col in self.window_cols
            }
        else:
            # index by windows
            subject_data = self.index[idx]
            windows = {
                col: [subject_data[f"{col}.start_idx"], subject_data[f"{col}.end_idx"]]
                for col in self.window_cols
            }
        subject_id = subject_data["subject_id"]

        shard = self.pytorch_dataset.subj_map[subject_id]
        subject_idx = self.pytorch_dataset.subj_indices[subject_id]
        out = {}
        for window in self.window_cols:
            st, end = windows[window][0], windows[window][1]
            subject_dynamic_data = JointNestedRaggedTensorDict(
                tensors_fp=Path(self.config.data_dir) / "data" / f"{shard}.nrt"
            )[subject_idx, st:end]

            if st >= end:
                raise ValueError(f"start index {st} >= end index {end}")

            out[window] = self.pytorch_dataset.load_subject(subject_dynamic_data, subject_id, st, end)

        if self.config.do_include_subject_id:
            out["subject_id"] = subject_id
        return out
