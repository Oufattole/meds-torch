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

from meds_torch.data.components.pytorch_dataset import PytorchDataset


def get_window_indexes(timeseries_df: pl.DataFrame, windows_df: pl.DataFrame) -> pl.DataFrame:
    """Computes the start and end indexes of time windows for each entry in the provided DataFrame. This
    function assumes that the "time" in `timestamps_series` is sorted. It finds the index of timestamps that
    fall between 'start' and 'end' times specified in `windows_df`.

    Parameters:
    - timeseries_df (pl.Series): A Polars dataframe containing sorted datetime values for each subject.
    - windows_df (pl.DataFrame): A DataFrame with columns 'name', 'start', and 'end' specifying the time
        windows.

    Returns:
    - pl.DataFrame: A DataFrame with the original columns of `windows_df` plus for each <COL_NAME>.start
        and <COL_NAME>.end in windows_df, the column '<COL_NAME>.start_idx' and '<COL_NAME>.end_idx'
        indicating the inclusive index range of timestamps within each window is added.

    Example:
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
    """
    datetime_cols = [col for col in windows_df.columns if col.endswith(".start") or col.endswith(".end")]
    windows_df = windows_df.join(how="inner", other=timeseries_df, on="subject_id")
    expr = [
        pl.col("time").explode().search_sorted(pl.col(col).explode()).alias(f"{col}_idx")
        for col in datetime_cols
    ]
    return windows_df.group_by(pl.col("subject_id")).agg(expr)


def cache_window_indexes(cfg: DictConfig, split: str, static_dfs) -> pl.DataFrame:
    # TODO add support for windows between different subjects
    # Parse windows
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
    timeseries_df = pl.concat(static_dfs.values()).select("subject_id", "time")
    cached_window_df = get_window_indexes(timeseries_df, window_df)
    cache_window_fp = Path(cfg.cache_dir) / f"{split}.parquet"
    cache_window_fp.parent.mkdir(parents=True, exist_ok=True)
    cached_window_df.write_parquet(cache_window_fp)


class MultiWindowSamplingStrategy(StrEnum):
    """Enumeration of sampling strategies for multi-window datasets.

    Attributes:     RANDOM: Randomly sample and window from the dataset, partitioning it into separate
    subwindows.     PREDEFINED: Use predefined windows around events to sample from the dataset.
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

    Args:     cfg (DictConfig): Configuration options for the dataset.     split (str): The data split to use
    (e.g., 'train', 'validation', 'test').

    Attributes:     config (DictConfig): The configuration object for the dataset.     split (str): The
    current data split being used.     pytorch_dataset (PytorchDataset): The underlying PyTorch dataset.
    window_cols (list): List of window column names.     index (list): List of dictionaries, each representing
    a subject or a window,                   depending on the sampling strategy.
    """

    def __init__(self, cfg: DictConfig, split: str):
        """Initialize the MultiWindowPytorchDataset.

        This method sets up the dataset by loading or creating cached window indexes, initializing the
        underlying PytorchDataset, and preparing the index based on the specified sampling strategy.

        Args:     cfg (DictConfig): Configuration options for the dataset.     split (str): The data split to
        use (e.g., 'train', 'validation', 'test').
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
        if self.config.subject_level_sampling:
            # index by subject_id
            self.index = window_df.to_dicts()
        else:
            # index by windows
            self.index = window_df.explode(
                [col for col in window_df.columns if col.endswith("_idx")]
            ).to_dicts()

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

        Args:     batch (dict[str, np.array]): A dictionary of arrays, each representing a batch of data for a
        specific feature.

        Returns:     dict[str, torch.Tensor]: A dictionary of tensors, representing the collated batch data.
        """
        out = {}
        for col in self.window_cols:
            out[col] = self.pytorch_dataset.collate([x[col] for x in batch])
        for key in batch[0].keys():
            if key not in self.window_cols:
                out[key] = batch[key]
        return out

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        """Get a single item from the dataset.

        This method retrieves data for a single subject, returning a dictionary where each key represents a
        window, and the corresponding value is the output of PytorchDataset.__getitem__ for that window.

        Args:     idx (int): The index of the item to retrieve.

        Returns:     dict[str, dict]: A dictionary where keys are window names (e.g., 'pre', 'post') and
        values are dictionaries containing the data for each window.                      The structure of
        each window's data typically includes:         - 'static_indices': List of static categorical metadata
        elements.         - 'static_values': List of static numerical metadata elements.         - 'dynamic':
        Dictionary containing:             - 'time_delta_days': List of time deltas between events. -
        'dim1/code': List of dynamic categorical metadata elements.             - 'dim1/numeric_value': List
        of dynamic numerical metadata elements.
        """
        return self._seeded_getitem(idx)

    @SeedableMixin.WithSeed
    def _seeded_getitem(self, idx: int) -> dict[str, list[float]]:
        """Seedable version of __getitem__.

        This method is a seedable version of __getitem__, allowing for reproducible data retrieval. It handles
        both subject-level and window-level sampling, returning a dictionary of window data for the selected
        subject.

        Args:     idx (int): The index of the item to retrieve.

        Returns:     dict[str, dict]: A dictionary where keys are window names (e.g., 'pre', 'post') and
        values are dictionaries containing the data for each window,                      as returned by
        PytorchDataset.load_subject().
        """
        if self.config.subject_level_sampling:
            # index by subject_id
            subject_data = self.index[idx]
            num_windows = len(subject_data[f"{self.window_cols[0]}.start_idx"])
            selected_window_idx = np.random.choice(num_windows)
            subject_id = subject_data["subject_id"]
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
            subject_dynamic_data = JointNestedRaggedTensorDict.load_slice(
                Path(self.config.data_dir) / "data" / f"{shard}.nrt", subject_idx
            )
            out[window] = self.pytorch_dataset.load_subject(
                subject_dynamic_data, subject_id, windows[window][0], windows[window][1]
            )

        if self.config.do_include_subject_id:
            out["subject_id"] = subject_id
        return out
