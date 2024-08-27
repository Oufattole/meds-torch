from pathlib import Path

MAX_POLARS_THREADS = 1
import numpy as np
import polars as pl
import torch
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
    - timeseries_df (pl.Series): A Polars dataframe containing sorted datetime values for each patient.
    - windows_df (pl.DataFrame): A DataFrame with columns 'name', 'start', and 'end' specifying the time
        windows.

    Returns:
    - pl.DataFrame: A DataFrame with the original columns of `windows_df` plus for each <COL_NAME>.start
        and <COL_NAME>.end in windows_df, the column '<COL_NAME>.start_idx' and '<COL_NAME>.end_idx'
        indicating the inclusive index range of timestamps within each window is added.

    Example:
    >>> timeseries_df = pl.DataFrame({
    ...     "patient_id": [1, 2],
    ...     "time": [
    ...         pl.Series(["1978-03-09 00:00:00", "2010-05-26 02:30:56", "2010-05-26 04:51:52"]
    ...             ).str.strptime(pl.Datetime),
    ...         pl.Series(["1970-03-09 00:00:00", "1972-05-26 02:30:56", "1975-05-26 04:51:52"]
    ...             ).str.strptime(pl.Datetime)
    ...     ]
    ... })
    >>> windows_df = pl.DataFrame({
    ...     "patient_id": [1, 2],
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
    │ patient_id ┆ timestamp                       │
    │ ---        ┆ ---                             │
    │ i64        ┆ list[datetime[μs]]              │
    ╞════════════╪═════════════════════════════════╡
    │ 1          ┆ [1978-03-09 00:00:00, 2010-05-… │
    │ 2          ┆ [1970-03-09 00:00:00, 1972-05-… │
    └────────────┴─────────────────────────────────┘
    >>> windows_df
    shape: (2, 5)
    ┌────────────┬─────────────────────┬─────────────────────┬────────────────────┬────────────────────┐
    │ patient_id ┆ pre.start           ┆ pre.end             ┆ post.start         ┆ post.end           │
    │ ---        ┆ ---                 ┆ ---                 ┆ ---                ┆ ---                │
    │ i64        ┆ list[datetime[μs]]  ┆ list[datetime[μs]]  ┆ list[datetime[μs]] ┆ list[datetime[μs]] │
    ╞════════════╪═════════════════════╪═════════════════════╪════════════════════╪════════════════════╡
    │ 1          ┆ [1978-03-09         ┆ [2010-05-26         ┆ [1978-03-09        ┆ [2010-05-26        │
    │            ┆ 00:00:00, 2010-05-… ┆ 02:30:56, 2010-05-… ┆ 00:00:00,          ┆ 02:30:56,          │
    │            ┆                     ┆                     ┆ 2010-05-…          ┆ 2010-05-…          │
    │ 2          ┆ [1969-05-26         ┆ [1971-05-26         ┆ [1971-05-26        ┆ [1980-05-26        │
    │            ┆ 02:30:56]           ┆ 04:51:52]           ┆ 02:30:56]          ┆ 04:51:52]          │
    └────────────┴─────────────────────┴─────────────────────┴────────────────────┴────────────────────┘
    >>> get_window_indexes(
    ...     timeseries_df, windows_df).select("patient_id", pl.col("^.*_idx$")).sort("patient_id")
    shape: (2, 5)
    ┌────────────┬───────────────┬─────────────┬────────────────┬──────────────┐
    │ patient_id ┆ pre.start_idx ┆ pre.end_idx ┆ post.start_idx ┆ post.end_idx │
    │ ---        ┆ ---           ┆ ---         ┆ ---            ┆ ---          │
    │ i64        ┆ list[u32]     ┆ list[u32]   ┆ list[u32]      ┆ list[u32]    │
    ╞════════════╪═══════════════╪═════════════╪════════════════╪══════════════╡
    │ 1          ┆ [0, 1]        ┆ [1, 2]      ┆ [0, 1]         ┆ [1, 2]       │
    │ 2          ┆ [0]           ┆ [1]         ┆ [1]            ┆ [3]          │
    └────────────┴───────────────┴─────────────┴────────────────┴──────────────┘
    """
    datetime_cols = [col for col in windows_df.columns if col.endswith(".start") or col.endswith(".end")]
    windows_df = windows_df.join(how="inner", other=timeseries_df, on="patient_id")
    expr = [
        pl.col("time").explode().search_sorted(pl.col(col).explode()).alias(f"{col}_idx")
        for col in datetime_cols
    ]
    return windows_df.group_by(pl.col("patient_id")).agg(expr)


def cache_window_indexes(cfg: DictConfig, split: str, static_dfs) -> pl.DataFrame:
    # TODO add support for windows between different patients
    # Parse windows
    window_df = pl.read_parquet(cfg.raw_windows_fp)
    window_cols = [col for col in window_df.columns if col.endswith("_summary")]
    col = "pre.start_summary"
    exprs = [pl.col("patient_id")]
    for col in window_cols:
        for side in ("start", "end"):
            parsed_col = col.split(".")[0] + f".{side}"
            exprs.append(pl.col(col).struct.field(f"timestamp_at_{side}").alias(parsed_col))
    window_df = window_df.select(exprs)
    window_df = window_df.group_by("patient_id").agg(pl.all())
    timeseries_df = pl.concat(static_dfs.values()).select("patient_id", "time")
    cached_window_df = get_window_indexes(timeseries_df, window_df)
    cache_window_fp = Path(cfg.cached_windows_dir) / f"{split}.parquet"
    cache_window_fp.parent.mkdir(parents=True, exist_ok=True)
    cached_window_df.write_parquet(cache_window_fp)


class MultiWindowPytorchDataset(SeedableMixin, torch.utils.data.Dataset):
    """A Multi Window PyTorch Dataset class, enabling contrastive learning pretraining.

    Args:     config: Configuration options for the dataset, in an `omegaconf.DictConfig` object.     split:
    The split of data which should be used in this dataset (e.g., ``'train'``, ``'tuning'``, ``'held_out'``).
    This will dictate where the system looks for files.
    """

    def __init__(self, cfg: DictConfig, split: str):
        super().__init__()

        self.config = cfg
        self.split = split
        self.pytorch_dataset = PytorchDataset(cfg, split)
        cached_windows_fp = Path(cfg.cached_windows_dir) / f"{split}.parquet"
        if cached_windows_fp.exists():
            window_df = pl.read_parquet(cached_windows_fp)
        else:
            logger.info("No cached windows found. Caching now.")
            cache_window_indexes(cfg, split, self.pytorch_dataset.static_dfs)
            window_df = pl.read_parquet(cached_windows_fp)
        self.window_cols = sorted(
            list({col.split(".")[0] for col in window_df.columns if col.endswith("_idx")})
        )
        if self.config.patient_level_sampling:
            # index by patient_id
            self.index = window_df.to_dicts()
        else:
            # index by windows
            self.index = window_df.explode(
                [col for col in window_df.columns if col.endswith("_idx")]
            ).to_dicts()

    def collate(self, batch: dict[str : np.array]) -> dict[str : torch.Tensor]:
        out = {}
        for col in self.window_cols:
            out[col] = self.pytorch_dataset.collate([x[col] for x in batch])
        for key in batch[0].keys():
            if key not in self.window_cols:
                out[key] = batch[key]
        return out

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
    def _seeded_getitem(self, idx: int) -> dict[str, list[float]]:
        """Returns a Returns a dictionary corresponding to a single subject's data.

        This function is a seedable version of `__getitem__`.
        """
        if self.config.patient_level_sampling:
            # index by patient_id
            patient_data = self.index[idx]
            num_windows = len(patient_data[f"{self.window_cols[0]}.start_idx"])
            selected_window_idx = np.random.choice(num_windows)
            patient_id = patient_data["patient_id"]
            windows = {
                col: [
                    patient_data[f"{col}.start_idx"][selected_window_idx],
                    patient_data[f"{col}.end_idx"][selected_window_idx],
                ]
                for col in self.window_cols
            }
        else:
            # index by windows
            patient_data = self.index[idx]
            windows = {
                col: [patient_data[f"{col}.start_idx"], patient_data[f"{col}.end_idx"]]
                for col in self.window_cols
            }
            patient_id = patient_data["patient_id"]

        shard = self.pytorch_dataset.subj_map[patient_id]
        patient_idx = self.pytorch_dataset.subj_indices[patient_id]
        out = {}
        for window in self.window_cols:
            patient_dynamic_data = JointNestedRaggedTensorDict.load_slice(
                Path(self.config.data_dir) / "data" / f"{shard}.nrt", patient_idx
            )
            out[window] = self.pytorch_dataset.load_patient(
                patient_dynamic_data, patient_id, windows[window][0], windows[window][1]
            )

        if self.config.do_include_patient_id:
            out["patient_id"] = patient_id
        return out
