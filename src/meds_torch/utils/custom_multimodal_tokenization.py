#!/usr/bin/env python
"""Functions for tokenizing MEDS datasets.

Here, _tokenization_ refers specifically to the process of converting a longitudinal, irregularly sampled,
continuous time sequence into a temporal sequence at the level that will be consumed by deep-learning models.

All these functions take in _normalized_ data -- meaning data where there are _no longer_ any code modifiers,
as those have been normalized alongside codes into integer indices (in the output code column). The only
columns of concern here thus are `subject_id`, `time`, `code`, `numeric_value`, and `modality_fp`.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path

import hydra
import numpy as np
import polars as pl
import torch
from loguru import logger
from MEDS_transforms import PREPROCESS_CONFIG_YAML
from MEDS_transforms.mapreduce.utils import rwlock_wrap, shard_iterator
from MEDS_transforms.utils import hydra_loguru_init, write_lazyframe
from omegaconf import DictConfig, OmegaConf
from safetensors.torch import save_file

SECONDS_PER_MINUTE = 60.0
SECONDS_PER_HOUR = SECONDS_PER_MINUTE * 60.0
SECONDS_PER_DAY = SECONDS_PER_HOUR * 24.0


@dataclass
class MultimodalReader(ABC):
    """Abstract base class for reading multimodal data."""

    base_path: str

    @abstractmethod
    def read_modality(self, relative_modality_fp: str) -> torch.Tensor:
        """Read and return modality data as a tensor or other structured format.

        Args:
            relative_modality_fp: Relative path to the modality data file.

        Returns:
            Dictionary containing the processed modality data.
        """


class NpyReader(MultimodalReader):
    def read_modality(self, relative_modality_fp: str) -> torch.Tensor:
        """Read and return modality data from a NumPy binary file (npz).

        Args:
            modality_fp: Relative path to the modality data file.

        Returns:
            Dictionary containing the processed modality data in NumPy format.
        """
        data = np.load(Path(self.base_path) / relative_modality_fp)
        return torch.tensor(data)


class DummyReader(MultimodalReader):
    def read_modality(self, relative_modality_fp: str):
        # Return a simple tensor for testing
        return torch.tensor([1, 2, 3])


def fill_to_nans(col: str | pl.Expr) -> pl.Expr:
    """This function fills infinite and null values with NaN.

    This enables the downstream functions to naturally tensorize data into numpy or Torch tensors.

    Args:
        col: The input column.

    Returns:
        A `pl.Expr` object that fills infinite and null values with NaN.

    Examples:
        >>> print(fill_to_nans("value")) # doctest: +NORMALIZE_WHITESPACE
        .when([(col("value").is_infinite()) |
               (col("value").is_null())]).then(dyn float: NaN).otherwise(col("value"))
        >>> df = pl.DataFrame({"value": [1.0, float("inf"), None, -float("inf"), 2.0]})
        >>> df.select(fill_to_nans("value").alias("value"))["value"].to_list()
        [1.0, nan, nan, nan, 2.0]
    """
    if isinstance(col, str):
        col = pl.col(col)

    return pl.when(col.is_infinite() | col.is_null()).then(float("nan")).otherwise(col)


def split_static_and_dynamic(df: pl.LazyFrame) -> tuple[pl.LazyFrame, pl.LazyFrame]:
    """This function splits the input data into static and dynamic data.

    Static data is data that has a null time, and dynamic data is everything else.
    For dynamic data, a modality index is added for non-null modality paths.

    Args:
        df: The input data.

    Returns:
        A tuple of two `pl.LazyFrame` objects, the first being the static data and the second being the
        dynamic data.

    Examples:
        >>> from datetime import datetime
        >>> df = pl.DataFrame({
        ...     "subject_id": [1, 1, 2, 2],
        ...     "time": [None, datetime(2021, 1, 1), None, datetime(2021, 1, 2)],
        ...     "code": [100, 101, 200, 201],
        ...     "numeric_value": [1.0, 2.0, 3.0, 4.0],
        ...     "modality_fp": [None, "path/to/image1.jpg", None, "path/to/image2.jpg"]
        ... }).lazy()
        >>> static, dynamic = split_static_and_dynamic(df)
        >>> static.collect()
        shape: (2, 4)
        ┌────────────┬──────┬───────────────┬─────────────┐
        │ subject_id ┆ code ┆ numeric_value ┆ modality_fp │
        │ ---        ┆ ---  ┆ ---           ┆ ---         │
        │ i64        ┆ i64  ┆ f64           ┆ str         │
        ╞════════════╪══════╪═══════════════╪═════════════╡
        │ 1          ┆ 100  ┆ 1.0           ┆ null        │
        │ 2          ┆ 200  ┆ 3.0           ┆ null        │
        └────────────┴──────┴───────────────┴─────────────┘
        >>> dynamic.collect()
        shape: (2, 6)
        ┌────────────┬─────────────────────┬──────┬───────────────┬────────────────────┬──────────────┐
        │ subject_id ┆ time                ┆ code ┆ numeric_value ┆ modality_fp        ┆ modality_idx │
        │ ---        ┆ ---                 ┆ ---  ┆ ---           ┆ ---                ┆ ---          │
        │ i64        ┆ datetime[μs]        ┆ i64  ┆ f64           ┆ str                ┆ f32          │
        ╞════════════╪═════════════════════╪══════╪═══════════════╪════════════════════╪══════════════╡
        │ 1          ┆ 2021-01-01 00:00:00 ┆ 101  ┆ 2.0           ┆ path/to/image1.jpg ┆ 0.0          │
        │ 2          ┆ 2021-01-02 00:00:00 ┆ 201  ┆ 4.0           ┆ path/to/image2.jpg ┆ 1.0          │
        └────────────┴─────────────────────┴──────┴───────────────┴────────────────────┴──────────────┘
    """
    static = df.filter(pl.col("time").is_null()).drop("time")
    dynamic = df.filter(pl.col("time").is_not_null())

    # Add modality index for modality paths
    if "modality_fp" in df.columns:
        dynamic = dynamic.with_columns(
            [
                pl.when(pl.col("modality_fp").is_not_null())
                .then(pl.col("modality_fp").rank("dense") - 1)
                .otherwise(None)
                .cast(pl.Float32)
                .alias("modality_idx")
            ]
        )

    return static, dynamic


def process_modality_data(df: pl.DataFrame, reader: MultimodalReader) -> dict[str, dict]:
    """Process modality data using the provided reader.

    Args:
        df: DataFrame containing modality paths and their corresponding codes and modality indices.
        reader: Instance of MultimodalReader to process the modality data.

    Returns:
        Dictionary mapping f"{modality_idx}" to processed modality data.

    Examples:
        >>> df = pl.DataFrame({
        ...     "code": [101, 201],
        ...     "modality_fp": ["path1.jpg", "path2.jpg"],
        ...     "modality_idx": [0, 1]
        ... })
        >>> result = process_modality_data(df, DummyReader("."))
        >>> sorted(result.keys())
        ['0', '1']
        >>> result['0']
        tensor([1, 2, 3])
        >>> # Check empty case
        >>> df_empty = pl.DataFrame({
        ...     "code": [101],
        ...     "modality_fp": [None],
        ...     "modality_idx": [None]
        ... })
        >>> process_modality_data(df_empty, DummyReader("."))
        {}
    """
    modality_mapping = {}

    # Filter to rows with non-null modality paths
    modality_df = df.filter(pl.col("modality_fp").is_not_null())

    for row in modality_df.iter_rows(named=True):
        key = f"{round(row['modality_idx'])}"
        modality_mapping[key] = reader.read_modality(row["modality_fp"])

    return modality_mapping


def extract_statics_and_schema(df: pl.LazyFrame) -> pl.LazyFrame:
    """This function extracts static data and schema information.

    Args:
        df: The input data.

    Returns:
        A `pl.LazyFrame` object containing the static data and the unique times of the subject.

    Examples:
        >>> from datetime import datetime
        >>> df = pl.DataFrame({
        ...     "subject_id": [1, 1, 1, 2, 2],
        ...     "time": [None, datetime(2021, 1, 1), datetime(2021, 1, 13),
        ...             None, datetime(2021, 1, 2)],
        ...     "code": [100, 101, 102, 200, 201],
        ...     "numeric_value": [1.0, 2.0, 3.0, 4.0, 5.0],
        ...     "modality_fp": [None, "path1.jpg", "path2.jpg", None, "path3.jpg"]
        ... }).lazy()
        >>> result_df = extract_statics_and_schema(df)
        >>> result_df.collect()
        shape: (2, 5)
        ┌────────────┬───────────┬───────────────┬─────────────────────┬─────────────────────────────────┐
        │ subject_id ┆ code      ┆ numeric_value ┆ start_time          ┆ time                            │
        │ ---        ┆ ---       ┆ ---           ┆ ---                 ┆ ---                             │
        │ i64        ┆ list[i64] ┆ list[f64]     ┆ datetime[μs]        ┆ list[datetime[μs]]              │
        ╞════════════╪═══════════╪═══════════════╪═════════════════════╪═════════════════════════════════╡
        │ 1          ┆ [100]     ┆ [1.0]         ┆ 2021-01-01 00:00:00 ┆ [2021-01-01 00:00:00, 2021-01-… │
        │ 2          ┆ [200]     ┆ [4.0]         ┆ 2021-01-02 00:00:00 ┆ [2021-01-02 00:00:00]           │
        └────────────┴───────────┴───────────────┴─────────────────────┴─────────────────────────────────┘
    """
    static, dynamic = split_static_and_dynamic(df)

    # This collects static data by subject ID and stores only (as a list) the codes and numeric values
    static_by_subject = static.group_by("subject_id", maintain_order=True).agg("code", "numeric_value")

    # This collects the unique times for each subject
    schema_by_subject = dynamic.group_by("subject_id", maintain_order=True).agg(
        pl.col("time").min().alias("start_time"), pl.col("time").unique(maintain_order=True)
    )

    result = static_by_subject.join(schema_by_subject, on="subject_id", how="full", coalesce=True)
    return result


def extract_seq_of_subject_events(
    df: pl.LazyFrame, reader: MultimodalReader
) -> tuple[pl.LazyFrame, dict[str, dict]]:
    """This function extracts sequences of subject events and processes modality data.

    Args:
        df: The input data.
        reader: Instance of MultimodalReader to process the modality data.

    Returns:
        A tuple containing:
        - A `pl.LazyFrame` object containing the sequences of subject events
        - A dictionary mapping modality_idx to processed modality data

    Examples:
        >>> from datetime import datetime
        >>> df = pl.DataFrame({
        ...     "subject_id": [1, 1, 1, 2, 2],
        ...     "time": [None, datetime(2021, 1, 1), datetime(2021, 1, 13),
        ...             None, datetime(2021, 1, 2)],
        ...     "code": [100, 101, 102, 200, 201],
        ...     "numeric_value": [1.0, 2.0, 3.0, 4.0, 5.0],
        ...     "modality_fp": [None, "path1.jpg", None, None, "path2.jpg"]
        ... }).lazy()
        >>> result_df, modality_mapping = extract_seq_of_subject_events(df, DummyReader("."))
        >>> result_df.collect()
        shape: (2, 5)
        ┌────────────┬─────────────────┬─────────────────┬─────────────────┬─────────────────┐
        │ subject_id ┆ time_delta_days ┆ code            ┆ numeric_value   ┆ modality_idx    │
        │ ---        ┆ ---             ┆ ---             ┆ ---             ┆ ---             │
        │ i64        ┆ list[f32]       ┆ list[list[i64]] ┆ list[list[f64]] ┆ list[list[f32]] │
        ╞════════════╪═════════════════╪═════════════════╪═════════════════╪═════════════════╡
        │ 1          ┆ [NaN, 12.0]     ┆ [[101], [102]]  ┆ [[2.0], [3.0]]  ┆ [[0.0], [NaN]]  │
        │ 2          ┆ [NaN]           ┆ [[201]]         ┆ [[5.0]]         ┆ [[1.0]]         │
        └────────────┴─────────────────┴─────────────────┴─────────────────┴─────────────────┘
        >>> modality_mapping  # Check modality mapping was created
        {'0': tensor([1, 2, 3]), '1': tensor([1, 2, 3])}
    """
    _, dynamic = split_static_and_dynamic(df)

    # Process modality data if paths exist
    modality_mapping = {}
    if "modality_fp" in df.columns:
        modality_mapping = process_modality_data(dynamic.collect(), reader)

    time_delta_days_expr = (pl.col("time").diff().dt.total_seconds() / SECONDS_PER_DAY).cast(pl.Float32)

    result = (
        dynamic.group_by("subject_id", "time", maintain_order=True)
        .agg(
            pl.col("code").name.keep(),
            fill_to_nans("numeric_value").name.keep(),
            fill_to_nans("modality_idx").name.keep() if "modality_fp" in df.columns else None,
        )
        .group_by("subject_id", maintain_order=True)
        .agg(
            fill_to_nans(time_delta_days_expr).alias("time_delta_days"),
            "code",
            "numeric_value",
            "modality_idx" if "modality_fp" in df.columns else None,
        )
    )

    return result, modality_mapping


@hydra.main(
    version_base=None, config_path=str(PREPROCESS_CONFIG_YAML.parent), config_name=PREPROCESS_CONFIG_YAML.stem
)
def main(cfg: DictConfig):
    hydra_loguru_init()
    tokenize(cfg)


def tokenize(cfg: DictConfig):
    """Main function for tokenizing MEDS datasets.

    Examples:
        >>> import tempfile
        >>> import polars as pl
        >>> from datetime import datetime
        >>> from omegaconf import OmegaConf
        >>> from safetensors import safe_open
        >>> import torch
        >>> # Create temporary directory and test data
        >>> tmpdir_object = tempfile.TemporaryDirectory()
        >>> tmpdir = str(tmpdir_object.name)
        >>> test_df = pl.DataFrame({
        ...     "subject_id": [1, 1, 1, 2, 2],
        ...     "time": [None, datetime(2021,1,1), datetime(2021,1,2), None, datetime(2021,1,3)],
        ...     "code": [100, 101, 102, 200, 201],
        ...     "numeric_value": [1.0, 2.0, 3.0, 4.0, 5.0],
        ...     "modality_fp": [None, "image1.jpg", None, None, "image2.jpg"]
        ... })
        >>>
        >>> # Save test data
        >>> in_fp = Path(tmpdir) / "shard_0.parquet"
        >>> test_df.write_parquet(in_fp)
        >>>
        >>> # Create config
        >>> cfg = OmegaConf.create({
        ...     "stage": "tokenize",
        ...     "stage_cfg": {
        ...         "input_dir": str(tmpdir),
        ...         "data_input_dir": str(tmpdir),
        ...         "output_dir": str(tmpdir),
        ...         "file_pattern": "shard_*.parquet",
        ...         "do_sequential": True,
        ...         "reader": {
        ...             "_target_": "meds_torch.utils.custom_multimodal_tokenization.DummyReader",
        ...             "base_path": "."},
        ...     },
        ...     "do_overwrite": True
        ... })
        >>>
        >>> # Run tokenize
        >>> tokenize(cfg)
        >>>
        >>> # Verify outputs
        >>> assert (Path(tmpdir) / "schemas" / "shard_0.parquet").exists()
        >>> assert (Path(tmpdir) / "event_seqs" / "shard_0.parquet").exists()
        >>> assert (Path(tmpdir) / "modalities" / "shard_0.safetensors").exists()
        >>>
        >>> # Check schema output
        >>> schema_df = pl.read_parquet(Path(tmpdir) / "schemas" / "shard_0.parquet")
        >>> assert len(schema_df) == 2  # Two subjects
        >>> assert all(col in schema_df.columns for col in [
        ...     "subject_id", "code", "numeric_value", "start_time"])
        >>>
        >>> # Check event sequences output
        >>> events_df = pl.read_parquet(Path(tmpdir) / "event_seqs" / "shard_0.parquet")
        >>> assert len(events_df) == 2  # Two subjects
        >>> assert all(col in events_df.columns for col in [
        ...     "subject_id", "time_delta_days", "code", "numeric_value", "modality_idx"])
        >>>
        >>> # Check modalities output
        >>> with safe_open(
        ...     Path(tmpdir) / "modalities" / "shard_0.safetensors", framework="pt", device="cpu") as f:
        ...     assert set(f.keys()) == {'0', '1'}  # Two modality indices
        ...     assert torch.equal(f.get_tensor('0'), torch.tensor([1, 2, 3]))
        ...     assert torch.equal(f.get_tensor('1'), torch.tensor([1, 2, 3]))
        >>>
        >>> tmpdir_object.cleanup()
    """
    logger.info(
        f"Running with config:\n{OmegaConf.to_yaml(cfg)}\n"
        f"Stage: {cfg.stage}\n\n"
        f"Stage config:\n{OmegaConf.to_yaml(cfg.stage_cfg)}"
    )

    output_dir = Path(cfg.stage_cfg.output_dir)
    reader = hydra.utils.instantiate(cfg.stage_cfg.reader)
    if not hasattr(reader, "read_modality"):
        raise ValueError("cfg.stage_cfg.reader must have a read_modality function.")
    if train_only := cfg.stage_cfg.get("train_only", False):
        raise ValueError(f"train_only={train_only} is not supported for this stage.")
    shards_single_output, include_only_train = shard_iterator(cfg)

    for in_fp, out_fp in shards_single_output:
        sharded_path = out_fp.relative_to(output_dir)

        schema_out_fp = output_dir / "schemas" / sharded_path
        event_seq_out_fp = output_dir / "event_seqs" / sharded_path
        modality_out_fp = (output_dir / "modalities" / sharded_path).with_suffix(".safetensors")

        logger.info(f"Tokenizing {str(in_fp.resolve())} into schemas at {str(schema_out_fp.resolve())}")

        # Add output path for the LazyFrame to use in compute functions
        rwlock_wrap(
            in_fp,
            schema_out_fp,
            pl.scan_parquet,
            write_lazyframe,
            extract_statics_and_schema,
            do_overwrite=cfg.do_overwrite,
        )

        logger.info(f"Tokenizing {str(in_fp.resolve())} into event_seqs at {str(event_seq_out_fp.resolve())}")

        def write_fn(inputs, out_fp):
            df, modality_mapping = inputs
            modality_out_fp.parent.mkdir(parents=True, exist_ok=True)
            save_file(modality_mapping, modality_out_fp)
            write_lazyframe(df, out_fp)

        # Add output path for the LazyFrame to use in compute functions
        rwlock_wrap(
            in_fp,
            event_seq_out_fp,
            pl.scan_parquet,
            write_fn,
            lambda df: extract_seq_of_subject_events(df, reader),
            do_overwrite=cfg.do_overwrite,
        )

    logger.info(f"Done with {cfg.stage}")


if __name__ == "__main__":  # pragma: no cover
    main()
