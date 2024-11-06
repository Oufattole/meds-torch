#!/usr/bin/env python
"""Functions for tokenizing MEDS datasets.

Here, _tokenization_ refers specifically to the process of converting a longitudinal, irregularly sampled,
continuous time sequence into a temporal sequence at the level that will be consumed by deep-learning models.

All these functions take in _normalized_ data -- meaning data where there are _no longer_ any code modifiers,
as those have been normalized alongside codes into integer indices (in the output code column). The only
columns of concern here thus are `subject_id`, `time`, `code`, `numeric_value`.
"""

from pathlib import Path

import hydra
import polars as pl
from loguru import logger
from MEDS_transforms import PREPROCESS_CONFIG_YAML
from MEDS_transforms.mapreduce.utils import rwlock_wrap, shard_iterator
from MEDS_transforms.utils import hydra_loguru_init, write_lazyframe
from omegaconf import DictConfig, OmegaConf
from safetensors.torch import save_file
from transformers import AutoTokenizer

TOKENIZER = AutoTokenizer.from_pretrained("emilyalsentzer/Bio_ClinicalBERT")

SECONDS_PER_MINUTE = 60.0
SECONDS_PER_HOUR = SECONDS_PER_MINUTE * 60.0
SECONDS_PER_DAY = SECONDS_PER_HOUR * 24.0


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
        >>> print(fill_to_nans(pl.col("time_delta"))) # doctest: +NORMALIZE_WHITESPACE
        .when([(col("time_delta").is_infinite()) |
               (col("time_delta").is_null())]).then(dyn float: NaN).otherwise(col("time_delta"))
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
    For dynamic data, a modality index is added for non-null text values.

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
        ...     "text_value": [None, "fever", None, "cough"]
        ... }).lazy()
        >>> static, dynamic = split_static_and_dynamic(df)
        >>> static.collect()
        shape: (2, 4)
        ┌────────────┬──────┬───────────────┬────────────┐
        │ subject_id ┆ code ┆ numeric_value ┆ text_value │
        │ ---        ┆ ---  ┆ ---           ┆ ---        │
        │ i64        ┆ i64  ┆ f64           ┆ str        │
        ╞════════════╪══════╪═══════════════╪════════════╡
        │ 1          ┆ 100  ┆ 1.0           ┆ null       │
        │ 2          ┆ 200  ┆ 3.0           ┆ null       │
        └────────────┴──────┴───────────────┴────────────┘
        >>> dynamic.collect()
        shape: (2, 6)
        ┌────────────┬─────────────────────┬──────┬───────────────┬────────────┬──────────────┐
        │ subject_id ┆ time                ┆ code ┆ numeric_value ┆ text_value ┆ modality_idx │
        │ ---        ┆ ---                 ┆ ---  ┆ ---           ┆ ---        ┆ ---          │
        │ i64        ┆ datetime[μs]        ┆ i64  ┆ f64           ┆ str        ┆ f32          │
        ╞════════════╪═════════════════════╪══════╪═══════════════╪════════════╪══════════════╡
        │ 1          ┆ 2021-01-01 00:00:00 ┆ 101  ┆ 2.0           ┆ fever      ┆ 1.0          │
        │ 2          ┆ 2021-01-02 00:00:00 ┆ 201  ┆ 4.0           ┆ cough      ┆ 0.0          │
        └────────────┴─────────────────────┴──────┴───────────────┴────────────┴──────────────┘
    """
    static = df.filter(pl.col("time").is_null()).drop("time")
    dynamic = df.filter(pl.col("time").is_not_null())

    # Add modality index for text values
    if "text_value" in df.columns:
        dynamic = dynamic.with_columns(
            [
                pl.when(pl.col("text_value").is_not_null())
                .then(pl.col("text_value").rank("dense") - 1)
                .otherwise(None)
                .cast(pl.Float32)
                .alias("modality_idx")
            ]
        )

    return static, dynamic


def tokenize_text_values(df: pl.DataFrame) -> dict[str, dict]:
    """Tokenize text values and create a mapping of code_modality to tokens.

    Args:
        df: DataFrame containing text values and their corresponding codes and modality indices.

    Returns:
        Dictionary mapping f"{code}_{modality_idx}" to tokenized text.

    Examples:
        >>> df = pl.DataFrame({
        ...     "code": [101, 201],
        ...     "text_value": ["fever", "cough"],
        ...     "modality_idx": [0, 1]
        ... })
        >>> result = tokenize_text_values(df)
        >>> sorted(result.keys())  # Check keys are formatted correctly
        ['0', '1']
        >>> result['0']
        tensor([  101, 10880,   102])
        >>> result['1']
        tensor([  101, 21810,   102])
        >>> # Check empty case
        >>> df_empty = pl.DataFrame({
        ...     "code": [101],
        ...     "text_value": [None],
        ...     "modality_idx": [None]
        ... })
        >>> tokenize_text_values(df_empty)
        {}
    """
    text_mapping = {}

    # Filter to rows with non-null text values
    text_df = df.filter(pl.col("text_value").is_not_null())

    for row in text_df.iter_rows(named=True):
        key = f"{round(row['modality_idx'])}"
        tokens = TOKENIZER(row["text_value"], return_tensors="pt")
        text_mapping[key] = tokens["input_ids"].squeeze()

    return text_mapping


def extract_statics_and_schema(df: pl.LazyFrame) -> tuple[pl.LazyFrame, dict[str, dict]]:
    """This function extracts static data and schema information (sequence of subject unique times).

    Args:
        df: The input data.

    Returns:
        A tuple containing:
        - A `pl.LazyFrame` object containing the static data and the unique times of the subject
        - A dictionary mapping code_modality to tokenized text

    Examples:
        >>> from datetime import datetime
        >>> df = pl.DataFrame({
        ...     "subject_id": [1, 1, 1, 2, 2],
        ...     "time": [None, datetime(2021, 1, 1), datetime(2021, 1, 13),
        ...             None, datetime(2021, 1, 2)],
        ...     "code": [100, 101, 102, 200, 201],
        ...     "numeric_value": [1.0, 2.0, 3.0, 4.0, 5.0],
        ...     "text_value": [None, "fever", "cough", None, "pain"]
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


def extract_seq_of_subject_events(df: pl.LazyFrame) -> tuple[pl.LazyFrame, dict[str, dict]]:
    """This function extracts sequences of subject events, which are sequences of measurements.

    Args:
        df: The input data.

    Returns:
        A tuple containing:
        - A `pl.LazyFrame` object containing the sequences of subject events
        - A dictionary mapping code_modality to tokenized text

    Examples:
        >>> from datetime import datetime
        >>> df = pl.DataFrame({
        ...     "subject_id": [1, 1, 1, 2, 2],
        ...     "time": [None, datetime(2021, 1, 1), datetime(2021, 1, 13),
        ...             None, datetime(2021, 1, 2)],
        ...     "code": [100, 101, 102, 200, 201],
        ...     "numeric_value": [1.0, 2.0, 3.0, 4.0, 5.0],
        ...     "text_value": [None, "fever", None, None, "pain"]
        ... }).lazy()
        >>> result_df, text_mapping = extract_seq_of_subject_events(df)
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
        >>> sorted(text_mapping.keys())  # Check text mapping was created
        ['0', '1']
    """
    _, dynamic = split_static_and_dynamic(df)

    # Process text values if they exist
    text_mapping = {}
    if "text_value" in df.columns:
        text_mapping = tokenize_text_values(dynamic.collect())

    time_delta_days_expr = (pl.col("time").diff().dt.total_seconds() / SECONDS_PER_DAY).cast(pl.Float32)

    result = (
        dynamic.group_by("subject_id", "time", maintain_order=True)
        .agg(
            pl.col("code").name.keep(),
            fill_to_nans("numeric_value").name.keep(),
            fill_to_nans("modality_idx").name.keep() if "text_value" in df.columns else None,
        )
        .group_by("subject_id", maintain_order=True)
        .agg(
            fill_to_nans(time_delta_days_expr).alias("time_delta_days"),
            "code",
            "numeric_value",
            "modality_idx" if "text_value" in df.columns else None,
        )
    )

    return result, text_mapping


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
        >>>
        >>> # Create temporary directory for test data
        >>> with tempfile.TemporaryDirectory() as tmpdir:
        ...     # Create test input data
        ...     test_df = pl.DataFrame({
        ...         "subject_id": [1, 1, 1, 2, 2],
        ...         "time": [None, datetime(2021,1,1), datetime(2021,1,2), None, datetime(2021,1,3)],
        ...         "code": [100, 101, 102, 200, 201],
        ...         "numeric_value": [1.0, 2.0, 3.0, 4.0, 5.0],
        ...         "text_value": [None, "normal", None, None, "abnormal"]
        ...     })
        ...
        ...     # Save test data
        ...     in_fp = Path(tmpdir) / "shard_0.parquet"
        ...     test_df.write_parquet(in_fp)
        ...
        ...     # Create config
        ...     cfg = OmegaConf.create({
        ...         "stage": "tokenize",
        ...         "stage_cfg": {
        ...             "input_dir": str(tmpdir),
        ...             "data_input_dir": str(tmpdir),
        ...             "output_dir": str(tmpdir),
        ...             "file_pattern": "shard_*.parquet",
        ...             "do_sequential": True
        ...         },
        ...         "do_overwrite": True
        ...     })
        ...
        ...     # Run tokenize
        ...     tokenize(cfg)
        ...
        ...     # Verify outputs
        ...     assert (Path(tmpdir) / "schemas" / "shard_0.parquet").exists()
        ...     assert (Path(tmpdir) / "event_seqs" / "shard_0.parquet").exists()
        ...     assert (Path(tmpdir) / "modalities" / "shard_0.safetensors").exists()
        ...
        ...     # Check schema output
        ...     schema_df = pl.read_parquet(Path(tmpdir) / "schemas" / "shard_0.parquet")
        ...     assert len(schema_df) == 2  # Two subjects
        ...     assert all(col in schema_df.columns for col in [
        ...         "subject_id", "code", "numeric_value", "start_time"])
        ...
        ...     # Check event sequences output
        ...     events_df = pl.read_parquet(Path(tmpdir) / "event_seqs" / "shard_0.parquet")
        ...     assert len(events_df) == 2  # Two subjects
        ...     assert all(col in events_df.columns for col in [
        ...         "subject_id", "time_delta_days", "code", "numeric_value", "modality_idx"])
        ...
        ...     # Check event sequences output
        ...     with safe_open(
        ...         Path(tmpdir) / "modalities" / "shard_0.safetensors",
        ...         framework="pt", device="cpu") as f:
        ...         assert set(f.keys()) == {'1', '0'}
        ...         print(f.get_tensor('1'))
        ...         print(f.get_tensor('0'))
        tensor([ 101, 2999,  102])
        tensor([  101, 22832,   102])
    """

    logger.info(
        f"Running with config:\n{OmegaConf.to_yaml(cfg)}\n"
        f"Stage: {cfg.stage}\n\n"
        f"Stage config:\n{OmegaConf.to_yaml(cfg.stage_cfg)}"
    )

    output_dir = Path(cfg.stage_cfg.output_dir)
    if train_only := cfg.stage_cfg.get("train_only", False):
        raise ValueError(f"train_only={train_only} is not supported for this stage.")
    shards_single_output, include_only_train = shard_iterator(cfg)

    for in_fp, out_fp in shards_single_output:
        sharded_path = out_fp.relative_to(output_dir)

        schema_out_fp = output_dir / "schemas" / sharded_path
        event_seq_out_fp = output_dir / "event_seqs" / sharded_path
        text_out_fp = (output_dir / "modalities" / sharded_path).with_suffix(".safetensors")

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
            df, text_mapping = inputs
            text_out_fp.parent.mkdir(parents=True, exist_ok=True)
            save_file(text_mapping, text_out_fp)
            write_lazyframe(df, out_fp)

        # Add output path for the LazyFrame to use in compute functions
        rwlock_wrap(
            in_fp,
            event_seq_out_fp,
            pl.scan_parquet,
            write_fn,
            extract_seq_of_subject_events,
            do_overwrite=cfg.do_overwrite,
        )

    logger.info(f"Done with {cfg.stage}")


if __name__ == "__main__":  # pragma: no cover
    main()
