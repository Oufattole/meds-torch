#!/usr/bin/env python
"""Transformations for splitting quantile codes back into their components."""

import hydra
import polars as pl
from MEDS_transforms import PREPROCESS_CONFIG_YAML
from MEDS_transforms.mapreduce.mapper import map_over
from omegaconf import DictConfig


def split_quantile_codes(df: pl.LazyFrame, ignore_pattern: str = None) -> pl.LazyFrame:
    """Split codes with quantile suffixes (e.g., 'lab//A//_Q_1') into separate code and quantile columns.

    Args:
        df: LazyFrame with columns including 'code' that may contain quantile suffixes

    Returns:
        LazyFrame with original code and extracted quantile value

    Examples:
    >>> from datetime import datetime
    >>> df = pl.DataFrame({
    ...     "subject_id": [1, 1, 2, 2],
    ...     "time": [
    ...         datetime(2021, 1, 1),
    ...         datetime(2021, 1, 2),
    ...         datetime(2022, 10, 2),
    ...         datetime(2022, 10, 2),
    ...     ],
    ...     "code": ["lab//A//_Q_1", "dx//B", "lab//A//_Q_4", "lab//C"],
    ...     "numeric_value": [1.0, None, 3.0, None],
    ... })
    >>> result = split_quantile_codes(df.lazy()).collect()
    >>> result
    shape: (6, 5)
    ┌────────────┬─────────────────────┬────────┬───────────────┬──────────┐
    │ subject_id ┆ time                ┆ code   ┆ numeric_value ┆ quantile │
    │ ---        ┆ ---                 ┆ ---    ┆ ---           ┆ ---      │
    │ i64        ┆ datetime[μs]        ┆ str    ┆ f64           ┆ i64      │
    ╞════════════╪═════════════════════╪════════╪═══════════════╪══════════╡
    │ 1          ┆ 2021-01-01 00:00:00 ┆ lab//A ┆ 1.0           ┆ 1        │
    │ 1          ┆ 2021-01-01 00:00:00 ┆ 1      ┆ 1.0           ┆ 1        │
    │ 1          ┆ 2021-01-02 00:00:00 ┆ dx//B  ┆ null          ┆ null     │
    │ 2          ┆ 2022-10-02 00:00:00 ┆ lab//A ┆ 3.0           ┆ 4        │
    │ 2          ┆ 2022-10-02 00:00:00 ┆ 4      ┆ 3.0           ┆ 4        │
    │ 2          ┆ 2022-10-02 00:00:00 ┆ lab//C ┆ null          ┆ null     │
    └────────────┴─────────────────────┴────────┴───────────────┴──────────┘
    >>> result = split_quantile_codes(df.lazy(), "^lab.*$").collect()
    >>> result
    shape: (4, 5)
    ┌────────────┬─────────────────────┬──────────────┬───────────────┬──────────┐
    │ subject_id ┆ time                ┆ code         ┆ numeric_value ┆ quantile │
    │ ---        ┆ ---                 ┆ ---          ┆ ---           ┆ ---      │
    │ i64        ┆ datetime[μs]        ┆ str          ┆ f64           ┆ i64      │
    ╞════════════╪═════════════════════╪══════════════╪═══════════════╪══════════╡
    │ 1          ┆ 2021-01-01 00:00:00 ┆ lab//A//_Q_1 ┆ 1.0           ┆ 1        │
    │ 1          ┆ 2021-01-02 00:00:00 ┆ dx//B        ┆ null          ┆ null     │
    │ 2          ┆ 2022-10-02 00:00:00 ┆ lab//A//_Q_4 ┆ 3.0           ┆ 4        │
    │ 2          ┆ 2022-10-02 00:00:00 ┆ lab//C       ┆ null          ┆ null     │
    └────────────┴─────────────────────┴──────────────┴───────────────┴──────────┘
    """
    # Create regex pattern to match quantile suffix
    quantile_pattern = r"^(.+?)(?://_Q_(\d+))?$"

    ignore_expr = pl.col("code").str.contains(ignore_pattern) if ignore_pattern else pl.lit(False)

    # First create the DataFrame with split codes and quantiles
    df_split = (
        df.with_row_count("original_row_idx")
        .with_columns(
            [
                # Extract original code by removing quantile suffix
                pl.col("code").str.extract(quantile_pattern, 1).alias("original_code"),
                # Extract quantile value, cast to integer, will be null if no match
                pl.col("code").str.extract(quantile_pattern, 2).cast(pl.Int64).alias("quantile"),
            ]
        )
        .with_columns(
            [
                # Use original code as new code column
                pl.when(ignore_expr).then(pl.col("code")).otherwise(pl.col("original_code")).alias("code"),
                # Add row number for ordering
                pl.lit(0).alias("is_quantile_row"),
            ]
        )
        .drop("original_code")
    )

    # Create a second DataFrame with quantile numbers as codes
    df_quantiles = df_split.filter(pl.col("quantile").is_not_null() & ~ignore_expr).with_columns(
        [
            pl.col("quantile").cast(pl.Utf8).alias("code"),  # Convert quantile to string for code column
            pl.lit(1).alias("is_quantile_row"),  # Mark as quantile row for ordering
        ]
    )

    # Combine original and quantile rows, maintaining order
    return (
        pl.concat([df_split, df_quantiles])
        .sort(["subject_id", "time", "original_row_idx", "is_quantile_row"])
        .drop(["is_quantile_row", "original_row_idx"])
    )


@hydra.main(
    version_base=None, config_path=str(PREPROCESS_CONFIG_YAML.parent), config_name=PREPROCESS_CONFIG_YAML.stem
)
def main(cfg: DictConfig):
    """Split quantile codes in MEDS dataset."""

    def split_codes(df, code_metadata, code_modifiers=None):
        return split_quantile_codes(df, cfg.stage_cfg.ignore_regex)

    map_over(cfg, compute_fn=split_codes)


if __name__ == "__main__":
    main()
