#!/usr/bin/env python
"""Transformations for normalizing MEDS datasets, across both categorical and continuous dimensions."""
from pathlib import Path

import hydra
import polars as pl
from loguru import logger
from MEDS_transforms import PREPROCESS_CONFIG_YAML
from MEDS_transforms.mapreduce.mapper import map_over
from omegaconf import DictConfig, OmegaConf


def convert_to_discrete_quantiles(
    meds_data: pl.DataFrame, code_metadata: pl.DataFrame, custom_quantiles
) -> pl.DataFrame:
    """Converts the numeric values in a MEDS dataset to discrete quantiles that are added to the code name.

    Returns:
        - A new DataFrame with the quantile values added to the code name.
        - A new DataFrame with the metadata for the quantile codes.

    Examples:
    >>> from datetime import datetime
    >>> MEDS_df = pl.DataFrame(
    ...     {
    ...         "patient_id": [1, 1, 1, 2, 2, 2, 3],
    ...         "time": [
    ...             datetime(2021, 1, 1),
    ...             datetime(2021, 1, 1),
    ...             datetime(2021, 1, 2),
    ...             datetime(2022, 10, 2),
    ...             datetime(2022, 10, 2),
    ...             datetime(2022, 10, 2),
    ...             datetime(2022, 10, 2),
    ...         ],
    ...         "code": ["lab//A", "lab//C", "dx//B", "lab//A", "dx//D", "lab//C", "lab//F"],
    ...         "numeric_value": [1, 3, None, 3, None, None, None],
    ...     },
    ...     schema = {
    ...         "patient_id": pl.UInt32,
    ...         "time": pl.Datetime,
    ...         "code": pl.Utf8,
    ...         "numeric_value": pl.Float64,
    ...    },
    ... )
    >>> code_metadata = pl.DataFrame(
    ...     {
    ...         "code": ["lab//A", "lab//C", "dx//B", "dx//E", "lab//F", "dx//D"],
    ...         "code/vocab_index": [0, 1, 2, 3, 4, 5],
    ...         "values/quantiles": [ # [[-3,-1,1,3], [-3,-1,1,3], [], [-3,-1,1,3], [-3,-1,1,3], [-3,-1,1,3]],
    ...             {"values/quantile/0.2": -3, "values/quantile/0.4": -1,
    ...                 "values/quantile/0.6": 1, "values/quantile/0.8": 3},
    ...             {"values/quantile/0.2": -3, "values/quantile/0.4": -1,
    ...                 "values/quantile/0.6": 1, "values/quantile/0.8": 3},
    ...             {"values/quantile/0.2": None, "values/quantile/0.4": None,
    ...                 "values/quantile/0.6": None, "values/quantile/0.8": None},
    ...             {"values/quantile/0.2": -3, "values/quantile/0.4": -1,
    ...                 "values/quantile/0.6": 1, "values/quantile/0.8": 3},
    ...             {"values/quantile/0.2": -3, "values/quantile/0.4": -1,
    ...                 "values/quantile/0.6": 1, "values/quantile/0.8": 3},
    ...             {"values/quantile/0.2": -3, "values/quantile/0.4": -1,
    ...                 "values/quantile/0.6": 1, "values/quantile/0.8": 3},
    ...         ],
    ...     },
    ...     schema = {
    ...         "code": pl.Utf8,
    ...         "code/vocab_index": pl.UInt32,
    ...         "values/quantiles": pl.Struct([
    ...             pl.Field("values/quantile/0.2", pl.Float64),
    ...             pl.Field("values/quantile/0.4", pl.Float64),
    ...             pl.Field("values/quantile/0.6", pl.Float64),
    ...             pl.Field("values/quantile/0.8", pl.Float64),
    ...         ]), # pl.List(pl.Float64),
    ...     },
    ... )
    >>> custom_quantiles = {"lab//C": {"values/quantile/0.5": 0}}
    >>> result = convert_to_discrete_quantiles(
    ...    MEDS_df, code_metadata, custom_quantiles)
    >>> result.sort("patient_id", "time", "code")
    shape: (7, 4)
    ┌────────────┬─────────────────────┬──────────────┬───────────────┐
    │ patient_id ┆ time                ┆ code         ┆ numeric_value │
    │ ---        ┆ ---                 ┆ ---          ┆ ---           │
    │ u32        ┆ datetime[μs]        ┆ str          ┆ f64           │
    ╞════════════╪═════════════════════╪══════════════╪═══════════════╡
    │ 1          ┆ 2021-01-01 00:00:00 ┆ lab//A//_Q_3 ┆ 1.0           │
    │ 1          ┆ 2021-01-01 00:00:00 ┆ lab//C//_Q_2 ┆ 3.0           │
    │ 1          ┆ 2021-01-02 00:00:00 ┆ dx//B        ┆ null          │
    │ 2          ┆ 2022-10-02 00:00:00 ┆ dx//D        ┆ null          │
    │ 2          ┆ 2022-10-02 00:00:00 ┆ lab//A//_Q_4 ┆ 3.0           │
    │ 2          ┆ 2022-10-02 00:00:00 ┆ lab//C       ┆ null          │
    │ 3          ┆ 2022-10-02 00:00:00 ┆ lab//F       ┆ null          │
    └────────────┴─────────────────────┴──────────────┴───────────────┘
    >>> custom_quantiles = {"lab//A": {"values/quantile/0.5": 3}}
    >>> convert_to_discrete_quantiles(MEDS_df, code_metadata, custom_quantiles
    ...     ).sort("patient_id", "time", "code")
    shape: (7, 4)
    ┌────────────┬─────────────────────┬──────────────┬───────────────┐
    │ patient_id ┆ time                ┆ code         ┆ numeric_value │
    │ ---        ┆ ---                 ┆ ---          ┆ ---           │
    │ u32        ┆ datetime[μs]        ┆ str          ┆ f64           │
    ╞════════════╪═════════════════════╪══════════════╪═══════════════╡
    │ 1          ┆ 2021-01-01 00:00:00 ┆ lab//A//_Q_1 ┆ 1.0           │
    │ 1          ┆ 2021-01-01 00:00:00 ┆ lab//C//_Q_4 ┆ 3.0           │
    │ 1          ┆ 2021-01-02 00:00:00 ┆ dx//B        ┆ null          │
    │ 2          ┆ 2022-10-02 00:00:00 ┆ dx//D        ┆ null          │
    │ 2          ┆ 2022-10-02 00:00:00 ┆ lab//A//_Q_1 ┆ 3.0           │
    │ 2          ┆ 2022-10-02 00:00:00 ┆ lab//C       ┆ null          │
    │ 3          ┆ 2022-10-02 00:00:00 ┆ lab//F       ┆ null          │
    └────────────┴─────────────────────┴──────────────┴───────────────┘
    """
    # Step 1: Add custom_quantiles column to code_metadata
    code_metadata = add_custom_quantiles_column(code_metadata, custom_quantiles)

    # Step 2: Join meds_data with the updated code_metadata
    result = meds_data.join(code_metadata, on="code", how="left")

    # Step 3: Process quantiles, prioritizing custom_quantiles
    result = process_quantiles(result)

    return result


def convert_metadata_codes_to_discrete_quantiles(
    code_metadata: pl.DataFrame, custom_quantiles
) -> pl.DataFrame:
    """Converts the numeric values in a MEDS dataset to discrete quantiles that are added to the code name.

    Returns:
        - A new DataFrame with the quantile values added to the code name.
        - A new DataFrame with the metadata for the quantile codes.

    Examples:
    >>> from datetime import datetime
    >>> code_metadata = pl.DataFrame(
    ...     {
    ...         "code": ["lab//A", "lab//C", "dx//B", "dx//E", "lab//F", "dx//D"],
    ...         "code/vocab_index": [0, 1, 2, 3, 4, 5],
    ...         "values/quantiles": [ # [[-3,-1,1,3], [-3,-1,1,3], [], [-3,-1,1,3], [-3,-1,1,3], [-3,-1,1,3]],
    ...             {"values/quantile/0.2": -3, "values/quantile/0.4": -1,
    ...                 "values/quantile/0.6": 1, "values/quantile/0.8": 3},
    ...             {"values/quantile/0.2": -3, "values/quantile/0.4": -1,
    ...                 "values/quantile/0.6": 1, "values/quantile/0.8": 3},
    ...             {"values/quantile/0.2": None, "values/quantile/0.4": None,
    ...                 "values/quantile/0.6": None, "values/quantile/0.8": None},
    ...             {"values/quantile/0.2": -3, "values/quantile/0.4": -1,
    ...                 "values/quantile/0.6": 1, "values/quantile/0.8": 3},
    ...             {"values/quantile/0.2": -3, "values/quantile/0.4": -1,
    ...                 "values/quantile/0.6": 1, "values/quantile/0.8": 3},
    ...             {"values/quantile/0.2": -3, "values/quantile/0.4": -1,
    ...                 "values/quantile/0.6": 1, "values/quantile/0.8": 3},
    ...         ],
    ...     },
    ...     schema = {
    ...         "code": pl.Utf8,
    ...         "code/vocab_index": pl.UInt32,
    ...         "values/quantiles": pl.Struct([
    ...             pl.Field("values/quantile/0.2", pl.Float64),
    ...             pl.Field("values/quantile/0.4", pl.Float64),
    ...             pl.Field("values/quantile/0.6", pl.Float64),
    ...             pl.Field("values/quantile/0.8", pl.Float64),
    ...         ]), # pl.List(pl.Float64),
    ...     },
    ... )
    >>> custom_quantiles = {"lab//C": {"values/quantile/0.5": 0}}
    >>> quantile_code_metadata = convert_metadata_codes_to_discrete_quantiles(
    ...    code_metadata, custom_quantiles)
    >>> quantile_code_metadata.sort("code/vocab_index")
    shape: (27, 3)
    ┌──────────────┬──────────────────┬─────────────────────┐
    │ code         ┆ code/vocab_index ┆ values/quantiles    │
    │ ---          ┆ ---              ┆ ---                 │
    │ str          ┆ u32              ┆ struct[4]           │
    ╞══════════════╪══════════════════╪═════════════════════╡
    │ lab//A       ┆ 0                ┆ {-3.0,-1.0,1.0,3.0} │
    │ lab//A//_Q_1 ┆ 1                ┆ {-3.0,-1.0,1.0,3.0} │
    │ lab//A//_Q_2 ┆ 2                ┆ {-3.0,-1.0,1.0,3.0} │
    │ lab//A//_Q_3 ┆ 3                ┆ {-3.0,-1.0,1.0,3.0} │
    │ lab//A//_Q_4 ┆ 4                ┆ {-3.0,-1.0,1.0,3.0} │
    │ …            ┆ …                ┆ …                   │
    │ dx//D        ┆ 22               ┆ {-3.0,-1.0,1.0,3.0} │
    │ dx//D//_Q_1  ┆ 23               ┆ {-3.0,-1.0,1.0,3.0} │
    │ dx//D//_Q_2  ┆ 24               ┆ {-3.0,-1.0,1.0,3.0} │
    │ dx//D//_Q_3  ┆ 25               ┆ {-3.0,-1.0,1.0,3.0} │
    │ dx//D//_Q_4  ┆ 26               ┆ {-3.0,-1.0,1.0,3.0} │
    └──────────────┴──────────────────┴─────────────────────┘
    """
    # Step 1: Add custom_quantiles column to code_metadata
    final_metadata_columns = code_metadata.columns
    code_metadata = add_custom_quantiles_column(code_metadata, custom_quantiles)

    # Step 2: Generate quantile_code_metadata
    quantile_code_metadata = generate_quantile_code_metadata(code_metadata)

    return quantile_code_metadata.select(final_metadata_columns)


def add_custom_quantiles_column(code_metadata: pl.DataFrame, custom_quantiles: dict) -> pl.DataFrame:
    # Convert custom_quantiles dict to a Polars Series
    if isinstance(custom_quantiles, DictConfig):
        custom_quantiles = OmegaConf.to_container(custom_quantiles)
    custom_quantiles_series = pl.Series(
        name="custom_quantiles",
        values=[custom_quantiles.get(code) if code is not None else None for code in code_metadata["code"]],
    )

    # Add the custom_quantiles column to code_metadata
    return code_metadata.with_columns(custom_quantiles_series)


def process_quantiles(df: pl.DataFrame) -> pl.DataFrame:
    # Use custom_quantiles if available, otherwise use values/quantiles
    df = df.with_columns(
        effective_quantiles=pl.when(pl.col("custom_quantiles").is_not_null())
        .then(pl.col("custom_quantiles"))
        .otherwise(pl.col("values/quantiles"))
    )

    # Unnest the effective_quantiles and calculate the quantile
    quantile_columns = pl.selectors.starts_with("values/quantile/")
    df = df.unnest("effective_quantiles").with_columns(
        quantile=pl.when(pl.col("numeric_value").is_not_null()).then(
            pl.sum_horizontal(quantile_columns.lt(pl.col("numeric_value"))).add(1)
        )
    )

    # Create the new code with quantile information
    code_quantile_concat = pl.concat_str(pl.col("code"), pl.lit("//_Q_"), pl.col("quantile"))
    df = df.with_columns(
        code=pl.when(pl.col("quantile").is_not_null()).then(code_quantile_concat).otherwise(pl.col("code"))
    )

    # Clean up intermediate columns
    df = df.drop("quantile", "code/vocab_index", "values/quantiles", "custom_quantiles", quantile_columns)

    return df


def generate_quantile_code_metadata(code_metadata: pl.DataFrame) -> pl.DataFrame:
    # Step 1: Determine the number of quantiles for each code
    num_quantiles = len(code_metadata.schema["values/quantiles"].fields)
    if isinstance(code_metadata.schema["custom_quantiles"], pl.Struct):
        num_custom_quantiles = len(code_metadata.schema["custom_quantiles"].fields)
        num_quantiles_expr = (
            pl.when(pl.col("custom_quantiles").is_not_null())
            .then(pl.lit(num_custom_quantiles) + 1)
            .otherwise(pl.lit(num_quantiles) + 1)
        )
    else:
        num_quantiles_expr = pl.lit(num_quantiles)

    code_metadata = code_metadata.with_columns(num_quantiles_expr.alias("num_quantiles"))

    # Step 2: Generate rows for each code and its quantiles
    expanded_metadata = (
        code_metadata.select(
            pl.col("code").repeat_by(pl.col("num_quantiles")),
            pl.col("code/vocab_index").repeat_by(pl.col("num_quantiles")).alias("base_index"),
            pl.exclude("code", "code/vocab_index"),
        )
        .explode(["code", "base_index"])
        .with_row_index()
    )

    # Step 3: Generate quantile indices and adjust vocab indices
    offset = (
        expanded_metadata.group_by("code", maintain_order=True)
        .agg(pl.col("base_index").first(), pl.col("num_quantiles").first())
        .select(pl.col("code"), pl.col("num_quantiles").cum_sum().alias("offset") - pl.col("num_quantiles"))
    )
    expanded_metadata = expanded_metadata.join(offset, on="code", how="left")
    assert expanded_metadata["base_index"].is_sorted()
    expanded_metadata = expanded_metadata.with_columns((pl.col("index") - pl.col("offset")).alias("quantile"))
    expanded_metadata = expanded_metadata.rename({"index": "code/vocab_index"}).drop(
        "custom_quantiles", "num_quantiles", "offset"
    )

    # Step 4: Generate quantile codes
    expanded_metadata = expanded_metadata.with_columns(
        quantile_code=pl.when(pl.col("quantile") != 0)
        .then(pl.concat_str(pl.col("code"), pl.lit("//_Q_"), pl.col("quantile").cast(pl.Utf8)))
        .otherwise(pl.col("code"))
    )

    # Step 5: Select and rename final columns
    final_metadata = expanded_metadata.drop("base_index", "code", "quantile").rename(
        {"quantile_code": "code"}
    )

    return final_metadata


def quantile_normalize(
    df: pl.LazyFrame,
    code_metadata: pl.DataFrame,
    code_modifiers: list[str] | None = None,
    custom_quantiles: dict = {},
) -> pl.LazyFrame:
    """Normalize a MEDS dataset across both categorical and continuous dimensions.

    This function expects a MEDS dataset in flattened form, with columns for:
      - `patient_id`
      - `time`
      - `code`
      - `numeric_value`

    In addition, the `code_metadata` dataset should contain information about the codes in the MEDS dataset,
    including the mandatory columns:
      - `code` (`categorical`)
      - `code/vocab_index` (`int`)
      - Any `code_modifiers` columns, if specified

    Additionally, it must either have:
      - Pre-computed means and standard deviations for the numeric values of the codes in the MEDS dataset,
        via:
        - `values/mean` (`float`)
        - `values/std` (`float`)
      - Or the necessary statistics to compute the per-occurrence mean and standard deviation of the numeric
        values of the codes in the MEDS dataset, via:
        - `values/n_occurrences` (`int`)
        - `values/sum` (`float`)
        - `values/sum_sqd` (`float`)


    The `values/*` functions will be used to normalize the code numeric values to have a mean of 0 and a
    standard deviation of 1. The output dataframe will further be filtered to only contain rows where the
    `code` in the MEDS dataset appears in the `code_metadata` dataset, and the output `code` column will be
    converted to the `code/vocab_index` integral ID from the `code_metadata` dataset.

    This function can further be customized by specifying additional columns to join on, via the
    `code_modifiers` parameter, which must appear in both the MEDS dataset and the code metadata. These
    columns will be discarded from the output dataframe, which will only contain the four expected input
    columns, though normalized.

    Args:
        df: The MEDS dataset to normalize. See above for the expected schema.
        code_metadata: Metadata about the codes in the MEDS dataset. See above for the expected schema.
        code_modifiers: Additional columns to join on, which will be discarded from the output dataframe.

    Returns:
        The normalized MEDS dataset, with the schema described above.

    Examples:
        >>> from datetime import datetime
        >>> MEDS_df = pl.DataFrame(
        ...     {
        ...         "patient_id": [1, 1, 1, 2, 2, 2, 3],
        ...         "time": [
        ...             datetime(2021, 1, 1),
        ...             datetime(2021, 1, 1),
        ...             datetime(2021, 1, 2),
        ...             datetime(2022, 10, 2),
        ...             datetime(2022, 10, 2),
        ...             datetime(2022, 10, 2),
        ...             datetime(2022, 10, 2),
        ...         ],
        ...         "code": ["lab//A", "lab//A", "dx//B", "lab//A", "dx//D", "lab//C", "lab//F"],
        ...         "numeric_value": [1, 3, None, 3, None, None, None],
        ...     },
        ...     schema = {
        ...         "patient_id": pl.UInt32,
        ...         "time": pl.Datetime,
        ...         "code": pl.Utf8,
        ...         "numeric_value": pl.Float64,
        ...    },
        ... )
        >>> code_metadata = pl.DataFrame(
        ...     {
        ...         "code": ["lab//A", "lab//C", "dx//B", "dx//E", "lab//F"],
        ...         "code/vocab_index": [0, 2, 3, 4, 5],
        ...         "values/quantiles": [ # [[-3,-1,1,3], [-3,-1,1,3], [], [-3,-1,1,3], [-3,-1,1,3]],
        ...             {"values/quantile/0.2": -3, "values/quantile/0.4": -1,
        ...                 "values/quantile/0.6": 1, "values/quantile/0.8": 3},
        ...             {"values/quantile/0.2": -3, "values/quantile/0.4": -1,
        ...                 "values/quantile/0.6": 1, "values/quantile/0.8": 3},
        ...             {"values/quantile/0.2": None, "values/quantile/0.4": None,
        ...                 "values/quantile/0.6": None, "values/quantile/0.8": None},
        ...             {"values/quantile/0.2": -3, "values/quantile/0.4": -1,
        ...                 "values/quantile/0.6": 1, "values/quantile/0.8": 3},
        ...             {"values/quantile/0.2": -3, "values/quantile/0.4": -1,
        ...                 "values/quantile/0.6": 1, "values/quantile/0.8": 3},
        ...         ],
        ...     },
        ...     schema = {
        ...         "code": pl.Utf8,
        ...         "code/vocab_index": pl.UInt32,
        ...         "values/quantiles": pl.Struct([
        ...             pl.Field("values/quantile/0.2", pl.Float64),
        ...             pl.Field("values/quantile/0.4", pl.Float64),
        ...             pl.Field("values/quantile/0.6", pl.Float64),
        ...             pl.Field("values/quantile/0.8", pl.Float64),
        ...         ]), # pl.List(pl.Float64),
        ...     },
        ... )
        >>> code_metadata
        shape: (5, 3)
        ┌────────┬──────────────────┬───────────────────────┐
        │ code   ┆ code/vocab_index ┆ values/quantiles      │
        │ ---    ┆ ---              ┆ ---                   │
        │ str    ┆ u32              ┆ struct[4]             │
        ╞════════╪══════════════════╪═══════════════════════╡
        │ lab//A ┆ 0                ┆ {-3.0,-1.0,1.0,3.0}   │
        │ lab//C ┆ 2                ┆ {-3.0,-1.0,1.0,3.0}   │
        │ dx//B  ┆ 3                ┆ {null,null,null,null} │
        │ dx//E  ┆ 4                ┆ {-3.0,-1.0,1.0,3.0}   │
        │ lab//F ┆ 5                ┆ {-3.0,-1.0,1.0,3.0}   │
        └────────┴──────────────────┴───────────────────────┘
        >>> quantile_normalize(MEDS_df.lazy(), code_metadata).collect().sort("patient_id", "time", "code")
        shape: (4, 4)
        ┌────────────┬─────────────────────┬──────┬───────────────┐
        │ patient_id ┆ time                ┆ code ┆ numeric_value │
        │ ---        ┆ ---                 ┆ ---  ┆ ---           │
        │ u32        ┆ datetime[μs]        ┆ u32  ┆ f64           │
        ╞════════════╪═════════════════════╪══════╪═══════════════╡
        │ 1          ┆ 2021-01-01 00:00:00 ┆ 3    ┆ 1.0           │
        │ 1          ┆ 2021-01-02 00:00:00 ┆ 8    ┆ null          │
        │ 2          ┆ 2022-10-02 00:00:00 ┆ 4    ┆ null          │
        │ 3          ┆ 2022-10-02 00:00:00 ┆ 16   ┆ null          │
        └────────────┴─────────────────────┴──────┴───────────────┘
        >>> custom_quantiles = {"lab//A": {"values/quantile/0.5": 2}}
        >>> quantile_normalize(MEDS_df.lazy(), code_metadata, custom_quantiles=custom_quantiles
        ...     ).collect().sort("patient_id", "time", "code")
        shape: (4, 4)
        ┌────────────┬─────────────────────┬──────┬───────────────┐
        │ patient_id ┆ time                ┆ code ┆ numeric_value │
        │ ---        ┆ ---                 ┆ ---  ┆ ---           │
        │ u32        ┆ datetime[μs]        ┆ u32  ┆ f64           │
        ╞════════════╪═════════════════════╪══════╪═══════════════╡
        │ 1          ┆ 2021-01-01 00:00:00 ┆ 1    ┆ 1.0           │
        │ 1          ┆ 2021-01-02 00:00:00 ┆ 7    ┆ null          │
        │ 2          ┆ 2022-10-02 00:00:00 ┆ 2    ┆ null          │
        │ 3          ┆ 2022-10-02 00:00:00 ┆ 17   ┆ null          │
        └────────────┴─────────────────────┴──────┴───────────────┘
    """
    # TODO: add support for original values/mean and values/std normalization of continuous values
    df = convert_to_discrete_quantiles(df.collect(), code_metadata, custom_quantiles)
    final_columns = df.columns
    metadata = convert_metadata_codes_to_discrete_quantiles(code_metadata, custom_quantiles)
    df = df.join(metadata["code", "code/vocab_index"], on="code", how="inner")
    df = df.drop("code").rename({"code/vocab_index": "code"})[final_columns]
    return df.lazy()


@hydra.main(
    version_base=None, config_path=str(PREPROCESS_CONFIG_YAML.parent), config_name=PREPROCESS_CONFIG_YAML.stem
)
def main(cfg: DictConfig):
    """TODO."""

    def normalize(df, code_metadata, code_modifiers=None):
        return quantile_normalize(
            df,
            code_metadata,
            code_modifiers=code_modifiers,
            custom_quantiles=cfg.stage_cfg.get("custom_quantiles", {}),
        )

    map_over(cfg, compute_fn=normalize)

    custom_quantiles = cfg.stage_cfg.get("custom_quantiles", {})

    metadata_input_dir = Path(cfg.stage_cfg.metadata_input_dir)
    code_metadata = pl.read_parquet(metadata_input_dir / "codes.parquet", use_pyarrow=True)
    quantile_code_metadata = convert_metadata_codes_to_discrete_quantiles(code_metadata, custom_quantiles)

    output_fp = metadata_input_dir / "codes.parquet"
    output_fp.parent.mkdir(parents=True, exist_ok=True)
    logger.info(f"Indices assigned. Writing to {output_fp}")
    quantile_code_metadata.write_parquet(output_fp, use_pyarrow=True)


if __name__ == "__main__":
    main()
