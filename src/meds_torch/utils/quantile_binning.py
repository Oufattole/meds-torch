#!/usr/bin/env python
"""Transformations for normalizing MEDS datasets, across both categorical and continuous
dimensions.

DO NOT RUN THIS WITH PARALLELISM
"""
from pathlib import Path

import hydra
import polars as pl
from loguru import logger
from MEDS_transforms import PREPROCESS_CONFIG_YAML
from MEDS_transforms.mapreduce.mapper import map_over
from omegaconf import DictConfig, OmegaConf


def process_quantiles(df: pl.DataFrame) -> pl.DataFrame:
    """Process quantiles in a DataFrame, using custom quantiles if available.

    Args:
        df: A Polars DataFrame containing columns for values/quantiles and optionally custom_quantiles

    Returns:
        A DataFrame with processed quantiles and updated code column

    Examples:
    >>> import polars as pl
    >>> df = pl.DataFrame({
    ...     "code": ["lab//A", "lab//B", "lab//C", "lab//D"],
    ...     "numeric_value": [-1.0, 2.0, None, 0.0],
    ...     "values/quantiles": [
    ...         {"values/quantile/0.2": 0, "values/quantile/0.4": 1,
    ...          "values/quantile/0.6": 2, "values/quantile/0.8": 3},
    ...         {"values/quantile/0.2": 0, "values/quantile/0.4": 1,
    ...          "values/quantile/0.6": 2, "values/quantile/0.8": 3},
    ...         {"values/quantile/0.2": 0, "values/quantile/0.4": 1,
    ...          "values/quantile/0.6": 2, "values/quantile/0.8": 3},
    ...         {"values/quantile/0.2": 0, "values/quantile/0.4": 1,
    ...          "values/quantile/0.6": 2, "values/quantile/0.8": 3}
    ...     ],
    ...     "custom_quantiles": [None,
    ...         {"values/quantile/0.5": 1.5},
    ...         None,
    ...         None
    ...     ],
    ...     "code/vocab_index": [0, 1, 2, 3],
    ... })
    >>> result = process_quantiles(df)
    >>> result.select(["code", "numeric_value"])
    shape: (4, 2)
    ┌──────────────┬───────────────┐
    │ code         ┆ numeric_value │
    │ ---          ┆ ---           │
    │ str          ┆ f64           │
    ╞══════════════╪═══════════════╡
    │ lab//A//_Q_1 ┆ -1.0          │
    │ lab//B//_Q_2 ┆ 2.0           │
    │ lab//C       ┆ null          │
    │ lab//D//_Q_1 ┆ 0.0           │
    └──────────────┴───────────────┘
    >>> # Test with only custom quantiles
    >>> df_custom = pl.DataFrame({
    ...     "code": ["lab//A", "lab//A", "lab//A"],
    ...     "numeric_value": [-0.5, 1.0, 4.0],
    ...     "values/quantiles": [
    ...         {"values/quantile/0.2": 0, "values/quantile/0.4": 1,
    ...          "values/quantile/0.6": 2, "values/quantile/0.8": 3},
    ...         {"values/quantile/0.2": 0, "values/quantile/0.4": 1,
    ...          "values/quantile/0.6": 2, "values/quantile/0.8": 3},
    ...         {"values/quantile/0.2": 0, "values/quantile/0.4": 1,
    ...          "values/quantile/0.6": 2, "values/quantile/0.8": 3},
    ...     ],
    ...     "custom_quantiles": [
    ...         None,
    ...         None,
    ...         None,
    ...     ],
    ...     "code/vocab_index": [0, 0, 0],
    ... })
    >>> result_custom = process_quantiles(df_custom)
    >>> result_custom.select(["code", "numeric_value"])
    shape: (3, 2)
    ┌──────────────┬───────────────┐
    │ code         ┆ numeric_value │
    │ ---          ┆ ---           │
    │ str          ┆ f64           │
    ╞══════════════╪═══════════════╡
    │ lab//A//_Q_1 ┆ -0.5          │
    │ lab//A//_Q_2 ┆ 1.0           │
    │ lab//A//_Q_5 ┆ 4.0           │
    └──────────────┴───────────────┘
    """
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
    df = df.drop("quantile", "values/quantiles", "custom_quantiles", quantile_columns)

    return df


def add_custom_quantiles_column(code_metadata: pl.DataFrame, custom_quantiles: dict) -> pl.DataFrame:
    """Add a custom_quantiles column to code_metadata DataFrame based on provided
    dictionary.

    Args:
        code_metadata: A Polars DataFrame containing code information
        custom_quantiles: A dictionary mapping codes to their custom quantile values

    Returns:
        A DataFrame with an added custom_quantiles column

    Examples:
    >>> import polars as pl
    >>> code_metadata = pl.DataFrame({
    ...     "code": ["lab//A", "lab//B", "lab//C"],
    ... })
    >>> custom_quantiles = {
    ...     "lab//A": {"values/quantile/0.5": 1.5},
    ...     "lab//B": {"values/quantile/0.5": 2.5}
    ... }
    >>> result = add_custom_quantiles_column(code_metadata, custom_quantiles)
    >>> result
    shape: (3, 2)
    ┌────────┬──────────────────┐
    │ code   ┆ custom_quantiles │
    │ ---    ┆ ---              │
    │ str    ┆ struct[1]        │
    ╞════════╪══════════════════╡
    │ lab//A ┆ {1.5}            │
    │ lab//B ┆ {2.5}            │
    │ lab//C ┆ null             │
    └────────┴──────────────────┘
    >>> # Test with empty custom_quantiles
    >>> result = add_custom_quantiles_column(code_metadata, {})
    >>> result
    shape: (3, 2)
    ┌────────┬──────────────────┐
    │ code   ┆ custom_quantiles │
    │ ---    ┆ ---              │
    │ str    ┆ null             │
    ╞════════╪══════════════════╡
    │ lab//A ┆ null             │
    │ lab//B ┆ null             │
    │ lab//C ┆ null             │
    └────────┴──────────────────┘
    """
    # Convert custom_quantiles dict to a Polars Series
    if isinstance(custom_quantiles, DictConfig):
        custom_quantiles = OmegaConf.to_container(custom_quantiles)
    custom_quantiles_series = pl.Series(
        name="custom_quantiles",
        values=[custom_quantiles.get(code) if code is not None else None for code in code_metadata["code"]],
    )

    # Add the custom_quantiles column to code_metadata
    return code_metadata.with_columns(custom_quantiles_series)


def convert_to_discrete_quantiles(
    meds_data: pl.DataFrame, code_metadata: pl.DataFrame, custom_quantiles
) -> pl.DataFrame:
    """Converts the numeric values in a MEDS dataset to discrete quantiles that are
    added to the code name.

    Returns:
        - A new DataFrame with the quantile values added to the code name.
        - A new DataFrame with the metadata for the quantile codes.

    Examples:
    >>> from datetime import datetime
    >>> MEDS_df = pl.DataFrame(
    ...     {
    ...         "subject_id": [1, 1, 1, 2, 2, 2, 3],
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
    ...         "subject_id": pl.UInt32,
    ...         "time": pl.Datetime,
    ...         "code": pl.Utf8,
    ...         "numeric_value": pl.Float64,
    ...    },
    ... )
    >>> code_metadata = pl.DataFrame(
    ...     {
    ...         "code": ["lab//A", "lab//C", "dx//B", "dx//E", "lab//F", "dx//D"],
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
    >>> result.sort("subject_id", "time", "code")
    shape: (7, 4)
    ┌────────────┬─────────────────────┬──────────────┬───────────────┐
    │ subject_id ┆ time                ┆ code         ┆ numeric_value │
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
    ...     ).sort("subject_id", "time", "code")
    shape: (7, 4)
    ┌────────────┬─────────────────────┬──────────────┬───────────────┐
    │ subject_id ┆ time                ┆ code         ┆ numeric_value │
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
    """Converts the numeric values in a MEDS dataset to discrete quantiles that are
    added to the code name.

    Returns:
        - A new DataFrame with the quantile values added to the code name.
        - A new DataFrame with the metadata for the quantile codes.

    Examples:
    >>> from datetime import datetime
    >>> code_metadata = pl.DataFrame(
    ...     {
    ...         "code": ["lab//A", "lab//C", "dx//B", "dx//E", "lab//F", "dx//D"],
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
    ...             {"values/quantile/0.2": None, "values/quantile/0.4": None,
    ...                 "values/quantile/0.6": None, "values/quantile/0.8": None},
    ...         ],
    ...     },
    ...     schema = {
    ...         "code": pl.Utf8,
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
    >>> quantile_code_metadata.sort("code")
    shape: (25, 2)
    ┌──────────────┬───────────────────────┐
    │ code         ┆ values/quantiles      │
    │ ---          ┆ ---                   │
    │ str          ┆ struct[4]             │
    ╞══════════════╪═══════════════════════╡
    │ dx//B        ┆ {null,null,null,null} │
    │ dx//D        ┆ {null,null,null,null} │
    │ dx//E        ┆ {-3.0,-1.0,1.0,3.0}   │
    │ dx//E//_Q_1  ┆ {-3.0,-1.0,1.0,3.0}   │
    │ dx//E//_Q_2  ┆ {-3.0,-1.0,1.0,3.0}   │
    │ …            ┆ …                     │
    │ lab//F//_Q_1 ┆ {-3.0,-1.0,1.0,3.0}   │
    │ lab//F//_Q_2 ┆ {-3.0,-1.0,1.0,3.0}   │
    │ lab//F//_Q_3 ┆ {-3.0,-1.0,1.0,3.0}   │
    │ lab//F//_Q_4 ┆ {-3.0,-1.0,1.0,3.0}   │
    │ lab//F//_Q_5 ┆ {-3.0,-1.0,1.0,3.0}   │
    └──────────────┴───────────────────────┘
    """
    # Step 1: Add custom_quantiles column to code_metadata
    final_metadata_columns = code_metadata.columns
    code_metadata = add_custom_quantiles_column(code_metadata, custom_quantiles)

    # Step 2: Generate quantile_code_metadata
    quantile_code_metadata = generate_quantile_code_metadata(code_metadata)

    return quantile_code_metadata.select(final_metadata_columns)


def generate_quantile_code_metadata(code_metadata: pl.DataFrame) -> pl.DataFrame:
    """Modifies the code_metadata DataFrame to include quantile codes.

    Args:
        code_metadata (pl.DataFrame): Current code metadata DataFrame

    Returns:
        pl.DataFrame: dataframe where we duplicate rows for each quantile code.

    Examples:
    >>> from datetime import datetime
    >>> code_metadata = pl.DataFrame(
    ...     {
    ...         "code": ["lab//A", "lab//C", "dx//B", "dx//E", "lab//F", "dx//D"],
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
    ...             {"values/quantile/0.2": None, "values/quantile/0.4": None,
    ...                 "values/quantile/0.6": None, "values/quantile/0.8": None},
    ...         ],
    ...     },
    ...     schema = {
    ...         "code": pl.Utf8,
    ...         "values/quantiles": pl.Struct([
    ...             pl.Field("values/quantile/0.2", pl.Float64),
    ...             pl.Field("values/quantile/0.4", pl.Float64),
    ...             pl.Field("values/quantile/0.6", pl.Float64),
    ...             pl.Field("values/quantile/0.8", pl.Float64),
    ...         ]), # pl.List(pl.Float64),
    ...     },
    ... )
    >>> quantile_code_metadata = generate_quantile_code_metadata(code_metadata)
    >>> quantile_code_metadata.sort("code")
    shape: (26, 2)
    ┌───────────────────────┬──────────────┐
    │ values/quantiles      ┆ code         │
    │ ---                   ┆ ---          │
    │ struct[4]             ┆ str          │
    ╞═══════════════════════╪══════════════╡
    │ {null,null,null,null} ┆ dx//B        │
    │ {null,null,null,null} ┆ dx//D        │
    │ {-3.0,-1.0,1.0,3.0}   ┆ dx//E        │
    │ {-3.0,-1.0,1.0,3.0}   ┆ dx//E//_Q_1  │
    │ {-3.0,-1.0,1.0,3.0}   ┆ dx//E//_Q_2  │
    │ …                     ┆ …            │
    │ {-3.0,-1.0,1.0,3.0}   ┆ lab//F//_Q_1 │
    │ {-3.0,-1.0,1.0,3.0}   ┆ lab//F//_Q_2 │
    │ {-3.0,-1.0,1.0,3.0}   ┆ lab//F//_Q_3 │
    │ {-3.0,-1.0,1.0,3.0}   ┆ lab//F//_Q_4 │
    │ {-3.0,-1.0,1.0,3.0}   ┆ lab//F//_Q_5 │
    └───────────────────────┴──────────────┘
    """
    # Step 1: Get the number of quantiles
    quantile_fields = code_metadata.schema["values/quantiles"].fields
    num_bins = len(quantile_fields) + 1  # Add 1 for n+1 bins
    codes_per_entry = num_bins + 1  # Add 1 for the original code

    # Get first quantile name
    first_quantile = quantile_fields[0].name
    is_null_quantile = code_metadata.select(pl.col("values/quantiles").struct.field(first_quantile).is_null())

    # Determine number of bins per row
    codes_per_entry_expr = pl.when(is_null_quantile).then(1).otherwise(codes_per_entry)

    # If custom quantiles exist, update the expression
    if "custom_quantiles" in code_metadata.columns and isinstance(
        code_metadata.schema["custom_quantiles"], pl.Struct
    ):
        custom_bins = len(code_metadata.schema["custom_quantiles"].fields) + 2
        codes_per_entry_expr = (
            pl.when(pl.col("custom_quantiles").is_not_null())
            .then(custom_bins + 2)
            .otherwise(codes_per_entry_expr)
        )

    code_metadata = code_metadata.with_columns(codes_per_entry_expr.alias("codes_per_entry_expr"))

    # Step 2: Generate rows for each code and its quantiles
    code_metadata = code_metadata.with_row_index("code_index")
    expanded_metadata = (
        code_metadata.select(
            pl.col("code").repeat_by(pl.col("codes_per_entry_expr")),
            pl.col("code_index").repeat_by(pl.col("codes_per_entry_expr")).alias("base_index"),
            pl.exclude("code", "code_index"),
        )
        .explode(["code", "base_index"])
        .with_row_index()
    )

    # Step 3: Generate binned code vocab indices and adjust vocab indices
    offset = (
        expanded_metadata.group_by("code", maintain_order=True)
        .agg(pl.col("base_index").first(), pl.col("codes_per_entry_expr").first())
        .select(
            pl.col("code"),
            pl.col("codes_per_entry_expr").cum_sum().alias("offset") - pl.col("codes_per_entry_expr"),
        )
    )
    expanded_metadata = expanded_metadata.join(offset, on="code", how="left")
    assert expanded_metadata["base_index"].is_sorted()
    expanded_metadata = expanded_metadata.with_columns((pl.col("index") - pl.col("offset")).alias("quantile"))
    expanded_metadata = expanded_metadata.rename({"index": "code_index"}).drop(
        "codes_per_entry_expr", "offset", "code_index"
    )
    if "custom_quantiles" in expanded_metadata.columns:
        expanded_metadata = expanded_metadata.drop("custom_quantiles")

    # Step 4: Generate quantile codes
    expanded_metadata = expanded_metadata.with_columns(
        binned_code=pl.when(pl.col("quantile") != 0)
        .then(pl.concat_str(pl.col("code"), pl.lit("//_Q_"), pl.col("quantile").cast(pl.Utf8)))
        .otherwise(pl.col("code"))
    )

    # Step 5: Select and rename final columns
    final_metadata = expanded_metadata.drop("base_index", "code", "quantile").rename({"binned_code": "code"})

    return final_metadata


def quantile_normalize(
    df: pl.LazyFrame,
    code_metadata: pl.DataFrame,
    code_modifiers: list[str] | None = None,
    custom_quantiles: dict = {},
) -> pl.LazyFrame:
    """Normalize a MEDS dataset across both categorical and continuous dimensions.

    This function expects a MEDS dataset in flattened form, with columns for:
      - `subject_id`
      - `time`
      - `code`
      - `numeric_value`

    In addition, the `code_metadata` dataset should contain information about the codes in the MEDS dataset,
    including the mandatory columns:
      - `code` (`categorical`)
      - Any `code_modifiers` columns, if specified
      - `values/quantiles`

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
        ...         "subject_id": [1, 1, 1, 2, 2, 2, 3],
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
        ...         "subject_id": pl.UInt32,
        ...         "time": pl.Datetime,
        ...         "code": pl.Utf8,
        ...         "numeric_value": pl.Float64,
        ...    },
        ... )
        >>> code_metadata = pl.DataFrame(
        ...     {
        ...         "code": ["lab//A", "lab//C", "dx//B", "dx//E", "lab//F"],
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
        ...         "values/quantiles": pl.Struct([
        ...             pl.Field("values/quantile/0.2", pl.Float64),
        ...             pl.Field("values/quantile/0.4", pl.Float64),
        ...             pl.Field("values/quantile/0.6", pl.Float64),
        ...             pl.Field("values/quantile/0.8", pl.Float64),
        ...         ]), # pl.List(pl.Float64),
        ...     },
        ... )
        >>> code_metadata
        shape: (5, 2)
        ┌────────┬───────────────────────┐
        │ code   ┆ values/quantiles      │
        │ ---    ┆ ---                   │
        │ str    ┆ struct[4]             │
        ╞════════╪═══════════════════════╡
        │ lab//A ┆ {-3.0,-1.0,1.0,3.0}   │
        │ lab//C ┆ {-3.0,-1.0,1.0,3.0}   │
        │ dx//B  ┆ {null,null,null,null} │
        │ dx//E  ┆ {-3.0,-1.0,1.0,3.0}   │
        │ lab//F ┆ {-3.0,-1.0,1.0,3.0}   │
        └────────┴───────────────────────┘
        >>> quantile_normalize(MEDS_df.lazy(), code_metadata).collect().sort("subject_id", "time", "code")
        shape: (7, 4)
        ┌────────────┬─────────────────────┬──────────────┬───────────────┐
        │ subject_id ┆ time                ┆ code         ┆ numeric_value │
        │ ---        ┆ ---                 ┆ ---          ┆ ---           │
        │ u32        ┆ datetime[μs]        ┆ str          ┆ f64           │
        ╞════════════╪═════════════════════╪══════════════╪═══════════════╡
        │ 1          ┆ 2021-01-01 00:00:00 ┆ lab//A//_Q_3 ┆ 1.0           │
        │ 1          ┆ 2021-01-01 00:00:00 ┆ lab//A//_Q_4 ┆ 3.0           │
        │ 1          ┆ 2021-01-02 00:00:00 ┆ dx//B        ┆ null          │
        │ 2          ┆ 2022-10-02 00:00:00 ┆ dx//D        ┆ null          │
        │ 2          ┆ 2022-10-02 00:00:00 ┆ lab//A//_Q_4 ┆ 3.0           │
        │ 2          ┆ 2022-10-02 00:00:00 ┆ lab//C       ┆ null          │
        │ 3          ┆ 2022-10-02 00:00:00 ┆ lab//F       ┆ null          │
        └────────────┴─────────────────────┴──────────────┴───────────────┘
        >>> custom_quantiles = {"lab//A": {"values/quantile/0.5": 2}}
        >>> quantile_normalize(MEDS_df.lazy(), code_metadata, custom_quantiles=custom_quantiles
        ...     ).collect().sort("subject_id", "time", "code")
        shape: (7, 4)
        ┌────────────┬─────────────────────┬──────────────┬───────────────┐
        │ subject_id ┆ time                ┆ code         ┆ numeric_value │
        │ ---        ┆ ---                 ┆ ---          ┆ ---           │
        │ u32        ┆ datetime[μs]        ┆ str          ┆ f64           │
        ╞════════════╪═════════════════════╪══════════════╪═══════════════╡
        │ 1          ┆ 2021-01-01 00:00:00 ┆ lab//A//_Q_1 ┆ 1.0           │
        │ 1          ┆ 2021-01-01 00:00:00 ┆ lab//A//_Q_2 ┆ 3.0           │
        │ 1          ┆ 2021-01-02 00:00:00 ┆ dx//B        ┆ null          │
        │ 2          ┆ 2022-10-02 00:00:00 ┆ dx//D        ┆ null          │
        │ 2          ┆ 2022-10-02 00:00:00 ┆ lab//A//_Q_2 ┆ 3.0           │
        │ 2          ┆ 2022-10-02 00:00:00 ┆ lab//C       ┆ null          │
        │ 3          ┆ 2022-10-02 00:00:00 ┆ lab//F       ┆ null          │
        └────────────┴─────────────────────┴──────────────┴───────────────┘
    """
    # TODO: add support for original values/mean and values/std normalization of continuous values
    df = convert_to_discrete_quantiles(df.collect(), code_metadata, custom_quantiles)
    return df.lazy()


@hydra.main(
    version_base=None, config_path=str(PREPROCESS_CONFIG_YAML.parent), config_name=PREPROCESS_CONFIG_YAML.stem
)
def main(cfg: DictConfig):
    """Bins the numeric values and collapses the bin number into the code name.

    DO NOT RUN THIS WITH PARALLELISM as it will recursively perform quantile binning N workers times.
    """

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
