#!/usr/bin/env python
"""Transformations for normalizing MEDS datasets, across both categorical and continuous dimensions."""


import hydra
import polars as pl
from MEDS_transforms import PREPROCESS_CONFIG_YAML
from MEDS_transforms.mapreduce.mapper import map_over
from omegaconf import DictConfig


def normalize(
    df: pl.LazyFrame, code_metadata: pl.DataFrame, code_modifiers: list[str] | None = None
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
        >>> normalize(MEDS_df.lazy(), code_metadata).collect()
        shape: (6, 4)
        ┌────────────┬─────────────────────┬──────┬───────────────┐
        │ patient_id ┆ time                ┆ code ┆ numeric_value │
        │ ---        ┆ ---                 ┆ ---  ┆ ---           │
        │ u32        ┆ datetime[μs]        ┆ u32  ┆ f64           │
        ╞════════════╪═════════════════════╪══════╪═══════════════╡
        │ 1          ┆ 2021-01-01 00:00:00 ┆ 12   ┆ 1.0           │
        │ 1          ┆ 2021-01-01 00:00:00 ┆ 18   ┆ 3.0           │
        │ 1          ┆ 2021-01-02 00:00:00 ┆ 3    ┆ null          │
        │ 2          ┆ 2022-10-02 00:00:00 ┆ 18   ┆ 3.0           │
        │ 2          ┆ 2022-10-02 00:00:00 ┆ 2    ┆ null          │
        │ 3          ┆ 2022-10-02 00:00:00 ┆ 5    ┆ null          │
        └────────────┴─────────────────────┴──────┴───────────────┘
    """
    if code_modifiers is None:
        code_modifiers = []

    cols_to_select = ["code", "code/vocab_index", "values/quantiles"] + code_modifiers

    idx_col = "_row_idx"
    df_cols = df.collect_schema().names()
    while idx_col in df_cols:
        idx_col = f"_{idx_col}"

    max_code_index = code_metadata["code/vocab_index"].max() + 1

    code_metadata = code_metadata.with_columns(
        code_metadata.select(pl.col("values/quantiles"))
        .unnest("values/quantiles")
        .select(pl.concat_list(pl.all()).alias("values/quantiles"))
    )

    def compute_quantile_index(value, quantiles):
        if value is None or len(quantiles) == 0 or None in quantiles:
            return 0
        for i, q in enumerate(quantiles):
            if value <= q:
                return i + 1
        return len(quantiles)

    # Step 1: Add row index
    df_with_index = df.with_row_index(idx_col)

    # Step 2: Join with code_metadata
    df_joined = df_with_index.join(
        code_metadata.lazy().select(cols_to_select),
        on=["code"] + code_modifiers,
        how="inner",
        join_nulls=True,
    )

    # Step 3: Compute new code
    df_with_new_code = df_joined.with_columns(
        [
            pl.when(pl.col("numeric_value").is_not_null())
            .then(
                pl.struct(["numeric_value", "values/quantiles", "code/vocab_index"]).map_elements(
                    lambda row: max_code_index
                    * compute_quantile_index(row["numeric_value"], row["values/quantiles"])
                    + row["code/vocab_index"],
                    return_dtype=pl.UInt32,
                )
            )
            .otherwise(pl.col("code/vocab_index"))
            .alias("code")
        ]
    )

    # Step 4: Select desired columns
    df_selected = df_with_new_code.select(
        idx_col,
        "patient_id",
        "time",
        "code",
        "numeric_value",
    )

    # Step 5: Sort and drop index column
    df_final = df_selected.sort(idx_col).drop(idx_col)

    return df_final


@hydra.main(
    version_base=None, config_path=str(PREPROCESS_CONFIG_YAML.parent), config_name=PREPROCESS_CONFIG_YAML.stem
)
def main(cfg: DictConfig):
    """TODO."""
    map_over(cfg, compute_fn=normalize)


if __name__ == "__main__":
    main()
