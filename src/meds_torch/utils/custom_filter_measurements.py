#!/usr/bin/env python
"""A polars-to-polars transformation function for filtering subjects by sequence
length."""
from collections.abc import Callable

import hydra
import polars as pl
from MEDS_transforms import PREPROCESS_CONFIG_YAML
from MEDS_transforms.mapreduce.mapper import map_over
from omegaconf import DictConfig


def filter_measurements_fntr(
    stage_cfg: DictConfig, code_metadata: pl.LazyFrame, code_modifiers: list[str] | None = None
) -> Callable[[pl.LazyFrame], pl.LazyFrame]:
    """Returns a function that filters subject events to only encompass those with a set
    of permissible codes.

    Args:
        df: The input DataFrame.
        stage_cfg: The configuration for the code filtering stage.

    Returns:
        The processed DataFrame.

    Examples:
        >>> code_metadata_df = pl.DataFrame({
        ...     "code":               ["A", "A", "B", "C"],
        ...     "modifier1":          [1,   2,   1,   2],
        ...     "code/n_subjects":    [2,   1,   3,   2],
        ...     "code/n_occurrences": [4,   5,   3,   2],
        ... })
        >>> data = pl.DataFrame({
        ...     "subject_id": [1,   1,   2,   2],
        ...     "code":       ["A", "B", "A", "C"],
        ...     "modifier1":  [1,   1,   2,   2],
        ... }).lazy()
        >>> stage_cfg = DictConfig({"min_subjects_per_code": 2, "min_occurrences_per_code": 3})
        >>> fn = filter_measurements_fntr(stage_cfg, code_metadata_df, ["modifier1"])
        >>> fn(data).collect()
        shape: (2, 3)
        ┌────────────┬──────┬───────────┐
        │ subject_id ┆ code ┆ modifier1 │
        │ ---        ┆ ---  ┆ ---       │
        │ i64        ┆ str  ┆ i64       │
        ╞════════════╪══════╪═══════════╡
        │ 1          ┆ A    ┆ 1         │
        │ 1          ┆ B    ┆ 1         │
        └────────────┴──────┴───────────┘
        >>> stage_cfg = DictConfig({"min_subjects_per_code": 1, "min_occurrences_per_code": 4})
        >>> fn = filter_measurements_fntr(stage_cfg, code_metadata_df, ["modifier1"])
        >>> fn(data).collect()
        shape: (2, 3)
        ┌────────────┬──────┬───────────┐
        │ subject_id ┆ code ┆ modifier1 │
        │ ---        ┆ ---  ┆ ---       │
        │ i64        ┆ str  ┆ i64       │
        ╞════════════╪══════╪═══════════╡
        │ 1          ┆ A    ┆ 1         │
        │ 2          ┆ A    ┆ 2         │
        └────────────┴──────┴───────────┘
        >>> stage_cfg = DictConfig({"min_subjects_per_code": 1})
        >>> fn = filter_measurements_fntr(stage_cfg, code_metadata_df, ["modifier1"])
        >>> fn(data).collect()
        shape: (4, 3)
        ┌────────────┬──────┬───────────┐
        │ subject_id ┆ code ┆ modifier1 │
        │ ---        ┆ ---  ┆ ---       │
        │ i64        ┆ str  ┆ i64       │
        ╞════════════╪══════╪═══════════╡
        │ 1          ┆ A    ┆ 1         │
        │ 1          ┆ B    ┆ 1         │
        │ 2          ┆ A    ┆ 2         │
        │ 2          ┆ C    ┆ 2         │
        └────────────┴──────┴───────────┘
        >>> stage_cfg = DictConfig({"min_subjects_per_code": None, "min_occurrences_per_code": None})
        >>> fn = filter_measurements_fntr(stage_cfg, code_metadata_df, ["modifier1"])
        >>> fn(data).collect()
        shape: (4, 3)
        ┌────────────┬──────┬───────────┐
        │ subject_id ┆ code ┆ modifier1 │
        │ ---        ┆ ---  ┆ ---       │
        │ i64        ┆ str  ┆ i64       │
        ╞════════════╪══════╪═══════════╡
        │ 1          ┆ A    ┆ 1         │
        │ 1          ┆ B    ┆ 1         │
        │ 2          ┆ A    ┆ 2         │
        │ 2          ┆ C    ┆ 2         │
        └────────────┴──────┴───────────┘
        >>> stage_cfg = DictConfig({"min_occurrences_per_code": 5})
        >>> fn = filter_measurements_fntr(stage_cfg, code_metadata_df, ["modifier1"])
        >>> fn(data).collect()
        shape: (1, 3)
        ┌────────────┬──────┬───────────┐
        │ subject_id ┆ code ┆ modifier1 │
        │ ---        ┆ ---  ┆ ---       │
        │ i64        ┆ str  ┆ i64       │
        ╞════════════╪══════╪═══════════╡
        │ 2          ┆ A    ┆ 2         │
        └────────────┴──────┴───────────┘

        >>> data = pl.DataFrame({
        ...     "subject_id": [1,   1,   2,   2],
        ...     "code":       ["BIRTH", "VISIT//PULMONARY", "BIRTH", "BIRTH//VISIT"],
        ...     "modifier1":  [1,   1,   2,   2],
        ... }).lazy()
        >>> code_metadata_df = pl.DataFrame({
        ...     "code":               ["BIRTH", "VISIT//PULMONARY", "BIRTH", "BIRTH//VISIT"],
        ...     "modifier1":          [1,   1,   2,   2],
        ...     "code/n_subjects":    [2,   1,   3,   2],
        ...     "code/n_occurrences": [4,   5,   3,   2],
        ... })
        >>> stage_cfg = DictConfig({"additional_codes": ["^BIRTH$", "^VISIT.*$"]})
        >>> fn = filter_measurements_fntr(stage_cfg, code_metadata_df, ["modifier1"])
        >>> fn(data).collect()
        shape: (3, 3)
        ┌────────────┬──────────────────┬───────────┐
        │ subject_id ┆ code             ┆ modifier1 │
        │ ---        ┆ ---              ┆ ---       │
        │ i64        ┆ str              ┆ i64       │
        ╞════════════╪══════════════════╪═══════════╡
        │ 1          ┆ BIRTH            ┆ 1         │
        │ 1          ┆ VISIT//PULMONARY ┆ 1         │
        │ 2          ┆ BIRTH            ┆ 2         │
        └────────────┴──────────────────┴───────────┘

    This stage works even if the default row index column exists:
        >>> code_metadata_df = pl.DataFrame({
        ...     "code":               ["A", "A", "B", "C"],
        ...     "modifier1":          [1,   2,   1,   2],
        ...     "code/n_subjects":    [2,   1,   3,   2],
        ...     "code/n_occurrences": [4,   5,   3,   2],
        ... })
        >>> data = pl.DataFrame({
        ...     "subject_id": [1,   1,   2,   2],
        ...     "code":       ["A", "B", "A", "C"],
        ...     "modifier1":  [1,   1,   2,   2],
        ...     "_row_idx":   [1,   1,   1,   1],
        ... }).lazy()
        >>> stage_cfg = DictConfig({"min_subjects_per_code": 2, "min_occurrences_per_code": 3})
        >>> fn = filter_measurements_fntr(stage_cfg, code_metadata_df, ["modifier1"])
        >>> fn(data).collect()
        shape: (2, 4)
        ┌────────────┬──────┬───────────┬──────────┐
        │ subject_id ┆ code ┆ modifier1 ┆ _row_idx │
        │ ---        ┆ ---  ┆ ---       ┆ ---      │
        │ i64        ┆ str  ┆ i64       ┆ i64      │
        ╞════════════╪══════╪═══════════╪══════════╡
        │ 1          ┆ A    ┆ 1         ┆ 1        │
        │ 1          ┆ B    ┆ 1         ┆ 1        │
        └────────────┴──────┴───────────┴──────────┘
    """

    min_subjects_per_code = stage_cfg.get("min_subjects_per_code", None)
    min_occurrences_per_code = stage_cfg.get("min_occurrences_per_code", None)
    additional_codes = stage_cfg.get("additional_codes", None)

    filter_exprs = []
    if min_subjects_per_code is not None:
        filter_exprs.append(pl.col("code/n_subjects") >= min_subjects_per_code)
    if min_occurrences_per_code is not None:
        filter_exprs.append(pl.col("code/n_occurrences") >= min_occurrences_per_code)

    if additional_codes is not None:
        additional_codes_expr = pl.col("code").str.contains("|".join(additional_codes))
        if filter_exprs:
            filter_expr = pl.any_horizontal([pl.all_horizontal(filter_exprs), additional_codes_expr])
        else:
            filter_expr = additional_codes_expr

    else:
        if not filter_exprs:
            return lambda df: df
        filter_expr = pl.all_horizontal(filter_exprs)

    join_cols = ["code"]
    if code_modifiers:
        join_cols.extend(code_modifiers)

    allowed_code_metadata = (code_metadata.filter(filter_expr).select(join_cols)).lazy()

    def filter_measurements_fn(df: pl.LazyFrame) -> pl.LazyFrame:
        f"""Filters subject events to only encompass those with a set of permissible codes.

        In particular, this function filters the DataFrame to only include (code, modifier) pairs that have
        at least {min_subjects_per_code} subjects and {min_occurrences_per_code} occurrences.
        """

        idx_col = "_row_idx"
        df_columns = set(df.collect_schema().names())
        while idx_col in df_columns:
            idx_col = f"_{idx_col}"

        return (
            df.with_row_index(idx_col)
            .join(allowed_code_metadata, on=join_cols, how="inner")
            .sort(idx_col)
            .drop(idx_col)
        )

    return filter_measurements_fn


@hydra.main(
    version_base=None, config_path=str(PREPROCESS_CONFIG_YAML.parent), config_name=PREPROCESS_CONFIG_YAML.stem
)
def main(cfg: DictConfig):
    """TODO."""
    map_over(cfg, compute_fn=filter_measurements_fntr)


if __name__ == "__main__":  # pragma: no cover
    main()
