#!/usr/bin/env python
"""
Functions for tensorizing MEDS datasets.

This stage takes the output of the tokenization step (which may include
columns like `time_delta_days`, `code`, `numeric_value`, and optionally
`modality_idx` if text data was embedded) and converts them into a
JointNestedRaggedTensorDict (NRT).
"""

from functools import partial

import hydra
import polars as pl
from loguru import logger
from MEDS_transforms import PREPROCESS_CONFIG_YAML
from MEDS_transforms.mapreduce.mapper import map_over
from MEDS_transforms.mapreduce.utils import shard_iterator
from nested_ragged_tensors.ragged_numpy import JointNestedRaggedTensorDict
from omegaconf import DictConfig


def convert_to_NRT(df: pl.LazyFrame) -> JointNestedRaggedTensorDict:
    """
    Converts a tokenized dataframe into a nested ragged tensor using
    JointNestedRaggedTensorDict. Typically, we expect columns like:

        - `time_delta_<something>`  (e.g. `time_delta_days`)
        - `code` (ragged list of codes)
        - `numeric_value` (ragged list of numeric values)
        - `modality_idx` (optional, if text embeddings or other modalities are present)

    Args:
        df: A Polars lazy DataFrame containing the above columns.

    Returns:
        A JointNestedRaggedTensorDict object representing the selected columns.

    Raises:
        ValueError: If there are no columns beginning with `time_delta_`
                    or if there is more than one such column.

    Examples:
        Example input dataframe to this stage: 

        subject_id   time_delta_days     code              numeric_value         modality_idx
        ------------------------------------------------------------------------------------------
        1            [0.0, 3.5]          [[101], [102,103]] [[10.0], [5.5, 6.1]]  [0, 1, 2]
        2            [0.0]               [[201]]            [[2.2]]               [3]

        >>> import polars as pl
        >>> from nested_ragged_tensors.ragged_numpy import JointNestedRaggedTensorDict
        >>> # Minimal example: some rows have 2 timesteps, some have 1
        >>> df = pl.DataFrame({
        ...     "subject_id": [1, 2],
        ...     "time_delta_days": [[float("nan"), 3.5], [float("nan")]],
        ...     "code": [[[101], [102, 103]], [[201]]],
        ...     "numeric_value": [[[10.0], [5.5, 6.1]], [[2.2]]],
        ... })
        >>> nrt = convert_to_NRT(df.lazy())
        >>> isinstance(nrt, JointNestedRaggedTensorDict)
        True
        >>> sorted(nrt.keys())
        ['code', 'numeric_value', 'time_delta_days']
    """
    # 1) Identify the one time_delta column (e.g., "time_delta_days")
    schema = df.collect_schema()
    all_columns = schema.names()
    time_delta_cols = [c for c in all_columns if c.startswith("time_delta_")]

    if len(time_delta_cols) == 0:
        raise ValueError("Expected at least one 'time_delta_' column, found none.")
    elif len(time_delta_cols) > 1:
        raise ValueError(f"Expected exactly one time delta column, found: {time_delta_cols}")

    time_delta_col = time_delta_cols[0]

    # 2) Build a list of columns to select safely
    columns_to_select = [time_delta_col, "code", "numeric_value"]

    # Only include modality_idx if it actually exists in the DF
    if "modality_idx" in all_columns:
        columns_to_select.append("modality_idx")

    # 3) Collect data into a dictionary of lists (or lists-of-lists)
    df_collected = df.select(columns_to_select).collect()
    tensors_dict = df_collected.to_dict(as_series=False)

    # 4) Quick sanity checks for empty columns
    if all(len(v) == 0 for v in tensors_dict.values()):
        logger.warning("All columns are empty. Returning an empty tensor dict.")
        return JointNestedRaggedTensorDict({})

    for k, v in tensors_dict.items():
        if len(v) == 0:
            raise ValueError(f"Column {k} is empty (0 rows).")

    # 5) Convert to a JointNestedRaggedTensorDict
    return JointNestedRaggedTensorDict(tensors_dict)


@hydra.main(
    version_base=None,
    config_path=str(PREPROCESS_CONFIG_YAML.parent),
    config_name=PREPROCESS_CONFIG_YAML.stem
)
def main(cfg: DictConfig):
    """
    Orchestrates the "tensorization" stage, reading the tokenized (parquet) data,
    converting each shard's data into nested ragged tensors, and writing them out.
    """
    # This calls map_over(...), which will:
    # - Iterate over shards from "event_seqs" subdirectory
    # - Pass each shard to compute_fn=convert_to_NRT
    # - Then call write_fn=JointNestedRaggedTensorDict.save on the result
    map_over(
        cfg,
        compute_fn=convert_to_NRT,
        write_fn=JointNestedRaggedTensorDict.save,
        shard_iterator_fntr=partial(shard_iterator, in_prefix="event_seqs/", out_suffix=".nrt"),
    )


if __name__ == "__main__":
    main()