#!/usr/bin/env python
"""Transformations for adding time-derived measurements (e.g., a patient's age) to a MEDS dataset."""
from collections.abc import Callable

import hydra
import polars as pl
from loguru import logger
from MEDS_transforms import PREPROCESS_CONFIG_YAML
from MEDS_transforms.mapreduce.mapper import map_over
from omegaconf import DictConfig, OmegaConf

TIME_START_TOKEN = "TIME//START//TOKEN"
TIME_DELTA_TOKEN = "TIME//DELTA//TOKEN"


def add_new_events_fntr(fn: Callable[[pl.DataFrame], pl.DataFrame]) -> Callable[[pl.DataFrame], pl.DataFrame]:
    """Creates a "meta" functor that computes the input functor on a MEDS shard then combines both dataframes.

    Args:
        fn: The function that computes the new events.

    Returns:
        A function that computes the new events and combines them with the original DataFrame, returning a
        result in proper MEDS sorted order.

    Examples:
        >>> from datetime import datetime
        >>> df = pl.DataFrame(
        ...     {
        ...         "patient_id": [1, 1, 1, 1, 2, 2, 3, 3],
        ...         "time": [
        ...             None,
        ...             datetime(1990, 1, 1),
        ...             datetime(2021, 1, 1),
        ...             datetime(2021, 1, 1),
        ...             datetime(1988, 1, 2),
        ...             datetime(2023, 1, 3),
        ...             datetime(2022, 1, 1),
        ...             datetime(2022, 1, 1),
        ...         ],
        ...         "code": ["static", "DOB", "lab//A", "lab//B", "DOB", "lab//A", "lab//B", "dx//1"],
        ...     },
        ...     schema={"patient_id": pl.UInt32, "time": pl.Datetime, "code": pl.Utf8},
        ... )
        >>> df
        shape: (8, 3)
        ┌────────────┬─────────────────────┬────────┐
        │ patient_id ┆ time                ┆ code   │
        │ ---        ┆ ---                 ┆ ---    │
        │ u32        ┆ datetime[μs]        ┆ str    │
        ╞════════════╪═════════════════════╪════════╡
        │ 1          ┆ null                ┆ static │
        │ 1          ┆ 1990-01-01 00:00:00 ┆ DOB    │
        │ 1          ┆ 2021-01-01 00:00:00 ┆ lab//A │
        │ 1          ┆ 2021-01-01 00:00:00 ┆ lab//B │
        │ 2          ┆ 1988-01-02 00:00:00 ┆ DOB    │
        │ 2          ┆ 2023-01-03 00:00:00 ┆ lab//A │
        │ 3          ┆ 2022-01-01 00:00:00 ┆ lab//B │
        │ 3          ┆ 2022-01-01 00:00:00 ┆ dx//1  │
        └────────────┴─────────────────────┴────────┘
        >>> # As an example, we'll use the age functor defined elsewhere in this module.
        >>> age_cfg = DictConfig({"DOB_code": "DOB", "age_code": "AGE", "age_unit": "years"})
        >>> age_fn = age_fntr(age_cfg)
        >>> age_fn(df)
        shape: (2, 4)
        ┌────────────┬─────────────────────┬──────┬───────────────┐
        │ patient_id ┆ time                ┆ code ┆ numeric_value │
        │ ---        ┆ ---                 ┆ ---  ┆ ---           │
        │ u32        ┆ datetime[μs]        ┆ str  ┆ f32           │
        ╞════════════╪═════════════════════╪══════╪═══════════════╡
        │ 1          ┆ 2021-01-01 00:00:00 ┆ AGE  ┆ 31.001347     │
        │ 2          ┆ 2023-01-03 00:00:00 ┆ AGE  ┆ 35.004169     │
        └────────────┴─────────────────────┴──────┴───────────────┘
        >>> # Now, we'll use the add_new_events functor to add these age events to the original DataFrame.
        >>> add_age_fn = add_new_events_fntr(age_fn)
        >>> add_age_fn(df)
        shape: (10, 4)
        ┌────────────┬─────────────────────┬────────┬───────────────┐
        │ patient_id ┆ time                ┆ code   ┆ numeric_value │
        │ ---        ┆ ---                 ┆ ---    ┆ ---           │
        │ u32        ┆ datetime[μs]        ┆ str    ┆ f32           │
        ╞════════════╪═════════════════════╪════════╪═══════════════╡
        │ 1          ┆ null                ┆ static ┆ null          │
        │ 1          ┆ 1990-01-01 00:00:00 ┆ DOB    ┆ null          │
        │ 1          ┆ 2021-01-01 00:00:00 ┆ AGE    ┆ 31.001347     │
        │ 1          ┆ 2021-01-01 00:00:00 ┆ lab//A ┆ null          │
        │ 1          ┆ 2021-01-01 00:00:00 ┆ lab//B ┆ null          │
        │ 2          ┆ 1988-01-02 00:00:00 ┆ DOB    ┆ null          │
        │ 2          ┆ 2023-01-03 00:00:00 ┆ AGE    ┆ 35.004169     │
        │ 2          ┆ 2023-01-03 00:00:00 ┆ lab//A ┆ null          │
        │ 3          ┆ 2022-01-01 00:00:00 ┆ lab//B ┆ null          │
        │ 3          ┆ 2022-01-01 00:00:00 ┆ dx//1  ┆ null          │
        └────────────┴─────────────────────┴────────┴───────────────┘
    """

    def out_fn(df: pl.DataFrame) -> pl.DataFrame:
        new_events = fn(df)
        df = df.with_row_index("__idx")
        new_events = new_events.with_columns(pl.lit(0, dtype=df.schema["__idx"]).alias("__idx"))
        return (
            pl.concat([df, new_events], how="diagonal").sort(by=["patient_id", "time", "__idx"]).drop("__idx")
        )

    return out_fn


TIME_DURATION_UNITS = {
    "seconds": (["s", "sec", "secs", "second", "seconds"], 1),
    "minutes": (["m", "min", "mins", "minute", "minutes"], 60),
    "hours": (["h", "hr", "hrs", "hour", "hours"], 60 * 60),
    "days": (["d", "day", "days"], 60 * 60 * 24),
    "weeks": (["w", "wk", "wks", "week", "weeks"], 60 * 60 * 24 * 7),
    "months": (["mo", "mos", "month", "months"], 60 * 60 * 24 * 30.436875),
    "years": (["y", "yr", "yrs", "year", "years"], 60 * 60 * 24 * 365.2422),
}


def normalize_time_unit(unit: str) -> tuple[str, float]:
    """Normalize a time unit string to a canonical form and return the number of seconds in that unit.

    Note that this function is designed for computing _approximate_ time durations over long periods, not
    canonical, local calendar time durations. E.g., a "month" is not a fixed number of seconds, but this
    function will return the average number of seconds in a month, accounting for leap years.

    TODO: consider replacing this function with the use of https://github.com/wroberts/pytimeparse

    Args:
        unit: The input unit to normalize.

    Returns:
        A tuple containing the canonical unit and the number of seconds in that unit.

    Raises:
        ValueError: If the input unit is not recognized.

    Examples:
        >>> normalize_time_unit("s")
        ('seconds', 1)
        >>> normalize_time_unit("min")
        ('minutes', 60)
        >>> normalize_time_unit("hours")
        ('hours', 3600)
        >>> normalize_time_unit("day")
        ('days', 86400)
        >>> normalize_time_unit("wks")
        ('weeks', 604800)
        >>> normalize_time_unit("month")
        ('months', 2629746.0)
        >>> normalize_time_unit("years")
        ('years', 31556926.080000002)
        >>> normalize_time_unit("fortnight")
        Traceback (most recent call last):
            ...
        ValueError: Unknown time unit 'fortnight'. Valid units include:
          * seconds: s, sec, secs, second, seconds
          * minutes: m, min, mins, minute, minutes
          * hours: h, hr, hrs, hour, hours
          * days: d, day, days
          * weeks: w, wk, wks, week, weeks
          * months: mo, mos, month, months
          * years: y, yr, yrs, year, years
    """
    for canonical_unit, (aliases, seconds) in TIME_DURATION_UNITS.items():
        if unit in aliases:
            return canonical_unit, seconds

    valid_unit_lines = []
    for canonical, (aliases, _) in TIME_DURATION_UNITS.items():
        valid_unit_lines.append(f"  * {canonical}: {', '.join(aliases)}")
    valid_units_str = "\n".join(valid_unit_lines)
    raise ValueError(f"Unknown time unit '{unit}'. Valid units include:\n{valid_units_str}")


def age_fntr(cfg: DictConfig) -> Callable[[pl.DataFrame], pl.DataFrame]:
    """Create a function that adds a patient's age to a DataFrame.

    Args:
        cfg: The configuration for the age function. This must contain the following mandatory keys:
            - "DOB_code": The code for the date of birth event in the raw data.
            - "age_code": The code for the age event in the output data.
            - "age_unit": The unit for the age event when converted to a numeric value in the output data.

    Returns:
        A function that returns the to-be-added "age" events with the patient's age for all input events with
        unique, non-null times in the data, for all patients who have an observed date of birth. It does
        not add an event for times that are equal to the date of birth.

    Raises:
        ValueError: If the input unit is not recognized.

    Examples:
        >>> from datetime import datetime
        >>> df = pl.DataFrame(
        ...     {
        ...         "patient_id": [1, 1, 1, 1, 1, 2, 2, 3, 3],
        ...         "time": [
        ...             None,
        ...             datetime(1990, 1, 1),
        ...             datetime(2021, 1, 1),
        ...             datetime(2021, 1, 1),
        ...             datetime(2021, 1, 2),
        ...             datetime(1988, 1, 2),
        ...             datetime(2023, 1, 3),
        ...             datetime(2022, 1, 1),
        ...             datetime(2022, 1, 1),
        ...         ],
        ...         "code": ["static", "DOB", "lab//A", "lab//B", "rx", "DOB", "lab//A", "lab//B", "dx//1"],
        ...     },
        ...     schema={"patient_id": pl.UInt32, "time": pl.Datetime, "code": pl.Utf8},
        ... )
        >>> df
        shape: (9, 3)
        ┌────────────┬─────────────────────┬────────┐
        │ patient_id ┆ time                ┆ code   │
        │ ---        ┆ ---                 ┆ ---    │
        │ u32        ┆ datetime[μs]        ┆ str    │
        ╞════════════╪═════════════════════╪════════╡
        │ 1          ┆ null                ┆ static │
        │ 1          ┆ 1990-01-01 00:00:00 ┆ DOB    │
        │ 1          ┆ 2021-01-01 00:00:00 ┆ lab//A │
        │ 1          ┆ 2021-01-01 00:00:00 ┆ lab//B │
        │ 1          ┆ 2021-01-02 00:00:00 ┆ rx     │
        │ 2          ┆ 1988-01-02 00:00:00 ┆ DOB    │
        │ 2          ┆ 2023-01-03 00:00:00 ┆ lab//A │
        │ 3          ┆ 2022-01-01 00:00:00 ┆ lab//B │
        │ 3          ┆ 2022-01-01 00:00:00 ┆ dx//1  │
        └────────────┴─────────────────────┴────────┘
        >>> age_cfg = DictConfig({"DOB_code": "DOB", "age_code": "AGE", "age_unit": "years"})
        >>> age_fn = age_fntr(age_cfg)
        >>> age_fn(df)
        shape: (3, 4)
        ┌────────────┬─────────────────────┬──────┬───────────────┐
        │ patient_id ┆ time                ┆ code ┆ numeric_value │
        │ ---        ┆ ---                 ┆ ---  ┆ ---           │
        │ u32        ┆ datetime[μs]        ┆ str  ┆ f32           │
        ╞════════════╪═════════════════════╪══════╪═══════════════╡
        │ 1          ┆ 2021-01-01 00:00:00 ┆ AGE  ┆ 31.001347     │
        │ 1          ┆ 2021-01-02 00:00:00 ┆ AGE  ┆ 31.004084     │
        │ 2          ┆ 2023-01-03 00:00:00 ┆ AGE  ┆ 35.004169     │
        └────────────┴─────────────────────┴──────┴───────────────┘
        >>> age_cfg = DictConfig({"DOB_code": "DOB", "age_code": "AGE", "age_unit": "scores"})
        >>> import pytest
        >>> with pytest.raises(ValueError):
        ...     age_fntr(age_cfg)
    """

    canonical_unit, seconds_in_unit = normalize_time_unit(cfg.age_unit)
    microseconds_in_unit = int(1e6) * seconds_in_unit

    def fn(df: pl.LazyFrame) -> pl.LazyFrame:
        dob_expr = pl.when(pl.col("code") == cfg.DOB_code).then(pl.col("time")).min().over("patient_id")
        age_expr = (pl.col("time") - dob_expr).dt.total_microseconds() / microseconds_in_unit
        age_expr = age_expr.cast(pl.Float32, strict=False)

        return (
            df.drop_nulls(subset=["time"])
            .unique(subset=["patient_id", "time"], maintain_order=True)
            .select(
                "patient_id",
                "time",
                pl.lit(cfg.age_code, dtype=df.schema["code"]).alias("code"),
                age_expr.alias("numeric_value"),
            )
            .drop_nulls(subset=["numeric_value"])
            .filter(pl.col("numeric_value") > 0)
        )

    return fn


def time_delta_to_quantile_sequence(
    time_delta: float, quantile_list: pl.DataFrame, max_length: int = 3
) -> list:
    result = []
    remaining_time = time_delta

    while remaining_time > 0 and len(result) < max_length:
        # Find the largest quantile that's less than or equal to the remaining time
        chosen_quantile = next((q for v, q in quantile_list if v <= remaining_time), quantile_list[-1][1])

        result.append(chosen_quantile)
        remaining_time -= next(v for v, q in quantile_list if q == chosen_quantile)

    return result


def time_delta_fntr(cfg: DictConfig) -> Callable[[pl.DataFrame], pl.DataFrame]:
    """Create a function that adds time_delta event rows to a DataFrame.

    Args:
        cfg: The configuration for the age function. This must contain the following mandatory keys:
            - "quantile_fp": The path to a parquet file with pre-defined time delta quantiles.
                Should have a quantile column of type Int64 and a value column of type Float64
                indicating the numerical value for the quantile.
            - "max_length": The maximum number of time_delta qunatile tokens to use to approximate
                the continuous value.
            - "time_unit": The unit for the time deltas when dates are converted to a numeric value
                in the output data.

    Returns:
        A function that returns the to-be-added "time_delta" events with the quantile of the time delta
        for all input events with unique, non-null times in the data. The very first event for a patient
        has a null time_delta, so it is imputed with a special time start token.

    Raises:
        ValueError: If the input unit is not recognized.

    Examples:
        >>> from datetime import datetime
        >>> df = pl.DataFrame(
        ...     {
        ...         "patient_id": [1, 1, 1, 1, 1, 2, 2, 3, 3],
        ...         "time": [
        ...             None,
        ...             datetime(1990, 1, 1),
        ...             datetime(2021, 1, 1),
        ...             datetime(2021, 1, 1),
        ...             datetime(2021, 1, 2),
        ...             datetime(1988, 1, 2),
        ...             datetime(2023, 1, 3),
        ...             datetime(2022, 1, 1),
        ...             datetime(2022, 1, 1),
        ...         ],
        ...         "code": ["static", "DOB", "lab//A", "lab//B", "rx", "DOB", "lab//A", "lab//B", "dx//1"],
        ...     },
        ...     schema={"patient_id": pl.UInt32, "time": pl.Datetime, "code": pl.Utf8},
        ... )
        >>> df
        shape: (9, 3)
        ┌────────────┬─────────────────────┬────────┐
        │ patient_id ┆ time                ┆ code   │
        │ ---        ┆ ---                 ┆ ---    │
        │ u32        ┆ datetime[μs]        ┆ str    │
        ╞════════════╪═════════════════════╪════════╡
        │ 1          ┆ null                ┆ static │
        │ 1          ┆ 1990-01-01 00:00:00 ┆ DOB    │
        │ 1          ┆ 2021-01-01 00:00:00 ┆ lab//A │
        │ 1          ┆ 2021-01-01 00:00:00 ┆ lab//B │
        │ 1          ┆ 2021-01-02 00:00:00 ┆ rx     │
        │ 2          ┆ 1988-01-02 00:00:00 ┆ DOB    │
        │ 2          ┆ 2023-01-03 00:00:00 ┆ lab//A │
        │ 3          ┆ 2022-01-01 00:00:00 ┆ lab//B │
        │ 3          ┆ 2022-01-01 00:00:00 ┆ dx//1  │
        └────────────┴─────────────────────┴────────┘
        >>> import tempfile
        >>> with tempfile.TemporaryDirectory() as tmpdir:
        ...     fp = f"{tmpdir}/quantile.parquet"
        ...     pl.DataFrame({"quantile": [25, 50, 75], "value": [1,2,3]}).write_parquet(fp)
        ...     time_delta_cfg = DictConfig({"quantile_fp": fp, "time_unit": "days", "max_length": 3})
        ...     time_delta_fn = time_delta_fntr(time_delta_cfg)
        ...     time_delta_fn(df)
        shape: (6, 4)
        ┌────────────┬─────────────────────┬───────────────┬────────────────────┐
        │ patient_id ┆ time                ┆ numeric_value ┆ code               │
        │ ---        ┆ ---                 ┆ ---           ┆ ---                │
        │ u32        ┆ datetime[μs]        ┆ f64           ┆ str                │
        ╞════════════╪═════════════════════╪═══════════════╪════════════════════╡
        │ 1          ┆ 1990-01-01 00:00:00 ┆ null          ┆ TIME//START//TOKEN │
        │ 1          ┆ 2021-01-01 00:00:00 ┆ 11323.0       ┆ TIME//DELTA//TOKEN │
        │ 1          ┆ 2021-01-02 00:00:00 ┆ 1.0           ┆ TIME//DELTA//TOKEN │
        │ 2          ┆ 1988-01-02 00:00:00 ┆ null          ┆ TIME//START//TOKEN │
        │ 2          ┆ 2023-01-03 00:00:00 ┆ 12785.0       ┆ TIME//DELTA//TOKEN │
        │ 3          ┆ 2022-01-01 00:00:00 ┆ null          ┆ TIME//START//TOKEN │
        └────────────┴─────────────────────┴───────────────┴────────────────────┘
    """

    _, seconds_in_unit = normalize_time_unit(cfg.time_unit)
    microseconds_in_unit = int(1e6) * seconds_in_unit

    def fn(df: pl.LazyFrame) -> pl.LazyFrame:
        # Apply the function to create the quantile sequence
        df = df.select("patient_id", "time").unique().sort(["patient_id", "time"]).drop_nulls()
        df = df.with_columns(
            (pl.col("time").diff().over("patient_id").dt.total_microseconds() / microseconds_in_unit)
            .alias("numeric_value")
            .cast(pl.Float32)
        )
        df = df.with_columns(
            pl.when(pl.col("numeric_value").is_null())
            .then(pl.lit(TIME_START_TOKEN))
            .otherwise(pl.lit(TIME_DELTA_TOKEN))
            .alias("code")
        )
        return df

    return fn


def add_time_derived_measurements_fntr(stage_cfg: DictConfig) -> Callable[[pl.LazyFrame], pl.LazyFrame]:
    INFERRED_STAGE_KEYS = {
        "is_metadata",
        "data_input_dir",
        "metadata_input_dir",
        "output_dir",
        "reducer_output_dir",
    }

    compute_fns = []
    # We use the raw stages object as the induced `stage_cfg` has extra properties like the input and output
    # directories.
    for feature_name, feature_cfg in stage_cfg.items():
        match feature_name:
            case "time_delta":
                compute_fns.append(add_new_events_fntr(time_delta_fntr(feature_cfg)))
            case str() if feature_name in INFERRED_STAGE_KEYS:
                continue
            case _:
                raise ValueError(f"Unknown time-derived measurement: {feature_name}")

        logger.info(f"Adding {feature_name} via config: {OmegaConf.to_yaml(feature_cfg)}")

    def fn(df: pl.LazyFrame) -> pl.LazyFrame:
        for compute_fn in compute_fns:
            df = compute_fn(df)
        return df

    return fn


@hydra.main(
    version_base=None, config_path=str(PREPROCESS_CONFIG_YAML.parent), config_name=PREPROCESS_CONFIG_YAML.stem
)
def main(cfg: DictConfig):
    """Adds time-derived measurements to a MEDS cohort as separate observations at each unique time."""

    map_over(cfg, compute_fn=add_time_derived_measurements_fntr)


if __name__ == "__main__":
    main()
