import doctest
from datetime import timedelta

import numpy as np
import polars as pl


def load_data(data_path: str) -> pl.DataFrame:
    """Load data from Parquet files located at a specified path.

    Args:
        data_path: The path pattern to load Parquet files from.

    Returns:
        A Polars DataFrame in MEDS format with the 'timestamp' column parsed to datetime
        and filled to first event for missing timestamps.

    Examples:
        >>> import polars as pl
        >>> from polars.datatypes import *
        >>> df = load_data("../../tests/test_data/final_cohort/train/0.parquet")
        >>> isinstance(df, pl.DataFrame)
        True
        >>> df_schema = df.schema
        >>> expected_schema = pl.Schema([('patient_id', Int64), ('code', String),
        ...                 ('timestamp', Datetime(time_unit='us', time_zone=None)),
        ...                 ('numerical_value', Float64)])
        >>> assert df_schema == expected_schema, f"Schema mismatch: {df_schema} != {expected_schema}"
    """
    df = pl.read_parquet(data_path)
    # Find minimum timestamp for each patient where timestamp is not null
    min_timestamps = (
        df.filter(pl.col("timestamp").is_not_null())
        .group_by("patient_id")
        .agg(pl.min("timestamp").alias("min_timestamp"))
    )

    df = df.join(min_timestamps, on="patient_id", how="left")

    # Fill null timestamps with the minimum timestamp
    df = df.with_columns(
        pl.when(pl.col("timestamp").is_null())
        .then(pl.col("min_timestamp"))
        .otherwise(pl.col("timestamp"))
        .alias("timestamp")
    )
    df = df.drop("min_timestamp")
    return df


def calculate_quantiles(df: pl.DataFrame) -> pl.DataFrame:
    """Calculate quantiles for the 'numerical_value' of each code.

    Args:
        df: DataFrame with the MEDS format.

    Returns:
        A DataFrame with corresponding quantile boundaries for each numerical_value of a code.

    Raises:
        ValueError: If the input DataFrame is empty or if it has no numerical values.

    Examples:
        >>> import polars as pl
        >>> import numpy as np
        >>> data = {'code': ['A', 'A', 'B', 'B', 'C'],
        ...         'numerical_value': [1.0, 2.0, 2.0, 3.0, 5.0]}
        >>> df = pl.DataFrame(data)
        >>> result = calculate_quantiles(df)
        >>> print(result.sort('code'))
        shape: (3, 11)
        ┌──────┬─────┬─────┬─────┬───┬─────┬─────┬─────┬─────┐
        │ code ┆ 1   ┆ 2   ┆ 3   ┆ … ┆ 7   ┆ 8   ┆ 9   ┆ 10  │
        │ ---  ┆ --- ┆ --- ┆ --- ┆   ┆ --- ┆ --- ┆ --- ┆ --- │
        │ str  ┆ f64 ┆ f64 ┆ f64 ┆   ┆ f64 ┆ f64 ┆ f64 ┆ f64 │
        ╞══════╪═════╪═════╪═════╪═══╪═════╪═════╪═════╪═════╡
        │ A    ┆ 1.0 ┆ 1.0 ┆ 1.0 ┆ … ┆ 2.0 ┆ 2.0 ┆ 2.0 ┆ 2.0 │
        │ B    ┆ 2.0 ┆ 2.0 ┆ 2.0 ┆ … ┆ 3.0 ┆ 3.0 ┆ 3.0 ┆ 3.0 │
        │ C    ┆ 5.0 ┆ 5.0 ┆ 5.0 ┆ … ┆ 5.0 ┆ 5.0 ┆ 5.0 ┆ 5.0 │
        └──────┴─────┴─────┴─────┴───┴─────┴─────┴─────┴─────┘

        >>> df_empty = pl.DataFrame({'code': [], 'numerical_value': []})
        >>> calculate_quantiles(df_empty)
        Traceback (most recent call last):
        ...
        ValueError: Input DataFrame is empty
    """
    quantile_levels = np.arange(0.1, 1.1, 0.1)

    if df.is_empty():
        raise ValueError("Input DataFrame is empty")

    if df.filter(pl.col("numerical_value").is_not_null()).is_empty():
        raise ValueError("Input DataFrame has no numerical data")

    # Initialize a list to hold the DataFrame for each quantile
    quantile_dfs = []
    for q in quantile_levels:
        quantile_df = (
            df.filter(pl.col("numerical_value").is_not_null())
            .group_by("code")
            .agg(pl.col("numerical_value").quantile(q).alias(f"{int(q*10)}"))
        )
        quantile_dfs.append(quantile_df)

    df_quantiles = quantile_dfs[0]

    for q_df in quantile_dfs[1:]:
        df_quantiles = df_quantiles.join(q_df, on="code", how="left")

    return df_quantiles


def assign_quantiles(df: pl.DataFrame) -> pl.DataFrame:
    """Assign quantile bins to each numerical value based on computed quantiles.

    Args:
        df: DataFrame with the MEDS format.

    Returns:
        The original DataFrame with an additional 'quantile' column indicating the quantile bin.

    Raises:
        ValueError: If the input DataFrame is empty.

    Examples:
        >>> import polars as pl
        >>> data = {'patient_id': [1, 1, 2, 2],
        ...         'code': ['A', 'A', 'B', 'B'],
        ...         'timestamp': [1, 2, 1, 2],
        ...         'numerical_value': [1.0, 2.0, 2.0, 3.0]}
        >>> df = pl.DataFrame(data)
        >>> result = assign_quantiles(df)
        >>> print(result)
        shape: (4, 5)
        ┌────────────┬──────┬───────────┬─────────────────┬──────────┐
        │ patient_id ┆ code ┆ timestamp ┆ numerical_value ┆ quantile │
        │ ---        ┆ ---  ┆ ---       ┆ ---             ┆ ---      │
        │ i64        ┆ str  ┆ i64       ┆ f64             ┆ str      │
        ╞════════════╪══════╪═══════════╪═════════════════╪══════════╡
        │ 1          ┆ A    ┆ 1         ┆ 1.0             ┆ 1        │
        │ 1          ┆ A    ┆ 2         ┆ 2.0             ┆ 10       │
        │ 2          ┆ B    ┆ 1         ┆ 2.0             ┆ 1        │
        │ 2          ┆ B    ┆ 2         ┆ 3.0             ┆ 10       │
        └────────────┴──────┴───────────┴─────────────────┴──────────┘

        # Test with empty DataFrame
        >>> df_empty = pl.DataFrame({'patient_id': [], 'code': [], 'timestamp': [], 'numerical_value': []})
        >>> assign_quantiles(df_empty)
        Traceback (most recent call last):
        ...
        ValueError: Input DataFrame is empty
    """

    if df.is_empty():
        raise ValueError("Input DataFrame is empty")

    df_quantiles = calculate_quantiles(df)
    df_original = df.clone()
    df = df.with_columns(((pl.col("numerical_value") * 100).floor() / 100).alias("rounded_numerical_value"))

    # Expand df_quantiles
    expanded_quantiles = df_quantiles.unpivot(
        index="code",
        on=[f"{i}" for i in range(1, 11)],
        variable_name="quantile",
        value_name="boundary",
    )

    df = df.join(expanded_quantiles, on="code", how="left")

    df = df.filter(
        pl.col("numerical_value").is_not_null() & (pl.col("rounded_numerical_value") <= pl.col("boundary"))
    )

    df = df.group_by(["patient_id", "code", "timestamp", "numerical_value"]).agg(pl.min("quantile"))

    result_df = df_original.join(df, on=["patient_id", "code", "timestamp", "numerical_value"], how="left")

    return result_df


def create_event_tokens(df: pl.DataFrame) -> pl.DataFrame:
    """Create event tokens for 'code' and 'quantile' combinations.

    Tokenizes events (or observations) by aggregating codes and quantiles, which can be split
    as ('code', 'quantile') or joint as 'code_quantile'. Quantiles are formatted as 'Q1' to 'Q10'
    for rows where 'numerical_value' is not null.

    Args:
        df: DataFrame on MEDS format with the added 'quantile' column.

    Returns:
        A DataFrame grouped by 'patient_id' and 'timestamp', with two event token columns, each
        represented by a list of code tokens, one with joint quantiles and one with separate.


    Examples:
        >>> import polars as pl
        >>> data = {'patient_id': [1, 1, 2, 2],
        ...         'timestamp': [1, 2, 1, 1],
        ...         'code': ['A', 'C', 'B', 'B'],
        ...         'numerical_value': [10.0, None, 20.0, 30.0],
        ...         'quantile': ['1', None, '2', '3']}
        >>> df = pl.DataFrame(data)
        >>> result = create_event_tokens(df)
        >>> print(result.sort(['patient_id', 'timestamp']))
        shape: (3, 4)
        ┌────────────┬───────────┬────────────────────┬─────────────────────┐
        │ patient_id ┆ timestamp ┆ event_tokens_joint ┆ event_tokens_split  │
        │ ---        ┆ ---       ┆ ---                ┆ ---                 │
        │ i64        ┆ i64       ┆ list[str]          ┆ list[str]           │
        ╞════════════╪═══════════╪════════════════════╪═════════════════════╡
        │ 1          ┆ 1         ┆ ["A_Q1"]           ┆ ["A", "Q1"]         │
        │ 1          ┆ 2         ┆ ["C"]              ┆ ["C"]               │
        │ 2          ┆ 1         ┆ ["B_Q2", "B_Q3"]   ┆ ["B", "Q2", … "Q3"] │
        └────────────┴───────────┴────────────────────┴─────────────────────┘

        # Test with empty DataFrame
        >>> df_empty = pl.DataFrame({'patient_id': [],
        ...                         'timestamp': [],
        ...                         'code': [],
        ...                         'numerical_value': [],
        ...                         'quantile': []})
        >>> create_event_tokens(df_empty)
        Traceback (most recent call last):
        ...
        ValueError: Input DataFrame is empty
    """
    if df.is_empty():
        raise ValueError("Input DataFrame is empty")

    # Create the 'code_token' expression for joint tokens
    code_token_expr = (
        pl.when(pl.col("numerical_value").is_not_null())
        .then(pl.concat_str([pl.col("code"), pl.lit("_Q"), pl.col("quantile").cast(pl.Utf8)]))
        .otherwise(pl.col("code"))
    )  # If numerical_value is null, keep the original code

    # Create the 'code_token_split' expression to separate 'code' and 'quantile'
    code_token_split_expr = (
        pl.when(pl.col("numerical_value").is_not_null())
        .then(
            pl.concat_list([pl.col("code"), pl.concat_str([pl.lit("Q"), pl.col("quantile").cast(pl.Utf8)])])
        )
        .otherwise(pl.concat_list(pl.col("code")))
    )

    df = df.with_columns(
        [code_token_expr.alias("code_token"), code_token_split_expr.alias("code_token_split")]
    )

    # Group by 'patient_id' and 'timestamp', and aggregate both token expressions into lists
    df = df.group_by(["patient_id", "timestamp"]).agg(
        [
            pl.col("code_token").alias("event_tokens_joint"),
            pl.col("code_token_split").flatten().alias("event_tokens_split"),
        ]
    )

    return df


def calculate_time_intervals(events_df: pl.DataFrame) -> pl.DataFrame:
    """Map time intervals between consecutive events for each patient.

    Args:
        events_df: DataFrame with 'patient_id', 'timestamp', and two event tokens' columns.

    Returns:
        A DataFrame with an additional 'time_interval' column indicating the time interval category between
        one event and its consecutive event.

    Examples:
        >>> import polars as pl
        >>> from datetime import datetime, timedelta
        >>> data = {
        ...     "patient_id": [1, 1, 1, 2],
        ...     "timestamp": [
        ...         datetime(2022, 1, 1, 0, 0),
        ...         datetime(2022, 1, 1, 0, 10),
        ...         datetime(2022, 1, 1, 1, 0),
        ...         datetime(2022, 1, 2)
        ...     ]
        ... }
        >>> df = pl.DataFrame(data)
        >>> result = calculate_time_intervals(df)
        >>> print(result)
        shape: (4, 3)
        ┌────────────┬─────────────────────┬───────────────┐
        │ patient_id ┆ timestamp           ┆ time_interval │
        │ ---        ┆ ---                 ┆ ---           │
        │ i64        ┆ datetime[μs]        ┆ list[str]     │
        ╞════════════╪═════════════════════╪═══════════════╡
        │ 1          ┆ 2022-01-01 00:00:00 ┆ null          │
        │ 1          ┆ 2022-01-01 00:10:00 ┆ ["5m-15m"]    │
        │ 1          ┆ 2022-01-01 01:00:00 ┆ ["15m-1h"]    │
        │ 2          ┆ 2022-01-02 00:00:00 ┆ null          │
        └────────────┴─────────────────────┴───────────────┘
        >>> long_interval_data = {
        ...     "patient_id": [1, 1, 1],
        ...     "timestamp": [
        ...         datetime(2022, 1, 1),
        ...         datetime(2022, 8, 1),
        ...         datetime(2024, 1, 1)
        ...     ]
        ... }
        >>> long_interval_df = pl.DataFrame(long_interval_data)
        >>> long_interval_result = calculate_time_intervals(long_interval_df)
        >>> print(long_interval_result)
        shape: (3, 3)
        ┌────────────┬─────────────────────┬─────────────────────────────┐
        │ patient_id ┆ timestamp           ┆ time_interval               │
        │ ---        ┆ ---                 ┆ ---                         │
        │ i64        ┆ datetime[μs]        ┆ list[str]                   │
        ╞════════════╪═════════════════════╪═════════════════════════════╡
        │ 1          ┆ 2022-01-01 00:00:00 ┆ null                        │
        │ 1          ┆ 2022-08-01 00:00:00 ┆ ["3m-6m", "3m-6m"]          │
        │ 1          ┆ 2024-01-01 00:00:00 ┆ ["3m-6m", "3m-6m", "3m-6m"] │
        └────────────┴─────────────────────┴─────────────────────────────┘

        # Test with an empty DataFrame
        >>> df_empty = pl.DataFrame({"patient_id": [], "timestamp": []})
        >>> calculate_time_intervals(df_empty)
        Traceback (most recent call last):
        ...
        ValueError: Input DataFrame is empty
    """
    if events_df.is_empty():
        raise ValueError("Input DataFrame is empty")

    events_df = events_df.sort(["patient_id", "timestamp"])

    # Ensure timestamp is in the right format and calculate differences
    events_df = events_df.with_columns(
        [pl.col("timestamp").diff().over("patient_id").cast(pl.Duration).alias("time_diff")]
    )

    error_message = ["Error: Invalid time difference interval"]

    # Time tokens
    def time_interval_tokens(time_diff):
        if time_diff is None or time_diff < timedelta(minutes=5):
            return None
        elif time_diff < timedelta(minutes=15):
            return ["5m-15m"]
        elif time_diff < timedelta(hours=1):
            return ["15m-1h"]
        elif time_diff < timedelta(hours=2):
            return ["1h-2h"]
        elif time_diff < timedelta(hours=6):
            return ["2h-6h"]
        elif time_diff < timedelta(hours=12):
            return ["6h-12h"]
        elif time_diff < timedelta(days=1):
            return ["12h-1d"]
        elif time_diff < timedelta(days=3):
            return ["1d-3d"]
        elif time_diff < timedelta(weeks=1):
            return ["3d-1w"]
        elif time_diff < timedelta(weeks=2):
            return ["1w-2w"]
        elif time_diff < timedelta(days=30):
            return ["2w-1m"]
        elif time_diff < timedelta(days=90):
            return ["1m-3m"]
        elif time_diff < timedelta(days=180):
            return ["3m-6m"]
        elif time_diff >= timedelta(days=180):
            num_tokens = (time_diff // timedelta(days=180)) + 1
            return ["3m-6m" for i in range(num_tokens)]
        else:
            return error_message

    events_df = events_df.with_columns(
        pl.when(pl.col("time_diff").is_not_null())
        .then(pl.col("time_diff").map_elements(time_interval_tokens, return_dtype=pl.List(pl.Utf8)))
        .otherwise(None)
        .alias("time_interval")
    )

    if events_df.filter(pl.col("time_interval") == pl.lit(error_message)).height > 0:
        raise ValueError("Invalid time difference interval found in the DataFrame.")

    events_df = events_df.drop(["time_diff"])

    return events_df


def generate_patient_timeline(events_df: pl.DataFrame) -> pl.DataFrame:
    """Generate patient timeline with tokens for the event codes and time intervals.

    Args:
        events_df: DataFrame with 'patient_id', 'timestamp', 'event_tokens', and 'time_interval' columns.

    Returns:
        DataFrame with two timeline columns for each patient containing the sequence of event
        and time interval tokens, with both code + quantile representations.

    Examples:
        >>> import polars as pl
        >>> from datetime import datetime
        >>> data = {
        ...     "patient_id": [1, 1, 1, 2],
        ...     "timestamp": [datetime(2022, 1, 1), datetime(2022, 2, 1),
        ...                   datetime(2022, 3, 1), datetime(2022, 4, 1)],
        ...     "event_tokens_split": [["A", "Q1"], ["B", "Q2"], ["C", "Q3"], ["D", "Q4"]],
        ...     "event_tokens_joint": ["A_Q1", "B_Q2", "C_Q3", "D_Q4"],
        ...     "time_interval": [None, "1m", "1m", None]
        ... }
        >>> df = pl.DataFrame(data)
        >>> result = generate_patient_timeline(df)
        >>> print(result.sort(["patient_id"]))
        shape: (2, 4)
        ┌────────────┬─────────────────────┬─────────────────────────┬──────────────────────────┐
        │ patient_id ┆ timestamp           ┆ split_quantile_timeline ┆ joint_quantile_timeline  │
        │ ---        ┆ ---                 ┆ ---                     ┆ ---                      │
        │ i64        ┆ datetime[μs]        ┆ list[str]               ┆ list[str]                │
        ╞════════════╪═════════════════════╪═════════════════════════╪══════════════════════════╡
        │ 1          ┆ 2022-01-01 00:00:00 ┆ ["A", "Q1", … "Q3"]     ┆ ["A_Q1", "1m", … "C_Q3"] │
        │ 2          ┆ 2022-04-01 00:00:00 ┆ ["D", "Q4"]             ┆ ["D_Q4"]                 │
        └────────────┴─────────────────────┴─────────────────────────┴──────────────────────────┘

        # Test with an empty DataFrame
        >>> df_empty = pl.DataFrame({"patient_id": [],
        ...                         "timestamp": [],
        ...                         "event_tokens_split": [],
        ...                         "event_tokens_joint": [],
        ...                         "time_interval": []})
        >>> generate_patient_timeline(df_empty)
        Traceback (most recent call last):
        ...
        ValueError: Input DataFrame is empty
    """
    if events_df.is_empty():
        raise ValueError("Input DataFrame is empty")

    # Shift time intervals to align with the subsequent event tokens
    events_df = events_df.with_columns(
        pl.col("time_interval").shift(-1).over("patient_id").alias("next_time_interval")
    )

    timeline_split_expr = (
        pl.when(pl.col("next_time_interval").is_not_null())
        .then(pl.concat_list([pl.col("event_tokens_split"), pl.col("next_time_interval").cast(pl.Utf8)]))
        .otherwise(pl.concat_list(pl.col("event_tokens_split")))
    )

    timeline_joint_expr = (
        pl.when(pl.col("next_time_interval").is_not_null())
        .then(pl.concat_list([pl.col("event_tokens_joint"), pl.col("next_time_interval").cast(pl.Utf8)]))
        .otherwise(pl.concat_list(pl.col("event_tokens_joint")))
    )

    events_df = events_df.with_columns(
        [
            timeline_split_expr.alias("split_quantile_timeline"),
            timeline_joint_expr.alias("joint_quantile_timeline"),
        ]
    )

    # Group by patient_id and aggregate sequences
    timeline_df = events_df.group_by("patient_id").agg(
        [
            pl.col("timestamp").first(),
            pl.col("split_quantile_timeline").flatten(),
            pl.col("joint_quantile_timeline").flatten(),
        ]
    )

    return timeline_df


# Block to run doctests when the script is executed directly
if __name__ == "__main__":
    doctest.testmod()
