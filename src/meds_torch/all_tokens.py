import doctest

import numpy as np
import polars as pl


def load_data(data_path: str) -> pl.DataFrame:
    """Load data from Parquet files located at a specified path.

    Args:
        data_path: The path pattern to load Parquet files from.

    Returns:
        A Polars DataFrame with the 'timestamp' column parsed to datetime and filled
        to first event for missing timestamps.

    Examples:
        >>> df = load_data("../../tests/test_data/final_cohort/train/0.parquet")
        >>> isinstance(df, pl.DataFrame)
        True
        >>> df.schema
        Schema([('patient_id', Int64), ('code', String),
        ...     ('timestamp', Datetime(time_unit='us', time_zone=None)),
        ...     ('numerical_value', Float64)])
    """
    df = pl.read_parquet(data_path)
    # Calculate the minimum timestamp for each patient where timestamp is not null
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
        df: Input dataframe with the MEDS schema.

    Returns:
        A DataFrame with 'code' and corresponding quantile boundaries for each numerical_value.

    Examples:
        >>> df = pl.DataFrame({
        ...     "code": ["A", "A", "B", "B", "B"],
        ...     "numerical_value": [10, 20, 15, 25, 5]
        ... })
        >>> df_quantiles = calculate_quantiles(df)
        >>> df_quantiles.to_dict(as_series=False)
        {'code': ['A', 'B'], '1': [10.0, 5.0], '2': [10.0, 5.0], '3': [10.0, 15.0], '4': [10.0, 15.0],
        ...                  '5': [20.0, 15.0], '6': [20.0, 15.0], '7': [20.0, 15.0], '8': [20.0, 25.0],
        ...                  '9': [20.0, 25.0], '10': [20.0, 25.0]}
    """
    quantile_levels = np.arange(0.1, 1.1, 0.1)

    # Calculate each quantile separately and store in a list
    quantile_dfs = []
    for q in quantile_levels:
        quantile_df = (
            df.filter(pl.col("numerical_value").is_not_null())
            .group_by("code")
            .agg([pl.col("numerical_value").quantile(q).alias(f"{int(q*10)}")])
        )
        quantile_dfs.append(quantile_df)

    # Merge all quantile DataFrames on 'code'
    df_quantiles = quantile_dfs[0]
    for q_df in quantile_dfs[1:]:
        df_quantiles = df_quantiles.join(q_df, on="code", how="left")

    return df_quantiles


def assign_quantiles(df: pl.DataFrame) -> pl.DataFrame:
    """Assign quantile bins to each numerical value based on computed quantiles.

    Args:
        df: Input dataframe with the MEDS schema.

    Returns:
        The original df with an additional 'quantile' column indicating the quantile bin.

    Examples:
        >>> df = pl.DataFrame({
        ...     "patient_id": [1, 1, 1, 1, 1],
        ...     "timestamps": [None, None, None, None, None],
        ...     "code": ["A", "A", "B", "B", "B"],
        ...     "numerical_value": [10, 20, 15, 25, 5]
        ... })
        >>> df_quantiles = calculate_quantiles(df)
        >>> df = assign_quantiles(df, df_quantiles)
        >>> df.to_dict(as_series=False)
    """
    df_quantiles = calculate_quantiles(df)
    df_original = df.clone()
    df = df.with_columns(((pl.col("numerical_value") * 100).floor() / 100).alias("rounded_numerical_value"))

    # Expand df_quantiles
    expanded_quantiles = df_quantiles.melt(
        id_vars="code",
        value_vars=[f"{i}" for i in range(1, 11)],
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
    """Create a column that combines 'code' with 'quantile' (formatted as 'Q1' to 'Q10') for rows where
    'numerical_value' is not null, then group by patient and timestamp.

    Args:
        df: Input dataframe with 'code', 'numerical_value', and 'quantile' columns.

    Returns:
        A df grouped by 'patient_id' and 'timestamp', with an 'event_tokens' column represented
        by a list of 'code_token' for each group

    Examples:
        >>> df = pl.DataFrame({
        ...     "patient_id": [1, 1, 2, 2, 3],
        ...     "timestamp": [None, "2021-01-01 00:00:00", None, "2021-01-02 00:00:00", None],
        ...     "code_token": ["TOKEN_A", "TOKEN_B", "TOKEN_C", "TOKEN_D", "TOKEN_E"]
        ... }).with_column(pl.col("timestamp").str.strptime(pl.Datetime,fmt="%Y-%m-%d %H:%M:%S",strict=False))
        >>> df = preprocess_and_group_tokens(df)
        >>> df.to_dict(as_series=False)
    """
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
        events_df: Grouped DataFrame with 'patient_id', 'timestamp', and 'event_tokens' columns.

    Returns:
        DataFrame with an additional 'time_interval' column indicating the time interval category between
        one event and its consecutive event.

    Examples:
        >>> df = pl.DataFrame({
        ...     "patient_id": [1, 1, 1, 2, 2],
        ...     "timestamp": ["2021-01-01 12:00:00", "2021-01-01 12:05:00", "2021-01-01 13:00:00",
        ...                   "2021-01-02 14:00:00", "2021-01-02 14:30:00"],
        ...     "event_tokens": [["TOKEN_A"], ["TOKEN_B"], ["TOKEN_C"], ["TOKEN_D"], ["TOKEN_E"]]
        ... }).with_column(pl.col("timestamp").str.strptime(pl.Datetime))
        >>> df = calculate_time_intervals(df)
        >>> df.to_dict(as_series=False)
        {'patient_id': [1, 1, 1, 2, 2],
        'timestamp': [datetime(2021, 1, 1, 12, 0), datetime(2021, 1, 1, 12, 5), datetime(2021, 1, 1, 13, 0),
        datetime(2021, 1, 2, 14, 0), datetime(2021, 1, 2, 14, 30)],
        'event_tokens': [['TOKEN_A'], ['TOKEN_B'], ['TOKEN_C'], ['TOKEN_D'], ['TOKEN_E']],
        'time_interval': [None, '5m-15m', '15m-1h', None, '15m-1h']}
    """
    events_df = events_df.sort(["patient_id", "timestamp"])
    # Ensure timestamp is in the right format and calculate differences
    events_df = events_df.with_columns(
        [pl.col("timestamp").diff().over("patient_id").cast(pl.Duration).alias("time_diff")]
    )

    # Time tokens
    time_interval_expr = (
        pl.when(pl.col("time_diff").is_null())
        .then(pl.lit("No Previous Event"))
        .when(pl.col("time_diff") < pl.duration(minutes=15))
        .then(pl.lit("5m-15m"))
        .when((pl.col("time_diff") >= pl.duration(minutes=15)) & (pl.col("time_diff") < pl.duration(hours=1)))
        .then(pl.lit("15m-1h"))
        .when((pl.col("time_diff") >= pl.duration(hours=1)) & (pl.col("time_diff") < pl.duration(hours=2)))
        .then(pl.lit("1h-2h"))
        .when((pl.col("time_diff") >= pl.duration(hours=2)) & (pl.col("time_diff") < pl.duration(hours=6)))
        .then(pl.lit("2h-6h"))
        .when((pl.col("time_diff") >= pl.duration(hours=6)) & (pl.col("time_diff") < pl.duration(hours=12)))
        .then(pl.lit("6h-12h"))
        .when((pl.col("time_diff") >= pl.duration(hours=12)) & (pl.col("time_diff") < pl.duration(days=1)))
        .then(pl.lit("12h-1d"))
        .when((pl.col("time_diff") >= pl.duration(days=1)) & (pl.col("time_diff") < pl.duration(days=3)))
        .then(pl.lit("1d-3d"))
        .when((pl.col("time_diff") >= pl.duration(days=3)) & (pl.col("time_diff") < pl.duration(weeks=1)))
        .then(pl.lit("3d-1w"))
        .when((pl.col("time_diff") >= pl.duration(weeks=1)) & (pl.col("time_diff") < pl.duration(weeks=2)))
        .then(pl.lit("1w-2w"))
        .when((pl.col("time_diff") >= pl.duration(weeks=2)) & (pl.col("time_diff") < pl.duration(days=30)))
        .then(pl.lit("2w-1m"))
        .when((pl.col("time_diff") >= pl.duration(days=30)) & (pl.col("time_diff") < pl.duration(days=90)))
        .then(pl.lit("1m-3m"))
        .when((pl.col("time_diff") >= pl.duration(days=90)) & (pl.col("time_diff") < pl.duration(days=180)))
        .then(pl.lit("3m-6m"))
        .when(pl.col("time_diff") >= pl.duration(days=180))
        .then(pl.lit("6m+"))
        .otherwise(pl.lit("Unknown"))
    )

    # Apply the expression to the DataFrame
    events_df = events_df.with_columns(time_interval_expr.alias("time_interval"))

    return events_df


def generate_patient_timeline(events_df: pl.DataFrame) -> pl.DataFrame:
    """Generate patient timeline with tokens for the event codes and time intervals.

    Args:
        events_df: DataFrame with 'patient_id', 'timestamp', 'event_tokens', and 'time_interval' columns.

    Returns:
        DataFrame with a single 'timeline' column for each patient containing the sequence of event
        and time interval tokens.

    Examples:
        >>> df = pl.DataFrame({
        ...     "patient_id": [1, 1, 1, 2, 2],
        ...     "timestamp": ["2021-01-01 12:00:00", "2021-01-01 12:15:00", "2021-01-01 13:00:00",
        ...                   "2021-01-02 14:00:00", "2021-01-02 14:30:00"],
        ...     "event_tokens": [["TOKEN_A"], ["TOKEN_B"], ["TOKEN_C"], ["TOKEN_D"], ["TOKEN_E"]],
        ...     "time_interval": [None, "5m-15m", "15m-1h", None, "15m-1h"]
        ... }).with_column(pl.col("timestamp").str.strptime(pl.Datetime))
        >>> df = generate_patient_timeline(df)
        >>> df.to_dict(as_series=False)
        {'patient_id': [1, 2],
        'timeline': [['TOKEN_A', '5m-15m', 'TOKEN_B', '15m-1h', 'TOKEN_C'], ['TOKEN_D', '15m-1h', 'TOKEN_E']]}
    """

    # Shift the time intervals down to align with the subsequent event tokens
    shifted_time_intervals = events_df.with_columns(
        pl.col("time_interval").shift(-1).over("patient_id").alias("next_time_interval")
    )

    # Create an expression to concatenate event tokens with subsequent time intervals
    concat_expr = pl.concat_list(
        [
            pl.col("event_tokens_joint"),
            pl.when(pl.col("next_time_interval").is_not_null())
            .then(pl.concat_list([pl.col("next_time_interval")]))
            .otherwise(pl.lit([])),
        ]
    )

    # Group by patient_id and aggregate
    timeline_df = (
        shifted_time_intervals.group_by("patient_id")
        .agg(
            pl.col("event_tokens_joint").first().alias("start_token"),
            pl.col("next_time_interval").first().alias("start_interval"),
            pl.concat_list(concat_expr).alias("event_sequence"),
        )
        .with_columns(
            pl.concat_list(
                [
                    pl.col("start_token"),
                    pl.when(pl.col("start_interval").is_not_null())
                    .then(pl.concat_list([pl.col("start_interval")]))
                    .otherwise(pl.lit([])),
                    pl.col("event_sequence").flatten(),
                ]
            ).alias("timeline")
        )
        .select(["patient_id", "timeline"])
    )

    return timeline_df


# Block to run doctests when the script is executed directly
if __name__ == "__main__":
    doctest.testmod()
