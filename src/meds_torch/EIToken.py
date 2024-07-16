import doctest

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
        >>> df = load_data("../../tests/test_data/final_cohort/train/0.parquet")
        >>> isinstance(df, pl.DataFrame)
        True
        >>> df.schema
        Schema([('patient_id', Int64), ('code', String),
        ...     ('timestamp', Datetime(time_unit='us', time_zone=None)),
        ...     ('numerical_value', Float64)])
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
        df: DataFrame with the MEDS format.

    Returns:
        The original DataFrame with an additional 'quantile' column indicating the quantile bin.
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
    """Create event tokens for 'code' and 'quantile' combinations.

    Tokenizes events (or observations) by aggregating codes and quantiles, which can be split
    as ('code', 'quantile') or joint as 'code_quantile'. Quantiles are formatted as 'Q1' to 'Q10'
    for rows where 'numerical_value' is not null.

    Args:
        df: DataFrame on MEDS format with the added 'quantile' column.

    Returns:
        A DataFrame grouped by 'patient_id' and 'timestamp', with two event token columns, each
        represented by a list of code tokens, one with joint quantiles and one with separate.
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
        events_df: DataFrame with 'patient_id', 'timestamp', and two event tokens' columns.

    Returns:
        A DataFrame with an additional 'time_interval' column indicating the time interval category between
        one event and its consecutive event.
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

    events_df = events_df.with_columns(time_interval_expr.alias("time_interval"))

    return events_df


def generate_patient_timeline(events_df: pl.DataFrame) -> pl.DataFrame:
    """Generate patient timeline with tokens for the event codes and time intervals.

    Args:
        events_df: DataFrame with 'patient_id', 'timestamp', 'event_tokens', and 'time_interval' columns.

    Returns:
        DataFrame with two timeline columns for each patient containing the sequence of event
        and time interval tokens, with both code + quantile representations.
    """
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
