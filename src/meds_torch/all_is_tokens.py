import doctest

import numpy as np
import polars as pl


def load_data(data_path: str) -> pl.DataFrame:
    """Load data from Parquet files located at a specified path.

    Args:
        data_path (str): The path pattern to load Parquet files from.

    Returns:
        pl.DataFrame: A Polars DataFrame with the 'timestamp' column parsed to datetime.

    Examples:
        >>> df = load_data("scripts/trash/tutorial_6/test.parquet")
        >>> isinstance(df, pl.DataFrame)
        True
        >>> df.schema()
        {'timestamp': pl.Datetime, ...}
    """
    df = pl.read_parquet(data_path)
    df = df.with_columns(
        pl.col("timestamp").str.strptime(pl.Datetime, format="%Y-%m-%d %H:%M:%S", strict=False)
    )
    return df


def calculate_quantiles(df: pl.DataFrame) -> pl.DataFrame:
    """Calculate quantiles for the 'numeric_value' of each code.

    Args:
        df: Input dataframe with at least 'code' and 'numeric_value' columns.

    Returns:
        The original DataFrame with an additional 'quantile' column indicating the quantile bin.

    Examples:
        >>> df = pl.DataFrame({
        ...     "code": ["A", "A", "A", "A", "B", "B", "B", "C", "C"],
        ...     "numeric_value": [10, 15, 20, 25, 5, 10, 15, 1, 2]
        ... })
        >>> df = calculate_quantiles(df)
        >>> df.to_dict(as_series=False)
        {'code': ['A', 'A', 'A', 'A', 'B', 'B', 'B', 'C', 'C'],
        'numeric_value': [10, 15, 20, 25, 5, 10, 15, 1, 2],
        'quantile': [1, 4, 7, 10, 1, 5, 10, 1, 10]}
    """
    df_quantiles = (
        df.filter(pl.col("numeric_value").is_not_null())
        .groupby("code")
        .agg([pl.col("numeric_value").quantile(np.arange(0.1, 1.1, 0.1)).alias("quantiles")])
        .explode("quantiles")
    )

    df = df.join(df_quantiles, on="code", how="left")

    def map_to_quantiles(row):
        value = row["numeric_value"]
        quantiles = row["quantiles"]
        if value is None or quantiles is None:
            return None
        for i, q in enumerate(quantiles):
            if value <= q:
                return i + 1  # Adjust indices to go from 1 to 10
        return len(quantiles) + 1

    df = df.with_columns(pl.struct(["numeric_value", "quantiles"]).apply(map_to_quantiles).alias("quantile"))
    df = df.drop("quantiles")

    return df


def create_code_tokens(df: pl.DataFrame) -> pl.DataFrame:
    """Create a column that combines 'code' with 'quantile' (formatted as 'Q1' to 'Q10') for rows where
    'numeric_value' is not null.

    Args:
        df: Input dataframe with 'code', 'numeric_value', and 'quantile' columns.

    Returns:
        Updated DataFrame with a 'code_token' column.

    Examples:
        >>> df = pl.DataFrame({
        ...     "code": ["A", "B", "C", "D", "E"],
        ...     "numeric_value": [100, None, 50, 75, None],
        ...     "quantile": [1, None, 5, 8, None]
        ... })
        >>> df = create_code_tokens(df)
        >>> df.to_dict(as_series=False)
        {'code': ['A', 'B', 'C', 'D', 'E'],
        'numeric_value': [100, None, 50, 75, None],
        'quantile': [1, None, 5, 8, None],
        'code_token': ['A_Q1', 'B', 'C_Q5', 'D_Q8', 'E']}
    """
    token_expr = (
        pl.when(pl.col("numeric_value").is_not_null())
        .then(pl.concat_str([pl.col("code"), pl.lit("Q"), pl.col("quantile").cast(pl.Utf8)], separator="_"))
        .otherwise(pl.col("code"))
    )

    return df.with_column(token_expr.alias("code_token"))


def preprocess_and_group_tokens(df: pl.DataFrame) -> pl.DataFrame:
    """Fill df with the earliest timestamp per patient when null, then group by patient and timestamp.

    Args:
        df: Input dataframe with 'patient_id', 'timestamp', and 'code_token' columns.

    Returns:
        A DataFrame grouped by 'patient_id' and 'timestamp', with a list of 'code_token' for each group.

    Examples:
        >>> df = pl.DataFrame({
        ...     "patient_id": [1, 1, 2, 2, 3],
        ...     "timestamp": [None, "2021-01-01 00:00:00", None, "2021-01-02 00:00:00", None],
        ...     "code_token": ["TOKEN_A", "TOKEN_B", "TOKEN_C", "TOKEN_D", "TOKEN_E"]
        ... }).with_column(pl.col("timestamp").str.strptime(pl.Datetime,fmt="%Y-%m-%d %H:%M:%S",strict=False))
        >>> df = preprocess_and_group_tokens(df)
        >>> df.to_dict(as_series=False)
        {'patient_id': [1, 2, 3],
        'timestamp': [datetime(2021, 1, 1, 0, 0), datetime(2021, 1, 2, 0, 0), datetime(2021, 1, 2, 0, 0)],
        'event_tokens': [['TOKEN_A', 'TOKEN_B'], ['TOKEN_C', 'TOKEN_D'], ['TOKEN_E']]}
    """
    if df.get_column("timestamp").dtype != pl.Datetime:
        df = df.with_column(pl.col("timestamp").str.strptime(pl.Datetime, "%Y-%m-%d %H:%M:%S", strict=False))

    earliest_timestamps = df.groupby("patient_id").agg(pl.col("timestamp").min().alias("earliest_timestamp"))

    df = df.join(earliest_timestamps, on="patient_id", how="left")

    df = df.with_columns(
        pl.when(pl.col("timestamp").is_null())
        .then(pl.col("earliest_timestamp"))
        .otherwise(pl.col("timestamp"))
        .alias("timestamp")
    )
    df = df.drop("earliest_timestamp")

    grouped_df = df.groupby(["patient_id", "timestamp"]).agg(
        pl.col("code_token").list().alias("event_tokens")
    )

    return grouped_df


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
    intervals = [
        (pl.duration(minutes=0), pl.duration(minutes=15), "5m-15m"),
        (pl.duration(minutes=15), pl.duration(hours=1), "15m-1h"),
        (pl.duration(hours=1), pl.duration(hours=2), "1h-2h"),
        (pl.duration(hours=2), pl.duration(hours=6), "2h-6h"),
        (pl.duration(hours=6), pl.duration(hours=12), "6h-12h"),
        (pl.duration(hours=12), pl.duration(days=1), "12h-1d"),
        (pl.duration(days=1), pl.duration(days=3), "1d-3d"),
        (pl.duration(days=3), pl.duration(weeks=1), "3d-1w"),
        (pl.duration(weeks=1), pl.duration(weeks=2), "1w-2w"),
        (pl.duration(weeks=2), pl.duration(months=1), "2w-1mt"),
        (pl.duration(months=1), pl.duration(months=3), "1mt-3mt"),
        (pl.duration(months=3), pl.duration(months=6), "3mt-6mt"),
        (pl.duration(months=6), None, "6mt+"),
    ]

    def map_time_interval(diff):
        if diff is None:
            return None
        for start, end, label in intervals:
            if (end is None and diff >= start) or (diff >= start and diff < end):
                return label
        return "6mt+"

    events_df = events_df.with_columns([pl.col("timestamp").diff().over("patient_id").alias("time_diff")])

    events_df = events_df.with_columns([pl.col("time_diff").apply(map_time_interval).alias("time_interval")])

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

    def create_timeline(tokens):
        timeline = []
        for i in range(len(tokens["event_tokens"])):
            if i > 0:
                timeline.append(tokens["time_interval"][i])
            timeline.extend(tokens["event_tokens"][i])
        return timeline

    timeline_df = events_df.groupby("patient_id").agg(
        [pl.struct(["event_tokens", "time_interval"]).apply(create_timeline).alias("timeline")]
    )

    return timeline_df


# Block to run doctests when the script is executed directly
if __name__ == "__main__":
    doctest.testmod()
