"""
This file defines the output schema for generating trajectories using zero-shot models.

Example Usage:
# At the end of generate_trajectory script:
validated_df = validate_generated_data(df)
"""
from collections import OrderedDict

import pyarrow as pa

# Define struct fields in a fixed order using OrderedDict
MEDS_TRAJECTORY_SCHEMA = OrderedDict(
    [
        ("subject_id", pa.int64()),
        ("prediction_time", pa.timestamp("ns")),
        ("time", pa.timestamp("ns")),
        ("code", pa.string()),
        ("code/vocab_index", pa.int64()),
        ("numeric_value", pa.float64()),
        ("TRAJECTORY_TYPE", pa.string()),
    ]
)


def generation_analysis_schema():
    """
    Generate the schema for trajectory analysis data with consistent field ordering.

    Returns:
        pyarrow.Schema: The schema for trajectory analysis
    """
    return pa.schema([(name, dtype) for name, dtype in MEDS_TRAJECTORY_SCHEMA.items()])


def validate_generated_data(df):
    """
    Validate generated data against the schema and ensure correct ordering.

    Args:
        df: polars.DataFrame to validate

    Returns:
        pyarrow.Table: Validated data cast to the correct schema with ordered columns

    Example:
    >>> import polars as pl
    >>> from datetime import datetime
    >>> df = pl.DataFrame({
    ...     'time': [datetime(2024, 7, 18, 16, 21, 41)] * 3,
    ...     'code': ['A1', 'A2', 'A3'],
    ...     'code/vocab_index': [55, 59, 61],
    ...     'numeric_value': [0.625, 0.625, float('nan')],
    ...     'subject_id': [109767, 109767, 109767],
    ...     'prediction_time': [datetime(2024, 7, 18, 16, 21, 41)] * 3,
    ...     'TRAJECTORY_TYPE': ['INPUT_DATA'] * 3
    ... })
    >>> validate_generated_data(df)
    pyarrow.Table
    subject_id: int64
    prediction_time: timestamp[ns]
    time: timestamp[ns]
    code: string
    code/vocab_index: int64
    numeric_value: double
    TRAJECTORY_TYPE: string
    ----
    subject_id: [[109767,109767,109767]]
    prediction_time: [[2024-07-18 16:21:41.000000000,2024-07-18 16:21:41.000000000,2024-07-18 16:21:41.00...]]
    time: [[2024-07-18 16:21:41.000000000,2024-07-18 16:21:41.000000000,2024-07-18 16:21:41.000000000]]
    code: [["A1","A2","A3"]]
    code/vocab_index: [[55,59,61]]
    numeric_value: [[0.625,0.625,nan]]
    TRAJECTORY_TYPE: [["INPUT_DATA","INPUT_DATA","INPUT_DATA"]]
    """
    # Get the validated schema
    validated_schema = generation_analysis_schema()

    # Convert to arrow
    arrow_table = df.to_arrow()

    # Ensure columns are in the correct order
    expected_columns = validated_schema.names
    arrow_table = arrow_table.select(expected_columns)

    # Cast to the validated schema (this will handle struct field ordering)
    return arrow_table.cast(validated_schema)
