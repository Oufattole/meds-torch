# src/meds_torch/schemas/generate_analysis_schema.py

from collections import OrderedDict

import polars as pl
import pyarrow as pa

from meds_torch.models import ACTUAL_FUTURE, GENERATE_PREFIX, INPUT_DATA

# Define struct fields in a fixed order using OrderedDict
TRAJECTORY_FIELDS = OrderedDict(
    [
        ("code", pa.list_(pa.int64())),
        ("mask", pa.list_(pa.bool_())),
        ("numeric_value", pa.list_(pa.float64())),
        ("numeric_value_mask", pa.list_(pa.bool_())),
        ("time_delta_days", pa.list_(pa.float64())),
    ]
)

# Create trajectory_data_type with fixed field order
trajectory_data_type = pa.struct([(name, dtype) for name, dtype in TRAJECTORY_FIELDS.items()])


def generation_analysis_schema(num_samples=0, do_include_actual=True):
    """
    Generate the schema for trajectory analysis data with consistent field ordering.

    Args:
        num_samples (int): Number of generated samples
        do_include_actual (bool): Whether to include actual future data

    Returns:
        pyarrow.Schema: The schema for trajectory analysis
    """
    # Use OrderedDict to maintain consistent field order
    schema_fields = OrderedDict(
        [
            ("subject_id", pa.int64()),
            ("prediction_time", pa.timestamp("us")),
            (INPUT_DATA, trajectory_data_type),
        ]
    )

    # Add generated sample fields in order
    for i in range(num_samples):
        schema_fields[f"{GENERATE_PREFIX}{i}"] = trajectory_data_type

    # Add actual future field last if requested
    if do_include_actual:
        schema_fields[ACTUAL_FUTURE] = trajectory_data_type

    return pa.schema([(name, dtype) for name, dtype in schema_fields.items()])


def validate_generated_data(df):
    """
    Validate generated data against the schema and ensure correct ordering.

    Args:
        df: polars.DataFrame to validate

    Returns:
        pyarrow.Table: Validated data cast to the correct schema with ordered columns
    """
    # Determine if ACTUAL_FUTURE is present in the dataframe
    has_actual_future = ACTUAL_FUTURE in df.columns

    # Count number of generated samples
    num_samples = sum(1 for col in df.columns if col.startswith(GENERATE_PREFIX))

    # Get the validated schema
    validated_schema = generation_analysis_schema(
        num_samples=num_samples, do_include_actual=has_actual_future
    )

    # Convert to arrow
    arrow_table = df.to_arrow()

    # Ensure columns are in the correct order
    expected_columns = validated_schema.names
    arrow_table = arrow_table.select(expected_columns)

    # Cast to the validated schema (this will handle struct field ordering)
    return arrow_table.cast(validated_schema)


def reorder_struct_fields(df: pl.DataFrame, column_name: str) -> pl.DataFrame:
    """
    Reorder fields within a polars struct column to match the expected order.

    Args:
        df: polars.DataFrame containing the struct column
        column_name: name of the struct column to reorder

    Returns:
        polars.DataFrame with reordered struct fields
    """
    if column_name not in df.columns:
        return df

    # Create expressions to extract and reorder struct fields
    field_exprs = [
        pl.col(column_name).struct.field(field_name).alias(field_name)
        for field_name in TRAJECTORY_FIELDS.keys()
    ]

    # Create a new struct with fields in the correct order
    reordered_struct = pl.struct(field_exprs).alias(column_name)

    # Replace the original struct column with the reordered one
    return df.with_columns(reordered_struct)


# Example usage:
"""
# At the end of generate_trajectory script:
validated_df = validate_generated_data(df)

# If you need to handle struct field ordering separately:
for col in validated_df.column_names:
    if validated_df[col].type.equals(trajectory_data_type):
        validated_df = reorder_struct_fields(validated_df, col)
"""
