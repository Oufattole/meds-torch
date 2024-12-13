# src/meds_torch/schemas/prediction_schema.py

from collections import OrderedDict

import polars as pl
import pyarrow as pa

# Constants for field grouping and ordering
SCHEMA_FIELD_ORDER = OrderedDict(
    [
        # Required fields
        ("subject_id", pa.int64()),
        ("prediction_time", pa.timestamp("ns")),
        # Prediction fields
        ("boolean_value", pa.bool_()),
        ("predicted_boolean_value", pa.bool_()),
        ("predicted_boolean_probability", pa.float64()),
        # Optional fields
        ("embeddings", pa.list_(pa.float64())),
        ("logits", pa.list_(pa.float64())),
        ("logits_sequence", pa.list_(pa.list_(pa.float64()))),
        ("loss", pa.float64()),
    ]
)

REQUIRED_FIELDS = {"subject_id", "prediction_time"}
PREDICTION_FIELDS = {"boolean_value", "predicted_boolean_value", "predicted_boolean_probability"}
OPTIONAL_FIELDS = {"embeddings", "logits", "logits_sequence", "loss"}


def prediction_analysis_schema(
    include_prediction: bool = True,
    include_embeddings: bool = False,
    include_logits: bool = False,
    include_sequence_logits: bool = False,
    include_loss: bool = False,
) -> pa.Schema:
    """
    Generate the schema for prediction analysis data with optional components.
    Maintains consistent field ordering.
    """
    schema_fields = []

    # Add fields in the defined order
    for field_name, field_type in SCHEMA_FIELD_ORDER.items():
        if field_name in REQUIRED_FIELDS:
            schema_fields.append((field_name, field_type))
        elif field_name in PREDICTION_FIELDS and include_prediction:
            schema_fields.append((field_name, field_type))
        elif field_name == "embeddings" and include_embeddings:
            schema_fields.append((field_name, field_type))
        elif field_name == "logits" and include_logits:
            schema_fields.append((field_name, field_type))
        elif field_name == "logits_sequence" and include_sequence_logits:
            schema_fields.append((field_name, field_type))
        elif field_name == "loss" and include_loss:
            schema_fields.append((field_name, field_type))

    return pa.schema(schema_fields)


def validate_prediction_schema(df_schema: pa.Schema) -> bool:
    """
    Validates that the provided schema matches the required meds-evaluation schema.
    """
    df_fields: set[str] = set(df_schema.names)

    # Check required fields
    missing_required = REQUIRED_FIELDS - df_fields
    if missing_required:
        raise ValueError(f"Missing required fields: {missing_required}")

    # Check prediction fields (all or nothing)
    has_prediction = bool(PREDICTION_FIELDS & df_fields)
    if has_prediction and not PREDICTION_FIELDS.issubset(df_fields):
        raise ValueError(f"If any prediction field is present, all must be present: {PREDICTION_FIELDS}")

    # Validate field types and order
    for field_name in df_fields:
        expected_type = SCHEMA_FIELD_ORDER.get(field_name)
        if expected_type is None:
            raise ValueError(f"Unexpected field in schema: {field_name}")

        actual_type = df_schema.field(field_name).type
        if not expected_type.equals(actual_type):
            raise ValueError(
                f"Invalid type for {field_name}. " f"Expected {expected_type}, got {actual_type}"
            )

    return True


def validate_prediction_data(df: pl.DataFrame) -> pa.Table:
    """
    Validate prediction data against the schema and ensure correct ordering.

    Args:
        df: polars.DataFrame to validate

    Returns:
        polars.DataFrame: Validated data with correct schema and ordering
    """
    # Detect which optional components are present
    has_prediction = all(field in df.columns for field in PREDICTION_FIELDS)
    has_embeddings = "embeddings" in df.columns
    has_logits = "logits" in df.columns
    has_sequence_logits = "logits_sequence" in df.columns
    has_loss = "loss" in df.columns

    # Get the validated schema
    validated_schema = prediction_analysis_schema(
        include_prediction=has_prediction,
        include_embeddings=has_embeddings,
        include_logits=has_logits,
        include_sequence_logits=has_sequence_logits,
        include_loss=has_loss,
    )

    # Ensure columns are in the correct order before converting to arrow
    expected_columns = validated_schema.names
    df = df.select(expected_columns)

    # Convert to arrow and validate
    arrow_table = df.to_arrow().cast(validated_schema)

    # Validate the schema
    validate_prediction_schema(arrow_table.schema)

    # Cast to the validated schema
    return arrow_table


# Example usage:
"""
import polars as pl

df = pl.DataFrame({
    "subject_id": [1, 2, 3],
    "prediction_time": ["2024-01-01", "2024-01-02", "2024-01-03"],
    "boolean_value": [True, False, True],
    "predicted_boolean_value": [True, True, False],
    "predicted_boolean_probability": [0.8, 0.7, 0.3],
})

# Validate
validated_df = validate_prediction_data(df)
validated_df.write_parquet("predictions.parquet")
"""
