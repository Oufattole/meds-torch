# src/meds_torch/schemas/prediction_schema.py

from collections import OrderedDict

import polars as pl
import pyarrow as pa

from meds_torch.models import MODEL_PRED_PROBA_KEY, MODEL_PREFIX

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
    df_fields: set[str] = {name for name in df_schema.names if not name.startswith(MODEL_PREFIX)}

    # Check required fields
    missing_required = REQUIRED_FIELDS - df_fields
    if missing_required:
        raise ValueError(f"Missing required fields: {missing_required}")

    # Check prediction fields (all or nothing)
    has_prediction = bool(PREDICTION_FIELDS & df_fields)
    if has_prediction and not PREDICTION_FIELDS.issubset(df_fields):
        raise ValueError(f"If any prediction field is present, all must be present: {PREDICTION_FIELDS}")

    # Validate field types and order for non-MODEL// fields
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
    MODEL// prefixed columns are preserved without validation.

    Args:
        df: polars.DataFrame to validate

    Returns:
        pyarrow.Table: Validated data with correct schema and ordering

    Examples:
    >>> import polars as pl
    >>> import pyarrow as pa
    >>> # Create test data with both core and MODEL// prefixed columns
    >>> test_df = pl.DataFrame({
    ...     "subject_id": [1, 2],
    ...     "prediction_time": ["2024-01-01", "2024-01-02"],
    ...     "boolean_value": [True, False],
    ...     "predicted_boolean_value": [True, False],
    ...     "predicted_boolean_probability": [0.8, 0.3],
    ...     "MODEL//embeddings": [[1.0, 2.0], [3.0, 4.0]],  # Custom MODEL// column
    ...     "MODEL//extra_data": [100.0, 200.0],  # Another MODEL// column
    ...     "embeddings": [[5.0, 6.0], [7.0, 8.0]]  # Core schema column
    ... })
    >>>
    >>> # Validate the data
    >>> result = validate_prediction_data(test_df)
    >>>
    >>> # Check that both core and MODEL// columns are preserved
    >>> sorted(result.column_names)  # doctest: +NORMALIZE_WHITESPACE
    ['MODEL//embeddings', 'MODEL//extra_data', 'boolean_value', 'embeddings', 'predicted_boolean_probability',
     'predicted_boolean_value', 'prediction_time', 'subject_id']
    >>>
    >>> # Verify core schema types
    >>> str(result.schema.field("subject_id").type)
    'int64'
    >>> str(result.schema.field("prediction_time").type)
    'timestamp[ns]'
    >>> str(result.schema.field("embeddings").type)
    'list<item: double>'
    >>>
    >>> # Verify MODEL// columns are preserved with their types
    >>> str(result.schema.field("MODEL//embeddings").type)
    'large_list<item: double>'
    >>> str(result.schema.field("MODEL//extra_data").type)
    'double'
    >>>
    >>> # Test error case: missing required field
    >>> bad_df = test_df.drop("subject_id")
    >>> import pytest
    >>> with pytest.raises(Exception):
    ...     validate_prediction_data(bad_df)
    """
    # Separate MODEL// columns from core columns
    model_columns = [
        col for col in df.columns if col.startswith(MODEL_PREFIX) and col != MODEL_PRED_PROBA_KEY
    ]
    core_columns = [col for col in df.columns if not col.startswith(MODEL_PREFIX)]

    # Detect which optional components are present in core columns
    has_prediction = all(field in core_columns for field in PREDICTION_FIELDS)
    has_embeddings = "embeddings" in core_columns
    has_logits = "logits" in core_columns
    has_sequence_logits = "logits_sequence" in core_columns
    has_loss = "loss" in core_columns

    # Get the validated schema for core columns
    validated_schema = prediction_analysis_schema(
        include_prediction=has_prediction,
        include_embeddings=has_embeddings,
        include_logits=has_logits,
        include_sequence_logits=has_sequence_logits,
        include_loss=has_loss,
    )

    # Split dataframe into core and MODEL// parts
    core_df = df.select(core_columns)
    model_df = df.select(model_columns)

    # Ensure core columns are in the correct order
    expected_columns = validated_schema.names
    ordered_core_df = core_df.select(expected_columns)

    # Convert core columns to arrow with validated schema
    core_arrow = ordered_core_df.to_arrow().cast(validated_schema)

    # Convert MODEL// columns to arrow (without schema validation)
    model_arrow = model_df.to_arrow()

    # Combine the tables
    combined_schema = pa.schema(list(core_arrow.schema) + list(model_arrow.schema))
    combined_arrays = [core_arrow[name] for name in core_arrow.column_names] + [
        model_arrow[name] for name in model_arrow.column_names
    ]

    return pa.Table.from_arrays(combined_arrays, schema=combined_schema)


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
