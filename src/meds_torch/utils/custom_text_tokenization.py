#!/usr/bin/env python
"""Functions for tokenizing MEDS dataset, speficically thetext_value column using a pretrained tokenizer."""

from pathlib import Path

import hydra
import polars as pl
from loguru import logger
from MEDS_transforms.mapreduce.utils import rwlock_wrap, shard_iterator
from MEDS_transforms.utils import hydra_loguru_init, write_lazyframe
from omegaconf import DictConfig, OmegaConf
from pathlib import Path
from MEDS_transforms import PREPROCESS_CONFIG_YAML
from transformers import AutoTokenizer

SECONDS_PER_MINUTE = 60.0
SECONDS_PER_HOUR = SECONDS_PER_MINUTE * 60.0
SECONDS_PER_DAY = SECONDS_PER_HOUR * 24.0

tokenizer = AutoTokenizer.from_pretrained("emilyalsentzer/Bio_ClinicalBERT")

import polars as pl

def fill_to_nans(col: str | pl.Expr) -> pl.Expr:
    """
    Fill infinite and null values with NaN.

    Args:
        col: Column name or Polars expression to process

    Returns:
        Polars expression with nulls and infinities replaced with NaN

    Examples:
        >>> import polars as pl
        >>> df = pl.DataFrame({
        ...     "numeric_value": [1.0, None, float('inf'), -float('inf')]
        ... })
        >>> result = df.with_columns(fill_to_nans("numeric_value"))
        >>> result["numeric_value"].to_list()
        [1.0, nan, nan, nan]
    """
    if isinstance(col, str):
        col_expr = pl.col(col)
        return (
            pl.when(col_expr.is_infinite() | col_expr.is_null())
            .then(float("nan"))
            .otherwise(col_expr)
            .alias(col)
        )
    else:
        return pl.when(col.is_infinite() | col.is_null()).then(float("nan")).otherwise(col)

import polars as pl

def split_static_and_dynamic(df: pl.LazyFrame) -> tuple[pl.LazyFrame, pl.LazyFrame]:
    """
    Split data into static (null time) and dynamic (non-null time) parts.

    Args:
        df: Input LazyFrame with a 'time' column

    Returns:
        Tuple of (static_df, dynamic_df)

    Examples:
        >>> import polars as pl
        >>> df = pl.DataFrame({
        ...     "subject_id": [100000380, 100000380, 100000380],
        ...     "time": [None, "2017-11-18 10:26:09", "2018-01-09 19:18:09"],
        ...     "code": ["HNP_NOTE/STATIC", "HNP_NOTE/FHPHP", "HNP_NOTE/FHPHP"],
        ...     "numeric_value": [None, None, None],
        ...     "text_value": ["Static note", "Dynamic note 1", "Dynamic note 2"]
        ... }).lazy()
        >>> static, dynamic = split_static_and_dynamic(df)
        >>> static.collect().shape
        (1, 4)
        >>> dynamic.collect().shape
        (2, 5)
    """
    static = df.filter(pl.col("time").is_null()).drop("time")
    dynamic = df.filter(pl.col("time").is_not_null()).with_columns(
        pl.col("time").str.strptime(pl.Datetime("us"), format="%Y-%m-%d %H:%M:%S", strict=False)
    )
    return static, dynamic

def tokenize_text(text: str) -> list[int]:
    """Tokenize a text string using Bio_ClinicalBERT tokenizer.
    
    Args:
        text: Input text string to tokenize
        
    Returns:
        List of token IDs
        
    Examples:
        >>> # Test with actual clinical note text
        >>> text = "Patient presents with fever"
        >>> tokens = tokenize_text(text)
        >>> len(tokens) > 0  # Should return some tokens
        True
        >>> isinstance(tokens[0], int)  # Should be integer token IDs
        True
        
        >>> # Test empty/null cases
        >>> tokenize_text("")
        []
        >>> tokenize_text(None)
        []
    """
    if text is None or text == "":
        return []
    return tokenizer.encode(text, add_special_tokens=True)

def extract_statics_and_schema(df: pl.LazyFrame) -> pl.LazyFrame:
    """Extract static data, schema information, and tokenize text_value. An example of what the final output looks like is shown below.
    
    # Final Output (Schema + Static Data Combined):
    # | subject_id | code                                  | numeric_value | text_value (tokenized)                   | start_time            | time (unique)                                                     |
    # |------------|---------------------------------------|---------------|------------------------------------------|-----------------------|-------------------------------------------------------------------|
    # | 100000380  | ["HNP_NOTE//FHPHP", "HNP_NOTE//BHPHP"]| [null, null]  | [[101, 2054, ...], [101, 2054, ...], ...]| 2017-11-18 10:26:09   | ["2017-11-18 10:26:09", "2018-01-09 19:18:09", ..., "2021-08-23"] |
    
    Args:
        df: Input LazyFrame with required columns
        
    Returns:
        LazyFrame with static data and schema information
        
    Examples:
        >>> import polars as pl
        >>> df = pl.DataFrame({
        ...     "subject_id": [100000380, 100000380, 100000380],
        ...     "time": [None, "2017-11-18 10:26:09", "2018-01-09 19:18:09"],
        ...     "code": ["HNP_NOTE/STATIC", "HNP_NOTE/FHPHP", "HNP_NOTE/FHPHP"],
        ...     "numeric_value": [None, None, None],
        ...     "text_value": ["Static note", "Dynamic note 1", "Dynamic note 2"]
        ... }).lazy()
        >>> result = extract_statics_and_schema(df).collect()
        >>> # Check structure
        >>> "subject_id" in result.columns
        True
        >>> "start_time" in result.columns  # Should contain schema info
        True
        >>> # Check if text_value is tokenized
        >>> isinstance(result["text_value"][0].to_list()[0][0], int)  # Should be token IDs
        True
    """
    static, dynamic = split_static_and_dynamic(df)

    # Tokenize the text_value column using the pretrained tokenizer

    static = static.with_columns(
        pl.col("text_value").map_elements(lambda x: tokenize_text(x), return_dtype=pl.List(pl.Int64), skip_nulls=False).alias("text_value")
    )
    dynamic = dynamic.with_columns(
        pl.col("text_value").map_elements(lambda x: tokenize_text(x), return_dtype=pl.List(pl.Int64), skip_nulls=False).alias("text_value")
    )

    static_by_subject = static.group_by("subject_id", maintain_order=True).agg("code", "numeric_value", "text_value")
    schema_by_subject = dynamic.group_by("subject_id", maintain_order=True).agg(
        pl.col("time").min().alias("start_time"), pl.col("time").unique(maintain_order=True)
    )

    result_df = static_by_subject.join(schema_by_subject, on="subject_id", how="full", coalesce=True)

    return result_df

def extract_seq_of_subject_events(df: pl.LazyFrame) -> pl.LazyFrame:
    """Extract sequences of subject events and tokenize text_value. An example of the output can be found below.

    # Output DataFrame (after processing):
    #
    # | subject_id | time                  | time_delta_days | code                | numeric_value | text_value (tokenized)        |
    # |------------|-----------------------|-----------------|---------------------|---------------|-------------------------------|
    # | 100000380  | 2017-11-18 10:26:09   | NaN             | ["HNP_NOTE/FHPHP"]  | [NaN]         | [[101, 4534, 123, 102]]       |
    # | 100000380  | 2018-01-09 19:18:09   | 52.36           | ["HNP_NOTE/FHPHP"]  | [NaN]         | [[101, 4534, 124, 102]]       |
    # | 100000380  | 2018-03-10 13:57:50   | 59.94           | ["HNP_NOTE/BHPHP"]  | [NaN]         | [[101, 4534, 125, 102]]       |
        

    Args:
        df: Input LazyFrame with required columns
        
    Returns:
        LazyFrame with event sequences and tokenized text
        
    Examples:
        >>> import polars as pl
        >>> df = pl.DataFrame({
        ...     "subject_id": [100000380, 100000380],
        ...     "time": ["2017-11-18 10:26:09", "2018-01-09 19:18:09"],
        ...     "code": ["HNP_NOTE/FHPHP", "HNP_NOTE/FHPHP"],
        ...     "numeric_value": [None, None],
        ...     "text_value": ["Note 1", "Note 2"]
        ... }).lazy()
        >>> result = extract_seq_of_subject_events(df).collect()
        >>> "time_delta_days" in result.columns
        True
        >>> isinstance(result["text_value"][0].item()[0], int)
        True
        >>> result["time_delta_days"][1] > 0
        True
    """
    # Tokenize text_value only; let split_static_and_dynamic handle date parsing
    df = df.with_columns(
        pl.col("text_value").map_elements(
            lambda x: tokenize_text(x), 
            return_dtype=pl.List(pl.Int64), 
            skip_nulls=False
        )
    )
    
    # Split into static/dynamic, which parses time in split_static_and_dynamic
    _, dynamic = split_static_and_dynamic(df)

    # Group by subject_id and time to combine duplicates
    dynamic = dynamic.group_by("subject_id", "time", maintain_order=True).agg(
        pl.col("code").alias("code"),
        pl.col("numeric_value").alias("numeric_value"),
        pl.col("text_value").alias("text_value")
    ).sort("time").sort("subject_id")

    # Calculate time deltas within each subject group
    dynamic = dynamic.with_columns(
        (pl.col("time").diff().dt.total_seconds() / (24 * 3600))
        .cast(pl.Float32)
        .over("subject_id")
        .alias("time_delta_days")
    )

    return dynamic

@hydra.main(
    version_base=None, config_path=str(PREPROCESS_CONFIG_YAML.parent), config_name=PREPROCESS_CONFIG_YAML.stem
)
def main(cfg: DictConfig):
    """Main function for tokenization."""
    hydra_loguru_init()

    logger.info(
        f"Running with config:\n{OmegaConf.to_yaml(cfg)}\n"
        f"Stage: {cfg.stage}\n\n"
        f"Stage config:\n{OmegaConf.to_yaml(cfg.stage_cfg)}"
    )

    output_dir = Path(cfg.stage_cfg.output_dir)
    shards_single_output, include_only_train = shard_iterator(cfg)

    if include_only_train:
        raise ValueError("Not supported for this stage.")

    for in_fp, out_fp in shards_single_output:
        sharded_path = out_fp.relative_to(output_dir)

        schema_out_fp = output_dir / "schemas" / sharded_path
        event_seq_out_fp = output_dir / "event_seqs" / sharded_path

        logger.info(f"Tokenizing {str(in_fp.resolve())} into schemas at {str(schema_out_fp.resolve())}")

        rwlock_wrap(
            in_fp,
            schema_out_fp,
            pl.scan_parquet,
            write_lazyframe,
            extract_statics_and_schema,
            do_overwrite=cfg.do_overwrite,
        )

        logger.info(f"Tokenizing {str(in_fp.resolve())} into event_seqs at {str(event_seq_out_fp.resolve())}")

        rwlock_wrap(
            in_fp,
            event_seq_out_fp,
            pl.scan_parquet,
            write_lazyframe,
            extract_seq_of_subject_events,
            do_overwrite=cfg.do_overwrite,
        )

    logger.info(f"Done with {cfg.stage}")

if __name__ == "__main__":
    main()