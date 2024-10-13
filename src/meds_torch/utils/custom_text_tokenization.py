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

def fill_to_nans(col: str | pl.Expr) -> pl.Expr:
    """Fill infinite and null values with NaN."""
    if isinstance(col, str):
        col = pl.col(col)
    return pl.when(col.is_infinite() | col.is_null()).then(float("nan")).otherwise(col)

def split_static_and_dynamic(df: pl.LazyFrame) -> tuple[pl.LazyFrame, pl.LazyFrame]:
    """Split data into static (null time) and dynamic (non-null time) parts."""
    static = df.filter(pl.col("time").is_null()).drop("time")
    dynamic = df.filter(pl.col("time").is_not_null())
    return static, dynamic

def tokenize_text(text: str) -> list[int]:
    """Tokenize a text string using a pretrained tokenizer."""
    if text is None or text == "":
        return []
    return tokenizer.encode(text, add_special_tokens=True)

def extract_statics_and_schema(df: pl.LazyFrame) -> pl.LazyFrame:
    """Extract static data, schema information, and tokenize `text_value`."""
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

    return static_by_subject.join(schema_by_subject, on="subject_id", how="full", coalesce=True)

def extract_seq_of_subject_events(df: pl.LazyFrame) -> pl.LazyFrame:
    """Extract sequences of subject events and tokenize `text_value`."""
    df = df.with_columns(
        pl.col("text_value").map_elements(lambda x: tokenize_text(x), return_dtype=pl.List(pl.Int64), skip_nulls=False).alias("text_value")
    )
    _, dynamic = split_static_and_dynamic(df)

    time_delta_days_expr = (pl.col("time").diff().dt.total_seconds() / SECONDS_PER_DAY).cast(pl.Float32)

    return (
        dynamic.group_by("subject_id", "time", maintain_order=True)
        .agg(
            pl.col("code").name.keep(),
            fill_to_nans("numeric_value").name.keep(),
            "text_value"
        )
        .group_by("subject_id", maintain_order=True)
        .agg(
            fill_to_nans(time_delta_days_expr).alias("time_delta_days"),
            "code",
            "numeric_value",
            "text_value",
        )
    )

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