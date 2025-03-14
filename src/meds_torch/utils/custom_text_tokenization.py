#!/usr/bin/env python
"""Functions for tokenizing MEDS datasets - Optimized Version with Sample Limit."""

import multiprocessing as mp
import os
import time  # Added for timing functions
from abc import ABC, abstractmethod
from dataclasses import dataclass
from functools import wraps
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset

# Set the multiprocessing start method to 'spawn' to avoid CUDA issues
if __name__ == "__main__":  # Only in the main process
    mp.set_start_method("spawn", force=True)

import hydra
import polars as pl
from loguru import logger
from MEDS_transforms import PREPROCESS_CONFIG_YAML
from MEDS_transforms.mapreduce.utils import shard_iterator
from MEDS_transforms.utils import hydra_loguru_init, write_lazyframe
from omegaconf import DictConfig, OmegaConf
from safetensors.torch import save_file
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer, BertConfig

# Constants
SECONDS_PER_DAY = 86400.0
BATCH_SIZE = 512
SAMPLE_LIMIT = None  # Limit to 1000 samples

# Use GPU if available, with mixed precision
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
USE_MIXED_PRECISION = torch.cuda.is_available()

# Number of worker processes for parallel processing
NUM_WORKERS = max(1, os.cpu_count() - 1)

# Dictionary to store function execution times
function_times = {}
# Lock for accessing the function_times dictionary in multiprocessing
# We'll create the lock when needed instead of at module level
time_lock = None


def timing_decorator(func):
    """Decorator to measure function execution time"""

    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        elapsed_time = end_time - start_time

        func_name = func.__name__
        # Simply record the time locally - we'll collect from workers later if needed
        global function_times
        if func_name in function_times:
            function_times[func_name] += elapsed_time
        else:
            function_times[func_name] = elapsed_time

        return result

    return wrapper


@dataclass
class MultimodalReader(ABC):
    """Abstract base class for reading multimodal data."""

    base_path: str

    @abstractmethod
    def read_modality(self, relative_input: str) -> torch.Tensor:
        """Read and return modality data as a tensor."""


class TextDataset(Dataset):
    """Dataset for efficient batch processing of text data."""

    def __init__(self, texts, indices, tokenizer, max_length=512):
        self.texts = texts
        self.indices = indices
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        if not text or not isinstance(text, str):
            # Create empty tensors with the right shape for batch consistency
            # This ensures all elements in a batch have the same keys even if empty
            empty_input_ids = torch.zeros((self.max_length,), dtype=torch.long)
            empty_attention_mask = torch.zeros((self.max_length,), dtype=torch.long)

            return {
                "input_ids": empty_input_ids,
                "attention_mask": empty_attention_mask,
                "index": self.indices[idx],
                "empty": True,
            }

        # Tokenize with truncation
        try:
            encoding = self.tokenizer(
                text, truncation=True, max_length=self.max_length, padding="max_length", return_tensors="pt"
            )

            # Remove batch dimension
            encoding = {k: v.squeeze(0) for k, v in encoding.items()}
            encoding["index"] = self.indices[idx]
            encoding["empty"] = False

            return encoding
        except Exception as e:
            logger.warning(f"Error tokenizing text at index {idx}: {str(e)}")
            # Fall back to empty tensors on error
            empty_input_ids = torch.zeros((self.max_length,), dtype=torch.long)
            empty_attention_mask = torch.zeros((self.max_length,), dtype=torch.long)

            return {
                "input_ids": empty_input_ids,
                "attention_mask": empty_attention_mask,
                "index": self.indices[idx],
                "empty": True,
            }


class BioClinicalBertBatchEmbedder:
    """Optimized class for efficiently embedding clinical text using BioClinicalBERT."""

    def __init__(
        self, model_name="nlpie/tiny-clinicalbert", max_length=512, batch_size=BATCH_SIZE, device=None
    ):
        self.device = (
            torch.device("cuda" if torch.cuda.is_available() else "cpu")
            if device is None
            else torch.device(device)
        )
        logger.info(f"Loading model on: {self.device}")

        # Use smaller config if using tiny model for faster loading
        init_start_time = time.time()
        if "tiny" in model_name:
            config = BertConfig.from_pretrained(model_name)
            # Speed up by reducing number of attention heads if possible
            if hasattr(config, "num_attention_heads") and config.num_attention_heads > 4:
                config.num_attention_heads = 4
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModel.from_pretrained(model_name, config=config)
        else:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModel.from_pretrained(model_name)

        init_time = time.time() - init_start_time
        if "model_initialization" in function_times:
            function_times["model_initialization"] += init_time
        else:
            function_times["model_initialization"] = init_time

        self.model = self.model.to(self.device)
        self.model.eval()
        self.max_length = max_length
        self.batch_size = batch_size

    @timing_decorator
    def embed_texts_chunked(self, texts, indices, chunk_size=100000, callback=None):
        """Process texts in chunks with optimized batching."""
        dataset = TextDataset(texts, indices, self.tokenizer, self.max_length)

        # Check if we're in a worker process - if so, don't use additional workers in DataLoader
        in_worker_process = mp.current_process().name != "MainProcess"
        num_workers = 0 if in_worker_process else min(16, os.cpu_count() or 1)

        logger.info(
            f"DataLoader using {num_workers} workers (in {'worker' if in_worker_process else 'main'} process)"
        )

        dataloader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            num_workers=num_workers,  # No additional workers when in a worker process
            pin_memory=torch.cuda.is_available() and not in_worker_process,  # Only pin memory in main process
            # Add custom collate function to handle any remaining inconsistencies
            collate_fn=self._collate_batch,
        )

        all_results = []

        for batch in tqdm(dataloader, desc="Embedding texts", leave=False):
            # Skip empty batches
            if all(batch["empty"]):
                zeros = torch.zeros(len(batch["empty"]), self.model.config.hidden_size, device=self.device)
                indices = batch["index"]
                all_results.extend(zip(indices.tolist(), [z.cpu() for z in zeros]))
                continue

            # Process non-empty entries
            mask = ~batch["empty"]
            if not any(mask):
                continue

            # Extract and move input tensors to device
            input_ids = batch["input_ids"][mask].to(self.device)
            attention_mask = batch["attention_mask"][mask].to(self.device)

            # Generate embeddings with mixed precision if available
            with torch.no_grad():
                if USE_MIXED_PRECISION:
                    with torch.amp.autocast("cuda"):  # Fixed deprecated syntax
                        outputs = self.model(input_ids, attention_mask=attention_mask)
                else:
                    outputs = self.model(input_ids, attention_mask=attention_mask)

                # Use mean of hidden states as embedding
                batch_embs = outputs.last_hidden_state.mean(dim=1)

            # Create results for all entries
            results = []
            valid_idx = 0

            for i, is_empty in enumerate(batch["empty"]):
                idx = batch["index"][i].item()
                if is_empty:
                    # Create zero tensor for empty texts
                    emb = torch.zeros(self.model.config.hidden_size, device="cpu")
                else:
                    # Get embedding for valid text
                    emb = batch_embs[valid_idx].cpu()
                    valid_idx += 1
                results.append((idx, emb))

            all_results.extend(results)

        if callback:
            # Group embeddings by chunks for callback
            all_results.sort(key=lambda x: x[0])  # Sort by index
            for start_idx in range(0, len(all_results), chunk_size):
                end_idx = min(start_idx + chunk_size, len(all_results))
                chunk = all_results[start_idx:end_idx]
                callback([c[0] for c in chunk], [c[1] for c in chunk])
        else:
            for idx, emb in all_results:
                yield idx, emb

    def _collate_batch(self, batch):
        """Custom collate function to ensure batch consistency."""
        # First, check if all keys are consistent
        keys = set(batch[0].keys())
        for item in batch[1:]:
            if set(item.keys()) != keys:
                logger.warning(f"Inconsistent keys in batch: {set(item.keys())} vs {keys}")
                # Add missing keys with empty tensors
                for key in keys:
                    if key not in item:
                        if key == "input_ids":
                            item[key] = torch.zeros((self.max_length,), dtype=torch.long)
                        elif key == "attention_mask":
                            item[key] = torch.zeros((self.max_length,), dtype=torch.long)
                        elif key == "empty":
                            item[key] = True
                        else:
                            item[key] = None

        # Now combine the batch
        result = {}
        for key in keys:
            if key == "index":
                result[key] = torch.tensor([item[key] for item in batch])
            elif key == "empty":
                result[key] = torch.tensor([item[key] for item in batch], dtype=torch.bool)
            elif all(isinstance(item[key], torch.Tensor) for item in batch):
                result[key] = torch.stack([item[key] for item in batch])
            else:
                # For non-tensor fields, just collect them
                result[key] = [item[key] for item in batch]

        return result


class BioClinicalBertTextReader(MultimodalReader):
    """A MultimodalReader that embeds text using the optimized BioClinicalBertBatchEmbedder."""

    def __init__(
        self,
        base_path="",
        model_name="nlpie/tiny-clinicalbert",
        max_length=512,
        batch_size=BATCH_SIZE,
        device=None,
    ):
        super().__init__(base_path=base_path)
        self.embedder = BioClinicalBertBatchEmbedder(
            model_name=model_name, max_length=max_length, batch_size=batch_size, device=device
        )

    @timing_decorator
    def read_modality(self, text_value: str) -> torch.Tensor:
        """Embed a single piece of text into a tensor."""
        embeddings = list(self.embedder.embed_texts_chunked([text_value], [0]))
        return embeddings[0][1] if embeddings else torch.zeros(self.embedder.model.config.hidden_size)


@timing_decorator
def fill_to_nans(col: str | pl.Expr) -> pl.Expr:
    """Fill infinite and null values with NaN."""
    if isinstance(col, str):
        col = pl.col(col)
    return pl.when(col.is_infinite() | col.is_null()).then(float("nan")).otherwise(col)


@timing_decorator
def split_static_and_dynamic(df: pl.LazyFrame) -> tuple[pl.LazyFrame, pl.LazyFrame]:
    """Split the input data into static and dynamic components."""
    static = df.filter(pl.col("time").is_null()).drop("time")
    dynamic = df.filter(pl.col("time").is_not_null())

    # Add modality index for text values
    if "text_value" in df.collect_schema().names():
        dynamic = dynamic.with_columns(
            [
                pl.when(pl.col("text_value").is_not_null())
                .then(pl.col("text_value").rank("dense") - 1)
                .otherwise(None)
                .cast(pl.Float32)
                .alias("modality_idx")
            ]
        )

    return static, dynamic


@timing_decorator
def process_text_data(df: pl.DataFrame, reader: MultimodalReader) -> dict[str, torch.Tensor]:
    """Process and embed text data."""
    logger.info("Processing text data for embeddings")

    # Validate and clean text_value column
    df = df.with_columns(
        pl.col("text_value").map_elements(
            lambda x: x.encode("utf-8", "ignore").decode("utf-8", "ignore") if isinstance(x, str) else "",
            return_dtype=pl.Utf8,
        )
    )

    # Filter to rows with non-null text_value
    text_df = df.filter(pl.col("text_value").is_not_null())

    text_mapping = {}
    if len(text_df) == 0:
        logger.info("No text data found to process")
        return text_mapping

    # Get unique texts with modality indices
    unique_text_df = text_df.select(["text_value", "modality_idx"]).unique()
    texts = unique_text_df["text_value"].to_list()
    modality_idxs = unique_text_df["modality_idx"].to_list()

    logger.info(f"Processing {len(texts)} unique text entries")

    # Check for very large texts that might cause issues
    text_lengths = [len(t) if isinstance(t, str) else 0 for t in texts]
    if max(text_lengths) > 10000:
        logger.warning(
            f"Very long text detected: {max(text_lengths)} characters. " f"This might cause memory issues."
        )

    # Create chunked processing for very large datasets
    try:
        # Create embeddings dictionary
        for idx, emb in reader.embedder.embed_texts_chunked(texts, modality_idxs):
            text_mapping[f"{int(idx)}"] = emb.cpu()
    except Exception as e:
        logger.error(f"Error during text embedding: {str(e)}")
        # Try processing in smaller batches if we had an error
        if len(texts) > 100:
            logger.info("Retrying with smaller batches")
            chunk_size = max(1, len(texts) // 10)
            for i in range(0, len(texts), chunk_size):
                chunk_texts = texts[i : i + chunk_size]
                chunk_idxs = modality_idxs[i : i + chunk_size]
                logger.info(f"Processing chunk {i//chunk_size + 1}/10 with {len(chunk_texts)} texts")
                try:
                    for idx, emb in reader.embedder.embed_texts_chunked(chunk_texts, chunk_idxs):
                        text_mapping[f"{int(idx)}"] = emb.cpu()
                except Exception as chunk_e:
                    logger.error(f"Error processing chunk: {str(chunk_e)}")
                    # Continue with next chunk

    logger.info(f"Generated {len(text_mapping)} text embeddings")
    return text_mapping


@timing_decorator
def extract_statics_and_schema(df: pl.LazyFrame) -> pl.LazyFrame:
    """Extract static data and schema information."""
    logger.info("Extracting static data and schema")
    static, dynamic = split_static_and_dynamic(df)

    # Group static data by subject ID
    static_by_subject = static.group_by("subject_id", maintain_order=True).agg(
        pl.col("code"), pl.col("numeric_value")
    )

    # Collect unique times for each subject
    schema_by_subject = dynamic.group_by("subject_id", maintain_order=True).agg(
        pl.col("time").min().alias("start_time"),
        pl.col("time").unique(maintain_order=True),
    )

    # Join static and schema data
    result = static_by_subject.join(schema_by_subject, on="subject_id", how="full", coalesce=True)
    return result


@timing_decorator
def extract_seq_of_subject_events(
    df: pl.LazyFrame,
    reader: MultimodalReader,
    modality_out_fp: Path = None,
) -> tuple[pl.LazyFrame, dict[str, torch.Tensor]]:
    """Extract sequences of subject events."""
    logger.info("Extracting sequences of subject events")
    _, dynamic = split_static_and_dynamic(df)

    # Process text values if they exist
    text_mapping = {}
    if "text_value" in df.collect_schema().names():
        # Always use streaming approach
        logger.info("Using streaming approach for dataset")
        dynamic_collected = dynamic.collect(streaming=True)
        text_mapping = process_text_data(dynamic_collected, reader)

        # Save embeddings if output path is provided
        if modality_out_fp and text_mapping:
            modality_out_fp.parent.mkdir(parents=True, exist_ok=True)
            save_file(text_mapping, modality_out_fp)
            logger.info(f"Saved {len(text_mapping)} text embeddings to {modality_out_fp}")

    # Calculate time deltas in days
    time_delta_days_expr = (pl.col("time").diff().dt.total_seconds() / SECONDS_PER_DAY).cast(pl.Float32)

    # Convert back to LazyFrame for aggregation
    if isinstance(dynamic, pl.DataFrame):
        dynamic = dynamic.lazy()

    # Fix: Optimize aggregation with more explicit column references to avoid duplicates
    # First, group by subject_id and time to collect codes at each timepoint
    first_agg = dynamic.group_by(["subject_id", "time"], maintain_order=True).agg(
        [pl.col("code").alias("code"), fill_to_nans("numeric_value").alias("numeric_value")]
    )

    # If text_value exists, add modality_idx to the first aggregation
    if "text_value" in df.collect_schema().names():
        first_agg = dynamic.group_by(["subject_id", "time"], maintain_order=True).agg(
            [
                pl.col("code").alias("code"),
                fill_to_nans("numeric_value").alias("numeric_value"),
                fill_to_nans("modality_idx").alias("modality_idx"),
            ]
        )

    # Then group by subject_id to create sequences
    if "text_value" in df.collect_schema().names():
        result = first_agg.group_by("subject_id", maintain_order=True).agg(
            [
                fill_to_nans(time_delta_days_expr).alias("time_delta_days"),
                pl.col("code").alias("code"),
                pl.col("numeric_value").alias("numeric_value"),
                pl.col("modality_idx").alias("modality_idx"),
            ]
        )
    else:
        result = first_agg.group_by("subject_id", maintain_order=True).agg(
            [
                fill_to_nans(time_delta_days_expr).alias("time_delta_days"),
                pl.col("code").alias("code"),
                pl.col("numeric_value").alias("numeric_value"),
            ]
        )

    return result, text_mapping


@timing_decorator
def process_shard(
    in_fp, schema_out_fp, event_seq_out_fp, text_out_fp, reader, do_overwrite, sample_limit=None
):
    """Process a single shard with optional sample limit."""
    # Initialize a local function_times dictionary for this process
    global function_times
    function_times = {}

    shard_start_time = time.time()
    logger.info(f"Processing shard: {in_fp}")

    # Create a scan with sample limit if specified
    def limited_scan(file_path):
        df = pl.scan_parquet(file_path)
        if sample_limit:
            logger.info(f"Limiting to {sample_limit} samples")
            df = df.limit(sample_limit)
        return df

    # Extract static data and schema
    if not schema_out_fp.exists() or do_overwrite:
        schema_out_fp.parent.mkdir(parents=True, exist_ok=True)
        df = limited_scan(in_fp)
        schema_df = extract_statics_and_schema(df)
        write_lazyframe(schema_df, schema_out_fp)
        logger.info(f"Wrote schema to {schema_out_fp}")

    # Extract event sequences and text embeddings
    if not event_seq_out_fp.exists() or do_overwrite:
        event_seq_out_fp.parent.mkdir(parents=True, exist_ok=True)
        df = limited_scan(in_fp)
        event_seq_df, _ = extract_seq_of_subject_events(df, reader, text_out_fp)
        write_lazyframe(event_seq_df, event_seq_out_fp)
        logger.info(f"Wrote event sequences to {event_seq_out_fp}")

    shard_time = time.time() - shard_start_time
    logger.info(f"Shard processing time: {shard_time:.2f} seconds")

    # Log timing data from this process
    logger.info(f"Timing data for shard {in_fp}:")
    for func_name, exec_time in sorted(function_times.items(), key=lambda x: x[1], reverse=True):
        logger.info(f"  {func_name:<30}: {exec_time:.2f} seconds")

    return in_fp


@timing_decorator
def tokenize(cfg: DictConfig, sample_limit=SAMPLE_LIMIT):
    """Main function for tokenizing MEDS datasets - optimized version with sample limit."""
    logger.info(
        f"Running with config:\n{OmegaConf.to_yaml(cfg)}\n"
        f"Stage: {cfg.stage}\n"
        f"Stage config:\n{OmegaConf.to_yaml(cfg.stage_cfg)}\n"
        f"Using device: {DEVICE}, Mixed precision: {USE_MIXED_PRECISION}\n"
        f"Number of workers: {NUM_WORKERS}\n"
        f"SAMPLE LIMIT: {sample_limit} samples"
    )

    output_dir = Path(cfg.stage_cfg.output_dir)
    if train_only := cfg.stage_cfg.get("train_only", False):
        raise ValueError(f"train_only={train_only} is not supported for this stage.")

    shards_single_output, _ = shard_iterator(cfg)

    # Create output directories
    (output_dir / "schemas").mkdir(parents=True, exist_ok=True)
    (output_dir / "event_seqs").mkdir(parents=True, exist_ok=True)
    (output_dir / "modalities").mkdir(parents=True, exist_ok=True)

    # Initialize the text embedder reader (shared across processes)
    reader = BioClinicalBertTextReader(
        base_path="", model_name="nlpie/tiny-clinicalbert", max_length=512, batch_size=BATCH_SIZE, device=None
    )

    # Limit to processing just the first shard when sample_limit is set
    shards_list = list(shards_single_output)
    if sample_limit:
        logger.info(f"Sample limit set to {sample_limit}, will only process the first shard")
        if shards_list:
            shards_list = [shards_list[0]]

    # Force sequential processing to avoid nested multiprocessing issues
    logger.info("Using sequential processing to avoid multiprocessing issues")
    for in_fp, out_fp in tqdm(shards_list, desc="Processing shards"):
        sharded_path = out_fp.relative_to(output_dir)
        schema_out_fp = output_dir / "schemas" / sharded_path
        event_seq_out_fp = output_dir / "event_seqs" / sharded_path
        text_out_fp = (output_dir / "modalities" / sharded_path).with_suffix(".safetensors")

        process_shard(
            in_fp, schema_out_fp, event_seq_out_fp, text_out_fp, reader, cfg.do_overwrite, sample_limit
        )

    logger.info(f"Done with {cfg.stage} - processed with sample limit: {sample_limit}")

    # Calculate and display timing information
    print_timing_results()


def print_timing_results():
    """Calculate and print timing results for all functions."""
    total_time = sum(function_times.values())

    if total_time == 0:
        logger.warning("No timing data collected")
        return

    # Sort functions by execution time (descending)
    sorted_times = sorted(function_times.items(), key=lambda x: x[1], reverse=True)

    # Calculate percentages
    percentages = {}
    for func_name, exec_time in sorted_times:
        percentages[func_name] = (exec_time / total_time) * 100

    logger.info("\n\n===== FUNCTION TIMING RESULTS =====")
    logger.info(f"Total execution time: {total_time:.2f} seconds")
    logger.info(f"{'Function':<40} {'Time (s)':<15} {'Percentage':<10}")
    logger.info("-" * 65)

    for func_name, exec_time in sorted_times:
        percentage = percentages[func_name]
        logger.info(f"{func_name:<40} {exec_time:<15.2f} {percentage:<10.2f}%")

    # Also print as a sorted list for easier reading
    logger.info("\nTime Breakdown (most time-consuming first):")
    for func_name, exec_time in sorted_times:
        if exec_time > total_time * 0.01:  # Only show functions taking >1% of total time
            percentage = percentages[func_name]
            logger.info(f"{percentage:6.2f}% - {func_name}")

    return percentages  # Return percentages for potential further analysis


@hydra.main(
    version_base=None,
    config_path=str(PREPROCESS_CONFIG_YAML.parent),
    config_name=PREPROCESS_CONFIG_YAML.stem,
)
def main(cfg: DictConfig):
    overall_start_time = time.time()
    hydra_loguru_init()

    # Run with the sample limit
    tokenize(cfg, sample_limit=SAMPLE_LIMIT)

    overall_end_time = time.time()
    overall_time = overall_end_time - overall_start_time

    # Add overall execution time
    function_times["total_script_execution"] = overall_time

    # Print timing results once more at the very end
    print_timing_results()


if __name__ == "__main__":  # pragma: no cover
    # Set the multiprocessing start method to 'spawn' to avoid CUDA issues with fork
    mp.set_start_method("spawn", force=True)

    # Now we can initialize a Manager for shared state if needed
    # (though in this implementation we're not sharing timing data between processes)

    main()
