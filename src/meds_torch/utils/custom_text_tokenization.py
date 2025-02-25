#!/usr/bin/env python
"""Functions for tokenizing MEDS datasets.

Here, _tokenization_ refers specifically to the process of converting a longitudinal, irregularly sampled,
continuous time sequence into a temporal sequence at the level that will be consumed by deep-learning models.

All these functions take in _normalized_ data -- meaning data where there are _no longer_ any code modifiers,
as those have been normalized alongside codes into integer indices (in the output code column). The only
columns of concern here thus are `subject_id`, `time`, `code`, `numeric_value`.
"""

from pathlib import Path
from abc import ABC, abstractmethod
from dataclasses import dataclass

import hydra
import polars as pl
import torch
from loguru import logger
from MEDS_transforms import PREPROCESS_CONFIG_YAML
from MEDS_transforms.mapreduce.utils import rwlock_wrap, shard_iterator
from MEDS_transforms.utils import hydra_loguru_init, write_lazyframe
from omegaconf import DictConfig, OmegaConf
from safetensors.torch import save_file
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm

TOKENIZER = AutoTokenizer.from_pretrained("emilyalsentzer/Bio_ClinicalBERT")

SECONDS_PER_MINUTE = 60.0
SECONDS_PER_HOUR = SECONDS_PER_MINUTE * 60.0
SECONDS_PER_DAY = SECONDS_PER_HOUR * 24.0


@dataclass
class MultimodalReader(ABC):
    """Abstract base class for reading multimodal data."""

    base_path: str

    @abstractmethod
    def read_modality(self, relative_input: str) -> torch.Tensor:
        """
        Read and return modality data as a tensor or other structured format.

        Args:
            relative_input: Some string input denoting the data to read
                            (could be a filepath, text string, etc.)

        Returns:
            A torch.Tensor containing the processed modality data.
        """


class BioClinicalBertBatchEmbedder:
    """Class for efficiently embedding batches of clinical text using BioClinicalBERT."""
    
    def __init__(self, model_name="nlpie/tiny-clinicalbert", max_length=512, batch_size=128, device=None):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu") if device is None else torch.device(device)
        logger.info(f"Loading model on: {self.device}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(self.device)
        self.model.eval()
        self.max_length = max_length
        self.batch_size = batch_size  # Smaller batch size for memory efficiency

    def _chunk_text(self, text):
        # Encode the text to obtain a list of token IDs (without special tokens)
        token_ids = self.tokenizer.encode(text, add_special_tokens=False)
        # Define a chunk size that leaves room for [CLS] and [SEP] tokens
        chunk_size = self.max_length - 2
        # Split token_ids into chunks
        chunks = [token_ids[i:i + chunk_size] for i in range(0, len(token_ids), chunk_size)]
        # Add special tokens to each chunk
        chunks = [
            [self.tokenizer.cls_token_id] + chunk + [self.tokenizer.sep_token_id]
            for chunk in chunks
        ]
        return chunks

    def embed_texts_chunked(self, texts, indices, chunk_size=100000, callback=None):
        """Process texts in chunks to avoid memory issues with large datasets."""
        for start in range(0, len(texts), chunk_size):
            end = min(start + chunk_size, len(texts))
            chunk_texts = texts[start:end]
            chunk_indices = indices[start:end]

            # Process chunk into embeddings
            chunked_texts = []
            text_idx_map = []
            for i, text in enumerate(chunk_texts):
                if not text or not isinstance(text, str):
                    continue
                # Get list of chunks as lists of token IDs
                chunks = self._chunk_text(text)
                for chunk in chunks:
                    chunked_texts.append(chunk)
                    text_idx_map.append(i)

            if not chunked_texts:
                embeddings = [torch.zeros(self.model.config.hidden_size, device=self.device)] * len(chunk_texts)
            else:
                embeddings = [None] * len(chunk_texts)
                counts = [0] * len(chunk_texts)
                # Process in batches
                for batch_start in range(0, len(chunked_texts), self.batch_size):
                    batch_chunks = chunked_texts[batch_start:batch_start + self.batch_size]
                    # Convert each list of token IDs into a tensor
                    batch_tensors = [torch.tensor(chunk) for chunk in batch_chunks]
                    # Pad the batch to the same length
                    padded = torch.nn.utils.rnn.pad_sequence(
                        batch_tensors, batch_first=True, padding_value=self.tokenizer.pad_token_id
                    )
                    attention_mask = (padded != self.tokenizer.pad_token_id).long()
                    encoded = {
                        "input_ids": padded.to(self.device),
                        "attention_mask": attention_mask.to(self.device)
                    }
                    with torch.no_grad():
                        outputs = self.model(**encoded)
                        batch_embs = outputs.last_hidden_state.mean(dim=1)
                    for i_in_batch, emb in enumerate(batch_embs):
                        global_idx = batch_start + i_in_batch
                        t_idx = text_idx_map[global_idx]
                        if embeddings[t_idx] is None:
                            embeddings[t_idx] = emb
                        else:
                            embeddings[t_idx] += emb
                        counts[t_idx] += 1
                for t_idx in range(len(embeddings)):
                    if counts[t_idx] > 0:
                        embeddings[t_idx] /= counts[t_idx]
                    elif embeddings[t_idx] is None:
                        embeddings[t_idx] = torch.zeros(self.model.config.hidden_size, device=self.device)

            if callback:
                callback(chunk_indices, embeddings)
            else:
                yield from zip(chunk_indices, embeddings)

    def embed_texts(self, texts):
        """Embed each text by chunking and averaging token embeddings."""
        embeddings = []
        for text in texts:
            if not text or not isinstance(text, str):
                embeddings.append(torch.zeros(self.model.config.hidden_size, device=self.device))
                continue
                
            chunks = self._chunk_text(text)
            chunk_embeddings = []
            for chunk in chunks:
                input_ids = torch.tensor([chunk]).to(self.device)
                attention_mask = torch.ones_like(input_ids)
                with torch.no_grad():
                    output = self.model(input_ids, attention_mask=attention_mask)
                    chunk_emb = output.last_hidden_state.mean(dim=1)  # Average token embeddings
                chunk_embeddings.append(chunk_emb)
            if chunk_embeddings:
                final_embedding = torch.stack(chunk_embeddings).mean(dim=0)
            else:
                final_embedding = torch.zeros(self.model.config.hidden_size, device=self.device)
            embeddings.append(final_embedding)
        return embeddings


class BioClinicalBertTextReader(MultimodalReader):
    """
    A MultimodalReader that embeds a single text string at a time using BioClinicalBertBatchEmbedder.
    """

    def __init__(self,
                 base_path="",
                 model_name="nlpie/tiny-clinicalbert",
                 max_length=512,
                 batch_size=128,
                 device=None):
        super().__init__(base_path=base_path)
        self.embedder = BioClinicalBertBatchEmbedder(
            model_name=model_name,
            max_length=max_length,
            batch_size=batch_size,
            device=device
        )

    def read_modality(self, text_value: str) -> torch.Tensor:
        """
        Embed a single piece of text into a single torch.Tensor.
        """
        # embed_texts expects a list of text samples
        embeddings = self.embedder.embed_texts([text_value])
        return embeddings[0]  # single embedding


def fill_to_nans(col: str | pl.Expr) -> pl.Expr:
    """This function fills infinite and null values with NaN.

    This enables the downstream functions to naturally tensorize data into numpy or Torch tensors.

    Args:
        col: The input column.

    Returns:
        A `pl.Expr` object that fills infinite and null values with NaN.

    Examples:
        >>> print(fill_to_nans("value")) # doctest: +NORMALIZE_WHITESPACE
        .when([(col("value").is_infinite()) |
               (col("value").is_null())]).then(dyn float: NaN).otherwise(col("value"))
        >>> print(fill_to_nans(pl.col("time_delta"))) # doctest: +NORMALIZE_WHITESPACE
        .when([(col("time_delta").is_infinite()) |
               (col("time_delta").is_null())]).then(dyn float: NaN).otherwise(col("time_delta"))
        >>> df = pl.DataFrame({"value": [1.0, float("inf"), None, -float("inf"), 2.0]})
        >>> df.select(fill_to_nans("value").alias("value"))["value"].to_list()
        [1.0, nan, nan, nan, 2.0]
    """

    if isinstance(col, str):
        col = pl.col(col)

    return pl.when(col.is_infinite() | col.is_null()).then(float("nan")).otherwise(col)


def split_static_and_dynamic(df: pl.LazyFrame) -> tuple[pl.LazyFrame, pl.LazyFrame]:
    """This function splits the input data into static and dynamic data.

    Static data is data that has a null time, and dynamic data is everything else.
    For dynamic data, a modality index is added for non-null text values.

    Args:
        df: The input data.

    Returns:
        A tuple of two `pl.LazyFrame` objects, the first being the static data and the second being the
        dynamic data.

    Examples:
        >>> from datetime import datetime
        >>> df = pl.DataFrame({
        ...     "subject_id": [1, 1, 2, 2],
        ...     "time": [None, datetime(2021, 1, 1), None, datetime(2021, 1, 2)],
        ...     "code": [100, 101, 200, 201],
        ...     "numeric_value": [1.0, 2.0, 3.0, 4.0],
        ...     "text_value": [None, "fever", None, "cough"]
        ... }).lazy()
        >>> static, dynamic = split_static_and_dynamic(df)
        >>> static.collect()
        shape: (2, 4)
        ┌────────────┬──────┬───────────────┬────────────┐
        │ subject_id ┆ code ┆ numeric_value ┆ text_value │
        │ ---        ┆ ---  ┆ ---           ┆ ---        │
        │ i64        ┆ i64  ┆ f64           ┆ str        │
        ╞════════════╪══════╪═══════════════╪════════════╡
        │ 1          ┆ 100  ┆ 1.0           ┆ null       │
        │ 2          ┆ 200  ┆ 3.0           ┆ null       │
        └────────────┴──────┴───────────────┴────────────┘
        >>> dynamic.collect()
        shape: (2, 6)
        ┌────────────┬─────────────────────┬──────┬───────────────┬────────────┬──────────────┐
        │ subject_id ┆ time                ┆ code ┆ numeric_value ┆ text_value ┆ modality_idx │
        │ ---        ┆ ---                 ┆ ---  ┆ ---           ┆ ---        ┆ ---          │
        │ i64        ┆ datetime[μs]        ┆ i64  ┆ f64           ┆ str        ┆ f32          │
        ╞════════════╪═════════════════════╪══════╪═══════════════╪════════════╪══════════════╡
        │ 1          ┆ 2021-01-01 00:00:00 ┆ 101  ┆ 2.0           ┆ fever      ┆ 1.0          │
        │ 2          ┆ 2021-01-02 00:00:00 ┆ 201  ┆ 4.0           ┆ cough      ┆ 0.0          │
        └────────────┴─────────────────────┴──────┴───────────────┴────────────┴──────────────┘
    """
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


def process_text_data(df: pl.DataFrame, reader: MultimodalReader) -> dict[str, torch.Tensor]:
    """
    For each row that has a non-null `text_value`, we read and embed the text via `reader`.
    Return a dictionary where the key is the text index, and the value is the embedding tensor.

    Args:
        df: A polars DataFrame (already collected) with a `text_value` column.
        reader: An instance of MultimodalReader that can embed text.

    Returns:
        A dict[str, torch.Tensor] keyed by a rank-based index.
    """
    logger.info("Processing text data for embeddings")

    # 1) Ensure `text_value` is valid UTF-8
    df = df.with_columns(
        pl.col("text_value").map_elements(
            lambda x: x.encode("utf-8", "ignore").decode("utf-8", "ignore") if isinstance(x, str) else "",
            return_dtype=pl.Utf8  # Ensure correct output type
        )
    )

    # 2) Filter to rows with non-null `text_value`
    text_df = df.filter(pl.col("text_value").is_not_null())

    text_mapping = {}
    if len(text_df) == 0:
        return text_mapping

    # Get unique texts with their modality indices
    unique_text_df = text_df.select(["text_value", "modality_idx"]).unique()
    texts = unique_text_df["text_value"].to_list()
    modality_idxs = unique_text_df["modality_idx"].to_list()

    # Create embeddings dictionary
    for idx, emb in reader.embedder.embed_texts_chunked(texts, modality_idxs):
        text_mapping[f"{int(idx)}"] = emb.cpu()  # Move to CPU for saving

    return text_mapping


def extract_statics_and_schema(
    df: pl.LazyFrame,
) -> tuple[pl.LazyFrame, dict[str, dict]]:
    """This function extracts static data and schema information (sequence of subject unique times).

    Args:
        df: The input data.

    Returns:
        A tuple containing:
        - A `pl.LazyFrame` object containing the static data and the unique times of the subject
        - A dictionary mapping code_modality to tokenized text

    Examples:
        >>> from datetime import datetime
        >>> df = pl.DataFrame({
        ...     "subject_id": [1, 1, 1, 2, 2],
        ...     "time": [None, datetime(2021, 1, 1), datetime(2021, 1, 13),
        ...             None, datetime(2021, 1, 2)],
        ...     "code": [100, 101, 102, 200, 201],
        ...     "numeric_value": [1.0, 2.0, 3.0, 4.0, 5.0],
        ...     "text_value": [None, "fever", "cough", None, "pain"]
        ... }).lazy()
        >>> result_df = extract_statics_and_schema(df)
        >>> result_df.collect()
        shape: (2, 5)
        ┌────────────┬───────────┬───────────────┬─────────────────────┬─────────────────────────────────┐
        │ subject_id ┆ code      ┆ numeric_value ┆ start_time          ┆ time                            │
        │ ---        ┆ ---       ┆ ---           ┆ ---                 ┆ ---                             │
        │ i64        ┆ list[i64] ┆ list[f64]     ┆ datetime[μs]        ┆ list[datetime[μs]]              │
        ╞════════════╪═══════════╪═══════════════╪═════════════════════╪═════════════════════════════════╡
        │ 1          ┆ [100]     ┆ [1.0]         ┆ 2021-01-01 00:00:00 ┆ [2021-01-01 00:00:00, 2021-01-… │
        │ 2          ┆ [200]     ┆ [4.0]         ┆ 2021-01-02 00:00:00 ┆ [2021-01-02 00:00:00]           │
        └────────────┴───────────┴───────────────┴─────────────────────┴─────────────────────────────────┘
    """
    logger.info("Extracting statics and schema")
    static, dynamic = split_static_and_dynamic(df)

    # This collects static data by subject ID and stores only (as a list) the codes and numeric values
    static_by_subject = static.group_by("subject_id", maintain_order=True).agg("code", "numeric_value")

    # This collects the unique times for each subject
    schema_by_subject = dynamic.group_by("subject_id", maintain_order=True).agg(
        pl.col("time").min().alias("start_time"),
        pl.col("time").unique(maintain_order=True),
    )

    result = static_by_subject.join(schema_by_subject, on="subject_id", how="full", coalesce=True)
    return result


def extract_seq_of_subject_events(
    df: pl.LazyFrame,
    reader: MultimodalReader,
    modality_out_fp: Path = None,
) -> tuple[pl.LazyFrame, dict[str, torch.Tensor]]:
    """This function extracts sequences of subject events, which are sequences of measurements.

    Args:
        df: The input data.
        reader: MultimodalReader instance to embed text values.
        modality_out_fp: Path to save modality embeddings as safetensors.

    Returns:
        A tuple containing:
        - A `pl.LazyFrame` object containing the sequences of subject events
        - A dictionary mapping code_modality to embedded text tensors

    Examples:
        >>> from datetime import datetime
        >>> df = pl.DataFrame({
        ...     "subject_id": [1, 1, 1, 2, 2],
        ...     "time": [None, datetime(2021, 1, 1), datetime(2021, 1, 13),
        ...             None, datetime(2021, 1, 2)],
        ...     "code": [100, 101, 102, 200, 201],
        ...     "numeric_value": [1.0, 2.0, 3.0, 4.0, 5.0],
        ...     "text_value": [None, "fever", None, None, "pain"]
        ... }).lazy()
        >>> result_df, text_mapping = extract_seq_of_subject_events(df)
        >>> result_df.collect()
        shape: (2, 5)
        ┌────────────┬─────────────────┬─────────────────┬─────────────────┬─────────────────┐
        │ subject_id ┆ time_delta_days ┆ code            ┆ numeric_value   ┆ modality_idx    │
        │ ---        ┆ ---             ┆ ---             ┆ ---             ┆ ---             │
        │ i64        ┆ list[f32]       ┆ list[list[i64]] ┆ list[list[f64]] ┆ list[list[f32]] │
        ╞════════════╪═════════════════╪═════════════════╪═════════════════╪═════════════════╡
        │ 1          ┆ [NaN, 12.0]     ┆ [[101], [102]]  ┆ [[2.0], [3.0]]  ┆ [[0.0], [NaN]]  │
        │ 2          ┆ [NaN]           ┆ [[201]]         ┆ [[5.0]]         ┆ [[1.0]]         │
        └────────────┴─────────────────┴─────────────────┴─────────────────┴─────────────────┘
        >>> sorted(text_mapping.keys())  # Check text mapping was created
        ['0', '1']
    """
    logger.info("Extracting sequences of subject events")
    _, dynamic = split_static_and_dynamic(df)

    # Process text values if they exist
    text_mapping = {}
    if "text_value" in df.collect_schema().names():
        # Collect dynamic data to process text embeddings
        collected_dynamic = dynamic.collect()
        text_mapping = process_text_data(collected_dynamic, reader)
        
        # Save embeddings if output path is provided
        if modality_out_fp and text_mapping:
            modality_out_fp.parent.mkdir(parents=True, exist_ok=True)
            save_file(text_mapping, modality_out_fp)
            logger.info(f"Saved text embeddings to {modality_out_fp}")

    time_delta_days_expr = (pl.col("time").diff().dt.total_seconds() / SECONDS_PER_DAY).cast(pl.Float32)

    # Convert back to LazyFrame for aggregation
    if isinstance(dynamic, pl.DataFrame):
        dynamic = dynamic.lazy()

    result = (
        dynamic.group_by("subject_id", "time", maintain_order=True)
        .agg(
            pl.col("code").name.keep(),
            fill_to_nans("numeric_value").name.keep(),
            (fill_to_nans("modality_idx").name.keep() if "text_value" in df.collect_schema().names() else None),
        )
        .group_by("subject_id", maintain_order=True)
        .agg(
            fill_to_nans(time_delta_days_expr).alias("time_delta_days"),
            "code",
            "numeric_value",
            "modality_idx" if "text_value" in df.collect_schema().names() else None,
        )
    )

    return result, text_mapping


@hydra.main(
    version_base=None,
    config_path=str(PREPROCESS_CONFIG_YAML.parent),
    config_name=PREPROCESS_CONFIG_YAML.stem,
)
def main(cfg: DictConfig):
    hydra_loguru_init()
    tokenize(cfg)


def tokenize(cfg: DictConfig):
    """Main function for tokenizing MEDS datasets.

    Examples:
        >>> import tempfile
        >>> import polars as pl
        >>> from datetime import datetime
        >>> from omegaconf import OmegaConf
        >>> from safetensors import safe_open
        >>>
        >>> # Create temporary directory for test data
        >>> with tempfile.TemporaryDirectory() as tmpdir:
        ...     # Create test input data
        ...     test_df = pl.DataFrame({
        ...         "subject_id": [1, 1, 1, 2, 2],
        ...         "time": [None, datetime(2021,1,1), datetime(2021,1,2), None, datetime(2021,1,3)],
        ...         "code": [100, 101, 102, 200, 201],
        ...         "numeric_value": [1.0, 2.0, 3.0, 4.0, 5.0],
        ...         "text_value": [None, "normal", None, None, "abnormal"]
        ...     })
        ...
        ...     # Save test data
        ...     in_fp = Path(tmpdir) / "shard_0.parquet"
        ...     test_df.write_parquet(in_fp)
        ...
        ...     # Create config
        ...     cfg = OmegaConf.create({
        ...         "stage": "tokenize",
        ...         "stage_cfg": {
        ...             "input_dir": str(tmpdir),
        ...             "data_input_dir": str(tmpdir),
        ...             "output_dir": str(tmpdir),
        ...             "file_pattern": "shard_*.parquet",
        ...             "do_sequential": True
        ...         },
        ...         "do_overwrite": True
        ...     })
        ...
        ...     # Run tokenize
        ...     tokenize(cfg)
        ...
        ...     # Verify outputs
        ...     assert (Path(tmpdir) / "schemas" / "shard_0.parquet").exists()
        ...     assert (Path(tmpdir) / "event_seqs" / "shard_0.parquet").exists()
        ...     assert (Path(tmpdir) / "modalities" / "shard_0.safetensors").exists()
        ...
        ...     # Check schema output
        ...     schema_df = pl.read_parquet(Path(tmpdir) / "schemas" / "shard_0.parquet")
        ...     assert len(schema_df) == 2  # Two subjects
        ...     assert all(col in schema_df.columns for col in [
        ...         "subject_id", "code", "numeric_value", "start_time"])
        ...
        ...     # Check event sequences output
        ...     events_df = pl.read_parquet(Path(tmpdir) / "event_seqs" / "shard_0.parquet")
        ...     assert len(events_df) == 2  # Two subjects
        ...     assert all(col in events_df.columns for col in [
        ...         "subject_id", "time_delta_days", "code", "numeric_value", "modality_idx"])
        ...
        ...     # Check event sequences output
        ...     with safe_open(
        ...         Path(tmpdir) / "modalities" / "shard_0.safetensors",
        ...         framework="pt", device="cpu") as f:
        ...         assert set(f.keys()) == {'1', '0'}
        ...         print(f.get_tensor('1'))
        ...         print(f.get_tensor('0'))
        tensor([ 101, 2999,  102])
        tensor([  101, 22832,   102])
    """

    logger.info(
        f"Running with config:\n{OmegaConf.to_yaml(cfg)}\n"
        f"Stage: {cfg.stage}\n\n"
        f"Stage config:\n{OmegaConf.to_yaml(cfg.stage_cfg)}"
    )

    output_dir = Path(cfg.stage_cfg.output_dir)
    if train_only := cfg.stage_cfg.get("train_only", False):
        raise ValueError(f"train_only={train_only} is not supported for this stage.")
    shards_single_output, include_only_train = shard_iterator(cfg)

    # Initialize the text embedder reader
    reader = BioClinicalBertTextReader(
        base_path="",
        model_name="nlpie/tiny-clinicalbert",  # Use smaller model for efficiency
        max_length=512,
        batch_size=128,
        device=None  # Auto-detect device
    )

    for in_fp, out_fp in tqdm(shards_single_output, desc="Processing shards"):
        sharded_path = out_fp.relative_to(output_dir)

        schema_out_fp = output_dir / "schemas" / sharded_path
        event_seq_out_fp = output_dir / "event_seqs" / sharded_path
        text_out_fp = (output_dir / "modalities" / sharded_path).with_suffix(".safetensors")

        logger.info(f"Tokenizing {str(in_fp.resolve())} into schemas at {str(schema_out_fp.resolve())}")

        # Extract static data and schema
        rwlock_wrap(
            in_fp,
            schema_out_fp,
            pl.scan_parquet,
            write_lazyframe,
            extract_statics_and_schema,
            do_overwrite=cfg.do_overwrite,
        )

        logger.info(f"Tokenizing {str(in_fp.resolve())} into event_seqs at {str(event_seq_out_fp.resolve())}")

        # Function to write event sequences and text embeddings
        def write_fn(df, out_fp):
            # Extract sequences and embeddings
            df_result, text_mapping = extract_seq_of_subject_events(df, reader, text_out_fp)
            write_lazyframe(df_result, out_fp)

        # Add output path for the LazyFrame to use in compute functions
        rwlock_wrap(
            in_fp,
            event_seq_out_fp,
            pl.scan_parquet,
            write_fn,
            lambda df: df,  # Pass through the DataFrame to write_fn
            do_overwrite=cfg.do_overwrite,
        )

    logger.info(f"Done with {cfg.stage}")


if __name__ == "__main__":  # pragma: no cover
    main()