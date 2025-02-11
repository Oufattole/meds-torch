#!/usr/bin/env python
"""
Functions for tokenizing MEDS datasets.

Here, _tokenization_ refers specifically to the process of converting a longitudinal, irregularly sampled,
continuous time sequence into a temporal sequence at the level that will be consumed by deep-learning models.

All these functions take in _normalized_ data -- meaning data where there are _no longer_ any code modifiers,
as those have been normalized alongside codes into integer indices (in the output code column). The only
columns of concern here thus are `subject_id`, `time`, `code`, `numeric_value`, `text_value`, etc.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from transformers import AutoTokenizer, AutoModel
import torch
import hydra
import numpy as np
import polars as pl
from loguru import logger
from MEDS_transforms import PREPROCESS_CONFIG_YAML
from MEDS_transforms.mapreduce.utils import rwlock_wrap, shard_iterator
from MEDS_transforms.utils import hydra_loguru_init, write_lazyframe
from omegaconf import DictConfig, OmegaConf
from safetensors.torch import save_file

# Constants
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


class NpyReader(MultimodalReader):
    """
    Example file-based modality reader for .npz or .npy files.
    """
    def read_modality(self, relative_modality_fp: str) -> torch.Tensor:
        """
        Read data from a NumPy binary file, returning a torch.Tensor.

        Args:
            relative_modality_fp: Relative path to the modality data file.

        Returns:
            Torch tensor containing the data.
        """
        data = np.load(Path(self.base_path) / relative_modality_fp)
        return torch.tensor(data)


class DummyReader(MultimodalReader):
    """
    Reader that always returns the same dummy tensor for demonstration.
    """
    def read_modality(self, relative_modality_fp: str) -> torch.Tensor:
        return torch.tensor([1, 2, 3])


# -----------------------------------------------------------------------------
# NEW: Define BioClinicalBertBatchEmbedder + BioClinicalBertTextReader
# -----------------------------------------------------------------------------
class BioClinicalBertBatchEmbedder:
    """
    Takes a list of text samples and returns a pooled embedding for each sample,
    using BioClinicalBERT in batches.

    Example usage (shown as a docstring, not a formal doctest):
    >>> embedder = BioClinicalBertBatchEmbedder(batch_size=2)
    >>> embeddings = embedder.embed_texts(["Short text", "Another text"])
    >>> len(embeddings)
    2
    >>> embeddings[0].shape  # e.g. [768] for standard BERT
    torch.Size([768])
    """

    def __init__(self,
                 model_name="emilyalsentzer/Bio_ClinicalBERT",
                 max_length=512,
                 batch_size=8,
                 device="cuda"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(device)
        self.model.eval()
        self.max_length = max_length
        self.batch_size = batch_size
        self.device = device

    def _chunk_text(self, text):
        """
        Splits `text` into sub-sequences of length ≤ (max_length - 2).

        Returns:
            A list of token *strings* for each chunk.

    >>> embedder = BioClinicalBertBatchEmbedder(max_length=10)
    >>> chunks = embedder._chunk_text("A B C D E F G H I J K L")
    >>> chunks  # Each chunk can hold up to 8 tokens plus [CLS], [SEP]
    [['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h'], ['i', 'j', 'k', 'l']]
        """
        tokens = self.tokenizer.tokenize(text)
        chunk_size = self.max_length - 2  # Reserve space for [CLS] and [SEP]
        return [tokens[i : i + chunk_size] for i in range(0, len(tokens), chunk_size)]

    def embed_texts(self, texts):
        """
        Given a list of text samples, returns a list of embeddings,
        one per text sample. Each embedding is shape [hidden_dim].

    >>> embedder = BioClinicalBertBatchEmbedder(
    ...     model_name="prajjwal1/bert-tiny",  # smaller model for quick demonstration
    ...     max_length=16, batch_size=2, device="cpu"
    ... )
    >>> test_texts = ["short text", "somewhat longer text for test purposes"]
    >>> result = embedder.embed_texts(test_texts)
    >>> len(result)
    2
    >>> # Each embedding is a 128-D vector if we used 'bert-tiny' (hidden_size=128)
    >>> result[0].shape
    torch.Size([128])
    >>> result[1].shape
    torch.Size([128])
        """
        # 1) For each text, chunk into sub-sequences
        chunked_texts = []
        text_idx_of_chunk = []

        for i, text in enumerate(texts):
            if not text or not isinstance(text, str):
                continue
            chunks = self._chunk_text(text)
            for chunk in chunks:
                chunked_texts.append(chunk)
                text_idx_of_chunk.append(i)

        if not chunked_texts:
            # If all are empty or invalid
            return [torch.zeros(self.model.config.hidden_size)] * len(texts)

        # 2) Convert chunks into model input in batches
        chunk_embeddings = [None] * len(chunked_texts)

        for start_idx in range(0, len(chunked_texts), self.batch_size):
            batch_chunk_tokens = chunked_texts[start_idx : start_idx + self.batch_size]
            encoded = self.tokenizer.batch_encode_plus(
                batch_chunk_tokens,
                is_split_into_words=True,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=self.max_length
            ).to(self.device)

            with torch.no_grad():
                outputs = self.model(**encoded)
                # outputs.last_hidden_state: [batch_size, seq_len, hidden_dim]
                batch_chunk_embs = outputs.last_hidden_state.mean(dim=1)

            for i_in_batch, emb in enumerate(batch_chunk_embs):
                global_idx = start_idx + i_in_batch
                chunk_embeddings[global_idx] = emb

        # 3) Aggregate chunk embeddings for each original text
        final_embeddings = [torch.zeros(self.model.config.hidden_size, device=self.device)
                            for _ in range(len(texts))]
        counts = [0] * len(texts)

        for chunk_idx, emb in enumerate(chunk_embeddings):
            t_idx = text_idx_of_chunk[chunk_idx]
            final_embeddings[t_idx] += emb
            counts[t_idx] += 1

        for t_idx in range(len(texts)):
            if counts[t_idx] > 0:
                final_embeddings[t_idx] /= counts[t_idx]
            else:
                final_embeddings[t_idx] = torch.zeros(self.model.config.hidden_size, device=self.device)

        return final_embeddings


# NEW: A reader class that wraps the batch embedder and implements read_modality
class BioClinicalBertTextReader(MultimodalReader):
    """
    A MultimodalReader that embeds a single text string at a time using BioClinicalBertBatchEmbedder.
    """

    def __init__(self,
                 base_path="",
                 model_name="emilyalsentzer/Bio_ClinicalBERT",
                 max_length=512,
                 batch_size=8,
                 device="cuda"):
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

    >>> reader = BioClinicalBertTextReader(
    ...     model_name="prajjwal1/bert-tiny",
    ...     max_length=16,
    ...     batch_size=2
    ... )
    >>> emb = reader.read_modality("Hello world!")  # doctest: +ELLIPSIS
    >>> emb.shape
    torch.Size([128])
        """
        # embed_texts expects a list of text samples
        embeddings = self.embedder.embed_texts([text_value])
        return embeddings[0]  # single embedding


def fill_to_nans(col: str | pl.Expr) -> pl.Expr:
    """
    This function fills infinite and null values with NaN.

    This enables the downstream functions to naturally tensorize data
    into numpy or Torch tensors.

    Args:
        col: The input column name or pl.Expr.

    Returns:
        A `pl.Expr` object that fills infinite and null values with NaN.

    Examples:
        >>> df = pl.DataFrame({"value": [1.0, float("inf"), None, -float("inf"), 2.0]})
        >>> df.select(fill_to_nans("value").alias("value"))["value"].to_list()
        [1.0, nan, nan, nan, 2.0]
    """
    if isinstance(col, str):
        col = pl.col(col)

    return pl.when(col.is_infinite() | col.is_null()).then(float("nan")).otherwise(col)


def split_static_and_dynamic(df: pl.LazyFrame) -> tuple[pl.LazyFrame, pl.LazyFrame]:
    """
    This function splits the input data into static and dynamic data.

    * Static data is data that has a null `time` column, so we drop that column.
    * Dynamic data is data that has a non-null `time`.
    * If there is a `text_value` column, we assign a unique numeric index (`modality_idx`)
      to each row that actually has text (non-null). This is useful for later
      looking up embeddings.

    Args:
        df: The input lazy DataFrame.

    Returns:
        A tuple of two `pl.LazyFrame` objects:
            1) static data (rows with `time` == null),
            2) dynamic data (rows with `time` != null).

    Examples:
        >>> from datetime import datetime
        >>> df = pl.DataFrame({
        ...     "subject_id": [1, 1, 2, 2],
        ...     "time": [None, datetime(2021, 1, 1), None, datetime(2021, 1, 2)],
        ...     "code": [100, 101, 200, 201],
        ...     "numeric_value": [1.0, 2.0, 3.0, 4.0],
        ...     "text_value": ["Static note", "Dynamic text #1", None, "Dynamic text #2"]
        ... }).lazy()

        >>> static, dynamic = split_static_and_dynamic(df)
        >>> static.collect()
        shape: (2, 4)
        ┌────────────┬──────┬───────────────┬─────────────┐
        │ subject_id ┆ code ┆ numeric_value ┆ text_value  │
        │ ---        ┆ ---  ┆ ---           ┆ ---         │
        │ i64        ┆ i64  ┆ f64           ┆ str         │
        ╞════════════╪══════╪═══════════════╪═════════════╡
        │ 1          ┆ 100  ┆ 1.0           ┆ Static note │
        │ 2          ┆ 200  ┆ 3.0           ┆ null        │
        └────────────┴──────┴───────────────┴─────────────┘

        >>> dynamic.collect()
        shape: (2, 6)
        ┌────────────┬─────────────────────┬──────┬───────────────┬─────────────────┬──────────────┐
        │ subject_id ┆ time                ┆ code ┆ numeric_value ┆ text_value      ┆ modality_idx │
        │ ---        ┆ ---                 ┆ ---  ┆ ---           ┆ ---             ┆ ---          │
        │ i64        ┆ datetime[μs]        ┆ i64  ┆ f64           ┆ str             ┆ f32          │
        ╞════════════╪═════════════════════╪══════╪═══════════════╪═════════════════╪══════════════╡
        │ 1          ┆ 2021-01-01 00:00:00 ┆ 101  ┆ 2.0           ┆ Dynamic text #1 ┆ 0.0          │
        │ 2          ┆ 2021-01-02 00:00:00 ┆ 201  ┆ 4.0           ┆ Dynamic text #2 ┆ 1.0          │
        └────────────┴─────────────────────┴──────┴───────────────┴─────────────────┴──────────────┘
    """
    # 1) Split into static vs dynamic
    static = df.filter(pl.col("time").is_null()).drop("time")
    dynamic = df.filter(pl.col("time").is_not_null())

    # 2) If we have a 'text_value' column, rank the rows that actually have text
    if "text_value" in df.collect_schema().names():
        dynamic = dynamic.with_columns(
            pl.when(pl.col("text_value").is_not_null())
            .then(pl.col("text_value").rank("dense") - 1)
            .otherwise(None)
            .cast(pl.Float32)
            .alias("modality_idx")
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

    Examples:
        >>> # We'll use a dummy text reader for demonstration:
        >>> class DummyTextReader(MultimodalReader):
        ...     def read_modality(self, text_value: str) -> torch.Tensor:
        ...         # For simplicity, just return tensor of size 2
        ...         return torch.tensor([len(text_value), 0])
        ...
        >>> df = pl.DataFrame({
        ...     "subject_id": [1, 2, 3],
        ...     "text_value": ["Hello", None, "Goodbye"]
        ... })
        >>> mapping = process_text_data(df, DummyTextReader(""))
        >>> # The keys are rank-based integers in string form:
        >>> len(mapping.keys())
        2
    """
    text_mapping = {}

    # Filter to rows with non-null text_value
    text_df = df.filter(pl.col("text_value").is_not_null())

    # Add "modality_idx" via rank
    text_df = text_df.with_columns(
        pl.col("text_value")
        .rank("dense") 
        .cast(pl.Float32)
        .alias("modality_idx")
    )

    for row in text_df.iter_rows(named=True):
        key = f"{int(row['modality_idx'])}"
        embedding = reader.read_modality(row["text_value"])
        text_mapping[key] = embedding

    return text_mapping


def extract_statics_and_schema(df: pl.LazyFrame) -> pl.LazyFrame:
    """
    This function extracts static data and schema information.

    1) Splits into static/dynamic by time=null vs. time!=null
    2) Groups static data per subject, storing codes & numeric values as lists
    3) Groups dynamic data to find the min time (start_time) and the unique times

    Args:
        df: The input lazy DataFrame.

    Returns:
        A `pl.LazyFrame` object containing:
          - subject_id
          - code (list)
          - numeric_value (list)
          - start_time (datetime)
          - time (list of unique times)
    Examples:
        >>> from datetime import datetime
        >>> df = pl.DataFrame({
        ...     "subject_id": [1, 1, 1, 2, 2],
        ...     "time": [None, datetime(2021, 1, 1), datetime(2021, 1, 13),
        ...              None, datetime(2021, 1, 2)],
        ...     "code": [100, 101, 102, 200, 201],
        ...     "numeric_value": [1.0, 2.0, 3.0, 4.0, 5.0],
        ...     "modality_fp": [None, "path1.jpg", "path2.jpg", None, "path3.jpg"]
        ... }).lazy()
        >>> result = extract_statics_and_schema(df).collect()
        >>> result.shape
        (2, 5)
        >>> sorted(result.columns)
        ['code', 'numeric_value', 'start_time', 'subject_id', 'time']
    """
    static, dynamic = split_static_and_dynamic(df)

    static_by_subject = static.group_by("subject_id", maintain_order=True).agg(
        [pl.col("code"), pl.col("numeric_value")]
    )

    schema_by_subject = dynamic.group_by("subject_id", maintain_order=True).agg(
        pl.col("time").min().alias("start_time"),
        pl.col("time").unique(maintain_order=True)
    )

    result = static_by_subject.join(schema_by_subject, on="subject_id", how="full", coalesce=True)
    return result


def extract_seq_of_subject_events(
    df: pl.LazyFrame, reader: MultimodalReader
) -> tuple[pl.LazyFrame, dict[str, torch.Tensor]]:
    """
    Splits the data into static/dynamic. Then, if `text_value` is present, uses
    `process_text_data` to embed the text. Returns a polars LazyFrame containing
    event sequences plus a dictionary (key->embedding).

    Args:
        df: The input lazy DataFrame.
        reader: An instance of MultimodalReader for text embedding.

    Returns:
        (lazyframe_of_events, dict_of_embeddings)

    Examples:
        >>> # We'll do a small example with a dummy text reader
        >>> class DummyTextReader(MultimodalReader):
        ...     def read_modality(self, txt: str) -> torch.Tensor:
        ...         return torch.tensor([len(txt)], dtype=torch.float)
        ...
        >>> data = pl.DataFrame({
        ...     "subject_id": [1, 1, 2, 2],
        ...     "time": [None, "2021-01-01", None, "2021-01-02"],
        ...     "text_value": ["Hello", "my friend", None, "Test"],
        ...     "code": [100, 101, 200, 201],
        ...     "numeric_value": [1.0, 2.0, 4.0, 5.0]
        ... }).lazy()
        >>> seq_df, mapping = extract_seq_of_subject_events(data, DummyTextReader(""))
        >>> isinstance(seq_df, pl.LazyFrame)
        True
        >>> # mapping is only created for rows with non-null text_value in dynamic portion
        >>> len(mapping.keys())
        2
    """
    _, dynamic = split_static_and_dynamic(df)

    text_mapping = {}
    if "text_value" in df.collect_schema().names():
        # Because process_text_data expects a collected DataFrame:
        text_mapping = process_text_data(dynamic.collect(), reader)
        # Re-generate the same "modality_idx" in the dynamic lazyframe
        text_collected = dynamic.with_columns(
            pl.when(pl.col("text_value").is_not_null())
            .then(pl.col("text_value").rank("dense") - 1)
            .otherwise(None)
            .cast(pl.Float32)
            .alias("modality_idx")
        )
        dynamic = text_collected

    time_delta_days_expr = (
        pl.col("time").diff().dt.total_seconds() / SECONDS_PER_DAY
    ).cast(pl.Float32)

    result = (
        dynamic.group_by("subject_id", "time", maintain_order=True)
        .agg(
            pl.col("code").alias("code").name.keep(),
            fill_to_nans("numeric_value").alias("numeric_value").name.keep(),
            fill_to_nans("modality_idx").alias("modality_idx").name.keep() if "text_value" in df.collect_schema().names() else None,
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
    config_name=PREPROCESS_CONFIG_YAML.stem
)
def main(cfg: DictConfig):
    """
    Entry point when running this script via Hydra CLI.
    """
    hydra_loguru_init()
    tokenize(cfg)


def tokenize(cfg: DictConfig):
    """
    Tokenization pipeline that:
      - Instantiates a text reader
      - Iterates over data shards
      - Writes out schema & event sequences
      - Saves text embeddings as safetensors
    """
    logger.info(
        f"Running with config:\n{OmegaConf.to_yaml(cfg)}\n"
        f"Stage: {cfg.stage}\n\n"
        f"Stage config:\n{OmegaConf.to_yaml(cfg.stage_cfg)}"
    )

    output_dir = Path(cfg.stage_cfg.output_dir)
    # Instantiate the new text reader
    reader = BioClinicalBertTextReader(
        base_path="",
        model_name="emilyalsentzer/Bio_ClinicalBERT",
        max_length=512,
        batch_size=8,      # or from cfg
        device="cpu"       # or from cfg
    )

    shards_single_output, include_only_train = shard_iterator(cfg)

    for in_fp, out_fp in shards_single_output:
        sharded_path = out_fp.relative_to(output_dir)

        schema_out_fp = output_dir / "schemas" / sharded_path
        event_seq_out_fp = output_dir / "event_seqs" / sharded_path
        modality_out_fp = (output_dir / "modalities" / sharded_path).with_suffix(".safetensors")

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

        def write_event_seqs_and_modalities(inputs, out_fp):
            df, text_mapping = inputs
            modality_out_fp.parent.mkdir(parents=True, exist_ok=True)
            save_file(text_mapping, modality_out_fp)  # store the embeddings as safetensors
            write_lazyframe(df, out_fp)

        rwlock_wrap(
            in_fp,
            event_seq_out_fp,
            pl.scan_parquet,
            write_event_seqs_and_modalities,
            lambda df: extract_seq_of_subject_events(df, reader),
            do_overwrite=cfg.do_overwrite,
        )

    logger.info(f"Done with {cfg.stage}")


if __name__ == "__main__":  # pragma: no cover
    main()