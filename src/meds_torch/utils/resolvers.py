"""This file contains all omegaconf yaml resolvers used in the project.

Make sure to add ```python from meds_torch.utils.resolvers import setup_resolvers setup_resolvers() ``` to the
top of any hydra script for this codebase to use the resolvers in this file.
"""
import polars as pl
from omegaconf import OmegaConf
from transformers import AutoTokenizer


def get_vocab_size(code_metadata_fp, num_special_tokens):
    vocab_size = pl.scan_parquet(code_metadata_fp).select("code/vocab_index").max().collect().item() + 1
    vocab_size += num_special_tokens
    return vocab_size


def get_eos_token_id(vocab_size, eos_offset):
    return vocab_size - eos_offset


def get_text_vocab_size(tokenizer_name):
    return AutoTokenizer.from_pretrained(tokenizer_name).vocab_size


def resolve_max_seq_len(max_seq_len, prepend_eos):
    return max_seq_len + int(prepend_eos)


def setup_resolvers():
    OmegaConf.register_new_resolver(
        "get_vocab_size",
        get_vocab_size,
        replace=True,
    )
    OmegaConf.register_new_resolver(
        "get_eos_token_id",
        get_eos_token_id,
        replace=True,
    )
    OmegaConf.register_new_resolver(
        "get_text_vocab_size",
        get_text_vocab_size,
        replace=True,
    )
    OmegaConf.register_new_resolver(
        "resolve_max_seq_len",
        resolve_max_seq_len,
        replace=True,
    )
