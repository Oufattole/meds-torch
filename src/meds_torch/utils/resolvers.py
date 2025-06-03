"""This file contains all omegaconf yaml resolvers used in the project.

Make sure to add ```python from meds_torch.utils.resolvers import setup_resolvers setup_resolvers() ``` to the
top of any hydra script for this codebase to use the resolvers in this file.
"""
import polars as pl
from omegaconf import OmegaConf


def get_vocab_size(code_metadata_fp, num_special_tokens):
    vocab_size = pl.scan_parquet(code_metadata_fp).select("code/vocab_index").max().collect().item() + 1
    vocab_size += num_special_tokens
    return vocab_size


def get_eos_token_id(vocab_size, eos_offset):
    return vocab_size - eos_offset


def int_prod(x: int, y: int) -> int:
    """Returns the closest integer to the product of x and y (available as an OmegaConf resolver).

    Examples:
        >>> int_prod(2, 3)
        6
        >>> int_prod(2, 3.5)
        7
        >>> int_prod(2.49, 3)
        7
    """
    return round(x * y)

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
        "int_prod",
        int_prod,
        replace=True,
    )
