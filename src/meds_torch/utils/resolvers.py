"""This file contains all omegaconf yaml resolvers used in the project.

Make sure to add ```python from meds_torch.utils.resolvers import setup_resolvers setup_resolvers() ``` to the
top of any hydra script for this codebase to use the resolvers in this file.
"""
import polars as pl
from omegaconf import OmegaConf


def get_vocab_size(code_metadata_fp, postpend_token, column):
    vocab_size = pl.scan_parquet(code_metadata_fp).select(column).max().collect().item() + 1
    vocab_size += int(postpend_token != "none")
    return vocab_size


def add_two(a):
    return a + 2


def add(a, b):
    return a + b


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


def get_subvocab_index(augmented_code_metadata_fp, token):
    metadata_df = pl.read_parquet(augmented_code_metadata_fp)
    return metadata_df.filter(pl.col("code") == token)["code/subvocab_index"][-1]


def get_vocab_index(augmented_code_metadata_fp, token):
    metadata_df = pl.read_parquet(augmented_code_metadata_fp)
    return metadata_df.filter(pl.col("code") == token)["code/vocab_index"][-1]


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
        "add_two",
        add_two,
        replace=True,
    )
    OmegaConf.register_new_resolver(
        "int_prod",
        int_prod,
        replace=True,
    )

    OmegaConf.register_new_resolver(
        "get_subvocab_index",
        get_subvocab_index,
        replace=True,
    )

    OmegaConf.register_new_resolver(
        "get_vocab_index",
        get_vocab_index,
        replace=True,
    )

    OmegaConf.register_new_resolver(
        "add",
        add,
        replace=True,
    )
