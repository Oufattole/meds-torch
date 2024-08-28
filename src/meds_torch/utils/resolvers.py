"""This file contains all omegaconf yaml resolvers used in the project.

Make sure to add ```python from meds_torch.utils.resolvers import setup_resolvers setup_resolvers() ``` to the
top of any hydra script for this codebase to use the resolvers in this file.
"""
import polars as pl
from omegaconf import OmegaConf


def get_vocab_size(code_metadata_fp, num_value_code_quantiles):
    if not num_value_code_quantiles:
        num_value_code_quantiles = 1
    num_codes = pl.scan_parquet(code_metadata_fp).select("code/vocab_index").max().collect().item() + 1
    vocab_size = num_codes * num_value_code_quantiles
    return vocab_size


def setup_resolvers():
    OmegaConf.register_new_resolver(
        "get_vocab_size",
        get_vocab_size,
        replace=True,
    )
