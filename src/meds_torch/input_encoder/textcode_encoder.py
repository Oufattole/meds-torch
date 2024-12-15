import dataclasses
import enum

import polars as pl
import torch
from mixins import TimeableMixin
from omegaconf import DictConfig
from torch import nn
from transformers import AutoModel, AutoTokenizer

from meds_torch.input_encoder import INPUT_ENCODER_MASK_KEY, INPUT_ENCODER_TOKENS_KEY
from meds_torch.utils.module_class import Module


@dataclasses.dataclass
class ModelOutput:
    rep: torch.Tensor
    hidden_states: torch.Tensor = None


class Triplet(enum.Enum):
    DATE = "date"
    VALUE = "value"
    VARIABLE = "variable"


def sequence_mask(lengths, maxlen, dtype=torch.bool):
    row_vector = torch.arange(0, maxlen, 1, device=lengths.device)
    matrix = torch.unsqueeze(lengths, dim=-1)
    mask = row_vector < matrix

    mask.type(dtype)
    return mask


class CVE(nn.Module):
    """Continuous Value Encoder (CVE) module.

    Assumes input is a single continuous value, and encodes it as an `output_dim` size embedding vector.
    """

    def __init__(self, cfg):
        super().__init__()
        self.layer = nn.Linear(1, cfg.token_dim)

    def forward(self, x):
        return self.layer(x)


def fast_unique_with_inverse(x):
    """Efficiently computes unique elements and their inverse mapping for a 2D tensor.

    The function returns a tuple containing:
    - unique: tensor of unique values in sorted order
    - inverse: tensor of same shape as input, where each element is replaced by
              its index in the unique tensor

    Args:
        x (torch.Tensor): 2D input tensor with values in range [0, 10]

    Returns:
        tuple: (unique values tensor, inverse mapping tensor)

    Example:
        >>> x = torch.tensor([[0, 1, 0],
        ...                   [2, 1, 0]], device='cpu')
        >>> unique, inverse = fast_unique_with_inverse(x)
        >>> print(unique)
        tensor([0, 1, 2])
        >>> print(inverse)
        tensor([[0, 1, 0],
                [2, 1, 0]])

        >>> # Test with repeated values
        >>> x = torch.tensor([[5, 5, 5],
        ...                   [3, 3, 5]], device='cpu')
        >>> unique, inverse = fast_unique_with_inverse(x)
        >>> print(unique)
        tensor([3, 5])
        >>> print(inverse)
        tensor([[1, 1, 1],
                [0, 0, 1]])

        >>> # Test with all possible values
        >>> x = torch.tensor([[0, 10, 5],
        ...                   [7, 3, 1]], device='cpu')
        >>> unique, inverse = fast_unique_with_inverse(x)
        >>> print(unique)
        tensor([ 0,  1,  3,  5,  7, 10])
        >>> print(inverse)
        tensor([[0, 5, 3],
                [4, 2, 1]])
    """
    # Pre-allocate an empty tensor spanning the range of possible values
    B = torch.zeros(x.max().item() + 1, device=x.device, dtype=torch.int64)

    # First mark which positions have values (with 1s)
    B.scatter_(0, x.flatten(), torch.ones_like(x.flatten()))

    # Get unique values
    unique = torch.nonzero(B).flatten()

    # Create a dense mapping (0 to num_unique-1)
    B.zero_()  # Reset B
    B.scatter_(0, unique, torch.arange(len(unique), device=x.device))

    # Get inverse mapping using the dense indices
    inverse = B[x.flatten()]

    return unique, inverse.reshape(x.shape)


class TextCodeEmbedder(nn.Module, Module, TimeableMixin):
    def __init__(self, cfg: DictConfig):
        super().__init__()
        self.cfg = cfg

        # Build tokens map first
        token_map = self.build_code_to_tokens_map()

        # Initialize models
        self.code_embedder = AutoModel.from_pretrained(self.cfg.code_embedder)
        self.linear = nn.Linear(self.code_embedder.config.hidden_size, self.cfg.token_dim)

        # Register each tensor as a buffer
        self.key_to_buffer = {}
        for key, tensor in token_map.items():
            buffer_name = f"tokens_{key}"
            self.register_buffer(buffer_name, tensor)
            self.key_to_buffer[key] = buffer_name

    @TimeableMixin.TimeAs
    def build_code_to_tokens_map(self):
        """
        Builds a mapping from code to tokens

        Returns:
            code_to_tokens_map: A dictionary mapping from code to tokens
        """
        code_metadata = pl.scan_parquet(self.cfg.code_metadata_fp).select(
            ["code", "code/vocab_index", "description"]
        )
        code_metadata = code_metadata.sort("code/vocab_index").collect()
        # Process code names
        code_metadata = code_metadata.with_columns(
            pl.col("code").fill_null("").str.replace_all("//", " ").str.replace_all("_", " ")
        )
        # Merge code names into description when the description is missing
        code_metadata = code_metadata.with_columns(
            [
                pl.when(pl.col("description").is_null())
                .then(pl.col("code"))
                .otherwise(pl.col("description"))
                .alias("description")
            ]
        )

        # check that there is no 0 -- this should be reserved for the padding token
        assert (
            code_metadata.select(["code/vocab_index"]).min().item() == 1
        ), "Vocab index should start from 1."
        # check that there is no missing index
        assert (
            code_metadata.select(["code/vocab_index"]).max().item() == code_metadata.shape[0]
        ), "Vocab index should be continuous."

        tokenizer = AutoTokenizer.from_pretrained(self.cfg.code_tokenizer)
        tokenized_code_metadata = tokenizer(
            ["[PAD]"] + code_metadata.select(["description"]).fill_null("").to_series().to_list(),
            **self.cfg.tokenizer_config,
        )
        return tokenized_code_metadata

    @TimeableMixin.TimeAs
    def forward(self, codes, mask):
        with torch.no_grad():
            unique_codes, inverse_indices = fast_unique_with_inverse(codes)

        # Access the tensors through their registered buffer names
        embedder_inputs = {
            key: getattr(self, self.key_to_buffer[key])[unique_codes] for key in self.key_to_buffer.keys()
        }

        code_embeddings = self.code_embedder(**embedder_inputs).pooler_output
        code_embeddings = self.linear(code_embeddings)
        embeddings = code_embeddings[inverse_indices]

        return torch.zeros_like(embeddings)


class TextCodeEncoder(nn.Module, Module, TimeableMixin):
    """Container module with an encoder, a recurrent or transformer module, and a decoder.

    Copied from: https://github.com/pytorch/examples/blob/main/word_language_model/model.py
    """

    def __init__(self, cfg: DictConfig):
        super().__init__()
        self.cfg = cfg
        # Define Triplet Embedders
        self.date_embedder = CVE(cfg)
        self.code_embedder = TextCodeEmbedder(cfg)
        self.numeric_value_embedder = CVE(cfg)

    @TimeableMixin.TimeAs
    def embed_func(self, embedder, x):
        out = embedder.forward(x[None, :].transpose(2, 0)).permute(1, 2, 0)
        return out

    @TimeableMixin.TimeAs
    def get_embedding(self, batch):
        static_mask = batch["static_mask"]
        code = batch["code"]
        code_mask = batch["mask"]
        numeric_value = batch["numeric_value"]
        time_delta_days = batch["time_delta_days"]
        numeric_value_mask = batch["numeric_value_mask"]
        # Embed times and mask static value times
        time_emb = self.embed_func(self.date_embedder, time_delta_days) * ~static_mask.unsqueeze(dim=1)
        # Embed codes
        code_emb = self.code_embedder.forward(code, code_mask)
        code_emb = code_emb.permute(0, 2, 1)

        # Embed numerical values and mask nan values
        val_emb = self.embed_func(self.numeric_value_embedder, numeric_value) * numeric_value_mask.unsqueeze(
            dim=1
        )

        # Sum the (time, code, value) triplets and
        embedding = time_emb + code_emb + val_emb

        assert embedding.isfinite().all(), "Embedding is not finite"
        if embedding.shape[-1] > self.cfg.max_seq_len:
            raise ValueError(
                f"Triplet embedding length {embedding.shape[-1]} "
                "is greater than max_seq_len {self.cfg.max_seq_len}"
            )
        return embedding.transpose(1, 2)

    @TimeableMixin.TimeAs
    def forward(self, batch):
        embedding = self.get_embedding(batch)
        batch[INPUT_ENCODER_MASK_KEY] = batch["mask"]
        batch[INPUT_ENCODER_TOKENS_KEY] = embedding
        return batch
