import torch
from omegaconf import DictConfig
from torch import nn
from x_transformers import AutoregressiveWrapper

from meds_torch.input_encoder import INPUT_ENCODER_MASK_KEY, INPUT_ENCODER_TOKENS_KEY
from meds_torch.models import BACKBONE_EMBEDDINGS_KEY, BACKBONE_TOKENS_KEY
from meds_torch.models.components.utils import get_last_token
from meds_torch.utils.module_class import Module


class GenerationBudget:
    """A class to handle generation budgets with mutually exclusive sequence length or time length.

    This class ensures that only one of max_seq_len or min_time_len can be set.

    Examples:
        >>> # Create from sequence length
        >>> budget_seq = GenerationBudget.from_seq_len(100)
        >>> budget_seq.max_seq_len
        100
        >>> budget_seq.min_time_len is None
        True

        >>> # Create from time length
        >>> budget_time = GenerationBudget.from_time_len(60)
        >>> budget_time.min_time_len
        60
        >>> budget_time.max_seq_len is None
        True

        >>> # Values remain consistent
        >>> budget_seq = GenerationBudget.from_seq_len(500)
        >>> (budget_seq.max_seq_len, budget_seq.min_time_len)
        (500, None)

        >>> # Direct initialization isn't intended for use
        >>> budget = GenerationBudget(100)
    """

    def __init__(self, value: int):
        self._value = value

    @classmethod
    def from_seq_len(cls, value: int) -> "GenerationBudget":
        """Create a GenerationBudget from a maximum sequence length.

        Args:
            value: The maximum sequence length

        Returns:
            A GenerationBudget instance with max_seq_len set
        """
        instance = cls(value)
        instance._is_seq_len = True
        return instance

    @classmethod
    def from_time_len(cls, value: int) -> "GenerationBudget":
        """Create a GenerationBudget from a minimum time length.

        Args:
            value: The minimum time length

        Returns:
            A GenerationBudget instance with min_time_len set
        """
        instance = cls(value)
        return instance

    @property
    def max_seq_len(self) -> int | None:
        """The maximum sequence length, if this budget was created with from_seq_len."""
        return self._value if hasattr(self, "_is_seq_len") else None

    @property
    def min_time_len(self) -> int | None:
        """The minimum time length, if this budget was created with from_time_len."""
        return self._value if not hasattr(self, "_is_seq_len") else None


class TransformerDecoderModel(torch.nn.Module, Module):
    """Wrapper of Decoder Transformer for use in MEDS with triplet token embeddings."""

    def __init__(self, cfg: DictConfig):
        super().__init__()
        self.cfg = cfg
        self.model = cfg.model

        if cfg.token_emb:
            self.model.token_emb = cfg.token_emb

        # Setup LM Heads
        self.value_head = nn.Linear(cfg.token_dim, 1, bias=False)
        self.code_head = nn.Linear(cfg.token_dim, cfg.vocab_size, bias=False)
        self.time_head = nn.Linear(cfg.token_dim, 1, bias=False)

    def forward(self, batch):
        input_data, mask = batch[INPUT_ENCODER_TOKENS_KEY], batch[INPUT_ENCODER_MASK_KEY]
        output, embeddings = self.model(input_data, mask=mask, return_logits_and_embeddings=True)
        if self.cfg.get_last_token:
            embeddings = get_last_token(embeddings, ~mask)
        batch[BACKBONE_TOKENS_KEY] = output
        batch[BACKBONE_EMBEDDINGS_KEY] = embeddings
        return batch

    def generate(self, prompts, prompt_lengths, mask, **kwargs) -> torch.Tensor:
        """Generates autoregressive sequence continuations from input prompts.

        This method wraps the model with an AutoregressiveWrapper and generates sequences
        using the configured parameters. The output excludes the original prompt tokens.

        Args:
            prompts (torch.Tensor): Input token sequences to generate from
            prompt_lengths (torch.Tensor): Length of each prompt sequence, entries after the length are
                padding
            mask (torch.Tensor): Attention mask for the prompts
            **kwargs: Additional keyword arguments passed to the autoregressive generator

        Returns:
            torch.Tensor: Generated token sequences, excluding the prompt tokens.
                Shape is [batch_size, generated_length]

        Note:
            The generation uses the following configured parameters:
            - max_seq_len: Maximum sequence length for generation
            - temperature: Sampling temperature
            - eos_token_id: End of sequence token ID
        """
        model = AutoregressiveWrapper(self.model)
        out = model.generate(
            prompts,
            self.cfg.max_seq_len,
            prompt_lens=prompt_lengths,
            temperature=self.cfg.temperature,
            eos_token=self.cfg.eos_token_id,
            context_mask=mask,
            **kwargs,
        )[
            :, prompts.shape[1] :
        ]  # Remove the prompt
        return out
