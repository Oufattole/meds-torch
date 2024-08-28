"""Transformer Encoder Model.

Uses transformer encoder to get contextualized representations of all tokens and we take the CLS token
representation as the embedding.
"""

import torch
import torch.nn.functional as F
from omegaconf import DictConfig
from torch import nn
from x_transformers import Encoder, TransformerWrapper

from meds_torch.input_encoder import INPUT_ENCODER_MASK_KEY, INPUT_ENCODER_TOKENS_KEY
from meds_torch.models import BACKBONE_EMBEDDINGS_KEY, BACKBONE_TOKENS_KEY
from meds_torch.utils.module_class import Module


class AttentionAverager(torch.nn.Module):
    """Source paper: https://arxiv.org/pdf/1512.08756.pdf `FFAttention` is the Base Class for the Feed-Forward
    Attention Network. It is implemented as an abstract subclass of a PyTorch Module. You can then subclass
    this to create an architecture adapted to your problem.

    The FeedForward mechanism is implemented in five steps, three of which have to be implemented in your
    custom subclass:

    1. `embedding` (NotImplementedError) 2. `activation` (NotImplementedError) 3. `attention` (Already
    implemented) 4. `context` (Already implemented) 5. `out` (NotImplementedError)

    Attributes:     batch_size (int): The batch size, used for resizing the tensors.     T (int): The length
    of the sequence.     D_in (int): The dimension of each element of the sequence.     D_out (int): The
    dimension of the desired predicted quantity.     hidden (int): The dimension of the hidden state.
    batch_size=args.batch_size, T=args.seq_len, D_in=args.embed_size, D_out=, hidden=None
    """

    def __init__(self, cfg):
        super().__init__()
        # Net Config
        self.n_features = cfg.token_dim
        self.out_dim = 1
        self.layer = torch.nn.Linear(cfg.token_dim, self.out_dim)
        self.softmax = torch.nn.Softmax(dim=1)

    def forward(self, x, mask=None):
        """Forward pass for the Feed Forward Attention network."""
        # Compute the embedding activations
        x_a = F.tanh(self.layer(x))
        # Compute the probabilities alpha
        alpha = self.softmax(x_a + mask.unsqueeze(-1))
        # Compute the context vector c
        output = torch.bmm(alpha.view(x.shape[0], self.out_dim, x.shape[1]), x).squeeze(dim=1)
        return output, alpha


class AttentionAveragedTransformerEncoderModel(torch.nn.Module, Module):
    """Wrapper of Encoder Transformer for use in MEDS with triplet token embeddings.

    Attention averaging is used to get an embedding from tokens.
    """

    def __init__(self, cfg: DictConfig):
        super().__init__()
        dropout = cfg.dropout
        self.model = TransformerWrapper(
            num_tokens=cfg.token_dim,  # placeholder as this is not used
            max_seq_len=cfg.max_seq_len,
            emb_dropout=dropout,
            use_abs_pos_emb=(cfg.pos_encoding == "absolute_sinusoidal"),
            attn_layers=Encoder(
                dim=cfg.token_dim,
                depth=cfg.n_layers,
                heads=cfg.nheads,
                layer_dropout=dropout,  # stochastic depth - dropout entire layer
                attn_dropout=dropout,  # dropout post-attention
                ff_dropout=dropout,  # feedforward dropout
            ),
        )
        if cfg.pos_encoding != "absolute_sinusoidal" and cfg.pos_encoding is not None:
            raise ValueError(f"Unknown positional encoding: {cfg.pos_encoding}")
        if not cfg.use_xtransformers_token_emb:
            self.model.token_emb = nn.Identity()
        self.decoder = AttentionAverager(cfg)

    def forward(self, batch):
        input_data, mask = batch[INPUT_ENCODER_TOKENS_KEY], batch[INPUT_ENCODER_MASK_KEY]
        output = self.model(input_data.transpose(1, 2), mask=mask)
        batch[BACKBONE_TOKENS_KEY] = output
        output, _ = self.decoder(output, mask)
        batch[BACKBONE_EMBEDDINGS_KEY] = output
        return batch
