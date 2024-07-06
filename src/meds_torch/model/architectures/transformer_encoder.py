"""Transformer Encoder Model.

Uses transformer encoder to get contextualized representations of all tokens and we take the CLS token
representation as the embedding.
"""

import torch
import torch.nn.functional as F
from omegaconf import DictConfig
from torch import nn
from x_transformers import Encoder, TransformerWrapper


class TransformerEncoderModel(torch.nn.Module):
    """Wrapper of Encoder Transformer for use in MEDS with triplet token embeddings."""

    def __init__(self, cfg: DictConfig):
        super().__init__()
        dropout = cfg.model.params.dropout
        self.model = TransformerWrapper(
            num_tokens=cfg.model.embedder.token_dim,  # placeholder as this is not used
            max_seq_len=cfg.model.embedder.max_seq_len,
            emb_dropout=dropout,
            use_abs_pos_emb=False,
            attn_layers=Encoder(
                dim=cfg.model.embedder.token_dim,
                depth=cfg.model.params.n_layers,
                heads=cfg.model.params.nheads,
                layer_dropout=dropout,  # stochastic depth - dropout entire layer
                attn_dropout=dropout,  # dropout post-attention
                ff_dropout=dropout,  # feedforward dropout
            ),
        )
        self.model.token_emb = nn.Identity()
        self.rep_token = torch.nn.Parameter(torch.randn(1, 1, cfg.model.embedder.token_dim))

    def forward(self, batch, mask):
        # Add representation token to the beginning of the sequence
        repeated_rep_token = self.rep_token.repeat(batch.shape[0], 1, 1)
        batch = torch.column_stack((repeated_rep_token, batch.transpose(1, 2)))
        mask = torch.cat((torch.ones((2, 1), dtype=torch.bool), mask), dim=1)
        # pass tokens and attention mask to the transformer
        output = self.model(batch, mask=mask)
        # extract the representation token's embedding
        output = output[:, 0, :]
        return output


class AttentionAverager(torch.nn.Module):
    """
    Source paper: https://arxiv.org/pdf/1512.08756.pdf
    `FFAttention` is the Base Class for the Feed-Forward Attention Network.
    It is implemented as an abstract subclass of a PyTorch Module. You can
    then subclass this to create an architecture adapted to your problem.

    The FeedForward mechanism is implemented in five steps, three of
    which have to be implemented in your custom subclass:

    1. `embedding` (NotImplementedError)
    2. `activation` (NotImplementedError)
    3. `attention` (Already implemented)
    4. `context` (Already implemented)
    5. `out` (NotImplementedError)

    Attributes:
        batch_size (int): The batch size, used for resizing the tensors.
        T (int): The length of the sequence.
        D_in (int): The dimension of each element of the sequence.
        D_out (int): The dimension of the desired predicted quantity.
        hidden (int): The dimension of the hidden state.
        batch_size=args.batch_size, T=args.seq_len, D_in=args.embed_size, D_out=, hidden=None
    """

    def __init__(self, cfg):
        super().__init__()
        # Net Config
        self.n_features = cfg.model.embedder.token_dim
        self.out_dim = 1
        self.layer = torch.nn.Linear(cfg.model.embedder.token_dim, self.out_dim)
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


class AttentionAveragedTransformerEncoderModel(torch.nn.Module):
    """Wrapper of Encoder Transformer for use in MEDS with triplet token embeddings.

    Attention averaging is used to get an embedding from tokens.
    """

    def __init__(self, cfg: DictConfig):
        super().__init__()
        dropout = cfg.model.params.dropout
        self.model = TransformerWrapper(
            num_tokens=cfg.model.embedder.token_dim,  # placeholder as this is not used
            max_seq_len=cfg.model.embedder.max_seq_len,
            emb_dropout=dropout,
            use_abs_pos_emb=False,
            attn_layers=Encoder(
                dim=cfg.model.embedder.token_dim,
                depth=cfg.model.params.n_layers,
                heads=cfg.model.params.nheads,
                layer_dropout=dropout,  # stochastic depth - dropout entire layer
                attn_dropout=dropout,  # dropout post-attention
                ff_dropout=dropout,  # feedforward dropout
            ),
        )
        self.model.token_emb = nn.Identity()
        self.decoder = AttentionAverager(cfg)

    def forward(self, batch, mask):
        output = self.model(batch.transpose(1, 2), mask=mask)
        output, _ = self.decoder(output, mask)
        return output
