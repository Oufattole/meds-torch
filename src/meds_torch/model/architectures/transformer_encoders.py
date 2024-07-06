"""Transformer Encoder Model.

Uses transformer encoder to get contextualized representations of all tokens and we take the CLS token
representation as the embedding.
"""

import torch
import torch.nn.functional as F
from omegaconf import DictConfig
from torch import nn


class TransformerEncoderModel(torch.nn.Module):
    """Container module with an encoder, a recurrent or transformer module, and a decoder.

    Copied from: https://github.com/pytorch/examples/blob/main/word_language_model/model.py
    """

    def __init__(self, cfg: DictConfig):
        super().__init__()
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=cfg.model.embedder.token_dim,
            nhead=cfg.model.params.nheads,
            dim_feedforward=cfg.model.params.dim_feedforward,
            dropout=cfg.model.params.dropout,
            batch_first=True,
        )
        self.model = nn.TransformerEncoder(
            encoder_layer=encoder_layer,
            num_layers=cfg.model.params.n_layers,
        )
        self.rep_token = torch.nn.Parameter(torch.randn(1, 1, cfg.model.embedder.token_dim))

    def forward(self, batch, mask):
        # Add representation token to the beginning of the sequence
        repeated_rep_token = self.rep_token.repeat(batch.shape[0], 1, 1)
        batch = torch.column_stack((repeated_rep_token, batch.transpose(1, 2)))
        mask = torch.cat((torch.ones((2, 1), dtype=torch.bool), mask), dim=1)
        # pass tokens and attention mask to the transformer
        output = self.model(batch, src_key_padding_mask=mask)
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

    def activation(self, h_t):
        """
        Step 2:
        Compute the embedding activations e_t

        In : torch.Size([batch_size, sequence_length, hidden_dimensions])
        Out: torch.Size([batch_size, sequence_length, 1])
        """
        return F.tanh(self.layer(h_t))

    def attention(self, e_t, mask):
        """
        Step 3:
        Compute the probabilities alpha_t

        In : torch.Size([batch_size, sequence_length, 1])
        Out: torch.Size([batch_size, sequence_length, 1])
        """
        alphas = self.softmax(e_t + mask.unsqueeze(-1))
        return alphas

    def context(self, alpha_t, x_t):
        """
        Step 4:
        Compute the context vector c

        In : torch.Size([batch_size, sequence_length, 1])
             torch.Size([batch_size, sequence_length, sequence_dim])
        Out: torch.Size([batch_size, 1, hidden_dimensions])
        """
        batch_size = x_t.shape[0]
        return torch.bmm(alpha_t.view(batch_size, self.out_dim, x_t.shape[1]), x_t).squeeze(dim=1)

    def forward(self, x_e, mask=None, training=True):
        """Forward pass for the Feed Forward Attention network."""
        self.training = training
        x_a = self.activation(x_e)
        alpha = self.attention(x_a, mask)
        x_c = self.context(alpha, x_e)
        # x_o = self.out(x_c)
        return x_c, alpha


class AttentionAveragedTransformerEncoderModel(torch.nn.Module):
    """Container module with an encoder, a recurrent or transformer module, and a decoder.

    Copied from: https://github.com/pytorch/examples/blob/main/word_language_model/model.py
    """

    def __init__(self, cfg: DictConfig):
        super().__init__()
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=cfg.model.embedder.token_dim,
            nhead=cfg.model.params.nheads,
            dim_feedforward=cfg.model.params.dim_feedforward,
            dropout=cfg.model.params.dropout,
            batch_first=True,
        )
        self.model = nn.TransformerEncoder(
            encoder_layer=encoder_layer,
            num_layers=cfg.model.params.n_layers,
        )
        self.decoder = AttentionAverager(cfg)

    def forward(self, batch, mask):
        output = self.model(batch.transpose(1, 2), src_key_padding_mask=mask)
        output, _ = self.decoder(output, mask)
        return output
