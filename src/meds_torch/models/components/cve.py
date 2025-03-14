from torch import nn


class CVE(nn.Module):
    """
    Continuous Value Encoder (CVE) module.

    This module takes a single continuous value as input (e.g., numerical data or time deltas)
    and encodes it into a fixed-size embedding vector of dimension token_dim using a linear transformation.

    Args:
        cfg (DictConfig): Configuration object containing the following parameters:
            - token_dim (int): Dimensionality of the embeddings.

    Forward Pass:
        Input:
            x (torch.Tensor): A tensor of shape (batch_size, 1) containing the continuous values to encode.
        Output:
            torch.Tensor: A tensor of shape (batch_size, token_dim) representing the encoded embeddings.

    Example:
        >>> from omegaconf import OmegaConf
        >>> import torch
        >>> cfg = OmegaConf.create({"token_dim": 128})
        >>> cve = CVE(cfg)
        >>> x = torch.tensor([[1.0], [2.0], [3.0]])  # Batch of 3 samples
        >>> cve(x).shape
        torch.Size([3, 128])
    """

    def __init__(self, cfg):
        super().__init__()
        self.layer = nn.Linear(1, cfg.token_dim)

    def forward(self, x):
        return self.layer(x)


if __name__ == "__main__":
    import doctest

    doctest.testmod(verbose=True)
