import torch


def get_last_token(output: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Get the last non-masked token from the output tensor.

    Args: output (torch.Tensor): The output tensor of shape (batch_size, seq_len, hidden_dim) mask
    (torch.Tensor): The mask tensor of shape (batch_size, seq_len) where True indicates masked positions

    Returns: torch.Tensor: The last non-masked token for each sequence in the batch
    """
    # Find the last non-masked position
    last_non_masked = (~mask).float().cumsum(dim=1).argmax(dim=1)

    # Handle cases where all positions are masked
    all_masked = mask.all(dim=1)
    last_non_masked[all_masked] = 0

    if all_masked.any():
        raise ValueError(
            f"{all_masked.sum().item()} sequences have all positions masked. Mask should likely be negated"
        )

    # Create indices for gathering
    batch_size, _, hidden_dim = output.shape
    indices = last_non_masked.view(batch_size, 1, 1).expand(-1, 1, hidden_dim)

    # Gather the last non-masked tokens
    last_token = torch.gather(output, dim=1, index=indices).squeeze(1)

    return last_token
