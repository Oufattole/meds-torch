from dataclasses import dataclass

import torch


@dataclass
class TrajectoryBatch:
    """
    Initialize a batch of trajectories.

    Args:
        time (torch.Tensor): Tensor of shape (batch_size, sequence_length) containing days after prediction
            time. Values must be monotonically increasing within each sequence.
        code (torch.Tensor): Tensor of shape (batch_size, sequence_length) containing event codes
        mask (torch.Tensor): Tensor of shape (batch_size, sequence_length) indicates valid measurements/codes
        numeric_value (torch.Tensor): Tensor of shape (batch_size, sequence_length) containing numeric values
        numeric_value_mask (torch.Tensor): Tensor of shape (batch_size, sequence_length) indicating valid
            numeric values
    """

    time: torch.Tensor
    code: torch.Tensor
    mask: torch.Tensor
    numeric_value: torch.Tensor
    numeric_value_mask: torch.Tensor
