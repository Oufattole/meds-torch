from dataclasses import dataclass
from datetime import datetime

import numpy as np
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


def get_time_days_delta(
    pred_times: list[datetime], input_end_times: list[datetime], device: torch.device = torch.device("cpu")
) -> torch.Tensor:
    """
    Convert lists of prediction times and generation start times to the continuous number
    of days (including fractional days) between each pair.

    Args:
        pred_times (List[datetime]): List of prediction times
        input_end_times (List[datetime]): List of generation start times, one for each prediction time
        device (torch.device, optional): Device to place the output tensor on. Defaults to CPU.

    Returns:
        torch.Tensor: Tensor of shape (len(pred_times),) containing the number of days (including
            fractional parts) after each generation start time for each prediction time.

    Examples:
        >>> from datetime import datetime
        >>> import torch
        >>> pred_times = [
        ...     datetime(2022, 1, 1, 12, 0),  # Noon on Jan 1
        ...     datetime(2022, 1, 2, 0, 0)    # Midnight on Jan 2
        ... ]
        >>> end_times = [
        ...     datetime(2022, 1, 1, 0, 0),   # Midnight on Jan 1
        ...     datetime(2022, 1, 1, 0, 0)    # Midnight on Jan 1
        ... ]
        >>> get_time_days_delta(pred_times, end_times)
        tensor([0.5000, 1.0000])

        >>> pred_times = [
        ...     datetime(2022, 1, 1, 6, 0),   # 6 AM on Jan 1
        ...     datetime(2022, 1, 1, 18, 0)   # 6 PM on Jan 1
        ... ]
        >>> end_times = [
        ...     datetime(2022, 1, 1, 0, 0),   # Midnight on Jan 1
        ...     datetime(2022, 1, 1, 12, 0)   # Noon on Jan 1
        ... ]
        >>> get_time_days_delta(pred_times, end_times)
        tensor([0.2500, 0.2500])
    """
    if len(pred_times) != len(input_end_times):
        raise ValueError(
            f"Length mismatch: pred_times has {len(pred_times)} elements "
            f"but input_end_times has {len(input_end_times)} elements"
        )

    # Convert datetime lists to numpy arrays of timestamps
    pred_timestamps = np.array([dt.timestamp() for dt in pred_times])
    end_timestamps = np.array([dt.timestamp() for dt in input_end_times])

    # Calculate time differences in seconds
    time_deltas_seconds = pred_timestamps - end_timestamps

    # Convert to days (86400 seconds per day)
    days_delta = time_deltas_seconds / 86400

    return torch.tensor(days_delta, device=device, dtype=torch.float32)
