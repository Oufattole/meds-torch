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


@dataclass
class TimeToEventLabeler:
    """
    Initialize a time to event labeler.

    Args:
        target_codes: List of codes we're looking for in the sequence
        time_length: The maximum time window to look for the event
        min_time: Optional minimum time that must pass before events are considered valid
        numeric_value_min: Optional minimum value for numeric criteria
        numeric_value_max: Optional maximum value for numeric criteria

    Examples:
        # >>> # Case 1: Testing unknown_pred logic
        # >>> time = torch.tensor([
        # ...     [0.0, 2.0, 5.0, 12.0],  # Has event at day 5
        # ...     [0.0, 2.0, 3.0, 20.0],  # No event, but seen full window
        # ...     [0.0, 2.0, 4.0, 10.0],  # No event, haven't seen full window yet
        # ... ])
        # >>> code = torch.tensor([
        # ...     [1, 2, 100, 3],    # Has target code
        # ...     [1, 2, 3, 4],      # No target codes, full window seen
        # ...     [1, 2, 3, 4],      # No target codes, partial window
        # ... ])
        # >>> mask = torch.ones_like(time, dtype=torch.bool)
        # >>> numeric_value = torch.ones_like(time)
        # >>> numeric_value_mask = torch.ones_like(time, dtype=torch.bool)
        # >>> batch = TrajectoryBatch(time, code, mask, numeric_value, numeric_value_mask)
        # >>> labeler = TimeToEventLabeler(target_codes=[100], time_length=15.0)
        # >>> labels, unknown = labeler(batch)
        # >>> print(labels.tolist())  # Only first sequence has event
        # [[1.0], [0.0], [0.0]]
        # >>> print(unknown.tolist())  # Only third sequence is unknown (hasn't seen full window)
        # [False, False, True]

        >>> # Case 2: Complex case with multiple codes and criteria
        >>> time = torch.tensor([
        ...     [0.0, 3.0, 5.0, 35.0],   # Has valid event at day 5
        ...     [0.0, 2.0, 6.0, 35.0],   # No valid events, full window seen
        ...     [0.0, 2.0, 4.0, 10.0],   # No valid events, partial window
        ...     [0.0, 35.0, 40.0, 45.0], # Events outside time window
        ... ])
        >>> code = torch.tensor([
        ...     [1, 2, 100, 101],    # Has first target code
        ...     [100, 2, 3, 4],      # Has early target code
        ...     [1, 2, 3, 4],        # No target codes
        ...     [1, 100, 101, 2],    # Target codes but too late
        ... ])
        >>> mask = torch.ones_like(time, dtype=torch.bool)
        >>> numeric_value = torch.tensor([
        ...     [1.0, 1.0, 7.5, 8.0],  # Valid values
        ...     [6.0, 1.0, 4.5, 4.0],  # Valid values
        ...     [1.0, 1.0, 4.0, 4.0],  # Values don't matter (no target codes)
        ...     [1.0, 7.0, 8.0, 1.0],  # Valid values but events too late
        ... ])
        >>> numeric_value_mask = torch.ones_like(time, dtype=torch.bool)
        >>> batch = TrajectoryBatch(time, code, mask, numeric_value, numeric_value_mask)
        >>> labeler = TimeToEventLabeler(target_codes=[100, 101], time_length=30.0, min_time=5.0,
        ...                             numeric_value_min=5.0, numeric_value_max=10.0)
        >>> labels, unknown = labeler(batch)
        >>> print(labels.tolist())  # Only first sequence has valid event
        [[1.0], [0.0], [0.0], [0.0]]
        >>> print(unknown.tolist())  # Only third sequence is unknown
        [False, False, True, False]
    """

    target_codes: list[int]
    time_length: float
    min_time: float = 0.0
    numeric_value_min: float = None
    numeric_value_max: float = None
    include_min_time: bool = True
    include_max_time: bool = True

    def __call__(self, trajectory_batch: TrajectoryBatch) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Label sequences for time to event prediction.

        Args:
            trajectory (Trajectory): The trajectory data

        Returns:
            tuple of:
                pred_labels: Tensor of shape (batch_size, 1) containing binary labels
                unknown_pred: Tensor of shape (batch_size,) indicating if no prediction could be made
                            Only True if no event found AND max time < time_length
        """
        time = trajectory_batch.time
        code = trajectory_batch.code
        mask = trajectory_batch.mask
        numeric_value = trajectory_batch.numeric_value
        numeric_value_mask = trajectory_batch.numeric_value_mask

        # Find where any target code appears and is valid
        is_target_code = torch.zeros_like(code, dtype=torch.bool)
        for target_code in self.target_codes:
            is_target_code = is_target_code | (code == target_code)
        is_target_code = is_target_code & mask

        # Apply time window constraint
        in_time_window = time == torch.clamp(time, self.min_time, self.time_length)
        if not self.include_min_time:
            in_time_window = in_time_window & (time != self.min_time)
        if not self.include_max_time:
            in_time_window = in_time_window & (time != self.time_length)
        valid_events = is_target_code & in_time_window

        # Apply numeric constraints if specified
        if self.numeric_value_min is not None or self.numeric_value_max is not None:
            numeric_criteria = torch.ones_like(valid_events, dtype=torch.bool)

            if self.numeric_value_min is not None:
                numeric_criteria = numeric_criteria & (numeric_value >= self.numeric_value_min)

            if self.numeric_value_max is not None:
                numeric_criteria = numeric_criteria & (numeric_value <= self.numeric_value_max)

            valid_events = valid_events & numeric_criteria & numeric_value_mask

        # Find first valid event for each sequence
        has_event = valid_events.any(dim=1)

        # Create prediction labels
        pred_labels = has_event.unsqueeze(1).float()

        # Mark sequences as unknown only if:
        # 1. No valid events found AND
        # 2. Haven't seen the full time window yet (max time < time_length)
        max_times = torch.amax(time * mask.float(), dim=1)  # Use mask to ignore invalid times
        window_incomplete = max_times < self.time_length
        unknown_pred = ~has_event & window_incomplete

        return pred_labels, unknown_pred
