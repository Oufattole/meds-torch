from dataclasses import dataclass
from datetime import datetime

import numpy as np
import polars as pl
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
        time_scale: scale of the time, by default it is 'Y' (years). Any numpy datetime units can be used,
            see https://numpy.org/doc/2.1/reference/arrays.datetime.html#datetime-units
    """

    time: torch.Tensor
    code: torch.Tensor
    mask: torch.Tensor
    numeric_value: torch.Tensor
    numeric_value_mask: torch.Tensor
    time_scale: str = "Y"

    def to_meds(self, prediction_time: list[datetime], subject_id: list[str | int]) -> pl.DataFrame:
        """Convert the trajectory batch to MEDS format.

        Args:
            prediction_time: List of prediction times for each trajectory in the batch
            subject_id: List of subject IDs for each trajectory in the batch

        Returns:
            pl.DataFrame: MEDS format DataFrame with columns:
                - time: Absolute timestamp of the event
                - code: The medical code
                - numeric_value: The numeric value associated with the code (if any)
                - subject_id: ID of the subject
                - prediction_time: The prediction time for this trajectory

        Example:
        >>> batch = TrajectoryBatch(
        ...     time=torch.tensor([[0, .5, 2], [0, 3, 5]]),
        ...     code=torch.tensor([[1, 2, 3], [4, 5, 6]]),
        ...     mask=torch.tensor([[1, 1, 1], [1, 1, 0]]),
        ...     numeric_value=torch.tensor([[0.5, 1.0, 1.5], [2.0, 2.5, 3.0]]),
        ...     numeric_value_mask=torch.tensor([[1, 1, 0], [1, 0, 0]])
        ... )
        >>> prediction_times = [datetime(2024, 1, 1), datetime(2024, 1, 1)]
        >>> subject_ids = [1, 2]
        >>> df = batch.to_meds(prediction_times, subject_ids)
        >>> df.sort("subject_id")
        shape: (5, 5)
        ┌─────────────────────┬──────┬───────────────┬────────────┬─────────────────────┐
        │ time                ┆ code ┆ numeric_value ┆ subject_id ┆ prediction_time     │
        │ ---                 ┆ ---  ┆ ---           ┆ ---        ┆ ---                 │
        │ datetime[ns]        ┆ i32  ┆ f32           ┆ i32        ┆ datetime[ns]        │
        ╞═════════════════════╪══════╪═══════════════╪════════════╪═════════════════════╡
        │ 2024-01-01 00:00:00 ┆ 1    ┆ 0.5           ┆ 1          ┆ 2024-01-01 00:00:00 │
        │ 2024-07-01 14:54:36 ┆ 2    ┆ 1.0           ┆ 1          ┆ 2024-01-01 00:00:00 │
        │ 2025-12-31 11:38:24 ┆ 3    ┆ NaN           ┆ 1          ┆ 2024-01-01 00:00:00 │
        │ 2024-01-01 00:00:00 ┆ 4    ┆ 2.0           ┆ 2          ┆ 2024-01-01 00:00:00 │
        │ 2026-12-31 17:27:36 ┆ 5    ┆ NaN           ┆ 2          ┆ 2024-01-01 00:00:00 │
        └─────────────────────┴──────┴───────────────┴────────────┴─────────────────────┘
        """
        if len(prediction_time) != len(subject_id) or len(prediction_time) != self.time.shape[0]:
            raise ValueError("Number of prediction times and subject IDs must match batch size")
        schema = {
            "time": pl.Datetime,
            "code": pl.Int32,
            "numeric_value": pl.Float32,
            "subject_id": pl.Int32,
            "prediction_time": pl.Datetime,
        }

        # Pre-filter masked data using torch operations for efficiency
        batch_indices, seq_indices = torch.where(self.mask)
        if len(batch_indices) == 0:
            return pl.DataFrame(schema=schema)

        # Gather only the valid data points using index tensors
        time_values = self.time[batch_indices, seq_indices]
        code_values = self.code[batch_indices, seq_indices]
        numeric_values = self.numeric_value[batch_indices, seq_indices]
        numeric_value_masks = self.numeric_value_mask[batch_indices, seq_indices]

        # Convert to numpy for faster processing
        time_array = time_values.numpy()
        code_array = code_values.numpy().astype(np.int32)
        numeric_value_array = numeric_values.numpy()
        numeric_value_mask_array = numeric_value_masks.numpy()
        batch_indices = batch_indices.numpy()

        # Create arrays for prediction times and subject IDs
        pred_times = np.array(prediction_time, dtype="datetime64[ns]")[batch_indices]
        subject_ids = np.array(subject_id)[batch_indices].astype(np.int32)

        # Parallel processing using numpy vectorization
        time_deltas = time_array * np.timedelta64(1, self.time_scale).astype("timedelta64[ns]")
        timestamps = pred_times + time_deltas

        # Create the final dictionary with only valid data
        data_dict = {
            "time": timestamps,
            "code": code_array,
            "numeric_value": np.where(
                numeric_value_mask_array, numeric_value_array.astype(np.float32), np.nan
            ),
            "subject_id": subject_ids,
            "prediction_time": pred_times,
        }

        # Create DataFrame directly from the efficient dictionary
        return pl.from_dict(data_dict, schema=schema)


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
