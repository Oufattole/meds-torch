from dataclasses import dataclass
from datetime import datetime

import numpy as np
import polars as pl
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


@dataclass
class TrajectoryBatch:
    """
    Initialize a batch of trajectories.

    Args:
        time (torch.Tensor): Tensor of shape (batch_size, sequence_length) containing days after prediction
            time. Values must be monotonically increasing within each sequence.
        code (torch.Tensor): Tensor of shape (batch_size, sequence_length) containing event code vocabulary
            indices
        mask (torch.Tensor): Tensor of shape (batch_size, sequence_length) indicates valid measurements/codes
        numeric_value (torch.Tensor): Tensor of shape (batch_size, sequence_length) containing numeric values
        numeric_value_mask (torch.Tensor): Tensor of shape (batch_size, sequence_length) indicating valid
            numeric values
        metadata_df (pl.DataFrame): DataFrame containing code vocabulary mapping with 'code' and
            'code/vocab_index' columns
        time_scale: scale of the time, by default it is 'Y' (years). Any numpy datetime units can be used,
            see https://numpy.org/doc/2.1/reference/arrays.datetime.html#datetime-units
    """

    time: torch.Tensor
    code: torch.Tensor
    mask: torch.Tensor
    numeric_value: torch.Tensor
    numeric_value_mask: torch.Tensor
    metadata_df: pl.DataFrame
    time_scale: str = "Y"

    def to_meds(self, prediction_time: list[datetime], subject_id: list[str | int]) -> pl.DataFrame:
        """Convert the trajectory batch to MEDS format.

        Args:
            prediction_time: List of prediction times for each trajectory in the batch
            subject_id: List of subject IDs for each trajectory in the batch

        Returns:
            pl.DataFrame: MEDS format DataFrame with columns:
                - time: Absolute timestamp of the event
                - code: The medical code string
                - numeric_value: The numeric value associated with the code (if any)
                - subject_id: ID of the subject
                - prediction_time: The prediction time for this trajectory

        Example:
        >>> metadata_df = pl.DataFrame({
        ...     'code': ['A1', 'A2', 'A3', 'A4', 'A5', 'A6'],
        ...     'code/vocab_index': [1, 2, 3, 4, 5, 6]
        ... })
        >>> batch = TrajectoryBatch(
        ...     time=torch.tensor([[0, .5, 2], [0, 3, 5]]),
        ...     code=torch.tensor([[1, 2, 3], [4, 5, 6]]),
        ...     mask=torch.tensor([[1, 1, 1], [1, 1, 0]]),
        ...     numeric_value=torch.tensor([[0.5, 1.0, 1.5], [2.0, 2.5, 3.0]]),
        ...     numeric_value_mask=torch.tensor([[1, 1, 0], [1, 0, 0]]),
        ...     metadata_df=metadata_df
        ... )
        >>> prediction_times = [datetime(2024, 1, 1), datetime(2024, 1, 1)]
        >>> subject_ids = [1, 2]
        >>> df = batch.to_meds(prediction_times, subject_ids)
        >>> df.sort("subject_id", "code")
        shape: (5, 6)
        ┌────────────┬─────────────────────┬─────────────────────┬──────┬──────────────────┬───────────────┐
        │ subject_id ┆ prediction_time     ┆ time                ┆ code ┆ code/vocab_index ┆ numeric_value │
        │ ---        ┆ ---                 ┆ ---                 ┆ ---  ┆ ---              ┆ ---           │
        │ i32        ┆ datetime[ns]        ┆ datetime[ns]        ┆ str  ┆ i64              ┆ f32           │
        ╞════════════╪═════════════════════╪═════════════════════╪══════╪══════════════════╪═══════════════╡
        │ 1          ┆ 2024-01-01 00:00:00 ┆ 2024-01-01 00:00:00 ┆ A1   ┆ 1                ┆ 0.5           │
        │ 1          ┆ 2024-01-01 00:00:00 ┆ 2024-07-01 14:54:36 ┆ A2   ┆ 2                ┆ 1.0           │
        │ 1          ┆ 2024-01-01 00:00:00 ┆ 2025-12-31 11:38:24 ┆ A3   ┆ 3                ┆ NaN           │
        │ 2          ┆ 2024-01-01 00:00:00 ┆ 2024-01-01 00:00:00 ┆ A4   ┆ 4                ┆ 2.0           │
        │ 2          ┆ 2024-01-01 00:00:00 ┆ 2026-12-31 17:27:36 ┆ A5   ┆ 5                ┆ NaN           │
        └────────────┴─────────────────────┴─────────────────────┴──────┴──────────────────┴───────────────┘
        """
        if len(prediction_time) != len(subject_id) or len(prediction_time) != self.time.shape[0]:
            raise ValueError("Number of prediction times and subject IDs must match batch size")
        schema = {
            "time": pl.Datetime,
            "code": self.metadata_df.schema["code/vocab_index"],
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
        df = pl.from_dict(data_dict, schema=schema)

        # Convert code vocab indexes to strings using the metadata mapping
        df = df.join(
            self.metadata_df.select("code", "code/vocab_index"), left_on="code", right_on="code/vocab_index"
        ).rename({"code_right": "code", "code": "code/vocab_index"})

        return df["subject_id", "prediction_time", "time", "code", "code/vocab_index", "numeric_value"]


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
