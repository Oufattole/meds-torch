from pathlib import Path

import numpy as np
from loguru import logger
from mixins import SeedableMixin
from nested_ragged_tensors.ragged_numpy import JointNestedRaggedTensorDict
from omegaconf import DictConfig

from meds_torch.data.components.pytorch_dataset import PytorchDataset


class RandomWindowPytorchDataset(PytorchDataset):
    """A PyTorch Dataset class that generates random windows on-the-fly for contrastive learning pretraining.

    This class extends PytorchDataset to support random window generation without relying on predefined
    windows.

    Args:
        cfg (DictConfig): Configuration options for the dataset.
        split (str): The data split to use (e.g., 'train', 'validation', 'test').
        min_window_size (int): Minimum size of generated windows.
        max_window_size (int): Maximum size of generated windows.
        n_windows (int): Number of windows to generate for each sample.
    """

    def __init__(self, cfg: DictConfig, split: str):
        super().__init__(cfg, split)
        self.min_window_size = cfg.min_window_size
        self.max_window_size = cfg.max_window_size
        self.n_windows = cfg.n_windows
        self.window_cols = [f"window_{i}" for i in range(cfg.n_windows)]
        self.cfg = cfg
        self.filter_index()

    def filter_index(self):
        filtered_index = []
        for i in range(len(self.index)):
            seq_length = self.index[i][2] - self.index[i][1]
            window_size = self.get_random_window_size(seq_length)
            if window_size < self.min_window_size:
                logger.warning(
                    f"Sequence length {seq_length} is too short to accommodate "
                    f"{self.n_windows} windows of minimum size {self.min_window_size}. "
                    f"Skipping subject {self.index[i][0]}."
                )
            else:
                filtered_index.append(self.index[i])
        if len(filtered_index) < len(self.index):
            logger.warning(f"Filtered data to {len(filtered_index) / len(self.index)}% of original data.")
        self.index = filtered_index

    def get_random_window_size(self, seq_length: int) -> int:
        """Calculates the size of the generated random windows."""
        total_window_size = min(seq_length, self.max_window_size * self.n_windows)
        window_size = total_window_size // self.n_windows
        return window_size

    def generate_random_windows(self, seq_length: int) -> list[tuple[int, int]]:
        """Generate random windows within a sequence.

        This method supports two modes of operation: (a) Random windows: Windows can overlap and be in any
        order. (b) Consecutive random windows: Windows are side by side and in order.

        The window sizes are predetermined to be the same and add up to less than the length of the dataset.

        Args:
            seq_length (int): Length of the sequence to generate windows from.

        Returns:
            List[Tuple[int, int]]: List of (start, end) indices for each window.

        Raises:
            ValueError: If the sequence length is too short to accommodate all windows.
        """
        # Use user-defined window names if provided, otherwise use default names
        window_names = (
            self.cfg.window_names
            if hasattr(self.cfg, "window_names")
            else [f"window_{i}" for i in range(self.n_windows)]
        )

        # Ensure we have the correct number of window names
        if len(window_names) != self.n_windows:
            raise ValueError(
                f"Number of window names ({len(window_names)}) does not match n_windows ({self.n_windows})"
            )

        window_size = self.get_random_window_size(seq_length)
        total_window_size = window_size * self.n_windows
        windows = []

        if self.cfg.get("consecutive_windows", False):
            # Mode (b): Consecutive random windows
            start = np.random.randint(0, seq_length - total_window_size + 1)
            for i in range(self.n_windows):
                end = start + window_size
                windows.append((start, end))
                start = end
        else:
            # Mode (a): Random windows
            for _ in range(self.n_windows):
                start = np.random.randint(0, seq_length - window_size + 1)
                end = start + window_size
                windows.append((start, end))

        # Sort windows by start index to maintain order
        windows.sort(key=lambda x: x[0])

        # Create a dictionary with user-defined or default window names
        named_windows = {name: window for name, window in zip(window_names, windows)}

        return named_windows

    def partition_sequence(self, sequence: dict, windows: list[tuple[int, int]]) -> dict:
        """Partition a sequence into multiple windows.

        Args:
            sequence (dict): The full sequence data.
            windows (List[Tuple[int, int]]): List of (start, end) indices for each window.

        Returns:
            dict: A dictionary with partitioned data for each window.
        """
        partitioned = {}
        for window_name, (start, end) in windows.items():
            partitioned[window_name] = {
                k: v[start:end] if isinstance(v, (list, np.ndarray)) else v for k, v in sequence.items()
            }
        return partitioned

    @SeedableMixin.WithSeed
    def _seeded_getitem(self, idx: int) -> dict:
        """Get a randomly windowed item from the dataset.

        Args:
            idx (int): Index of the item to retrieve.

        Returns:
            dict: A dictionary containing randomly generated windows of the sequence.
        """
        subject_id, _, _ = self.index[idx]
        shard = self.subj_map[subject_id]
        subject_idx = self.subj_indices[subject_id]

        subject_dynamic_data = JointNestedRaggedTensorDict(
            tensors_fp=Path(self.config.data_dir) / "data" / f"{shard}.nrt"
        )[subject_idx]

        full_sequence = self.load_subject(subject_dynamic_data, subject_id, 0, len(subject_dynamic_data))

        seq_len = len(full_sequence["dynamic"])
        windows = self.generate_random_windows(seq_len)
        partitioned_sequence = self.partition_sequence(full_sequence, windows)

        if self.config.do_include_subject_id:
            partitioned_sequence["subject_id"] = subject_id

        return partitioned_sequence

    def collate(self, batch: list[dict]) -> dict:
        """Collate a batch of randomly windowed sequences.

        Args:
            batch (List[dict]): A list of dictionaries, each containing windowed sequences.

        Returns:
            dict: A dictionary with collated data for each window.
        """
        out = {}
        for col in self.window_cols:
            if col in batch[0]:
                out[col] = super().collate([x[col] for x in batch])

        # Collate any additional data that's not part of the windows
        for key in batch[0].keys():
            if key not in self.window_cols and key not in out:
                out[key] = [x[key] for x in batch]

        return out
