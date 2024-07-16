from pathlib import Path

import polars as pl
import torch
from mixins import SeedableMixin
from nested_ragged_tensors.ragged_numpy import (
    JointNestedRaggedTensorDict,
)
from omegaconf import DictConfig

from meds_torch.pytorch_dataset import PytorchDataset


class MultiWindowPytorchDataset(SeedableMixin, torch.utils.data.Dataset):
    """A Multi Window PyTorch Dataset class, enabling contrastive learning pretraining.

    Args:
        config: Configuration options for the dataset, in an `omegaconf.DictConfig` object.
        split: The split of data which should be used in this dataset (e.g., ``'train'``, ``'tuning'``,
            ``'held_out'``). This will dictate where the system looks for files.
    """

    def __init__(self, cfg: DictConfig, split: str):
        super().__init__()

        self.config = cfg
        self.split = split
        self.pytorch_dataset = PytorchDataset(cfg, split)
        self.load_and_process_windows()

    @property
    def patient_ids(self) -> list[int]:
        return [x[0] for x in self.index]

    def __len__(self):
        return len(self.index)

    @property
    def has_task(self) -> bool:
        return self.config.task_name is not None

    @property
    def max_seq_len(self) -> int:
        return self.config.max_seq_len

    def load_and_process_windows(self):
        # Assume load_windows() fetches and converts time windows to index windows
        window_df = pl.read_parquet(self.config.windows_fp)
        window_dict = {}
        for row in window_df.iter_rows():
            pid = row[0]
            if pid not in window_dict:
                window_dict[pid] = {}
            # Assuming multiple windows per patient can be delineated
            windows = row[2:]
            for w in windows:
                window_name = w["window_name"]
                start_idx = self.timestamp_to_index(pid, w["timestamp_at_start"])
                end_idx = self.timestamp_to_index(pid, w["timestamp_at_end"])
                if window_name not in window_dict[pid]:
                    window_dict[pid][window_name] = {"start": [], "end": []}
                window_dict[pid][window_name]["start"].append(start_idx)
                window_dict[pid][window_name]["end"].append(end_idx)
        self.window_dict = window_dict

    def timestamp_to_index(self, patient_id, timestamp):
        # converts from timestamps to nested ragged tensor indexes
        shard = self.pytorch_dataset.subj_map[patient_id]
        patient_idx = self.pytorch_dataset.subj_indices[patient_id]
        out = JointNestedRaggedTensorDict.load_slice(
            Path(self.config.tensorized_root) / f"{shard}.nrt", patient_idx
        )
        import pdb

        pdb.set_trace()

        raise NotImplementedError(
            "Implement timestamp to nested ragged tensor index conversion -- similar to what we do for tasks"
        )

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        """Returns a Returns a dictionary corresponding to a single subject's data.

        The output of this will not be tensorized as that work will need to be re-done in the collate function
        regardless. The output will have structure:
        ``
        {
            'time_delta_days': [seq_len],
            'dynamic_indices': [seq_len, n_data_per_event] (ragged),
            'dynamic_values': [seq_len, n_data_per_event] (ragged),
            'static_indices': [seq_len, n_data_per_event] (ragged),
        }
        ``

        1. ``time_delta_days`` captures the time between each event and the subsequent event in days.
        2. ``dynamic_indices`` captures the categorical metadata elements listed in `self.data_cols` in a
           unified vocabulary space spanning all metadata vocabularies.
        3. ``dynamic_values`` captures the numerical metadata elements listed in `self.data_cols`. If no
           numerical elements are listed in `self.data_cols` for a given categorical column, the according
           index in this output will be `np.NaN`.
        5. ``static_indices`` captures the categorical metadata elements listed in `self.static_cols` in a
           unified vocabulary.
        """
        return self._seeded_getitem(idx)

    @SeedableMixin.WithSeed
    def _seeded_getitem(self, idx: int) -> dict[str, list[float]]:
        """Returns a Returns a dictionary corresponding to a single subject's data.

        This function is a seedable version of `__getitem__`.
        """
        patient_id, st, end = self.pytorch_dataset.index[idx]

        shard = self.pytorch_dataset.subj_map[patient_id]
        patient_idx = self.pytorch_dataset.subj_indices[patient_id]

        out = {}
        out["dynamic"] = JointNestedRaggedTensorDict.load_slice(
            Path(self.config.tensorized_root) / f"{shard}.nrt", patient_idx
        )[st:end]
        import pdb

        pdb.set_trace()
        # TODO
