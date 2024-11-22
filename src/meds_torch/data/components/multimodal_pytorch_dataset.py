from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from loguru import logger
from mixins import SeedableMixin, TimeableMixin
from nested_ragged_tensors.ragged_numpy import JointNestedRaggedTensorDict
from omegaconf import DictConfig
from safetensors import safe_open

from meds_torch.data.components.pytorch_dataset import (
    BINARY_LABEL_COL,
    PytorchDataset,
    subsample_subject_data,
)


def get_nonnan_indices(arr: torch.Tensor):
    """Returns (row, col) indices of non-NaN values in row-major order

    >>> x = torch.tensor([[1., float('nan')], [float('nan'), 2.]])
    >>> get_nonnan_indices(x)
    (tensor([0, 1]), tensor([0, 1]))
    """
    rows, cols = torch.nonzero(~torch.isnan(arr), as_tuple=True)
    order = torch.argsort(rows * arr.shape[1] + cols)
    return rows[order], cols[order]


class MultiModalPytorchDataset(PytorchDataset):
    """A PyTorch Dataset class that supports MultiModal Data

    Args:
        cfg (DictConfig): Configuration options for the dataset.
        split (str): The data split to use (e.g., 'train', 'validation', 'test').
        min_window_size (int): Minimum size of generated windows.
        max_window_size (int): Maximum size of generated windows.
        n_windows (int): Number of windows to generate for each sample.
    """

    def __init__(self, cfg: DictConfig, split: str):
        super().__init__(cfg, split)
        self.filter_index()

    def filter_index(self):
        """Filters out subjects that have no modality data in their patient history."""
        # TODO: make this optional and only run for supervised learning maybe?
        new_index = []
        if self.has_task:
            for idx, (subject_id, start_idx, end_idx) in enumerate(self.index):
                subject_dynamic_data, subject_id, st, end = self.load_subject_dynamic_data(idx)

                out = self.load_subject(subject_dynamic_data, subject_id, st, end)
                modality_idxs = out["dynamic"].tensors["dim0/modality_idx"]
                has_modality_data = bool((~np.isnan(modality_idxs)).any())
                if has_modality_data:
                    new_index.append((subject_id, start_idx, end_idx))
            if len(new_index) < len(self.index):
                logger.warning(
                    f"Filtered index has {len(self.index)} entries "
                    f"while full index has {len(new_index)} entries."
                )
            self.index = new_index

    @SeedableMixin.WithSeed
    @TimeableMixin.TimeAs
    def load_subject(
        self, subject_dynamic_data, subject_id: int, global_st: int, global_end: int
    ) -> dict[str, list[float]]:
        """Load and process data for a single subject.

        Args:
            subject_dynamic_data: The dynamic data for the subject.
            subject_id (int): The ID of the subject to load.
            global_st (int): The start index of the sequence to load.
            global_end (int): The end index of the sequence to load.

        Returns:
            dict: A dictionary containing the processed data for the subject.
        """
        shard = self.subj_map[subject_id]
        subject_idx = self.subj_indices[subject_id]
        static_row = self.static_dfs[shard][subject_idx].to_dict()

        max_seq_len = self.config.max_seq_len

        out = {
            "static_indices": static_row["static_indices"].item().to_list(),
            "static_values": static_row["static_values"].item().to_list(),
        }

        if self.config.do_prepend_static_data:
            n_static = len(out["static_indices"])
            if n_static >= max_seq_len:
                raise ValueError(
                    f"Static data length {n_static} matches or exceeds "
                    f"max_seq_len {max_seq_len} for subject {subject_id}!"
                )

            max_seq_len -= n_static
        if self.config.postpend_eos_token:
            max_seq_len -= 1

        subject_dynamic_data, global_st, global_end = subsample_subject_data(
            subject_dynamic_data,
            max_seq_len,
            self.config.subsequence_sampling_strategy,
            self.config.do_flatten_tensors,
            global_st,
        )

        if self.config.do_include_subsequence_indices:
            out["start_idx"] = global_st
            out["end_idx"] = global_end

        tensors = subject_dynamic_data.tensors

        if self.config.do_prepend_static_data:
            tensors["dim0/time_delta_days"] = np.concatenate(
                [np.zeros(len(out["static_indices"])), tensors["dim0/time_delta_days"]]
            )
            tensors["dim0/static_mask"] = np.concatenate(
                [
                    np.ones(len(out["static_indices"]), dtype=bool),
                    np.zeros(len(tensors["dim0/code"]), dtype=bool),
                ]
            )
            tensors["dim0/code"] = np.concatenate([out["static_indices"], tensors["dim0/code"]])
            tensors["dim0/numeric_value"] = np.concatenate(
                [out["static_values"], tensors["dim0/numeric_value"]]
            )

            # We assume there is no static modality data
            tensors["dim0/modality_idx"] = np.concatenate(
                [np.full_like(out["static_values"], np.nan, dtype=float), tensors["dim0/modality_idx"]]
            )
        else:
            tensors["dim0/static_mask"] = np.zeros(len(tensors["dim0/code"]), dtype=bool)

        if self.config.postpend_eos_token:
            tensors["dim0/code"] = np.append(tensors["dim0/code"], [self.config.EOS_TOKEN_ID])
            tensors["dim0/static_mask"] = np.append(tensors["dim0/static_mask"], [False])
            tensors["dim0/numeric_value"] = np.append(tensors["dim0/numeric_value"], [0])
            tensors["dim0/time_delta_days"] = np.append(tensors["dim0/time_delta_days"], [0])
            tensors["dim0/modality_idx"] = np.append(tensors["dim0/modality_idx"], [np.nan])

        subject_dynamic_data = JointNestedRaggedTensorDict(processed_tensors=tensors)

        out["dynamic"] = subject_dynamic_data

        if self.config.do_include_start_time_min:
            out["start_time"] = static_row["time"].item().to_list()[global_st]
        if self.config.do_include_prediction_time:
            out["prediction_time"] = static_row["time"].item().to_list()[global_end - 1]

        return out

    @TimeableMixin.TimeAs
    def _seeded_getitem(self, idx: int) -> dict[str, list[float]]:
        """Returns a Returns a dictionary corresponding to a single subject's data.

        This function is a seedable version of `__getitem__`.
        """

        subject_dynamic_data, subject_id, st, end = self.load_subject_dynamic_data(idx)

        out = self.load_subject(subject_dynamic_data, subject_id, st, end)

        if self.config.do_include_subject_id:
            out["subject_id"] = subject_id

        if self.labels is not None:
            out[BINARY_LABEL_COL] = self.labels[idx]

        shard = self.subj_map[subject_id]
        safetensor_fp = Path(self.config.modality_files_root) / f"{shard}.safetensors"
        with safe_open(safetensor_fp, framework="pt") as f:
            modality_idxs = out["dynamic"].tensors["dim0/modality_idx"]
            out["modality_sequence_idx"] = torch.tensor(
                np.where(~np.isnan(modality_idxs))[0], dtype=torch.int32
            )
            nonnull_modality_idxs = modality_idxs[~np.isnan(modality_idxs)]
            out["modality"] = torch.stack([f.get_tensor(str(int(k))) for k in nonnull_modality_idxs]).to(
                torch.float32
            )

        return out

    @TimeableMixin.TimeAs
    def collate(self, batch: list[dict]) -> dict:
        """Combines a batch of data points into a single, tensorized batch.

        The collated output is a fully tensorized and padded dictionary, ready for input into an
        `input_encoder`. This method uses the JointNestedRaggedTensorDict API to collate and pad the data.

        Args:
            batch (list[dict]): A list of dictionaries, each representing a single sample as
                returned by the __getitem__ method.

        Returns:
            dict: A dictionary containing the collated batch data.
        """
        modality_data = [item.pop("modality") for item in batch]
        list_modality_sequence_idx = [item.pop("modality_sequence_idx") for item in batch]
        modality_sequence_idx = torch.concat(list_modality_sequence_idx)
        modality_batch_idx = torch.repeat_interleave(
            torch.arange(len(list_modality_sequence_idx)),
            torch.tensor([len(t) for t in list_modality_sequence_idx]),
        )

        data = JointNestedRaggedTensorDict.vstack([item["dynamic"] for item in batch]).to_dense()
        tensorized = {k: torch.as_tensor(v) for k, v in data.items()}
        tensorized["code"] = tensorized["code"].long()
        tensorized["mask"] = tensorized.pop("dim1/mask")
        tensorized["numeric_value_mask"] = ~torch.isnan(tensorized["numeric_value"])
        tensorized["time_delta_days"] = torch.nan_to_num(tensorized["time_delta_days"], nan=0).float()
        tensorized["numeric_value"] = torch.nan_to_num(tensorized["numeric_value"], nan=0).float()
        del tensorized["modality_idx"]

        # Add task labels to batch
        for k in batch[0].keys():
            if k not in ("dynamic", "static_values", "static_indices", "static_mask"):
                if isinstance(batch[0][k], datetime):
                    tensorized[k] = [item[k] for item in batch]
                else:
                    tensorized[k] = torch.Tensor([item[k] for item in batch])
        tensorized["modality"] = torch.concat(modality_data)
        tensorized["modality_sequence_idx"] = modality_sequence_idx
        tensorized["modality_batch_idx"] = modality_batch_idx
        assert (
            tensorized["modality"].shape[0] == len(modality_sequence_idx) == len(modality_batch_idx)
        ), "Modality shape mismatch"
        return tensorized
