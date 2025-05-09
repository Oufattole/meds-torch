import math

import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import IterableDataset
from tqdm.auto import trange

from meds_torch.utils import RankedLogger

# We need to import the resolvers so the cfg object has access to them
from meds_torch.utils.resolvers import setup_resolvers

log = RankedLogger(__name__, rank_zero_only=True)

setup_resolvers()


def collate_fn(batch_list):
    """
    Each item in batch_list is a dict:
      {
        "input_ids": Tensor of shape [window_len_i],
        "targets":   Tensor of shape [window_len_i],
        "skip_mask": Tensor of shape [window_len_i] (bool)
      }
    We'll pad them to [batch_size, max_window_len].
    """
    input_ids_list = [x["code"] for x in batch_list]
    mask_list = [x["mask"] for x in batch_list]
    # subject_id_list = torch.hstack([x["subject_id"] for x in batch_list])
    # prediction_time_list = [x["prediction_time"][0] for x in batch_list]
    # end_time_list = [x["end_time"][0] for x in batch_list]

    input_ids_padded = pad_sequence(input_ids_list, batch_first=True, padding_value=0)
    mask_padded = pad_sequence(mask_list, batch_first=True, padding_value=False)

    # Note: for bool Tensors, `padding_value=True` => skip by default
    # in any positions beyond the original window length

    data = {
        "code": input_ids_padded,
        "mask": mask_padded,
        # "subject_id": subject_id_list,
        # "prediction_time": prediction_time_list,
        # "end_time": end_time_list,
    }
    if "histogram" in batch_list[0]:
        histogram_list = [x["histogram"].squeeze(0) for x in batch_list]
        histogram_padded = pad_sequence(histogram_list, batch_first=True, padding_value=0)
        data["histogram"] = histogram_padded
    return data


class OverlapSkippingSlidingWindowDataset(IterableDataset):
    """
    For each sequence in `dataset`, produce overlapping windows of length `context_length`,
    sliding by `stride = (context_length - overlap)` each time. Then for each chunk after
    the first, mark the overlapped tokens at the beginning with a `skip_mask` so that
    they won't be double-counted in metrics.
    """

    def __init__(self, dataset, context_length=4, overlap=2, observation_lag=1):
        super().__init__()
        self.dataset = dataset
        self.context_length = context_length
        self.overlap = overlap
        self.stride = self.context_length - self.overlap
        self.carry_forward_keys = ["subject_id", "end_time", "prediction_time"]
        self.observation_lag = observation_lag
        if self.stride <= 0:
            raise ValueError("`overlap` must be strictly less than `context_length`.")

    def get_length(self):
        # TODO: Update to account for observation lag
        start = 0
        end = len(self.dataset)
        total_length = 0
        for i in trange(start, end):
            total_length += max(
                math.ceil(len(self.dataset[i]["dynamic"].tensors["dim0/code"]) / self.stride) - 1, 1
            )
        return total_length

    @property
    def collate(self) -> dict:
        return collate_fn

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is None or worker_info.num_workers == 1:
            start = 0
            end = len(self.dataset)
        else:
            num_workers = worker_info.num_workers
            worker_id = worker_info.id
            per_worker = int(math.ceil(len(self.dataset) / float(num_workers)))
            start = worker_id * per_worker
            end = min(start + per_worker, len(self.dataset))
            log.info(f"Worker {worker_id} iterating from {start} to {end}")
        for i in range(start, end):
            # 1) Get the full token sequence from the underlying dataset
            sample = self.dataset.collate([self.dataset[i]])
            # e.g. sample["code"] is a 1D Tensor of token IDs
            sequence = sample["code"].flatten()
            length = len(sequence)

            idx = 0
            window_count = 0
            while idx < length - self.observation_lag:
                window_end = min(idx + self.context_length, length)

                # The "input_ids" for this window
                input_ids = sequence[idx:window_end]

                # Build a skip mask
                # By default, do not skip any tokens
                skip_mask = torch.zeros_like(input_ids, dtype=torch.bool)

                # For the second (and subsequent) windows of a sequence,
                # skip the overlapped tokens at the beginning:
                if window_count > 0:  # i.e. not the very first chunk
                    # We overlapped `overlap` tokens, so skip them in metric evaluation
                    skip_mask[: self.overlap] = True

                data = {
                    "code": input_ids,
                    "mask": torch.ones_like(input_ids, dtype=torch.bool),
                    **{key: sample[key] for key in self.carry_forward_keys if key in sample},
                }
                if "histogram" in sample:
                    data["histogram"] = sample["histogram"][:, idx:window_end]
                yield data

                # Move forward by stride
                idx += self.stride
                window_count += 1
