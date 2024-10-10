import rootutils

root = rootutils.setup_root(__file__, dotenv=True, pythonpath=True, cwd=True)

from pathlib import Path

import numpy as np
import pytest
from mixins import SeedableMixin

from meds_torch.data.components.pytorch_dataset import PytorchDataset
from tests.conftest import create_cfg


class EveryQueryDataset(PytorchDataset):
    def __init__(self, cfg, split):
        # self.actual_max_seq_len = cfg.max_seq_len
        # cfg.max_seq_len = 1000000000 # artificially increaase to get full subject record
        # cfg.subsequence_sampling_strategy = 'from_start'
        cfg.do_include_subsequence_indices = True
        super().__init__(cfg, split)

        if self.config.min_query_offset < 0:
            raise ValueError("min_query_offset must be non-negative.")
        if self.config.min_query_duration < 0:
            raise ValueError("min_query_duration must be non-negative.")
        if self.config.min_query_offset >= self.config.max_query_offset:
            raise ValueError("min_query_offset must be less than max_query_offset.")
        if self.config.min_query_duration >= self.config.max_query_duration:
            raise ValueError("min_query_duration must be less than max_query_duration.")
        if self.config.query_sampling_strategy not in ("within_record", "random", "fixed"):
            raise ValueError("query_sampling_strategy must be one of 'within_record', 'random', or 'fixed'.")
        if self.config.query_sampling_strategy == "fixed":
            if self.config.query_fixed_duration is None:
                raise ValueError("query_fixed_duration must be specified for 'fixed' sampling strategy.")
            if self.config.query_fixed_duration < 0:
                raise ValueError("query_fixed_duration must be non-negative.")
            if self.config.query_fixed_offset is None:
                raise ValueError("query_fixed_offset must be specified for 'fixed' sampling strategy.")
            if self.config.query_fixed_offset < 0:
                raise ValueError("query_fixed_offset must be non-negative.")

    def sample_query_duration(self, times):
        record_duration = times[-1]
        min_input_duration = times[1]  # so that we have at least one event in input
        max_valid_query_duration = record_duration - min_input_duration

        match self.config.query_sampling_strategy:
            case "within_record":
                if max_valid_query_duration <= self.config.min_query_duration:
                    query_duration = max_valid_query_duration
                else:
                    query_duration = np.random.randint(
                        low=self.config.min_query_duration,
                        high=min(self.config.max_query_duration, max_valid_query_duration),
                    )
            case "random":
                query_duration = np.random.randint(
                    self.config.min_query_duration, self.config.max_query_duration
                )
            case "fixed":
                query_duration = self.config.query_fixed_duration

        assert query_duration > 0

        censored = query_duration > max_valid_query_duration

        if self.config.normalize_query:
            normalized_query_duration = (query_duration - self.config.min_query_duration) / (
                self.config.max_query_duration - self.config.min_query_duration
            )
        else:
            normalized_query_duration = query_duration

        return censored, query_duration, normalized_query_duration

    def sample_input_indices(self, times, query_duration, censored):
        seq_len = self.actual_max_seq_len

        record_duration = times[-1]
        if not censored and self.config.fit_inputs_before_query:
            max_input_end_time = record_duration - query_duration
            max_input_end_idx = np.max(np.argwhere((times) < max_input_end_time))
            if max_input_end_idx > seq_len:
                input_start_idx = np.random.choice(max_input_end_idx - seq_len)
                input_end_idx = input_start_idx + seq_len
            else:
                input_start_idx = 0
                input_end_idx = max_input_end_idx
        else:
            raise NotImplementedError
            # seq_len = tensors["dim1/code"].shape[0]
            # randomly sample over full input
            # should we allow existing FROM_SUBSEQUENCE_SAMPLE in meds torch
            # check if query fits in remaining data and update censored
            # start_offset = np.random.choice(seq_len - self.max_seq_len)
            # st += start_offset
            # end = min(end, st + self.max_seq_len)

        return censored, input_start_idx, input_end_idx

    def sample_query_offset(self, times, censored, input_end_idx, query_duration):
        record_duration = times[-1]
        input_duration = times[input_end_idx]
        max_valid_offset = record_duration - input_duration - query_duration

        match self.config.query_sampling_strategy:
            case "within_record":
                if max_valid_offset <= self.config.min_query_offset:
                    query_offset = max_valid_offset
                else:
                    query_offset = np.random.randint(
                        low=self.config.min_query_offset,
                        high=min(self.config.max_query_offset, max_valid_offset),
                    )
            case "random":
                query_offset = np.random.randint(self.config.min_query_offset, self.config.max_query_offset)
            case "fixed":
                query_offset = self.config.query_fixed_offset

        assert query_offset >= 0

        if not censored and query_offset > max_valid_offset:
            censored = True

        if self.config.normalize_query:
            normalized_query_offset = (query_offset - self.config.min_query_offset) / (
                self.config.max_query_offset - self.config.min_query_offset
            )
        else:
            normalized_query_offset = query_offset

        return censored, query_offset, normalized_query_offset

    def get_query_indices(self, times, query_offset, query_duration, input_end_idx):
        query_start_time = times[input_end_idx] + query_offset
        query_end_time = query_start_time + query_duration
        query_start_idx = np.min(np.argwhere((times) >= query_start_time))
        query_end_idx = np.max(np.argwhere((times) <= query_end_time), initial=query_start_idx)
        assert query_start_idx <= query_end_idx
        return query_start_idx, query_end_idx

    def count_query_occurrence(self, query_window_codes, query_window_values, query):
        count = 0
        for i in range(len(query_window_codes)):
            for j in range(len(query_window_codes[i])):
                if query_window_codes[i][j] == query["idx"]:
                    if query["has_value"]:
                        x = query_window_values[i][j]
                        if x is None:
                            continue  # todo: is None used for outlier removal?
                        if (x >= query["range_min"]) and (x <= query["range_max"]):
                            count += 1
                    else:
                        count += 1
        return count

    @SeedableMixin.WithSeed
    def _seeded_getitem(self, idx: int) -> dict[str, list[float]]:
        subject_dynamic_data, subject_id, st, end = super().load_subject_dynamic_data(idx)

        # context = super()._seeded_getitem(idx)

        record = super()._seeded_getitem(idx)
        record_dynamic = record["dynamic"].tensors
        """(issue) given very large self.config.max_seq_len given
        self.config.subsequence_sampling_strategy=FROM_START can we assume the full record is returned?"""

        time_delta = record_dynamic["dim0/time_delta_days"] * 1440  # days to minutes
        if np.isnan(time_delta[0]):
            time_delta[0] = 0
        times = np.cumsum(time_delta)

        censored, query_duration, normalized_query_duration = self.sample_query_duration(times)

        censored, input_start_idx, input_end_idx = self.sample_input_indices(times, query_duration, censored)

        # can move this above the time delta info

        # sample input first
        # then fit query within record

        censored, query_offset, normalized_query_offset = self.sample_query_offset(
            times, censored, input_end_idx, query_duration
        )

        if censored:
            pass
            # dont compute this

        query_start_idx, query_end_idx = self.get_query_indices(
            times, query_offset, query_duration, input_end_idx
        )
        query_window_codes = record_dynamic["dim1/code"][query_start_idx:query_end_idx]
        query_window_values = record_dynamic["dim1/numeric_value"][query_start_idx:query_end_idx]

        # query_code = self.config.sample_code()
        # fixed codes, valid subset of codes to sample: HydraConfig list
        # metadata/codes.parquet read from disk
        # look in triplet
        query_code = {
            "idx": 2,
            "has_value": True,
            "range_min": 0.4,
            "range_max": 0.7,
            # can also include code name and code type
        }
        query = query_code | {
            "offset": normalized_query_offset,
            "duration": normalized_query_duration,
        }

        count = self.count_query_occurrence(query_window_codes, query_window_values, query)
        answer = {"censored": censored, "count": count, "occurs": count != 0}

        # after calculating answer from the query window, reduce record to input window
        record["dynamic"] = record["dynamic"][input_start_idx:input_end_idx]

        item = {
            "context": record,
            "query": query,
            "answer": answer,
        }

        return item


@pytest.mark.parametrize(
    "query_offset, query_duration, input_end_idx, expected_indices, expected_exception",
    [
        # Test case 1: Normal case
        (
            2,
            3,
            4,
            (6, 9),
            None,
        ),
        # Test case 2: Zero offset and some duration
        (
            0,
            2,
            3,
            (3, 5),
            None,
        ),
        # Test case 3: Query extends beyond last time point
        # (issue: should this raise error if censored is false?)
        (
            7,
            5,
            2,
            (9, 10),
            None,
        ),
        # Test case 4: Query start time beyond times array (expect ValueError)
        (
            5,
            2,
            8,
            None,
            ValueError,
        ),
    ],
)
def test_get_query_indices(
    meds_dir: Path,
    query_offset,
    query_duration,
    input_end_idx,
    expected_indices,
    expected_exception,
):
    cfg = create_cfg(overrides=[], meds_dir=meds_dir)
    dataset = EveryQueryDataset(cfg.data, split="train")
    times = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    if expected_exception is not None:
        with pytest.raises(expected_exception):
            query_start_idx, query_end_idx = dataset.get_query_indices(
                times, query_offset, query_duration, input_end_idx
            )
    else:
        query_start_idx, query_end_idx = dataset.get_query_indices(
            times, query_offset, query_duration, input_end_idx
        )
        expected_query_start_idx, expected_query_end_idx = expected_indices
        assert query_start_idx == expected_query_start_idx
        assert query_end_idx == expected_query_end_idx


@pytest.mark.parametrize(
    "config_overrides, input_end_idx, query_duration, expected",
    [
        # Test case 1: Fixed strategy, query_offset within valid range
        (
            {"query_sampling_strategy": "fixed", "query_fixed_offset": 3, "normalize_query": False},
            3,
            2,
            (False, 3, 3),
        ),
        # Test case 2: Fixed strategy, query_offset exceeds max_valid_offset (censored set to True)
        (
            {"query_sampling_strategy": "fixed", "query_fixed_offset": 5, "normalize_query": False},
            3,
            2,
            (True, 5, 5),
        ),
        # Test case 3: Within_record strategy, max_valid_offset less than min_query_offset
        (
            {
                "query_sampling_strategy": "within_record",
                "min_query_offset": 2,
                "max_query_offset": 5,
                "normalize_query": False,
            },
            6,
            3,
            (False, 0, 0),
        ),
        # Test case 4: Within_record strategy with normalization
        (
            {
                "query_sampling_strategy": "within_record",
                "min_query_offset": 2,
                "max_query_offset": 5,
                "normalize_query": True,
            },
            3,
            2,
            (False, 2, 0),  # Normalized as (2 - 2) / (5 - 2)
        ),
        # Test case 5: Random strategy with random seed
        (
            {
                "query_sampling_strategy": "random",
                "min_query_offset": 2,
                "max_query_offset": 5,
                "normalize_query": False,
            },
            3,
            2,
            (False, 2, 2),  # With seed 0, np.random.randint(2,5) gives 2
        ),
    ],
)
def test_sample_query_offset(meds_dir: Path, config_overrides, input_end_idx, query_duration, expected):
    np.random.seed(0)
    times = list(range(10))  # times = [0, 1, 2, ..., 9]
    initial_censored = False
    cfg = create_cfg(overrides=[], meds_dir=meds_dir)
    for key, value in config_overrides.items():
        setattr(cfg.data, key, value)
    dataset = EveryQueryDataset(cfg.data, split="train")
    censored, query_offset, normalized_query_offset = dataset.sample_query_offset(
        times, initial_censored, input_end_idx, query_duration
    )
    expected_censored, expected_query_offset, expected_normalized_query_offset = expected
    assert censored == expected_censored
    assert query_offset == expected_query_offset
    assert normalized_query_offset == expected_normalized_query_offset


@pytest.mark.parametrize(
    "config_overrides, expected",
    [
        # Test case 1: Fixed strategy, query duration within valid range
        (
            {"query_sampling_strategy": "fixed", "query_fixed_duration": 5, "normalize_query": False},
            (False, 5, 5),
        ),
        # Test case 2: Fixed strategy, query duration exceeds valid range (censored)
        (
            {"query_sampling_strategy": "fixed", "query_fixed_duration": 11, "normalize_query": False},
            (True, 11, 11),
        ),
        # Test case 3: Within_record strategy
        (
            {
                "query_sampling_strategy": "within_record",
                "min_query_duration": 2,
                "max_query_duration": 5,
                "normalize_query": False,
            },
            (False, 2, 2),
        ),
        # Test case 4: Within_record strategy, max_valid_query_duration (8) less than min_query_duration
        (
            {
                "query_sampling_strategy": "within_record",
                "min_query_duration": 15,
                "max_query_duration": 20,
                "normalize_query": False,
            },
            (False, 8, 8),
        ),
        # # Test case 5: Within_record strategy with normalization
        (
            {
                "query_sampling_strategy": "within_record",
                "min_query_duration": 2,
                "max_query_duration": 5,
                "normalize_query": True,
            },
            (False, 2, (2 - 2) / (5 - 2)),
        ),
        # # Test case 6: Random strategy with random seed
        (
            {
                "query_sampling_strategy": "random",
                "min_query_duration": 2,
                "max_query_duration": 5,
                "normalize_query": False,
            },
            (False, 2, 2),
        ),
    ],
)
def test_sample_query_duration(meds_dir: Path, config_overrides, expected):
    np.random.seed(0)
    times = [*range(10)]
    cfg = create_cfg(overrides=[], meds_dir=meds_dir)
    for key, value in config_overrides.items():
        setattr(cfg.data, key, value)
    dataset = EveryQueryDataset(cfg.data, split="train")
    censored, query_duration, normalized_query_duration = dataset.sample_query_duration(times)
    expected_censored, expected_query_duration, expected_normalized_query_duration = expected
    assert censored == expected_censored
    assert query_duration == expected_query_duration
    if isinstance(expected_normalized_query_duration, float):
        assert np.isclose(normalized_query_duration, expected_normalized_query_duration)
    else:
        assert normalized_query_duration == expected_normalized_query_duration


@pytest.mark.parametrize(
    "collate_type",
    [
        "event_stream",
    ],
)
def test_pytorch_dataset(meds_dir: Path, collate_type):
    cfg = create_cfg(overrides=[], meds_dir=meds_dir)
    cfg.data.collate_type = collate_type
    pyd = EveryQueryDataset(cfg.data, split="train")
    assert not pyd.has_task
    item = pyd[0]
    assert item.keys() == {"context", "query", "answer"}
    assert item["context"].keys() == {"static_indices", "static_values", "dynamic"}
    # assert False
    batch = pyd.collate([pyd[i]["context"] for i in range(2)])
    if collate_type == "event_stream":
        assert batch.keys() == {
            "event_mask",
            "dynamic_values_mask",
            "time_delta_days",
            "dynamic_indices",
            "dynamic_values",
            "static_indices",
            "static_values",
        }
    else:
        raise NotImplementedError(f"{collate_type} not implemented")
