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
        cfg.max_seq_len = 5
        cfg.do_include_subsequence_indices = True
        super().__init__(cfg, split)

        if self.config.min_offset < 0:
            raise ValueError("min_query_offset must be non-negative.")
        if self.config.min_duration < 0:
            raise ValueError("min_query_duration must be non-negative.")
        if self.config.min_offset >= self.config.max_offset:
            raise ValueError("min_query_offset must be less than max_query_offset.")
        if self.config.min_duration >= self.config.max_duration:
            raise ValueError("min_query_duration must be less than max_query_duration.")
        if self.config.duration_sampling_strategy not in ("within_record", "random", "fixed"):
            raise ValueError(
                "duration_sampling_strategy must be one of 'within_record', 'random', or 'fixed'."
            )
        if self.config.duration_sampling_strategy == "fixed":
            if self.config.fixed_duration is None:
                raise ValueError("query_fixed_duration must be specified for 'fixed' sampling strategy.")
            if self.config.fixed_duration < 0:
                raise ValueError("query_fixed_duration must be non-negative.")
        if self.config.offset_sampling_strategy not in ("within_record", "random", "fixed"):
            raise ValueError("offset_sampling_strategy must be one of 'within_record', 'random', or 'fixed'.")
        if self.config.offset_sampling_strategy == "fixed":
            if self.config.fixed_offset is None:
                raise ValueError("query_fixed_offset must be specified for 'fixed' sampling strategy.")
            if self.config.fixed_offset < 0:
                raise ValueError("query_fixed_offset must be non-negative.")

    def normalize_future(self, query):
        # normalize offset and duration in place
        if self.config.normalize_query:
            query["duration"] = (query["duration"] - self.config.min_duration) / (
                self.config.max_duration - self.config.min_duration
            )
            query["offset"] = (query["offset"] - self.config.min_offset) / (
                self.config.max_offset - self.config.min_offset
            )
        return query

    def sample_future(self, max_valid_duration):
        if max_valid_duration < 0:
            raise ValueError(f"max_valid_duration must be non-negative, but got {max_valid_duration}")

        duration = self.sample_duration(max_valid_duration)

        max_valid_offset = max_valid_duration - duration
        if max_valid_offset < 0:
            raise ValueError(f"max_valid_offset must be non-negative, but got {max_valid_offset}")

        offset = self.sample_offset(max_valid_offset)

        if (duration > max_valid_duration) or (offset > max_valid_offset):
            is_censored = True
        else:
            is_censored = False

        future = {"offset": offset, "duration": duration}

        return future, is_censored

    def sample_duration(self, max_valid_duration):
        match self.config.duration_sampling_strategy:
            case "within_record":
                if max_valid_duration <= self.config.min_duration:
                    duration = max_valid_duration
                else:
                    duration = np.random.randint(
                        low=self.config.min_duration,
                        high=min(self.config.max_duration, max_valid_duration),
                    )
            case "random":
                duration = np.random.randint(self.config.min_duration, self.config.max_duration)
            case "fixed":
                duration = self.config.fixed_duration
        if duration < 0:
            raise ValueError(f"duration must be non-negative, but got {duration}")
        return duration

    def sample_offset(self, max_valid_offset):
        match self.config.offset_sampling_strategy:
            case "within_record":
                if max_valid_offset <= self.config.min_offset:
                    offset = max_valid_offset
                else:
                    offset = np.random.randint(
                        low=self.config.min_offset,
                        high=min(self.config.max_offset, max_valid_offset),
                    )
            case "random":
                offset = np.random.randint(self.config.min_offset, self.config.max_offset)
            case "fixed":
                offset = self.config.fixed_offset
        if offset < 0:
            raise ValueError(f"offset must be non-negative, but got {offset}")
        return offset

    def sample_event(self):
        NotImplemented
        # query_code = self.config.sample_code()
        # fixed codes, valid subset of codes to sample: HydraConfig list
        # metadata/codes.parquet read from disk
        # look in triplet
        event = {
            "idx": 2,
            "has_value": True,
            "range_min": 0.4,
            "range_max": 0.7,
            # can also include code name and code type
        }
        return event

    def tally_answer(self, future_dynamic, query):
        time_delta = future_dynamic.tensors["dim0/time_delta_days"] * 1440
        if np.isnan(time_delta[0]):
            time_delta[0] = 0
        times = np.cumsum(time_delta)

        start_time = query["offset"]
        end_time = query["offset"] + query["duration"]
        start_idx = np.min(np.argwhere((times) >= start_time))
        if end_time >= times[-1]:
            end_idx = None
        else:
            end_idx = np.min(np.argwhere((times) > end_time))
            assert start_idx <= end_idx

        if start_idx == end_idx:
            if start_idx == 0:
                # query is short, comes before first measurement, and has no data
                return 0
            else:
                # one event only
                return 0  # remove later once bug is fixed
                future_dynamic = future_dynamic[start_idx]  # this does not work
            """
            from nested_ragged_tensors.ragged_numpy import *
            J = JointNestedRaggedTensorDict({
                "dim0/time_delta_days": [0.00972222, 0.03333334],
                "dim1/lengths": [2, 1],
                "dim1/code": [[14, 15], [7]],
                "dim1/numeric_value": [[-0.0539279, 1.1927332], [float('nan')]],
                "dim1/bounds": [2, 3],
            }, schema={
                "dim1/time_delta_days": "float32",
                "dim2/code": "uint8",
                "dim2/numeric_value": "float32"
            }, pre_raggedified=True)
            len(J)
            J[0]
            """
        else:
            future_dynamic = future_dynamic[start_idx:end_idx]

        future_dynamic = future_dynamic.tensors

        count = 0
        for i in range(len(future_dynamic["dim1/code"])):
            for j in range(len(future_dynamic["dim1/code"][i])):
                if future_dynamic["dim1/code"][i][j] == query["idx"]:
                    if query["has_value"]:
                        x = future_dynamic["dim1/numeric_value"][i][j]
                        if x is None:
                            continue  # todo: is None used for outlier removal?
                        if (x >= query["range_min"]) and (x <= query["range_max"]):
                            count += 1
                    else:
                        count += 1

        return count

    def get_future_duration(self, subject_id, context_end_idx, record_end_idx):
        """
        alternative option to compute future duration
        future_dynamic = subj_dynamic[context["end_idx"] : record_end_idx].tensors
        time_delta = future_dynamic["dim0/time_delta_days"] * 1440
        if np.isnan(time_delta[0]):
            time_delta[0] = 0
        times = np.cumsum(time_delta)
        future_duration = times[-1]
        """
        shard = self.subj_map[subject_id]
        subject_idx = self.subj_indices[subject_id]
        static_row = self.static_dfs[shard][subject_idx].to_dict()
        context_end_time = static_row["time"].list.get(context_end_idx - 1)
        if record_end_idx == static_row["time"].list.len().item():
            record_end_time = static_row["time"].list.get(-1)
        else:
            record_end_time = static_row["time"].list.get(record_end_idx)
        future_duration = (record_end_time - context_end_time).dt.total_minutes().item()
        return future_duration

    @SeedableMixin.WithSeed
    def _seeded_getitem(self, idx: int) -> dict[str, list[float]]:
        context = super()._seeded_getitem(idx)

        subj_dynamic, subject_id, record_start_idx, record_end_idx = super().load_subject_dynamic_data(idx)

        future_duration = self.get_future_duration(subject_id, context["end_idx"], record_end_idx)
        future, is_censored = self.sample_future(max_valid_duration=future_duration)
        event = self.sample_event()
        query = future | event

        if is_censored:
            answer = {"censored": is_censored, "count": None, "occurs": None}
        else:
            future_dynamic = subj_dynamic[context["end_idx"] : record_end_idx]
            count = self.tally_answer(future_dynamic, query)
            answer = {"censored": is_censored, "count": count, "occurs": count != 0}

        query = self.normalize_future(query)

        item = {"context": context, "query": query, "answer": answer}

        return item


"""@pytest.mark.parametrize(
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
"""


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
    assert item["context"].keys() == {"static_indices", "static_values", "dynamic", "start_idx", "end_idx"}
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
            "start_idx",
            "end_idx",
        }
    else:
        raise NotImplementedError(f"{collate_type} not implemented")
