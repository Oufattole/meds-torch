import rootutils

root = rootutils.setup_root(__file__, dotenv=True, pythonpath=True, cwd=True)

import random
from pathlib import Path

import numpy as np
import omegaconf
import polars as pl
import pytest
import scipy
import torch
from mixins import SeedableMixin

from meds_torch.data.components.pytorch_dataset import PytorchDataset
from tests.conftest import create_cfg


class EveryQueryDataset(PytorchDataset):
    def __init__(self, cfg, split):
        cfg.max_seq_len = 5
        cfg.do_include_subsequence_indices = True
        super().__init__(cfg, split)

        self.code_strategies = ["uniform", "frequency"]
        if self.config.code_sampling_strategy not in self.code_strategies:
            raise ValueError(f"code_sampling_strategy must be one of {self.code_strategies}.")
        self.value_strategies = ["ignore", "random_quantile", "random_normal", "manual"]
        if self.config.default_value_sampling_strategy not in self.value_strategies:
            raise ValueError(f"default_value_sampling_strategy must be one of {self.value_strategies}")

        self.metadata = self._load_data()

        obj = self.config.get("codes", None)
        if obj is not None:
            if not isinstance(obj, omegaconf.listconfig.ListConfig):
                raise TypeError(f"codes must be a list, got {type(obj)}")
        self.set_codes(obj)

        for param, expected_type in [
            ("values_ignore", omegaconf.listconfig.ListConfig),
            ("values_random_quantile", omegaconf.listconfig.ListConfig),
            ("values_random_normal", omegaconf.listconfig.ListConfig),
            ("values_manual", omegaconf.dictconfig.DictConfig),
        ]:
            obj = self.config.get(param, None)
            if obj is not None:
                if not isinstance(obj, expected_type):
                    raise TypeError(f"{param} should be {expected_type}, but got {type(obj)}")
                self.set_values(strategy=param.replace("values_", ""), data=obj)

        # future
        self.future_strategies = ["within_record", "random", "fixed"]
        if self.config.min_offset < 0:
            raise ValueError("min_query_offset must be non-negative.")
        if self.config.min_duration < 0:
            raise ValueError("min_query_duration must be non-negative.")
        if self.config.min_offset >= self.config.max_offset:
            raise ValueError("min_query_offset must be less than max_query_offset.")
        if self.config.min_duration >= self.config.max_duration:
            raise ValueError("min_query_duration must be less than max_query_duration.")
        if self.config.duration_sampling_strategy not in self.future_strategies:
            raise ValueError(f"duration_sampling_strategy must be one of {self.future_strategies}.")
        if self.config.duration_sampling_strategy == "fixed":
            if self.config.fixed_duration is None:
                raise ValueError("query_fixed_duration must be specified for 'fixed' sampling strategy.")
            if self.config.fixed_duration < 0:
                raise ValueError("query_fixed_duration must be non-negative.")
        if self.config.offset_sampling_strategy not in self.future_strategies:
            raise ValueError(f"offset_sampling_strategy must be one of {self.future_strategies}.")
        if self.config.offset_sampling_strategy == "fixed":
            if self.config.fixed_offset is None:
                raise ValueError("query_fixed_offset must be specified for 'fixed' sampling strategy.")
            if self.config.fixed_offset < 0:
                raise ValueError("query_fixed_offset must be non-negative.")

    def _load_data(self):
        return (
            pl.read_parquet(self.config.code_metadata_fp)
            .filter(pl.col("code").is_not_null())
            .with_columns(
                pl.col("values/min").alias("values/quantile/0"),
                pl.col("values/quantiles").struct.field("values/quantile/0.25").alias("values/quantile/25"),
                pl.col("values/quantiles").struct.field("values/quantile/0.5").alias("values/quantile/50"),
                pl.col("values/quantiles").struct.field("values/quantile/0.75").alias("values/quantile/75"),
                pl.col("values/max").alias("values/quantile/100"),
                (pl.col("values/n_occurrences") > 0).alias("code/has_value"),
                (pl.col("values/sum") / pl.col("values/n_occurrences")).alias("values/mean"),
                pl.lit(self.config.default_value_sampling_strategy).alias("values/strategy"),
                pl.lit([]).cast(pl.List(pl.List(pl.Float64))).alias("values/range_options"),
            )
            .with_columns(
                (
                    (pl.col("values/sum_sqd") / pl.col("values/n_occurrences")) - (pl.col("values/mean")) ** 2
                ).alias("values/variance"),
            )
            .with_columns(
                pl.col("values/variance").sqrt().alias("values/std"),
            )
        )

    def _set_data_at_code(self, code, col, value):
        code = code.lower()
        self.metadata = self.metadata.with_columns(
            pl.when(pl.col("code").str.to_lowercase() == code)
            .then(pl.lit(value))
            .otherwise(pl.col(col))
            .alias(col)
        )

    def _get_data_at_code(self, code, col):
        code = code.lower()
        return self.metadata.filter(pl.col("code").str.to_lowercase() == code).select(col).item()

    def _validate_codes(self, codes):
        valid_codes = {x.lower() for x in self.metadata["code"].to_list()}
        for x in codes:
            x = x.lower()
            if x not in valid_codes:
                raise ValueError(f"Code '{x}' is not found in metadata\n\nValid options: {valid_codes}")
        return

    def _validate_range_bound(self, x):
        if isinstance(x, str):
            if not x.startswith("Q"):
                raise ValueError(f"String value '{x}' start with Q followed by the quantile.")
            if float(x.replace("Q", "")) not in self.config.quantiles:
                raise ValueError(f"Quantile '{x}' not supported, options are {self.config.quantiles}.")
        elif not isinstance(x, (int, float)):
            raise ValueError(f"Value '{x}' must be an int, float, or str.")

    def set_codes(self, codes: list[str] = None):
        if codes is None or not codes:
            self.code_options = self.metadata
        else:
            self._validate_codes(codes)
            codes = [x.lower() for x in codes]
            self.code_options = self.metadata.filter(pl.col("code").str.to_lowercase().is_in(codes))
        self.code_options = self.code_options.with_columns(
            (pl.col("code/n_occurrences") / pl.col("code/n_occurrences").sum()).alias("code/frequency")
        )

    def set_values(self, strategy: str, data: list | dict):
        assert strategy in self.value_strategies
        if strategy == "manual":
            assert isinstance(data, dict) or isinstance(data, omegaconf.dictconfig.DictConfig)
        else:
            assert isinstance(data, list) or isinstance(data, omegaconf.listconfig.ListConfig)

        codes = data.keys() if strategy == "manual" else data
        self._validate_codes(codes)

        for code in codes:
            self._set_data_at_code(code=code, col="values/strategy", value=strategy)
            if strategy == "manual":
                ranges = []
                for lower, upper in data[code]:
                    self._validate_range_bound(lower)
                    self._validate_range_bound(upper)
                    if isinstance(lower, str):
                        lower_quantile = int(lower.replace("Q", ""))
                        lower = self._get_data_at_code(code=code, col=f"values/quantile/{lower_quantile}")
                    if isinstance(upper, str):
                        upper_quantile = int(upper.replace("Q", ""))
                        upper = self._get_data_at_code(code=code, col=f"values/quantile/{upper_quantile}")
                    assert lower <= upper
                    # normalize the range based on mean/std from metadata
                    ranges.append([float(lower), float(upper)])
                self._set_data_at_code(code=code, col="values/range_options", value=ranges)

        # refresh code options with updated value info
        self.set_codes(codes=self.code_options["code"].to_list())

    def sample_code(self):
        match self.config.code_sampling_strategy:
            case "uniform":
                options = self.code_options
            case "frequency":
                num_buckets = int(self.code_options["code/frequency"].log(base=10).floor().to_numpy().min())
                bucket = np.random.choice([*range(num_buckets, 0)])
                lower, upper = np.logspace(bucket, bucket + 1, 2)
                options = self.code_options.filter(
                    pl.col("code/frequency").is_between(lower_bound=lower, upper_bound=upper)
                )
        code = options.sample().to_dicts()[0]
        return code

    def sample_value_range(self, code):
        match code["values/strategy"]:
            case "manual":
                lower, upper = random.choice(code["values/range_options"])
            case "random_quantile":
                lower_quantile, upper_quantile = sorted(random.sample(self.config.quantiles, 2))
                lower = code[f"values/quantile/{int(lower_quantile)}"]
                upper = code[f"values/quantile/{int(upper_quantile)}"]
            case "random_normal":
                # random interval from the support of the normal distribution sampled according to its density
                def _normal_support(mu, sigma):
                    return scipy.stats.norm.ppf(np.random.rand(), loc=mu, scale=sigma)

                mu, sigma = code["values/mean"], code["values/std"]
                lower, upper = sorted([_normal_support(mu, sigma), _normal_support(mu, sigma)])
        return lower, upper

    def sample_event(self):
        code = self.sample_code()
        if code["code/has_value"] and code["values/strategy"] != "ignore":
            use_value = True
            lower, upper = self.sample_value_range(code)
        else:
            use_value = False
            lower, upper = 0, 0  # mask later if needed
        event = {
            "name": code["code"],
            "vocab_index": code["code/vocab_index"],
            "has_value": code["code/has_value"],
            "use_value": use_value,
            "range_lower": lower,
            "range_upper": upper,
        }
        return event

    def normalize(self, query):
        if self.config.normalize_query:
            query["duration"] = (query["duration"] - self.config.min_duration) / (
                self.config.max_duration - self.config.min_duration
            )
            query["offset"] = (query["offset"] - self.config.min_offset) / (
                self.config.max_offset - self.config.min_offset
            )
            # tbd: query['range_lower'], query['range_upper']
            # range is provided in un-normalized units by user
            # can use mean/std from metadata
            # change in the preprocessing
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
            # end_idx is the first index you can't use
            # can use np.searchsorted, times list is sorted
            assert start_idx <= end_idx

        if start_idx == end_idx:
            # query is short and fits between two measurements, ie. has no data
            return 0
        else:
            future_dynamic = future_dynamic[start_idx:end_idx]

        future_dynamic = future_dynamic.tensors

        count = 0
        for i in range(len(future_dynamic["dim1/code"])):
            for j in range(len(future_dynamic["dim1/code"][i])):
                if future_dynamic["dim1/code"][i][j] == query["vocab_index"]:
                    if query["has_value"] and query["use_value"]:
                        x = future_dynamic["dim1/numeric_value"][i][j]
                        if x is None:
                            continue  # is None used for outlier removal?
                        if (x >= query["range_lower"]) and (x <= query["range_upper"]):
                            count += 1
                    else:
                        count += 1

        return count

    def get_subject_times(self, subject_id):
        """
        alternative option to compute times
        time_delta = dynamic["dim0/time_delta_days"] * 1440
        if np.isnan(time_delta[0]):
            time_delta[0] = 0
        times = np.cumsum(time_delta)
        """
        shard = self.subj_map[subject_id]
        subject_idx = self.subj_indices[subject_id]
        static_row = self.static_dfs[shard][subject_idx].to_dict()
        times = static_row["time"].to_numpy()[0].astype("datetime64[m]")
        return times

    def get_future_duration(self, subject_id, context_end_idx, record_end_idx):
        assert context_end_idx <= record_end_idx
        times = self.get_subject_times(subject_id)
        # should be the timestamp at which the context ends (and not the timestamp of the next event)
        context_end_time = times[context_end_idx - 1]
        # should be the last timestamp included in the record
        # not the first timestamp after the end of the record
        # last time you can use, not the first time you can't use
        record_end_time = times[record_end_idx - 1]
        future_duration = (record_end_time - context_end_time) / np.timedelta64(1, "m")
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

        query = self.normalize(query)

        item = {"context": context, "query": query, "answer": answer}

        return item

    def _query_collate(self, batch: list[dict]) -> dict:
        return {
            "offset": torch.tensor([x["offset"] for x in batch], dtype=torch.float64),
            "duration": torch.tensor([x["duration"] for x in batch], dtype=torch.float64),
            "vocab_index": torch.tensor([x["vocab_index"] for x in batch], dtype=torch.int64),
            "has_value": torch.tensor([x["has_value"] for x in batch], dtype=torch.bool),
            "use_value": torch.tensor([x["use_value"] for x in batch], dtype=torch.bool),
            "range_lower": torch.tensor([x["range_lower"] for x in batch], dtype=torch.float64),
            "range_upper": torch.tensor([x["range_upper"] for x in batch], dtype=torch.float64),
        }

    def _answer_collate(self, batch: list[dict]) -> dict:
        return {
            "censored": torch.tensor([x["censored"] for x in batch], dtype=torch.bool),
            "count": torch.tensor([x["count"] for x in batch], dtype=torch.int64),
            "occurs": torch.tensor([x["occurs"] for x in batch], dtype=torch.bool),
        }

    def collate(self, batch: list[dict]) -> dict:
        return {
            "context": super().collate([x["context"] for x in batch]),
            "query": self._query_collate([x["query"] for x in batch]),
            "answer": self._answer_collate([x["answer"] for x in batch]),
        }


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
    batch = pyd.collate([pyd[i] for i in range(2)])
    assert batch.keys() == {
        "context",
        "query",
        "answer",
    }
    assert batch["context"].keys() == {
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
