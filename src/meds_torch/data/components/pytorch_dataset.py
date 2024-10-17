from collections import defaultdict
from enum import StrEnum
from pathlib import Path

import numpy as np
import polars as pl
import torch
from loguru import logger
from mixins import SeedableMixin, TimeableMixin
from nested_ragged_tensors.ragged_numpy import (
    NP_FLOAT_TYPES,
    NP_INT_TYPES,
    NP_UINT_TYPES,
    JointNestedRaggedTensorDict,
)
from omegaconf import DictConfig

IDX_COL = "_row_index"
BINARY_LABEL_COL = "boolean_value"


class CollateType(StrEnum):
    event_stream = "event_stream"
    triplet = "triplet"
    triplet_prompt = "triplet_prompt"
    eic = "eic"


def subpad_vectors(a: np.ndarray, b: np.ndarray):
    """Create a new array by placing elements of 'a' at indices specified by 'b'.

    This function creates an array of zeros with length equal to the maximum value in 'b',
    then places the values from 'a' at the indices specified by 'b'.

    Args:
        a (numpy.ndarray): The source array containing values to be placed.
        b (numpy.ndarray): The array specifying the indices where values from 'a' should be placed.

    Returns:
        numpy.ndarray: A new array with values from 'a' placed at indices specified by 'b',
                       and zeros elsewhere.

    Example:
    >>> a = np.array([2, 4, 5])
    >>> b = np.array([3, 5, 10])
    >>> subpad_vectors(a, b)
    array([2, 0, 0, 4, 0, 5, 0, 0, 0, 0])
    """
    total_length = b[-1]
    result = np.zeros(total_length, dtype=a.dtype)
    result[[0] + list(b[:-1])] = a
    return result


class SubsequenceSamplingStrategy(StrEnum):
    """An enumeration of the possible subsequence sampling strategies for the dataset.

    Attributes:     RANDOM: Randomly sample a subsequence from the full sequence.     TO_END: Sample a
    subsequence from the end of the full sequence.         Note this starts at the last element and moves
    back.     FROM_START: Sample a subsequence from the start of the full sequence.
    """

    RANDOM = "random"
    TO_END = "to_end"
    FROM_START = "from_start"


class SeqPaddingSide(StrEnum):
    """An enumeration of the possible sequence padding sides for the dataset."""

    LEFT = "left"
    RIGHT = "right"


def merge_task_with_static(task_df: pl.DataFrame, static_dfs: dict[str, pl.DataFrame]):
    """Merges a DataFrame containing task information with multiple static DataFrames on the 'subject_id'
    column. The function performs a sequence of operations to merge these dataframes based on subject
    identifiers and respective timestamps.

    Parameters:
    - task_df (DataFrame): A DataFrame with columns 'subject_id', 'prediction_time', and 'label'.
    - static_dfs (dict of DataFrames): A dictionary of DataFrames indexed by their source names,
      each containing 'subject_id', 'static_indices', 'static_values', and "time".

    Returns:
    - DataFrame: The merged DataFrame containing data from task_df and all static_dfs.

    Example:
    >>> from datetime import datetime
    >>> import polars as pl
    >>> task_df = pl.DataFrame({
    ...     "subject_id": [1, 2, 1, 3],
    ...     "prediction_time": [
    ...         datetime(2020, 1, 2), datetime(2020, 1, 3), datetime(2021, 1, 4), datetime(2021, 1, 5)
    ...     ],
    ...     "boolean_value": [False, True, True, False]
    ... })
    >>> static_dfs = {
    ...     'train/0': pl.DataFrame({
    ...         "subject_id": [1, 2],
    ...         "start_time": [datetime(2020, 1, 1), datetime(2020, 1, 2)],
    ...         "time": [[datetime(2020, 1, 1, 1), datetime(2020, 1, 1, 3)],
    ...                       [datetime(2020, 1, 2), datetime(2020, 1, 1, 2, 3)]]
    ...     })
    ... }
    >>> merge_task_with_static(task_df, static_dfs)
    shape: (3, 4)
    ┌────────────┬─────────────────────┬───────────────┬─────────────────────────────────┐
    │ subject_id ┆ prediction_time     ┆ boolean_value ┆ time                            │
    │ ---        ┆ ---                 ┆ ---           ┆ ---                             │
    │ i64        ┆ datetime[μs]        ┆ bool          ┆ list[datetime[μs]]              │
    ╞════════════╪═════════════════════╪═══════════════╪═════════════════════════════════╡
    │ 1          ┆ 2020-01-02 00:00:00 ┆ False         ┆ [2020-01-01 01:00:00, 2020-01-… │
    │ 2          ┆ 2020-01-03 00:00:00 ┆ True          ┆ [2020-01-02 00:00:00, 2020-01-… │
    │ 1          ┆ 2021-01-04 00:00:00 ┆ True          ┆ [2020-01-01 01:00:00, 2020-01-… │
    └────────────┴─────────────────────┴───────────────┴─────────────────────────────────┘
    """
    static_df = pl.concat(static_dfs.values()).select("subject_id", "time")
    return task_df.join(static_df, on="subject_id", how="inner")


def get_task_indices_and_labels(
    task_df_joint: pl.DataFrame,
) -> tuple[list[tuple[int, int, int]], dict[str, list]]:
    """Processes the joint DataFrame to determine the index range for each subject's task.

    For each row in task_df_joint, it is assumed that `time` is a sorted column and the function
    computes the index of the last event at `prediction_time`.

    Parameters:
    - task_df_joint (DataFrame): A DataFrame resulting from the merge_task_with_static function.

    Returns:
    - list: list of index tuples of format (subject_id, start_idx, end_idx).
    - dict: dictionary of task names to lists of labels in the same order as the indexes.
    """
    end_idx_expr = (
        (pl.col("time").search_sorted(pl.col("prediction_time"), side="right")).last().alias("end_idx")
    )
    label_df = (
        task_df_joint.with_row_index(IDX_COL)
        .explode("time")
        .group_by(IDX_COL, "subject_id", "prediction_time", "boolean_value", maintain_order=True)
        .agg(end_idx_expr)
    )

    subject_ids = label_df["subject_id"]
    end_indices = label_df["end_idx"]
    labels = label_df[BINARY_LABEL_COL].to_list()

    indexes = list(zip(subject_ids, end_indices))

    return indexes, labels


class PytorchDataset(SeedableMixin, torch.utils.data.Dataset, TimeableMixin):
    """A PyTorch Dataset class for handling complex, multi-modal medical data.

    This dataset is designed to work with data from the MEDS (Medical Event Data Set) format, supporting
    various types of medical events, static patient information, and task-specific labels. It provides
    functionality for loading, processing, and collating data for use in PyTorch models.

    Key Features: - Handles different collation strategies (event stream, triplet, etc.) - Supports task-
    specific data handling for binary classification - Implements custom sampling strategies and sequence
    length constraints

    Args:     cfg (DictConfig): Configuration options for the dataset.     split (str): The data split to use
    (e.g., 'train', 'validation', 'test').

    Attributes:     config (DictConfig): The dataset configuration.     split (str): The current data split.
    code_metadata (pl.LazyFrame): Metadata for event codes.     static_dfs (dict): Dictionary of static
    DataFrames for each data shard.     subj_indices (dict): Mapping of subject IDs to their indices in the
    dataset.     subj_seq_bounds (dict): Sequence bounds (start, end) for each subject.     index (list): List
    of (subject_id, start, end) tuples for data access.     labels (dict): Task-specific labels for each data
    point.     tasks (list): List of task names.

    Methods:     __len__(): Returns the number of items in the dataset.     __getitem__(idx): Retrieves a
    single data point.     collate(batch): Collates a batch of data points based on the specified collation
    strategy.
    """

    def __init__(self, cfg: DictConfig, split: str):
        super().__init__()

        self.config = cfg
        self.split = split

        logger.info("Scanning code metadata")
        self.code_metadata = pl.scan_parquet(self.config.code_metadata_fp)

        logger.info("Reading subject schema and static data")
        self.read_subject_descriptors()

    def read_subject_descriptors(self):
        """Read subject schemas and static data from the dataset.

        This method processes the Parquet files for each shard in the dataset, extracting static data and
        creating various mappings and indices for efficient data access.

        The method populates the following instance attributes: - self.static_dfs: Dictionary of static
        DataFrames for each shard. - self.subj_indices: Mapping of subject IDs to their indices. -
        self.subj_seq_bounds: Dictionary of sequence bounds for each subject. - self.index: List of
        (subject_id, start, end) tuples for data access. - self.labels: Dictionary of task labels (if tasks
        are specified).

        If tasks are specified in the configuration, this method also processes the task labels and integrates
        them with the static data.

        Raises:     ValueError: If duplicate subjects are found across shards or if                 required
        task information is missing.     FileNotFoundError: If specified task files are not found.
        """

        schema_root = Path(self.config.schema_files_root)

        self.static_dfs = {}
        self.subj_indices = {}
        self.subj_seq_bounds = {}
        self.subj_map = {}

        schema_files = list(schema_root.glob(f"{self.split}/*.parquet"))
        if not schema_files:
            raise FileNotFoundError(
                f"No schema files found in {schema_root}! If your data is not sharded by split, this error "
                "may occur because this codebase does not handle non-split sharded data. See Issue #79 for "
                "tracking this issue."
            )

        for schema_fp in schema_files:
            shard = str(schema_fp.relative_to(schema_root).with_suffix(""))

            df = (
                pl.read_parquet(
                    schema_fp,
                    columns=[
                        "subject_id",
                        "start_time",
                        "code",
                        "numeric_value",
                        "time",
                    ],
                    use_pyarrow=True,
                )
                .rename({"code": "static_indices", "numeric_value": "static_values"})
                .with_columns(
                    pl.col("static_values").list.eval(pl.element().fill_null(0)),
                    pl.col("static_indices").list.eval(pl.element().fill_null(0)),
                )
            )

            self.static_dfs[shard] = df
            subject_ids = df["subject_id"]
            self.subj_map.update({subj: shard for subj in subject_ids})

            n_events = df.select(pl.col("time").list.len().alias("n_events")).get_column("n_events")
            for i, (subj, n_events_count) in enumerate(zip(subject_ids, n_events)):
                if subj in self.subj_indices or subj in self.subj_seq_bounds:
                    raise ValueError(f"Duplicate subject {subj} in {shard}!")

                self.subj_indices[subj] = i
                self.subj_seq_bounds[subj] = (0, n_events_count)

        if self.has_task:
            task_df_fp = Path(self.config.task_label_path)
            if not task_df_fp.is_file():
                logger.info(f"If the task file is not found at {task_df_fp}")
                task_df_fp = task_df_fp.with_suffix("") / "**/*.parquet"
                logger.info(f"Searching for task parquets over the glob {task_df_fp}")

            logger.info(f"Reading task constraints for {self.config.task_name} from {task_df_fp}")
            task_df = pl.read_parquet(task_df_fp)

            idx_col = "_row_index"
            while idx_col in task_df.columns:
                idx_col = f"_{idx_col}"

            task_df_joint = merge_task_with_static(task_df, self.static_dfs)
            # Convert dates to indexes in the nested ragged tensor, (for fast indexing of data)
            subjs_and_ends, self.labels = get_task_indices_and_labels(task_df_joint)
            self.index = [(subj, 0, end) for subj, end in subjs_and_ends]
        else:
            self.index = [(subj, *bounds) for subj, bounds in self.subj_seq_bounds.items()]
            self.labels = None

    @property
    def subject_ids(self) -> list[int]:
        return [x[0] for x in self.index]

    def __len__(self):
        return len(self.index)

    @property
    def has_task(self) -> bool:
        return self.config.task_name is not None

    @property
    def max_seq_len(self) -> int:
        return self.config.max_seq_len

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        """Retrieve a single data point from the dataset.

        This method returns a dictionary corresponding to a single subject's data at the specified index. The
        data is not tensorized in this method, as that work is typically done in the collate function.

        Args:     idx (int): The index of the data point to retrieve.

        Returns:     dict: A dictionary containing the data for the specified index. The structure typically
        includes:         - 'time_delta_days': List of time deltas between events.         -
        'dynamic_indices': List of categorical metadata elements.         - 'dynamic_values': List of
        numerical metadata elements.         - 'static_indices': List of static categorical metadata elements.
        Additional keys may be present based on the dataset configuration.

        Notes:     - The exact structure of the output dictionary depends on the dataset configuration and the
        collate type specified.     - This method uses the SeedableMixin to ensure reproducibility in data
        loading.     - The returned data is typically in a 'raw' format and may require further processing or
        tensorization in the collate function.
        """
        return self._seeded_getitem(idx)

    @TimeableMixin.TimeAs
    def load_subject_dynamic_data(self, idx: int):
        """Loads and returns the dynamic data slice for a given subject index, with subject ID and time
        range."""
        subject_id, st, end = self.index[idx]
        shard = self.subj_map[subject_id]
        subject_idx = self.subj_indices[subject_id]
        subject_dynamic_data = JointNestedRaggedTensorDict.load_slice(
            Path(self.config.data_dir) / "data" / f"{shard}.nrt", subject_idx
        )
        return subject_dynamic_data, subject_id, st, end

    @SeedableMixin.WithSeed
    @TimeableMixin.TimeAs
    def load_subject(
        self, subject_dynamic_data, subject_id: int, st: int, end: int
    ) -> dict[str, list[float]]:
        """Load and process data for a single subject.

        This method retrieves and processes the data for a specific subject, applying various transformations
        and filters based on the dataset configuration.

        Args:     subject_dynamic_data: The dynamic data for the subject.     subject_id (int): The ID of the
        subject to load.     st (int): The start index of the sequence to load.     end (int): The end index
        of the sequence to load.

        Returns:     dict: A dictionary containing the processed data for the subject. The exact contents
        depend on the dataset configuration but typically include:           - Static data (indices and
        values)           - Dynamic data (time series data, event codes, etc.)           - Sequence
        information (start and end indices, if configured)

        Raises:     ValueError: If the sequence length is invalid or if there are inconsistencies in the data
        shapes.

        Notes:     - This method applies sequence length constraints and sampling strategies       as
        specified in the dataset configuration.     - It handles different collate types (event_stream,
        triplet, etc.) differently.     - The method is decorated with @SeedableMixin.WithSeed to ensure
        reproducibility.
        """
        shard = self.subj_map[subject_id]
        subject_idx = self.subj_indices[subject_id]
        static_row = self.static_dfs[shard][subject_idx].to_dict()

        out = {
            "static_indices": static_row["static_indices"].item().to_list(),
            "static_values": static_row["static_values"].item().to_list(),
        }

        # TODO: remove this and handle flattening in the NRT class
        if self.config.collate_type == CollateType.event_stream:
            seq_len = end - st
        if self.config.collate_type != CollateType.event_stream:
            event_seq_len = end - st
            tensors = subject_dynamic_data.tensors
            seq_len = sum([array.size for array in tensors["dim1/code"][st:end]])
            if not seq_len >= event_seq_len:
                raise ValueError(
                    f"Measurement sequence length {seq_len} is less than event sequence length"
                    f" {event_seq_len}!"
                )
            tensors["dim1/numeric_value"] = np.concatenate(tensors["dim1/numeric_value"][st:end], axis=0)
            tensors["dim1/code"] = np.concatenate(tensors["dim1/code"][st:end], axis=0)
            seq_len = tensors["dim1/code"].shape[0]
            tensors["dim0/time_delta_days"] = subpad_vectors(
                tensors["dim0/time_delta_days"][st:end], tensors["dim1/bounds"][st:end]
            )
            st = 0
            end = st + seq_len

        if seq_len > self.config.max_seq_len:
            match self.config.subsequence_sampling_strategy:
                case SubsequenceSamplingStrategy.RANDOM:
                    start_offset = np.random.choice(seq_len - self.config.max_seq_len)
                case SubsequenceSamplingStrategy.TO_END:
                    start_offset = seq_len - self.config.max_seq_len
                case SubsequenceSamplingStrategy.FROM_START:
                    start_offset = 0
                case _:
                    raise ValueError(
                        f"Invalid subsequence sampling strategy {self.config.subsequence_sampling_strategy}!"
                    )

            st += start_offset
            end = min(end, st + self.config.max_seq_len)

        if self.config.do_include_subsequence_indices:
            out["start_idx"] = st
            out["end_idx"] = end

        if self.config.collate_type == CollateType.event_stream:
            out["dynamic"] = subject_dynamic_data[st:end]
        else:
            tensors["dim1/code"] = tensors["dim1/code"][st:end]
            tensors["dim1/numeric_value"] = tensors["dim1/numeric_value"][st:end]
            tensors["dim0/time_delta_days"] = tensors["dim0/time_delta_days"][st:end]
            out["dynamic"] = tensors

        if self.config.do_include_start_time_min:
            out["start_time"] = static_row["time"].item().to_list()[st]

        if end - st > self.config.max_seq_len:
            raise ValueError(f"Sequence length {end - st} exceeds max_seq_len {self.config.max_seq_len}!")

        if end == st:
            raise ValueError(f"Sequence length {end - st} is 0!")

        if self.config.postpend_eos_token:
            if self.config.collate_type == CollateType.event_stream:
                # Append EOS token to the end of the sequence
                eos_token = np.array([self.config.EOS_TOKEN_ID], dtype=out["dynamic"]["dim1/code"].dtype)
                out["dynamic"]["dim1/code"] = np.append(out["dynamic"]["dim1/code"], eos_token)

                # Extend other relevant arrays
                numeric_dtype = out["dynamic"]["dim1/numeric_value"].dtype
                time_dtype = out["dynamic"]["dim0/time_delta_days"].dtype
                out["dynamic"]["dim1/numeric_value"] = np.append(
                    out["dynamic"]["dim1/numeric_value"], np.array([0], dtype=numeric_dtype)
                )
                out["dynamic"]["dim0/time_delta_days"] = np.append(
                    out["dynamic"]["dim0/time_delta_days"], np.array([0], dtype=time_dtype)
                )

            else:
                # For other collate types
                eos_token = np.array([self.config.EOS_TOKEN_ID], dtype=out["dynamic"]["dim1/code"].dtype)
                out["dynamic"]["dim1/code"] = np.append(out["dynamic"]["dim1/code"], eos_token)

                numeric_dtype = out["dynamic"]["dim1/numeric_value"].dtype
                time_dtype = out["dynamic"]["dim0/time_delta_days"].dtype
                out["dynamic"]["dim1/numeric_value"] = np.append(
                    out["dynamic"]["dim1/numeric_value"], np.array([0], dtype=numeric_dtype)
                )
                out["dynamic"]["dim0/time_delta_days"] = np.append(
                    out["dynamic"]["dim0/time_delta_days"], np.array([0], dtype=time_dtype)
                )

        # Update end_idx if it's included
        if self.config.do_include_subsequence_indices:
            out["end_idx"] = end

        if self.config.collate_type != CollateType.event_stream and not (
            len(out["dynamic"]["dim1/code"])
            == len(out["dynamic"]["dim1/numeric_value"])
            == len(out["dynamic"]["dim0/time_delta_days"])
        ):
            code_shape = out["dynamic"]["dim1/code"].shape
            numeric_shape = out["dynamic"]["dim1/numeric_value"].shape
            time_shape = out["dynamic"]["dim0/time_delta_days"].shape
            raise ValueError(f"Shape mismatch: {code_shape} vs {numeric_shape} vs {time_shape}")

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

        if "dynamic" not in out:
            raise ValueError(f"Failed to load dynamic data for subject {subject_id} at idx {idx}!")
        return out

    @TimeableMixin.TimeAs
    def __dynamic_only_collate(self, batch: list[dict[str, list[float]]]) -> dict:
        """An internal collate function for only dynamic data."""
        keys = batch[0].keys()
        dense_keys = {k for k in keys if k not in ("dynamic", "static_indices", "static_values")}

        if dense_keys:
            dense_collated = torch.utils.data.default_collate([{k: x[k] for k in dense_keys} for x in batch])
        else:
            dense_collated = {}

        dynamic = JointNestedRaggedTensorDict.vstack([x["dynamic"] for x in batch]).to_dense(
            padding_side=self.config.seq_padding_side
        )
        dynamic["event_mask"] = dynamic.pop("dim1/mask")
        dynamic["dynamic_values"] = dynamic.pop("numeric_value")
        dynamic["dynamic_indices"] = dynamic.pop("code")
        dynamic["dynamic_values_mask"] = dynamic.pop("dim2/mask") & ~np.isnan(dynamic["dynamic_values"])

        dynamic_collated = {}
        for k, v in dynamic.items():
            if k.endswith("mask"):
                dynamic_collated[k] = torch.from_numpy(v)
            elif v.dtype in NP_UINT_TYPES + NP_INT_TYPES:
                dynamic_collated[k] = torch.from_numpy(v.astype(int)).long()
            elif v.dtype in NP_FLOAT_TYPES:
                dynamic_collated[k] = torch.from_numpy(v.astype(float)).float()
            else:
                raise TypeError(f"Don't know how to tensorify {k} of type {v.dtype}!")

        collated = {**dense_collated, **dynamic_collated}

        out_batch = {}
        out_batch["event_mask"] = collated["event_mask"]
        out_batch["dynamic_values_mask"] = collated["dynamic_values_mask"]
        out_batch["time_delta_days"] = torch.nan_to_num(collated["time_delta_days"].float(), nan=0)
        out_batch["dynamic_indices"] = collated["dynamic_indices"].long()
        out_batch["dynamic_values"] = torch.nan_to_num(collated["dynamic_values"].float(), nan=0)

        if self.config.do_include_start_time_min:
            out_batch["start_time"] = collated["start_time"].float()
        if self.config.do_include_subsequence_indices:
            out_batch["start_idx"] = collated["start_idx"].long()
            out_batch["end_idx"] = collated["end_idx"].long()
        if self.config.do_include_subject_id:
            out_batch["subject_id"] = collated["subject_id"].long()

        if not self.has_task:
            return out_batch

        # add task labels
        out_batch[BINARY_LABEL_COL] = torch.tensor([e[BINARY_LABEL_COL] for e in batch], dtype=torch.float32)

        return out_batch

    def collate_event_stream(self, batch: list[dict]) -> dict:
        """Combines the ragged dictionaries produced by `__getitem__` into a tensorized batch.

        This function handles conversion of arrays to tensors and padding of elements within the batch across
        static data elements, sequence events, and dynamic data elements.

        Args:     batch: A list of `__getitem__` format output dictionaries.

        Returns:     A fully collated, tensorized, and padded batch.
        """

        out_batch = self.__dynamic_only_collate(batch)

        max_n_static = max(len(x["static_indices"]) for x in batch)
        static_padded_fields = defaultdict(list)
        for e in batch:
            n_static = len(e["static_indices"])
            static_delta = max_n_static - n_static
            for k in ("static_indices", "static_values"):
                if static_delta > 0:
                    static_padded_fields[k].append(
                        torch.nn.functional.pad(
                            torch.tensor(e[k], dtype=torch.long), (0, static_delta), value=0
                        )
                    )
                else:
                    static_padded_fields[k].append(torch.tensor(e[k], dtype=torch.long))

        for k, v in static_padded_fields.items():
            out_batch[k] = torch.cat([T.unsqueeze(0) for T in v], dim=0)

        return out_batch

    @classmethod
    def process_triplet(cls, item: dict, do_prepend_static_data=True) -> dict:
        """Process a single triplet of dynamic and static data.

        This method takes a dictionary containing dynamic and static data for a single
        data point and processes it into a unified representation suitable for model input.

        Args:
            item (dict): A dictionary containing 'dynamic' and 'static' data.
            do_prepend_static_data (bool, optional): Whether to prepend static data
                                                     to the dynamic data. Defaults to True.

        Returns:
            dict: A processed dictionary containing:
                - mask: Boolean mask indicating valid data points.
                - static_mask: Boolean mask indicating static data points.
                - code: Concatenated static and dynamic codes.
                - numeric_value: Concatenated static and dynamic numerical values.
                - time_delta_days: Time deltas between events.
                - numeric_value_mask: Boolean mask for valid numeric values.

        Notes:
            - This method handles the integration of static and dynamic data.
            - It applies appropriate type conversions and handles missing values.
            - The resulting dictionary is suitable for further tensorization or batching.

        Examples:
            >>> import numpy as np
            >>> import tempfile, json, os
            >>> from omegaconf import DictConfig
            >>> item =  {
            ...         'dynamic': {
            ...                 'dim1/code': np.array([5, 6, 1, 2]),
            ...                 'dim1/numeric_value': np.array([50.0, 60.0, np.nan, np.nan]),
            ...                 'dim0/time_delta_days': np.array([0, 0, 12, 0])
            ...         },
            ...         'static_values': [70.0],
            ...         'static_indices': [2]
            ...     }
            >>> triplet_item = PytorchDataset.process_triplet(item)
            >>> for each in sorted(list(triplet_item.keys())): print(each)
            code
            mask
            numeric_value
            numeric_value_mask
            static_mask
            time_delta_days
            >>> for key, value in triplet_item.items(): print(key, value);
            mask [ True  True  True  True  True]
            static_mask [ True False False False False]
            code tensor([2, 5, 6, 1, 2])
            numeric_value [70. 50. 60.  0.  0.]
            time_delta_days [ 0  0  0 12  0]
            numeric_value_mask [ True  True  True False False]
        """
        dynamic_data = item["dynamic"]
        code = dynamic_data["dim1/code"]
        numeric_value = dynamic_data["dim1/numeric_value"]
        time_delta_days = dynamic_data["dim0/time_delta_days"]

        static_mask = np.zeros(len(code), dtype=bool)
        if do_prepend_static_data:
            static_values = np.asarray(item["static_values"], dtype=np.float32)
            static_indices = np.asarray(item["static_indices"], dtype=np.int32)
            code = np.concatenate([static_indices, code], dtype=np.int32, casting="unsafe")
            numeric_value = np.concatenate([static_values, numeric_value])
            static_mask = np.zeros(len(code), dtype=bool)
            static_mask[: len(static_values)] = True
            time_delta_days = np.concatenate(
                [np.zeros(len(static_values), dtype=time_delta_days.dtype), time_delta_days]
            )

        numeric_value_mask = ~np.isnan(numeric_value)
        # Replace NaNs with 0s
        np.nan_to_num(numeric_value, nan=0, copy=False)
        np.nan_to_num(time_delta_days, nan=0, copy=False)

        mask = np.ones(len(code), dtype=bool)
        if not len(mask) == len(code) == len(numeric_value) == len(time_delta_days):
            raise ValueError(
                f"Shape mismatch: {code.shape} vs {mask.shape} vs "
                f"{numeric_value.shape} vs {time_delta_days.shape}"
            )

        output = dict(
            mask=mask,
            static_mask=static_mask,
            code=torch.as_tensor(code, dtype=torch.int64),
            numeric_value=numeric_value,
            time_delta_days=time_delta_days,
            numeric_value_mask=numeric_value_mask,
        )
        return output

    @classmethod
    def collate_triplet(cls, batch: list[dict], do_prepend_static_data=True) -> dict:
        """Collate a batch of triplet format data into a unified batch dictionary.

        This method combines multiple data points in triplet format (times, codes, values)
        into a single batch, applying necessary padding and tensorization.

        Args:
            batch (list[dict]): A list of dictionaries, each representing a single data point.
            do_prepend_static_data (bool, optional): Whether to prepend static data to the
                                                     dynamic data. Defaults to True.

        Returns:
            dict: A dictionary containing the collated batch data, including:
                - mask: Tensor indicating valid data points across the batch.
                - static_mask: Tensor indicating static data points.
                - code: Tensor of concatenated static and dynamic codes.
                - numeric_value: Tensor of concatenated static and dynamic numerical values.
                - time_delta_days: Tensor of time deltas between events.
                - Additional task-specific labels if present in the input data.

        Notes:
            - This method handles padding to ensure uniform sequence lengths within the batch.
            - It converts all data to PyTorch tensors.
            - The method is flexible to handle additional task-specific data present in the input.

        Examples:
            >>> import torch
            >>> import numpy as np
            >>> batch = [
            ...     {
            ...         'dynamic': {
            ...              'dim1/code': np.array([1]),
            ...              'dim1/numeric_value': np.array([10.0]),
            ...              'dim0/time_delta_days': np.array([0])},
            ...         'static_values': [20.0],
            ...         'static_indices': [0],
            ...         'label': 0,
            ...     },
            ...     {
            ...         'dynamic':{
            ...              'dim1/code': np.array([5, 6, 1, 2]),
            ...              'dim1/numeric_value': np.array([50.0, 60.0, 0, 0]),
            ...              'dim0/time_delta_days': np.array([0, 0, 12, 0])},
            ...         'static_values': [70.0],
            ...         'static_indices': [2],
            ...         'label': 1,
            ...         },
            ... ]
            >>> collated_batch = PytorchDataset.collate_triplet(batch)
            >>> from pprint import pprint
            >>> pprint(collated_batch)
            {'code': tensor([[0, 1, 0, 0, 0],
                    [2, 5, 6, 1, 2]]),
             'label': tensor([0., 1.]),
             'mask': tensor([[ True,  True, False, False, False],
                    [ True,  True,  True,  True,  True]]),
             'numeric_value': tensor([[20., 10.,  0.,  0.,  0.],
                    [70., 50., 60.,  0.,  0.]], dtype=torch.float64),
             'numeric_value_mask': tensor([[ True,  True, False, False, False],
                    [ True,  True,  True,  True,  True]]),
             'static_mask': tensor([[ True, False, False, False, False],
                    [ True, False, False, False, False]]),
             'time_delta_days': tensor([[ 0,  0,  0,  0,  0],
                    [ 0,  0,  0, 12,  0]])}
        """
        processed_batch = [cls.process_triplet(item, do_prepend_static_data) for item in batch]
        tensorized_batch = {
            k: torch.nn.utils.rnn.pad_sequence(
                [torch.as_tensor(x[k]) for x in processed_batch],
                batch_first=True,
                padding_value=0,
            )
            for k in processed_batch[0].keys()
        }

        # Add task labels to batch
        for k in batch[0].keys():
            if k not in ("dynamic", "static_values", "static_indices"):
                tensorized_batch[k] = torch.Tensor([item[k] for item in batch])
        return tensorized_batch

    @TimeableMixin.TimeAs
    def collate(self, batch: list[dict]) -> dict:
        """Combine a batch of data points into a single, tensorized batch.

        This method serves as the main collation function, handling different collation strategies based on
        the dataset configuration. It delegates to specific collation methods depending on the collate_type
        specified in the configuration.

        Args:     batch (list[dict]): A list of dictionaries, each representing a single data point as
        returned by the __getitem__ method.

        Returns:     dict: A dictionary containing the collated batch data. The exact structure depends on the
        collation strategy used.

        Raises:     NotImplementedError: If an unsupported collate type is specified in the configuration.

        Notes:     - The method supports various collation strategies including event_stream,       triplet,
        triplet_prompt, eic.     - Each collation strategy is optimized for different model architectures and
        data representations.     - The collated output is fully tensorized and padded, ready for input into a
        PyTorch model.
        """
        collate_type = self.config.collate_type
        if collate_type == CollateType.event_stream:
            return self.collate_event_stream(batch)
        elif collate_type == CollateType.triplet_prompt:
            return self.collate_triplet(batch)
        elif collate_type == CollateType.eic:
            return self.collate_triplet(batch, self.config.do_prepend_static_data)
        elif collate_type == CollateType.triplet:
            return self.collate_triplet(batch, self.config.do_prepend_static_data)
        else:
            raise NotImplementedError(f"Unsupported collate type {collate_type}!")
