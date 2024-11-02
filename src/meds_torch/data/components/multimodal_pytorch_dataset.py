import numpy as np
import torch
from mixins import TimeableMixin
from nested_ragged_tensors.ragged_numpy import JointNestedRaggedTensorDict

from meds_torch.data.components.pytorch_dataset import PytorchDataset


def pop_key(
    jnrt: JointNestedRaggedTensorDict, key: str
) -> tuple[JointNestedRaggedTensorDict, JointNestedRaggedTensorDict]:
    """Pops a key from the JNRT.

    Args:
        jnrt: The source JointNestedRaggedTensorDict
        key: The key to remove

    Returns:
        A tuple of (jnrt_without_key, jnrt_with_only_key)

    Examples:
        >>> data = JointNestedRaggedTensorDict({
        ...     "subject_id": [1],
        ...     "time": [[0,1]],
        ...     "code": [[[1,2], [3]]],
        ... })
        >>> remainder, popped = pop_key(data, "code")
        >>> sorted(list(remainder.to_dense().keys()))
        ['dim1/mask', 'subject_id', 'time']
        >>> popped.to_dense()['code'].tolist()
        [[[1, 2], [3, 0]]]
        >>> dummy_ecg = [[0.2, 0.3], [0.4, 0.5]]
        >>> data = JointNestedRaggedTensorDict({
        ...     "subject_id": [1],
        ...     "time": [[0,1]],
        ...     "code": [[[1,2], [3]]],
        ...     "ecg": [[[dummy_ecg,[[]]], [[[]]]]]
        ... })
        >>> remainder, popped = pop_key(data, "ecg")
        >>> sorted(list(remainder.to_dense().keys()))
        ['code', 'dim1/mask', 'dim2/mask', 'subject_id', 'time']
        >>> remainder.to_dense()['code'].tolist()
        [[[1, 2], [3, 0]]]
        >>> popped.keys()
        {'ecg'}
        >>> ecg_data = popped.to_dense()['ecg']
        >>> ecg_data.shape
        (1, 2, 2, 2, 2)
        >>> ecg_data[0]
        array([[[[0.2, 0.3],
                 [0.4, 0.5]],
        <BLANKLINE>
                [[0. , 0. ],
                 [0. , 0. ]]],
        <BLANKLINE>
        <BLANKLINE>
               [[[0. , 0. ],
                 [0. , 0. ]],
        <BLANKLINE>
                [[0. , 0. ],
                 [0. , 0. ]]]], dtype=float32)
    """
    # Create two new dictionaries to hold the separated tensors
    remaining_tensors = {}
    popped_tensors = {}

    # Separate the tensors
    for tensor_key in jnrt.tensors.keys():
        dim_str, name = tensor_key.split("/")
        if name == key:
            popped_tensors[tensor_key] = jnrt.tensors[tensor_key]
        elif name == "bounds":
            # Include bounds for the popped dimension and higher
            popped_tensors[tensor_key] = jnrt.tensors[tensor_key]
            remaining_tensors[tensor_key] = jnrt.tensors[tensor_key]
        else:
            remaining_tensors[tensor_key] = jnrt.tensors[tensor_key]

    # Create new schema dicts
    remaining_schema = {k: v for k, v in jnrt.schema.items() if k != key}
    popped_schema = {key: jnrt.schema[key]}

    return (
        JointNestedRaggedTensorDict(processed_tensors=remaining_tensors, schema=remaining_schema),
        JointNestedRaggedTensorDict(processed_tensors=popped_tensors, schema=popped_schema),
    )


def extract_nested_data(jnrt: JointNestedRaggedTensorDict, key: str) -> tuple[list, dict]:
    """Extracts nested data (like ECGs) from a JNRT along with metadata about their original locations.

    Args:
        jnrt: The source JointNestedRaggedTensorDict
        key: The key containing nested data to extract

    Returns:
        A tuple of (extracted_data, location_map) where:
        - extracted_data is a list of the extracted nested data
        - location_map is a dict mapping from position in extracted_data to original indices

    Examples:
        >>> dummy_ecg = [[0.2, 0.3], [0.4, 0.9]]
        >>> data = JointNestedRaggedTensorDict({
        ...     "subject_id": [1],
        ...     "time": [[0,1]],
        ...     "code": [[[1,2],[3]]],
        ...     "ecg": [[[dummy_ecg,[[]]], [[[]]]]]
        ... })
        >>> ts_data, ecgs, loc_map = extract_nested_data(data, "ecg")
        >>> len(ecgs)  # Number of non-empty ECGs
        1
        >>> np.array(ecgs[0]).round(1)  # Verify we got the actual ECG data
        array([[0.2, 0.3],
               [0.4, 0.9]])
        >>> sorted(loc_map[0])  # Location should be a tuple of indices
        [0, 0, 0]
        >>> dummy_ecg_2 = [[1.2, 1.3], [1.4, 1.9]]
        >>> more_data = JointNestedRaggedTensorDict({
        ...     "subject_id": [1, 2],
        ...     "time": [[0,1], [0]],
        ...     "code": [[[1,2],[3]], [[4]]],
        ...     "ecg": [[[dummy_ecg,[[]]], [[[]]]], [[dummy_ecg_2]]]
        ... })
        >>> ts_data, ecgs, loc_map = extract_nested_data(more_data, "ecg")
        >>> len(ecgs) # Still only one real ECG
        2
        >>> np.array(ecgs[0]).round(1)
        array([[0.2, 0.3],
               [0.4, 0.9]])
        >>> np.array(ecgs[1]).round(1)
        array([[1.2, 1.3],
               [1.4, 1.9]])
    """
    extracted_data = []
    location_map = {}

    # First pop out just the key we want
    ts_data, key_data = pop_key(jnrt, key)
    max_modality_key_dim = key_data._get_dim(key)
    min_modality_key_dim = jnrt._get_dim("code")

    def is_valid_data(arr) -> bool:
        """Check if an array contains valid data (non-empty 2D array with actual values)."""
        if not isinstance(arr, (list, np.ndarray)) or len(arr) == 0:
            return False
        if not isinstance(arr[0], (list, np.ndarray)) or len(arr[0]) == 0:
            return False
        # Verify it's a 2D array with values
        return all(isinstance(x, (list, np.ndarray)) and len(x) > 0 for x in arr)

    def extract_data_recursive(key_data, extracted_data, location_map, dim, curr_indices):
        """Recursively extract nested data while tracking indices.

        Args:
            key_data: The JNRT containing just the target key data
            extracted_data: List to collect valid data arrays
            location_map: Dict to track original indices
            dim: Current dimension being processed
            curr_indices: List of current indices in the traversal
        """
        # Convert to dense at current level to iterate
        dense_data = key_data.to_dense()

        # Base case: if we're at a leaf node that could contains valid data
        if dim >= max_modality_key_dim:
            if isinstance(dense_data, dict) and key in dense_data:
                data = dense_data[key]
                if is_valid_data(data):
                    idx = len(extracted_data)
                    extracted_data.append(data.tolist() if isinstance(data, np.ndarray) else data)
                    location_map[idx] = tuple(curr_indices)
            return

        # Recursive case: traverse the structure
        if isinstance(dense_data, dict) and key in dense_data:
            data = dense_data[key]
            for i, subdata in enumerate(data):
                # Create slice for this index
                subdata_jnrt = key_data[i]
                extract_data_recursive(
                    subdata_jnrt, extracted_data, location_map, dim + 1, curr_indices + [i]
                )

    extract_data_recursive(key_data, extracted_data, location_map, min_modality_key_dim - 1, [])

    return ts_data, extracted_data, location_map


class MultimodalPytorchDataset(PytorchDataset):
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

    @TimeableMixin
    def collate(self, batch: list[dict]) -> dict:
        """Collate a batch of randomly windowed sequences.

        Args:
            batch (List[dict]): A list of dictionaries, each containing windowed sequences.

        Returns:
            dict: A dictionary with collated data for each window, including extracted modality data
                and location mapping.
        """
        return collate(batch)


def pyd_collate(batch):
    if isinstance(batch, list):
        data = JointNestedRaggedTensorDict.vstack([item["dynamic"] for item in batch]).to_dense()
    elif isinstance(batch, JointNestedRaggedTensorDict):
        data = batch.to_dense()
    else:
        raise ValueError(f"Invalid batch type {type(batch)}!")
    tensorized = {k: torch.as_tensor(v) for k, v in data.items()}
    tensorized["code"] = tensorized["code"].long()
    tensorized["mask"] = tensorized.pop("dim1/mask")
    tensorized["numeric_value_mask"] = ~torch.isnan(tensorized["numeric_value"])
    tensorized["time_delta_days"] = torch.nan_to_num(tensorized["time_delta_days"], nan=0).float()
    tensorized["numeric_value"] = torch.nan_to_num(tensorized["numeric_value"], nan=0).float()
    return tensorized


def collate(batch: list[dict]) -> dict:
    """Collate a batch of randomly windowed sequences.

    Args:
        batch (List[dict]): A list of dictionaries, each containing windowed sequences.

    Returns:
        dict: A dictionary with collated data for each window, including extracted modality data
            and location mapping.

    Examples:
        >>> # Create dummy ECG data
        >>> ecg1 = [[0.2, 0.3], [0.4, 0.5]]
        >>> ecg2 = [[0.6, 0.7], [0.8, 0.9]]
        >>>
        >>> # Create sample batch data
        >>> batch = [
        ...     {
        ...         "dynamic": JointNestedRaggedTensorDict({
        ...             "subject_id": [1],
        ...             "code": [[[1, 2], [3]]],
        ...             "ecg": [[[ecg1, [[]]], [[[]]]]],
        ...             "numeric_value": [[1.5, float('nan')]],
        ...             "time_delta_days": [[2.5, float('nan')]]
        ...         })
        ...     },
        ...     {
        ...         "dynamic": JointNestedRaggedTensorDict({
        ...             "subject_id": [2],
        ...             "code": [[[4]]],
        ...             "ecg": [[[ecg2]]],
        ...             "numeric_value": [[2.5]],
        ...             "time_delta_days": [[3.0]]
        ...         })
        ...     }
        ... ]
        >>>
        >>> # Create dataset instance and collate batch
        >>> result = collate(batch)
        >>>
        >>> # Check basic structure
        >>> for each in sorted(result.keys()): print(each)
        code
        dim2/mask
        dim3/mask
        ecg
        ecg_location_map
        mask
        numeric_value
        numeric_value_mask
        subject_id
        time_delta_days
        >>>
        >>> # Verify subject_id tensor
        >>> result['subject_id'].tolist()
        [[1], [2]]
        >>>
        >>> # Check ECG data extraction
        >>> len(result['ecg'])
        2
        >>>
        >>> # Verify first ECG
        >>> np.array(result['ecg'][0]).round(2).tolist()
        [[0.2, 0.3], [0.4, 0.5]]
        >>>
        >>> # Verify second ECG
        >>> np.array(result['ecg'][1]).round(2).tolist()
        [[0.6, 0.7], [0.8, 0.9]]
        >>>
        >>> # Check location mapping
        >>> sorted(result['ecg_location_map'].items())  # doctest: +NORMALIZE_WHITESPACE
        [(0, (0, 0, 0)), (1, (1, 0, 0))]
        >>>
        >>> # Verify numeric values and masks
        >>> result['numeric_value'].tolist()
        [1.5, 0.0, 2.5]
        >>> result['numeric_value_mask'].tolist()
        [True, False, True]
        >>>
        >>> # Verify time delta days
        >>> result['time_delta_days'].tolist()
        [2.5, 0.0, 3.0]
        >>>
        >>> # Test batch with no ECG data
        >>> batch_no_ecg = [
        ...     {
        ...         "dynamic": JointNestedRaggedTensorDict({
        ...             "subject_id": [1],
        ...             "code": [[[1]]],
        ...             "numeric_value": [[1.0]],
        ...             "time_delta_days": [[1.0]]
        ...         })
        ...     }
        ... ]
        >>> result_no_ecg = collate(batch_no_ecg)
        >>> 'ecg' in result_no_ecg
        False
        >>>
        >>> # Test batch with empty ECG arrays
        >>> batch_empty_ecg = [
        ...     {
        ...         "dynamic": JointNestedRaggedTensorDict({
        ...             "subject_id": [1],
        ...             "code": [[[1]]],
        ...             "ecg": [[[[]]]],
        ...             "numeric_value": [[1.0]],
        ...             "time_delta_days": [[1.0]]
        ...         })
        ...     }
        ... ]
        >>> result_empty_ecg = collate(batch_empty_ecg)
        >>> result_empty_ecg['ecg']
        []
        >>> result_empty_ecg['ecg_location_map']
        {}
    """
    jnrt = JointNestedRaggedTensorDict.vstack([item["dynamic"] for item in batch])
    modality_key = "ecg"

    # Only process modality if it exists in the data
    if modality_key in jnrt.keys():
        ts_data, extracted_data, location_map = extract_nested_data(jnrt, modality_key)
        tensorized = pyd_collate(ts_data)
        tensorized[modality_key] = extracted_data
        tensorized[f"{modality_key}_location_map"] = location_map
        return tensorized
    else:
        return pyd_collate(jnrt)
