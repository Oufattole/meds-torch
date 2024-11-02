import numpy as np
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
        >>> ecgs, loc_map = extract_nested_data(data, "ecg")
        >>> len(ecgs)  # Number of non-empty ECGs
        1
        >>> np.array(ecgs[0]).round(1)  # Verify we got the actual ECG data
        array([[0.2, 0.3],
               [0.4, 0.9]])
        >>> sorted(loc_map[0])  # Location should be a tuple of indices
        [0, 0, 0]
        >>> more_data = JointNestedRaggedTensorDict({
        ...     "subject_id": [1, 2],
        ...     "time": [[0,1], [0]],
        ...     "ecg": [[[dummy_ecg,[[]]], [[[]]]], [[[[]]]]]
        ... })
        >>> ecgs, loc_map = extract_nested_data(more_data, "ecg")

    """
    extracted_data = []
    location_map = {}

    # First pop out just the key we want
    _, key_data = pop_key(jnrt, key)
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

        # Base case: if we're at a leaf node that could contain valid data
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

    return extracted_data, location_map


# def create_multimodal_jnrt(data: list, dtype: np.dtype) -> JointNestedRaggedTensorDict:
#     """Creates a new JNRT from extracted multimodal data (like ECGs).

#     Args:
#         data: List of multimodal data arrays/lists
#         dtype: The numpy dtype for the data

#     Returns:
#         A new JointNestedRaggedTensorDict containing the padded multimodal data

#     Examples:
#         >>> dummy_ecgs = [
#         ...     [[0.2, 0.2], [0.0, 0.9]],
#         ...     [[0.1, 0.1], [0.2, 0.8], [0.3, 0.7]]
#         ... ]
#         >>> ecg_jnrt = create_multimodal_jnrt(dummy_ecgs, np.float32)
#     """
#     if not data:
#         return JointNestedRaggedTensorDict({"data": []}, schema={"data": dtype})

#     # Convert to numpy arrays if needed
#     arrays = [np.array(d, dtype=dtype) if not isinstance(d, np.ndarray) else d for d in data]

#     # Create the raw tensor structure
#     raw_tensors = {"data": arrays}

#     return JointNestedRaggedTensorDict(raw_tensors=raw_tensors, schema={"data": dtype})

# def flatten_except_keys(jnrt: JointNestedRaggedTensorDict,
#                        exclude_keys: list[str],
#                        dim: int) -> JointNestedRaggedTensorDict:
#     """Flattens a JNRT along a specified dimension while excluding certain keys.

#     Args:
#         jnrt: The source JointNestedRaggedTensorDict
#         exclude_keys: Keys to exclude from flattening
#         dim: The dimension to flatten along

#     Returns:
#         A flattened JointNestedRaggedTensorDict

#     Examples:
#         >>> dummy_ecg = [[0.2, 0.2], [0.0, 0.9]]
#         >>> data = JointNestedRaggedTensorDict({
#         ...     "subject_id": [1],
#         ...     "time": [[0,1]],
#         ...     "code": [[[1,2], [3]]],
#         ...     "ecg": [[[dummy_ecg,[[]]], [[[]]]]]
#         ... })
#         >>> flattened = flatten_except_keys(data, ["ecg"], 1)
#     """
#     # First pop all excluded keys
#     excluded_jnrts = []
#     current_jnrt = jnrt

#     for key in exclude_keys:
#         if key in current_jnrt.keys():
#             current_jnrt, excluded = pop_key(current_jnrt, key)
#             excluded_jnrts.append((key, excluded))

#     # Flatten the remaining data
#     flattened = current_jnrt.flatten(dim=-1) if dim == -1 else current_jnrt

#     # Reconstruct with excluded keys
#     for key, excluded_jnrt in excluded_jnrts:
#         tensors = flattened.tensors.copy()
#         tensors.update(excluded_jnrt.tensors)
#         schema = flattened.schema.copy()
#         schema.update(excluded_jnrt.schema)
#         flattened = JointNestedRaggedTensorDict(processed_tensors=tensors, schema=schema)

#     return flattened


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

    def collate(self, batch: list[dict]) -> dict:
        """Collate a batch of randomly windowed sequences.

        Args:
            batch (List[dict]): A list of dictionaries, each containing windowed sequences.

        Returns:
            dict: A dictionary with collated data for each window.
        """
        # return process_multimodal_batch(batch, do_flatten_tensors=self.config.do_flatten_tensors)
