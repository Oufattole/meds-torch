from nested_ragged_tensors.ragged_numpy import JointNestedRaggedTensorDict

from meds_torch.data.components.pytorch_dataset import PytorchDataset


def process_multimodal_batch(batch, do_flatten_tensors=False):
    """
    Process a batch of multimodal data, separating higher dimensional modalities (like images, ECGs)
    from regular time series data. Handles both flattened and non-flattened tensor formats.

    Args:
        batch (list): List of dictionaries containing "dynamic" JNRT data
        do_flatten_tensors (bool): Whether to flatten the tensors (changes mapping structure)

    Returns:
        dict: Contains processed data with keys:
            - 'data': Dense tensor of regular time series data
            - For each multimodal key:
                - 'data': The modal data array
                - 'bounds': List of bounds arrays for reconstruction
                - 'idx_map': Mapping from modal indices to (subject_id, [time_delta], code)

    Example:
        >>> import numpy as np
        >>> from nested_ragged_tensors.ragged_numpy import JointNestedRaggedTensorDict
        >>> # Create sample ECG data
        >>> dummy_ecg = np.random.rand(2,2).tolist()
        >>> # Create sample batch
        >>> batch = [{
        ...     "dynamic": JointNestedRaggedTensorDict(raw_tensors={
        ...         "subject_id": [1],
        ...         "time_delta_days": [[0,1]],
        ...         "code": [[[1,2], [3]]],
        ...         "ecg": [[[dummy_ecg,[[]]], [[[]]]]]
        ...     })
        ... }]
        >>> result = process_multimodal_batch(batch, do_flatten_tensors=True)
        >>> 'ecg' in result
        True
        >>> len(result['ecg']['idx_map']) == 3  # One mapping per code
        True
        >>> all(isinstance(idx[0], int) for idx in result['ecg']['idx_map'])  # Check subject_ids
        True

        >>> # Test with non-flattened tensors
        >>> result = process_multimodal_batch(batch, do_flatten_tensors=False)
        >>> len(result['ecg']['idx_map'][0]) == 3  # (subject_id, time_delta, code_idx)
        True
        >>> isinstance(result['ecg']['bounds'], list)  # Bounds stored as list
        True

        >>> # Test with regular time series data (no multimodal)
        >>> batch = [{
        ...     "dynamic": JointNestedRaggedTensorDict(raw_tensors={
        ...         "subject_id": [1],
        ...         "time_delta_days": [[0,1]],
        ...         "code": [[[1,2], [3]]]
        ...     })
        ... }]
        >>> result = process_multimodal_batch(batch)
        >>> isinstance(result['data'], dict)  # Regular data processed normally
        True
        >>> list(result.keys()) == ['data']  # No multimodal keys
        True
    """
    data = JointNestedRaggedTensorDict.vstack([item["dynamic"] for item in batch])
    multimodal_metadata = {}

    max_ts_dim = 3 - int(do_flatten_tensors)
    for key in data.keys():
        if data._get_dim(key) > max_ts_dim:
            multimodal_metadata[key] = data._get_dim(key)

    multimodal_data = {}
    multimodal_idx_map = {}

    if multimodal_metadata:
        # Pop multimodal data and their bounds
        for key in multimodal_metadata.keys():
            base_key = key.split("/")[-1]
            dim = multimodal_metadata[key]

            # Create index mapping for this modality
            idx_map = []
            current_idx = 0

            # Extract the data and bounds
            data_key = f"dim{dim}/{base_key}"
            modal_data = data.tensors.pop(data_key)

            # Pop all bounds for this modality
            bounds = []
            for d in range(max_ts_dim + 1, dim + 1):
                bounds_key = f"dim{d}/bounds"
                if bounds_key in data.tensors:
                    bounds.append(data.tensors.pop(bounds_key))

            # Build index mapping
            if do_flatten_tensors:
                # Map to (subject_id, code)
                subject_ids = data.tensors["dim0/subject_id"]
                code_bounds = data.tensors["dim2/bounds"]
                prev_code_idx = 0

                for i, code_end in enumerate(code_bounds):
                    for _ in range(prev_code_idx, code_end):
                        idx_map.append((subject_ids[i], current_idx))
                        current_idx += 1
                    prev_code_idx = code_end
            else:
                # Map to (subject_id, time_delta_days, code)
                subject_ids = data.tensors["dim0/subject_id"]
                time_bounds = data.tensors["dim1/bounds"]
                code_bounds = data.tensors["dim2/bounds"]
                time_deltas = data.tensors["dim1/time_delta_days"]

                prev_time_idx = 0
                for i, time_end in enumerate(time_bounds):
                    for t in range(prev_time_idx, time_end):
                        for _ in range(code_bounds[t], code_bounds[t + 1]):
                            idx_map.append((subject_ids[i], time_deltas[t], current_idx))
                            current_idx += 1
                    prev_time_idx = time_end

            # Store the processed data and mapping
            multimodal_data[base_key] = {"data": modal_data, "bounds": bounds}
            multimodal_idx_map[base_key] = idx_map

        # Reconstruct JNRT without multimodal data
        data = JointNestedRaggedTensorDict(processed_tensors=data.tensors)

    data = data.to_dense()

    # Add the multimodal data to the tensorized dict
    tensorized = {"data": data}
    for key in multimodal_data:
        tensorized[key] = {
            "data": multimodal_data[key]["data"],
            "bounds": multimodal_data[key]["bounds"],
            "idx_map": multimodal_idx_map[key],
        }

    return tensorized


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
        return process_multimodal_batch(batch, do_flatten_tensors=self.config.do_flatten_tensors)
