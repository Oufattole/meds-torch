import numpy as np
import torch
from mixins import TimeableMixin
from nested_ragged_tensors.ragged_numpy import JointNestedRaggedTensorDict
from meds_torch.data.components.pytorch_dataset import PytorchDataset

def pop_key(jnrt: JointNestedRaggedTensorDict, key: str) -> tuple[JointNestedRaggedTensorDict, JointNestedRaggedTensorDict]:
    """Pops a key from the JNRT.

    Args:
        jnrt: The source JointNestedRaggedTensorDict
        key: The key to remove

    Returns:
        A tuple of (jnrt_without_key, jnrt_with_only_key)

    Examples:
        >>> import polars as pl
        >>> data = JointNestedRaggedTensorDict({
        ...     "subject_id": [1, 2],
        ...     "time_delta_days": [[0, 1], [2, 3]],
        ...     "code": [[1, 2], [3, 4]],
        ...     "text_value": [[1, 2], [3, 4]],
        ...     "numeric_value": [[0.0, 1.0], [2.0, 3.0]],
        ...     "modality_idx": [[0, 0], [1, 1]]
        ... })
        >>> remainder, popped = pop_key(data, "text_value")
        >>> sorted(list(remainder.to_dense().keys()))
        ['code', 'dim1/mask', 'modality_idx', 'numeric_value', 'subject_id', 'time_delta_days']
        >>> popped.to_dense()['text_value'].tolist()
        [[1, 2], [3, 4]]
    """
    # Reset the schema, sometimes the schema is incorrect if you don't do this
    jnrt = JointNestedRaggedTensorDict(processed_tensors=jnrt.tensors, schema=jnrt.schema)

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
    popped_schema = {key: jnrt.schema[key]} if key in jnrt.schema else {}

    return (
        JointNestedRaggedTensorDict(processed_tensors=remaining_tensors, schema=remaining_schema),
        JointNestedRaggedTensorDict(processed_tensors=popped_tensors, schema=popped_schema),
    )


class MultiModalPytorchDataset(PytorchDataset):
    """A PyTorch Dataset class that handles multiple modalities including text for contrastive learning.

    This class extends PytorchDataset to support text and other modalities for contrastive learning
    between different data types.

    Args:
        cfg (DictConfig): Configuration options for the dataset.
        split (str): The data split to use (e.g., 'train', 'validation', 'test').
    """

    def collate(self, batch: list[dict]) -> dict:
        """Collate a batch of multimodal sequences.

        Args:
            batch (List[dict]): A list of dictionaries, each containing sequences with different modalities.

        Returns:
            dict: A dictionary with collated data for each modality, including:
                - text embeddings (if present)
                - codes
                - numeric values
                - time information
                - masks for each modality

        Examples:
            >>> # Create a mock dataset class that just implements collate
            >>> class MockDataset:
            ...     def collate(self, batch):
            ...         dynamic_data = [item["dynamic"] for item in batch]
            ...         combined = JointNestedRaggedTensorDict.vstack(dynamic_data)
            ...         dense_data = combined.to_dense()
            ...         output = {}
            ...         for key in dense_data:
            ...             output[key] = torch.as_tensor(dense_data[key])
            ...             if key == "numeric_value":
            ...                 output["numeric_value_mask"] = torch.ones_like(output[key], dtype=torch.bool)
            ...         return output
            >>> # Create sample batch data
            >>> batch = [
            ...     {
            ...         "dynamic": JointNestedRaggedTensorDict({
            ...             "subject_id": [1],
            ...             "time_delta_days": [[0, 1]],
            ...             "code": [[1, 2]],
            ...             "text_value": [[1, 2]],
            ...             "numeric_value": [[0.0, 1.0]],
            ...             "modality_idx": [[0, 0]]
            ...         })
            ...     },
            ...     {
            ...         "dynamic": JointNestedRaggedTensorDict({
            ...             "subject_id": [2],
            ...             "time_delta_days": [[2, 3]],
            ...             "code": [[3, 4]],
            ...             "text_value": [[3, 4]],
            ...             "numeric_value": [[2.0, 3.0]],
            ...             "modality_idx": [[1, 1]]
            ...         })
            ...     }
            ... ]
            >>> dataset = MockDataset()
            >>> result = dataset.collate(batch)
            >>> sorted(result.keys())
            ['code', 'dim1/mask', 'dim2/mask', 'modality_idx', 'numeric_value', 'numeric_value_mask', 'subject_id', 'text_value', 'time_delta_days']
        """
        # Extract all dynamic data from the batch
        dynamic_data = [item["dynamic"] for item in batch]
        
        # Combine all dynamic data
        combined = JointNestedRaggedTensorDict.vstack(dynamic_data)
        
        # Convert to dense tensors
        dense_data = combined.to_dense()
        
        # Create the output dictionary with all modalities
        output = {}
        
        # Add basic fields
        if "time_delta_days" in dense_data:
            output["time_delta_days"] = dense_data["time_delta_days"]
        
        if "code" in dense_data:
            output["code"] = dense_data["code"]
            
        if "numeric_value" in dense_data:
            output["numeric_value"] = dense_data["numeric_value"]
            output["numeric_value_mask"] = torch.ones_like(torch.as_tensor(dense_data["numeric_value"]), dtype=torch.bool)
            
        if "text_value" in dense_data:
            output["text_value"] = dense_data["text_value"]
            
        if "modality_idx" in dense_data:
            output["modality_idx"] = dense_data["modality_idx"]
            
        if "subject_id" in dense_data:
            output["subject_id"] = dense_data["subject_id"]
            
        # Add masks
        if "dim1/mask" in dense_data:
            output["dim1/mask"] = dense_data["dim1/mask"]

        if "dim2/mask" in dense_data:
            output["dim2/mask"] = dense_data["dim2/mask"]
        if "mask" in dense_data:
            output["mask"] = dense_data["mask"]
            
        return output
