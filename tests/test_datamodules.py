import rootutils

root = rootutils.setup_root(__file__, dotenv=True, pythonpath=True, cwd=True)

from pathlib import Path

import pytest
from torch.utils.data import DataLoader

from meds_torch.data.components.multiwindow_pytorch_dataset import (
    MultiWindowPytorchDataset,
)
from meds_torch.data.components.pytorch_dataset import PytorchDataset
from meds_torch.data.datamodule import MEDSDataModule

# from meds_torch.data.components.pytorch_dataset
from tests.conftest import SUPERVISED_TASK_NAME, create_cfg


@pytest.mark.parametrize(
    "collate_type",
    ["triplet", "event_stream", "text_code", "text_observation", "all_text", "triplet_prompt", "eic"],
)
def test_pytorch_dataset(meds_dir, collate_type):
    cfg = create_cfg(overrides=[], meds_dir=meds_dir)
    cfg.data.collate_type = collate_type
    cfg.data.tokenizer = "bert-base-uncased"
    pyd = PytorchDataset(cfg.data, split="train")
    assert not pyd.has_task
    item = pyd[0]
    assert item.keys() == {"static_indices", "static_values", "dynamic"}
    batch = pyd.collate([pyd[i] for i in range(2)])
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
    elif collate_type == "triplet":
        assert batch.keys() == {
            "mask",
            "static_mask",
            "code",
            "numeric_value",
            "time_delta_days",
            "numerical_value_mask",
        }
    elif collate_type == "text_code":
        assert set(batch.keys()) == {
            "mask",
            "static_mask",
            "code_tokens",
            "code_mask",
            "numeric_value",
            "time_delta_days",
            "numerical_value_mask",
        }
    elif collate_type == "text_observation":
        assert set(batch.keys()) == {
            "mask",
            "observation_tokens",
            "observation_mask",
        }
    elif collate_type == "all_text":
        assert set(batch.keys()) == {
            "mask",
            "observation_tokens",
            "observation_mask",
        }
    elif collate_type == "triplet_prompt":
        assert batch.keys() == {
            "mask",
            "static_mask",
            "code",
            "numeric_value",
            "time_delta_days",
            "numerical_value_mask",
        }
    elif collate_type == "eic":
        assert batch.keys() == {
            "mask",
            "static_mask",
            "code",
            "numeric_value",
            "time_delta_days",
            "numerical_value_mask",
        }
    else:
        raise NotImplementedError(f"{collate_type} not implemented")


@pytest.mark.parametrize("collate_type", ["triplet", "event_stream", "triplet_prompt"])
def test_pytorch_dataset_with_supervised_task(meds_dir, collate_type):
    cfg = create_cfg(overrides=[], meds_dir=meds_dir, supervised=True)
    cfg.data.collate_type = collate_type
    assert Path(cfg.data.task_label_path).exists(), f"Path does not exist: {cfg.data.task_label_path}"

    pyd = PytorchDataset(cfg.data, split="train")
    assert pyd.has_task
    item = pyd[0]
    assert item.keys() == {"static_indices", "static_values", "dynamic", "supervised_task"}
    batch = pyd.collate([pyd[i] for i in range(2)])
    if collate_type == "event_stream":
        assert batch.keys() == {
            "event_mask",
            "dynamic_values_mask",
            "time_delta_days",
            "dynamic_indices",
            "dynamic_values",
            "static_indices",
            "static_values",
            SUPERVISED_TASK_NAME,
        }
    elif collate_type in ["triplet", "triplet_prompt"]:
        assert batch.keys() == {
            "mask",
            "static_mask",
            "code",
            "numeric_value",
            "time_delta_days",
            "numerical_value_mask",
            SUPERVISED_TASK_NAME,
        }
    else:
        raise NotImplementedError(f"{collate_type} not implemented")


@pytest.mark.parametrize("patient_level_sampling", [False, True])
@pytest.mark.parametrize("collate_type", ["triplet"])
def test_contrastive_windows(meds_dir, patient_level_sampling, collate_type):
    cfg = create_cfg(overrides=["data=multiwindow_pytorch_dataset"], meds_dir=meds_dir)
    cfg.data.collate_type = collate_type
    cfg.data.patient_level_sampling = patient_level_sampling

    assert cfg.data.cached_windows_dir
    assert Path(cfg.data.raw_windows_fp).exists()

    pyd = MultiWindowPytorchDataset(cfg.data, split="train")
    item = pyd[0]
    assert item.keys() == {"pre", "post"}
    assert item["pre"].keys() == {"static_indices", "static_values", "dynamic"}
    assert item["post"].keys() == {"static_indices", "static_values", "dynamic"}

    batch = pyd.collate([pyd[i] for i in range(2)])
    assert batch.keys() == {"pre", "post"}
    for window in ["pre", "post"]:
        assert batch[window].keys() == {
            "mask",
            "static_mask",
            "code",
            "numeric_value",
            "time_delta_days",
            "numerical_value_mask",
        }


def test_full_datamodule(meds_dir):
    cfg = create_cfg(overrides=[], meds_dir=meds_dir)
    cfg.data.dataloader.batch_size = 1
    dm = MEDSDataModule(cfg.data)
    dm.prepare_data()

    assert not dm.data_train and not dm.data_val and not dm.data_test

    dm.setup()
    assert (
        isinstance(dm.data_train, PytorchDataset)
        and isinstance(dm.data_val, PytorchDataset)
        and isinstance(dm.data_test, PytorchDataset)
    )
    assert isinstance(dm.train_dataloader(), DataLoader) and isinstance(dm.val_dataloader(), DataLoader)

    num_datapoints = len(dm.data_train)
    assert num_datapoints == 70

    batch = next(iter(dm.train_dataloader()))
    for value in batch.values():
        assert len(value) == cfg.data.dataloader.batch_size
