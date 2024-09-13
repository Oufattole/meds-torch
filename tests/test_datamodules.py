import rootutils

root = rootutils.setup_root(__file__, dotenv=True, pythonpath=True, cwd=True)

import functools
import operator
from pathlib import Path

import polars as pl
import pytest
import torch
from omegaconf import open_dict
from torch.utils.data import DataLoader

from meds_torch.data.components.multiwindow_pytorch_dataset import (
    MultiWindowPytorchDataset,
)
from meds_torch.data.components.pytorch_dataset import PytorchDataset
from meds_torch.data.datamodule import MEDSDataModule
from tests.conftest import SUPERVISED_TASK_NAME, create_cfg


@pytest.mark.parametrize(
    "collate_type",
    [
        "triplet",
        "event_stream",
        "triplet_prompt",
        "eic",
        "text_code",
    ],
)
def test_pytorch_dataset(meds_dir, collate_type):
    cfg = create_cfg(overrides=[], meds_dir=meds_dir)
    cfg.data.collate_type = collate_type
    cfg.data.tokenizer = "emilyalsentzer/Bio_ClinicalBERT"
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
            "numeric_value_mask",
        }
    elif collate_type == "text_code":
        assert set(batch.keys()) == {
            "mask",
            "static_mask",
            "code",
            "code_tokens",
            "code_mask",
            "numeric_value",
            "time_delta_days",
            "numeric_value_mask",
        }
    elif collate_type == "triplet_prompt":
        assert batch.keys() == {
            "mask",
            "static_mask",
            "code",
            "numeric_value",
            "time_delta_days",
            "numeric_value_mask",
        }
    elif collate_type == "eic":
        assert batch.keys() == {
            "mask",
            "static_mask",
            "code",
            "numeric_value",
            "time_delta_days",
            "numeric_value_mask",
        }
    else:
        raise NotImplementedError(f"{collate_type} not implemented")


@pytest.mark.parametrize("collate_type", ["triplet", "event_stream", "triplet_prompt", "text_code", "eic"])
def test_pytorch_dataset_with_supervised_task(meds_dir, collate_type):
    cfg = create_cfg(overrides=[], meds_dir=meds_dir, supervised=True)
    with open_dict(cfg):
        cfg.data.collate_type = collate_type
        cfg.data.dataloader.batch_size = 70
    assert Path(cfg.data.task_label_path).exists(), f"Path does not exist: {cfg.data.task_label_path}"

    pyd = PytorchDataset(cfg.data, split="train")
    assert len(pyd) == 70
    assert pyd.has_task
    item = pyd[0]
    assert item.keys() == {"static_indices", "static_values", "dynamic", "supervised_task"}
    task_df = pl.read_parquet(meds_dir / "tasks/supervised_task.parquet")
    code_index_df = pl.read_parquet(meds_dir / "triplet_tensors/metadata/codes.parquet")[
        "code", "code/vocab_index"
    ]
    target_indices = code_index_df.filter(
        pl.col("code").eq("ADMISSION//ONCOLOGY") | pl.col("code").eq("ADMISSION//CARDIAC")
    )["code/vocab_index"].to_list()
    items = []
    for index in range(len(pyd)):
        subject_data = pyd[index]
        if not collate_type == "event_stream":
            subject_id, _, __loader__ = pyd.index[index]
            pyd.subj_map[subject_id]
            static_df_index = pyd.static_dfs[pyd.subj_map[subject_id]][pyd.subj_indices[subject_id]][
                "subject_id"
            ].item()
            # Check the subject ids match
            assert static_df_index == subject_id, f"Subject ids do not match for index {index}"

            # Check the supervised task matches the label
            assert (
                task_df.filter(pl.col("subject_id").eq(subject_id))[SUPERVISED_TASK_NAME].item()
                == subject_data[SUPERVISED_TASK_NAME]
            )
            assert subject_data[SUPERVISED_TASK_NAME] == pyd.labels[SUPERVISED_TASK_NAME][index]
            # Check the supervised task matches the target indices
            data_label = bool(
                functools.reduce(
                    operator.or_, [subject_data["dynamic"]["dim1/code"] == t for t in target_indices]
                ).any()
            )
            assert (
                data_label == subject_data[SUPERVISED_TASK_NAME]
            ), f"Supervised task does not match target indices for index {index}"

        items.append(subject_data)

    batch = pyd.collate(items)
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
            "numeric_value_mask",
            SUPERVISED_TASK_NAME,
        }
        code_tensor = batch["code"]
        is_target_tensor = torch.zeros_like(code_tensor)
        for target_index in target_indices:
            is_target_tensor = is_target_tensor | code_tensor == target_index
        labels = [each[SUPERVISED_TASK_NAME] for each in items]
        # Check labels are in the same order after collating
        assert batch[SUPERVISED_TASK_NAME].to(bool).tolist() == labels
        # Data should have the right order after collating, using data derived labels
        data_label = functools.reduce(operator.or_, [code_tensor == t for t in target_indices]).any(axis=1)
        assert data_label.tolist() == labels, "Supervised task does not match target indices"
    elif collate_type == "eic":
        assert batch.keys() == {
            "mask",
            "static_mask",
            "code",
            "numeric_value",
            "time_delta_days",
            "numeric_value_mask",
            SUPERVISED_TASK_NAME,
        }
    elif collate_type == "text_code":
        assert set(batch.keys()) == {
            "mask",
            "static_mask",
            "code",
            "code_tokens",
            "code_mask",
            "numeric_value",
            "time_delta_days",
            "numeric_value_mask",
            SUPERVISED_TASK_NAME,
        }
    else:
        raise NotImplementedError(f"{collate_type} not implemented")


@pytest.mark.parametrize("subject_level_sampling", [False, True])
@pytest.mark.parametrize("collate_type", ["triplet"])
def test_contrastive_windows(meds_dir, subject_level_sampling, collate_type):
    cfg = create_cfg(overrides=["data=multiwindow_pytorch_dataset"], meds_dir=meds_dir)
    cfg.data.collate_type = collate_type
    cfg.data.subject_level_sampling = subject_level_sampling

    assert cfg.data.cache_dir
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
            "numeric_value_mask",
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
