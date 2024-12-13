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
from meds_torch.data.components.random_windows_pytorch_dataset import (
    RandomWindowPytorchDataset,
)
from meds_torch.data.datamodule import MEDSDataModule
from tests.conftest import SUPERVISED_TASK_NAME, create_cfg


@pytest.mark.parametrize(
    "collate_type",
    [
        "triplet",
        "triplet_prompt",
        "eic",
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
    if collate_type == "triplet":
        assert batch.keys() == {
            "mask",
            "static_mask",
            "code",
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


@pytest.mark.parametrize("collate_type", ["triplet", "triplet_prompt", "eic"])
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
    assert item.keys() == {"static_indices", "static_values", "dynamic", "boolean_value"}
    task_df = pl.read_parquet(meds_dir / "tasks/boolean_value.parquet")
    code_index_df = pl.read_parquet(meds_dir / "triplet_tensors/metadata/codes.parquet")[
        "code", "code/vocab_index"
    ]
    target_indices = code_index_df.filter(
        pl.col("code").eq("ADMISSION//ONCOLOGY") | pl.col("code").eq("ADMISSION//CARDIAC")
    )["code/vocab_index"].to_list()
    items = []
    for index in range(len(pyd)):
        subject_data = pyd[index]
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
        assert subject_data[SUPERVISED_TASK_NAME] == pyd.labels[index]
        # Check the supervised task matches the target indices
        # The task is a deterministic function of the codes in this case
        # so we can check the codes themeselves are loaded correctly via this check.
        data_label = bool(
            functools.reduce(
                operator.or_, [subject_data["dynamic"].tensors["dim0/code"] == t for t in target_indices]
            ).any()
        )
        assert (
            data_label == subject_data[SUPERVISED_TASK_NAME]
        ), f"Supervised task does not match target indices for index {index}"

        items.append(subject_data)

    batch = pyd.collate(items)
    if collate_type in ["triplet", "triplet_prompt"]:
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
    else:
        raise NotImplementedError(f"{collate_type} not implemented")


@pytest.mark.parametrize("do_early_fuse_windows", [False, True])
@pytest.mark.parametrize("collate_type", ["triplet"])
@pytest.mark.parametrize("subject_level_sampling", [False, True])
def test_contrastive_windows(meds_dir, subject_level_sampling, collate_type, do_early_fuse_windows):
    overrides = [
        "data=multiwindow_pytorch_dataset",
        "data.do_include_subject_id=true",
    ]
    if do_early_fuse_windows:
        overrides.append("data.early_fusion_windows=[pre,post]")
    cfg = create_cfg(overrides=overrides, meds_dir=meds_dir)
    cfg.data.collate_type = collate_type
    cfg.data.subject_level_sampling = subject_level_sampling
    expected_item_keys = {"pre", "post", "subject_id"}

    assert cfg.data.cache_dir
    assert Path(cfg.data.raw_windows_fp).exists()

    pyd = MultiWindowPytorchDataset(cfg.data, split="train")
    item = pyd[0]
    assert item.keys() == expected_item_keys
    assert item["pre"].keys() == {"static_indices", "static_values", "dynamic"}
    assert item["post"].keys() == {"static_indices", "static_values", "dynamic"}

    expected_batch_keys = expected_item_keys | ({"FUSED"} if do_early_fuse_windows else set())
    batch = pyd.collate([pyd[i] for i in range(2)])
    assert batch.keys() == expected_batch_keys
    expected_window_keys = {
        "mask",
        "static_mask",
        "code",
        "numeric_value",
        "time_delta_days",
        "numeric_value_mask",
    }
    for window in expected_batch_keys - {"subject_id"}:
        assert batch[window].keys() & expected_window_keys == expected_window_keys


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


@pytest.mark.parametrize(
    "collate_type",
    [
        "triplet",
        "triplet_prompt",
        "eic",
    ],
)
def test_random_windows_pytorch_dataset(meds_dir, collate_type):
    cfg = create_cfg(overrides=["data=random_windows_pytorch_dataset"], meds_dir=meds_dir)
    with open_dict(cfg):
        cfg.data.collate_type = collate_type
        cfg.data.tokenizer = "emilyalsentzer/Bio_ClinicalBERT"
        cfg.data.min_window_size = 10
        cfg.data.max_window_size = 50
        cfg.data.n_windows = 1

    rwd = RandomWindowPytorchDataset(cfg.data, split="train")
    assert not rwd.has_task
    item = rwd[0]
    assert set(item.keys()) == {"window_0"}
    window = item["window_0"]
    assert set(window.keys()) == {"static_indices", "static_values", "dynamic"}

    batch = rwd.collate([rwd[i] for i in range(2)])
    assert set(batch.keys()) == {"window_0"}

    window = batch["window_0"]
    assert set(window.keys()) == {
        "mask",
        "static_mask",
        "code",
        "numeric_value",
        "time_delta_days",
        "numeric_value_mask",
    }


def test_random_window_generation(meds_dir):
    cfg = create_cfg(overrides=["data=random_windows_pytorch_dataset"], meds_dir=meds_dir)
    with open_dict(cfg):
        cfg.data.collate_type = "triplet"
        cfg.data.min_window_size = 10
        cfg.data.max_window_size = 50
        cfg.data.n_windows = 1

    rwd = RandomWindowPytorchDataset(
        cfg.data,
        split="train",
    )

    # Test generate_random_windows method
    windows = rwd.generate_random_windows(100)
    assert len(windows) == 1
    start, end = windows["window_0"]
    assert 10 <= end - start <= 50
    assert 0 <= start < end <= 100

    # Test that windows are different for different calls
    windows2 = rwd.generate_random_windows(100)
    assert windows["window_0"] != windows2["window_0"]


def test_random_window_batch_sizes(meds_dir):
    cfg = create_cfg(overrides=["data=random_windows_pytorch_dataset"], meds_dir=meds_dir)
    with open_dict(cfg):
        cfg.data.collate_type = "triplet"
        cfg.data.min_window_size = 10
        cfg.data.max_window_size = 50
        cfg.data.n_windows = 1

    rwd = RandomWindowPytorchDataset(cfg.data, split="train")

    batch = rwd.collate([rwd[i] for i in range(len(rwd))])
    assert len(batch["window_0"]["mask"]) == len(rwd)

    # Check that window sizes are within the specified range
    window_sizes = batch["window_0"]["mask"].sum(dim=1)
    assert all(10 <= size <= 50 for size in window_sizes)


def test_random_window_generation_modes(meds_dir):
    cfg = create_cfg(overrides=["data=random_windows_pytorch_dataset"], meds_dir=meds_dir)
    with open_dict(cfg):
        cfg.data.collate_type = "triplet"
        cfg.data.n_windows = 3
        cfg.data.window_names = ["window_1", "window_2", "window_3"]
        cfg.data.max_window_size = 50

    # Test consecutive windows
    with open_dict(cfg):
        cfg.data.consecutive_windows = True

    rwd_consecutive = RandomWindowPytorchDataset(cfg.data, split="train")

    # Generate windows for a hypothetical sequence of length 200
    consecutive_windows = rwd_consecutive.generate_random_windows(200)

    # Check if windows are consecutive
    window_starts = [w[0] for w in consecutive_windows.values()]
    window_ends = [w[1] for w in consecutive_windows.values()]

    assert all(
        window_ends[i] == window_starts[i + 1] for i in range(len(window_starts) - 1)
    ), "Consecutive windows are not truly consecutive"

    assert window_starts == sorted(window_starts), "Consecutive windows are not in order"

    # Test non-consecutive windows
    with open_dict(cfg):
        cfg.data.consecutive_windows = False

    rwd_non_consecutive = RandomWindowPytorchDataset(cfg.data, split="train")

    # Generate windows for a hypothetical sequence of length 200
    non_consecutive_windows = rwd_non_consecutive.generate_random_windows(200)

    # Check if windows are non-consecutive (at least one pair is non-consecutive)
    window_starts = [w[0] for w in non_consecutive_windows.values()]
    window_ends = [w[1] for w in non_consecutive_windows.values()]

    assert any(
        window_ends[i] != window_starts[i + 1] for i in range(len(window_starts) - 1)
    ), "Non-consecutive windows are all consecutive"

    # Additional checks
    for mode, windows in [("Consecutive", consecutive_windows), ("Non-consecutive", non_consecutive_windows)]:
        # Check if all windows are within the sequence length
        assert all(
            0 <= start < end <= 200 for start, end in windows.values()
        ), f"{mode} windows exceed sequence length"

        # Check if window sizes are correct
        assert all(
            0 < end - start <= cfg.data.max_window_size for start, end in windows.values()
        ), f"{mode} window sizes are incorrect"

        # Check if the correct number of windows is generated
        assert len(windows) == cfg.data.n_windows, f"Incorrect number of {mode.lower()} windows generated"

        # Check if window names are correct
        assert set(windows.keys()) == set(
            cfg.data.window_names
        ), f"Incorrect window names for {mode.lower()} windows"

    print("All window generation tests passed successfully!")
