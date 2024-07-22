import rootutils

root = rootutils.setup_root(__file__, dotenv=True, pythonpath=True, cwd=True)

from pathlib import Path

import polars as pl
import pytest
from omegaconf import open_dict
from torch.utils.data import DataLoader

from meds_torch.data.components.multiwindow_pytorch_dataset import (
    MultiWindowPytorchDataset,
)
from meds_torch.data.components.pytorch_dataset import PytorchDataset
from meds_torch.data.meds_datamodule import MEDSDataModule

# from meds_torch.data.components.pytorch_dataset
from tests.conftest import SUPERVISED_TASK_NAME
from tests.helpers.run_sh_command import run_command


@pytest.mark.parametrize("collate_type", ["triplet", "event_stream"])
def test_meds_pytorch_dataset(cfg_meds_train, collate_type):
    cfg = cfg_meds_train
    cfg.data.collate_type = collate_type
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
            "numerical_value",
            "time_delta_days",
            "numerical_value_mask",
        }
    else:
        raise NotImplementedError(f"{collate_type} not implemented")


@pytest.mark.parametrize("collate_type", ["triplet", "event_stream"])
def test_meds_pytorch_dataset_with_supervised_task(cfg_meds_train, collate_type):
    cfg = cfg_meds_train
    cfg.data.collate_type = collate_type
    with open_dict(cfg):
        cfg.data.task_name = SUPERVISED_TASK_NAME
    assert Path(cfg.data.task_label_path).exists()

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
    elif collate_type == "triplet":
        assert batch.keys() == {
            "mask",
            "static_mask",
            "code",
            "numerical_value",
            "time_delta_days",
            "numerical_value_mask",
            SUPERVISED_TASK_NAME,
        }
    else:
        raise NotImplementedError(f"{collate_type} not implemented")


RANDOM_LOCAL_WINDOWS = """
predicates:
    event:
        code: _IGNORE

trigger: _ANY_EVENT

windows:
    pre:
        start: null
        end: trigger
        start_inclusive: True
        end_inclusive: False
    post:
        start: pre.end
        end: null
        start_inclusive: True
        end_inclusive: True
"""


@pytest.mark.parametrize("patient_level_sampling", [False, True])
@pytest.mark.parametrize("window_config", [RANDOM_LOCAL_WINDOWS])
@pytest.mark.parametrize("collate_type", ["triplet"])
def test_contrastive_windows(
    cfg_meds_multiwindow_train, tmp_path, window_config, patient_level_sampling, collate_type
):
    cfg = cfg_meds_multiwindow_train
    cfg.data.collate_type = collate_type
    cfg.data.patient_level_sampling = patient_level_sampling
    assert cfg.data.cached_windows_dir
    raw_windows_path = cfg.data.raw_windows_fp
    aces_task_cfg_path = tmp_path / "aces_config.yaml"

    aces_task_cfg_path.write_text(window_config)
    aces_kwargs = {
        "data.path": str((Path(cfg.paths.meds_dir) / "*/*.parquet").resolve()),
        "data.standard": "meds",
        "cohort_dir": str(aces_task_cfg_path.parent.resolve()),
        "cohort_name": "aces_config",
        "output_filepath": raw_windows_path,
        "hydra.verbose": True,
    }

    run_command("aces-cli", aces_kwargs, "aces-cli")

    meds_df = pl.read_parquet(str((Path(cfg.paths.meds_dir) / "*/*.parquet").resolve()))
    # TODO: ACES is duplicating these outputs should be fixed
    aces_df = pl.read_parquet(raw_windows_path).unique()

    number_of_unique_event_times = (
        meds_df.unique(["patient_id", "timestamp"]).drop_nulls("timestamp").shape[0]
    )
    assert number_of_unique_event_times == aces_df.shape[0]

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
            "numerical_value",
            "time_delta_days",
            "numerical_value_mask",
        }


def test_meds_datamodule(cfg_meds_train):
    cfg = cfg_meds_train.copy()
    cfg.data.dataloader.batch_size = 1
    dm = MEDSDataModule(cfg.data)
    dm.prepare_data()

    assert not dm.data_train and not dm.data_val and not dm.data_test

    dm.setup(stage="fit")
    assert (
        isinstance(dm.data_train, PytorchDataset)
        and isinstance(dm.data_val, PytorchDataset)
        and (dm.data_test is None)
    )
    assert isinstance(dm.train_dataloader(), DataLoader) and isinstance(dm.val_dataloader(), DataLoader)

    num_datapoints = len(dm.data_train)
    assert num_datapoints == 4

    batch = next(iter(dm.train_dataloader()))
    for value in batch.values():
        assert len(value) == cfg.data.dataloader.batch_size
