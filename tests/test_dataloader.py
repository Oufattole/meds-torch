import rootutils

root = rootutils.setup_root(__file__, dotenv=True, pythonpath=True, cwd=True)

import shutil
import subprocess
from datetime import datetime
from pathlib import Path

import polars as pl
import torch
from hydra import compose, initialize

from meds_torch.data.components.multiwindow_pytorch_dataset import MultiWindowPytorchDataset
from meds_torch.data.components.pytorch_dataset import PytorchDataset


def test_event_stream(tmp_path):
    MEDS_cohort_dir = tmp_path / "processed" / "final_cohort"
    shutil.copytree(Path("./tests/test_data"), MEDS_cohort_dir.parent)

    kwargs = {
        "raw_MEDS_cohort_dir": str(MEDS_cohort_dir.parent.resolve()),
        "MEDS_cohort_dir": str(MEDS_cohort_dir.resolve()),
        "max_seq_len": 512,
        "model.embedder.token_dim": 4,
        "collate_type": "event_stream",
    }

    with initialize(version_base=None, config_path="../configs"):  # path to config.yaml
        overrides = [f"{k}={v}" for k, v in kwargs.items()]
        cfg = compose(config_name="pytorch_dataset", overrides=overrides)  # config.yaml

    pyd = PytorchDataset(cfg, split="train")
    item = pyd[0]
    assert item.keys() == {"static_indices", "static_values", "dynamic"}
    batch = pyd.collate([pyd[i] for i in range(2)])
    assert batch.keys() == {
        "event_mask",
        "dynamic_values_mask",
        "time_delta_days",
        "dynamic_indices",
        "dynamic_values",
        "static_indices",
        "static_values",
    }


def test_task(tmp_path):
    MEDS_cohort_dir = tmp_path / "processed" / "final_cohort"
    shutil.copytree(Path("./tests/test_data"), MEDS_cohort_dir.parent)

    task_name = "bababooey"
    kwargs = {
        "raw_MEDS_cohort_dir": str(MEDS_cohort_dir.parent.resolve()),
        "MEDS_cohort_dir": str(MEDS_cohort_dir.resolve()),
        "max_seq_len": 512,
        "model.embedder.token_dim": 4,
        "collate_type": "triplet",
        "task_name": task_name,
    }

    task_df = pl.DataFrame(
        {
            "patient_id": [239684, 1195293, 68729, 814703],
            "start_time": [
                datetime(1980, 12, 28),
                datetime(1978, 6, 20),
                datetime(1978, 3, 9),
                datetime(1976, 3, 28),
            ],
            "end_time": [
                datetime(2010, 5, 11, 18, 25, 35),
                datetime(2010, 6, 20, 20, 12, 31),
                datetime(2010, 5, 26, 2, 30, 56),
                datetime(2010, 2, 5, 5, 55, 39),
            ],
            task_name: [0, 1, 0, 1],
        }
    )
    tasks_dir = MEDS_cohort_dir / "tasks"
    tasks_dir.mkdir(parents=True, exist_ok=True)
    task_df.write_parquet(tasks_dir / f"{task_name}.parquet")

    with initialize(version_base=None, config_path="../configs"):  # path to config.yaml
        overrides = [f"{k}={v}" for k, v in kwargs.items()]
        cfg = compose(config_name="pytorch_dataset", overrides=overrides)  # config.yaml

    # triplet collating
    pyd = PytorchDataset(cfg, split="train")
    batch = pyd.collate([pyd[i] for i in range(2)])
    assert batch.keys() == {
        "mask",
        "static_mask",
        "code",
        "numerical_value",
        "time_delta_days",
        "numerical_value_mask",
        task_name,
    }
    for key in batch.keys() - {task_name}:
        assert batch[key].shape == torch.Size([2, 10])
    assert batch[task_name].shape == torch.Size([2])


def run_command(script: str, hydra_kwargs: dict[str, str], test_name: str, expected_returncode: int = 0):
    command_parts = [script] + [f"{k}={v}" for k, v in hydra_kwargs.items()]
    command_out = subprocess.run(" ".join(command_parts), shell=True, capture_output=True)
    stderr = command_out.stderr.decode()
    stdout = command_out.stdout.decode()
    if command_out.returncode != expected_returncode:
        raise AssertionError(
            f"{test_name} returned {command_out.returncode} (expected {expected_returncode})!\n"
            f"stdout:\n{stdout}\nstderr:\n{stderr}"
        )
    return stderr, stdout


RANDOM_EBCL_CFG = """
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


def test_ebcl_patient_level(tmp_path):
    MEDS_cohort_dir = tmp_path / "processed" / "final_cohort"
    shutil.copytree(Path("./tests/test_data"), MEDS_cohort_dir.parent)

    windows_path = tmp_path / "windows.parquet"

    aces_task_cfg_path = tmp_path / "aces_config.yaml"

    aces_task_cfg_path.write_text(RANDOM_EBCL_CFG)
    aces_kwargs = {
        "data.path": str((MEDS_cohort_dir / "*/*.parquet").resolve()),
        "data.standard": "meds",
        "cohort_dir": str(aces_task_cfg_path.parent.resolve()),
        "cohort_name": "aces_config",
        "output_filepath": str(windows_path.resolve()),
        "hydra.verbose": True,
    }

    stderr, stdout = run_command("aces-cli", aces_kwargs, "extract_ebcl")

    meds_df = pl.read_parquet(str((MEDS_cohort_dir / "*/*.parquet").resolve()))
    # TODO: ACES is duplicating these outputs should be fixed
    aces_df = pl.read_parquet(str(windows_path.resolve())).unique()

    number_of_unique_event_times = (
        meds_df.unique(["patient_id", "timestamp"]).drop_nulls("timestamp").shape[0]
    )

    assert number_of_unique_event_times == aces_df.shape[0]

    kwargs = {
        "raw_MEDS_cohort_dir": str(MEDS_cohort_dir.parent.resolve()),
        "MEDS_cohort_dir": str(MEDS_cohort_dir.resolve()),
        "max_seq_len": 512,
        "model.embedder.token_dim": 4,
        "collate_type": "triplet",
        "raw_windows_fp": str(windows_path.resolve()),
        "cached_windows_dir": str((windows_path.parent / "cached_windows").resolve()),
        "patient_level_sampling": True,
    }

    with initialize(version_base=None, config_path="../configs"):  # path to config.yaml
        overrides = [f"{k}={v}" for k, v in kwargs.items()]
        cfg = compose(config_name="data.multiwindow_pytorch_dataset", overrides=overrides)  # config.yaml

    pyd = MultiWindowPytorchDataset(cfg, split="train")
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


def test_ebcl_event_level(tmp_path):
    MEDS_cohort_dir = tmp_path / "processed" / "final_cohort"
    shutil.copytree(Path("./tests/test_data"), MEDS_cohort_dir.parent)

    windows_path = tmp_path / "windows.parquet"

    aces_task_cfg_path = tmp_path / "aces_config.yaml"

    aces_task_cfg_path.write_text(RANDOM_EBCL_CFG)
    aces_kwargs = {
        "data.path": str((MEDS_cohort_dir / "*/*.parquet").resolve()),
        "data.standard": "meds",
        "cohort_dir": str(aces_task_cfg_path.parent.resolve()),
        "cohort_name": "aces_config",
        "output_filepath": str(windows_path.resolve()),
        "hydra.verbose": True,
    }

    stderr, stdout = run_command("aces-cli", aces_kwargs, "extract_ebcl")

    meds_df = pl.read_parquet(str((MEDS_cohort_dir / "*/*.parquet").resolve()))
    # TODO: ACES is duplicating these outputs should be fixed
    aces_df = pl.read_parquet(str(windows_path.resolve())).unique()

    number_of_unique_event_times = (
        meds_df.unique(["patient_id", "timestamp"]).drop_nulls("timestamp").shape[0]
    )

    assert number_of_unique_event_times == aces_df.shape[0]

    kwargs = {
        "raw_MEDS_cohort_dir": str(MEDS_cohort_dir.parent.resolve()),
        "MEDS_cohort_dir": str(MEDS_cohort_dir.resolve()),
        "max_seq_len": 512,
        "model.embedder.token_dim": 4,
        "collate_type": "triplet",
        "raw_windows_fp": str(windows_path.resolve()),
        "cached_windows_dir": str((windows_path.parent / "cached_windows").resolve()),
    }

    with initialize(version_base=None, config_path="../configs"):  # path to config.yaml
        overrides = [f"{k}={v}" for k, v in kwargs.items()]
        cfg = compose(config_name="multiwindow_pytorch_dataset", overrides=overrides)  # config.yaml

    pyd = MultiWindowPytorchDataset(cfg, split="train")
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


OCP_CFG = """
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

def test_ocp_event_level(tmp_path):
    MEDS_cohort_dir = tmp_path / "processed" / "final_cohort"
    shutil.copytree(Path("./tests/test_data"), MEDS_cohort_dir.parent)

    windows_path = tmp_path / "windows.parquet"

    aces_task_cfg_path = tmp_path / "aces_config.yaml"

    aces_task_cfg_path.write_text(OCP_CFG)
    aces_kwargs = {
        "data.path": str((MEDS_cohort_dir / "*/*.parquet").resolve()),
        "data.standard": "meds",
        "cohort_dir": str(aces_task_cfg_path.parent.resolve()),
        "cohort_name": "aces_config",
        "output_filepath": str(windows_path.resolve()),
        "hydra.verbose": True,
    }

    stderr, stdout = run_command("aces-cli", aces_kwargs, "extract_ebcl")

    meds_df = pl.read_parquet(str((MEDS_cohort_dir / "*/*.parquet").resolve()))
    aces_df = pl.read_parquet(str(windows_path.resolve())).unique()

    number_of_unique_event_times = (
        meds_df.unique(["patient_id", "timestamp"]).drop_nulls("timestamp").shape[0]
    )

    assert number_of_unique_event_times == aces_df.shape[0]

    kwargs = {
        "raw_MEDS_cohort_dir": str(MEDS_cohort_dir.parent.resolve()),
        "MEDS_cohort_dir": str(MEDS_cohort_dir.resolve()),
        "max_seq_len": 512,
        "model.embedder.token_dim": 4,
        "collate_type": "triplet",
        "raw_windows_fp": str(windows_path.resolve()),
        "cached_windows_dir": str((windows_path.parent / "cached_windows").resolve()),
    }

    with initialize(version_base=None, config_path="../configs"):  # path to config.yaml
        overrides = [f"{k}={v}" for k, v in kwargs.items()]
        cfg = compose(config_name="multiwindow_pytorch_dataset", overrides=overrides)  # config.yaml

    pyd = MultiWindowPytorchDataset(cfg, split="train")
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