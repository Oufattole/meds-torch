# TODO: finish the sweep tests
import os
from pathlib import Path

import pytest

from tests.conftest import SUPERVISED_TASK_NAME
from tests.helpers.run_if import RunIf
from tests.helpers.run_sh_command import run_command


@RunIf(sh=True)
@pytest.mark.slow
def test_train_experiments_fast_dev(tmp_path: Path, meds_dir) -> None:
    """Test running all available experiment configs with `fast_dev_run=True.`

    :param tmp_path: The temporary logging path.
    """
    script = "meds-torch-train"
    args = ["-m"]
    kwargs = {
        "logger": "[]",
        "experiment": "example",
        "++trainer.fast_dev_run": "true",
        "paths.data_dir": str(meds_dir / "triplet_tensors"),
        "paths.meds_cohort_dir": str(meds_dir / "MEDS_cohort"),
        "paths.output_dir": str(tmp_path / "train"),
        "data.task_name": SUPERVISED_TASK_NAME,
        "data.task_root_dir": str(meds_dir / "tasks"),
        "callbacks.model_checkpoint.save_top_k": -1,
    }
    run_command(script=script, args=args, hydra_kwargs=kwargs, test_name="train")


@RunIf(sh=True, min_gpus=1)
@pytest.mark.slow
def test_train_experiments(tmp_path: Path, meds_dir) -> None:
    """Test running all available experiment configs with `fast_dev_run=True.`

    :param tmp_path: The temporary logging path.
    """
    script = "meds-torch-train"
    args = ["-m"]
    kwargs = {
        "logger": "[]",
        "experiment": "example",
        "paths.data_dir": str(meds_dir / "triplet_tensors"),
        "paths.meds_cohort_dir": str(meds_dir / "MEDS_cohort"),
        "paths.output_dir": str(tmp_path / "train"),
        "data.task_name": SUPERVISED_TASK_NAME,
        "data.task_root_dir": str(meds_dir / "tasks"),
        "callbacks.model_checkpoint.save_top_k": -1,
        "trainer.max_epochs": 2,
        "trainer.accelerator": "gpu",
    }
    run_command(script=script, args=args, hydra_kwargs=kwargs, test_name="train")
    assert "last.ckpt" in os.listdir(tmp_path / "train" / "checkpoints")


@RunIf(sh=True, min_gpus=1)
@pytest.mark.slow
def test_tune_experiments(tmp_path: Path, meds_dir) -> None:
    """Test running all available experiment configs with `fast_dev_run=True.`

    :param tmp_path: The temporary logging path.
    """
    script = "meds-torch-tune"
    args = ["-m"]
    kwargs = {
        "logger": "[]",
        "hparams_search": "ray_tune",
        "callbacks": "ray",
        "hparams_search.ray.num_samples": 2,
        "experiment": "example",
        "paths.data_dir": str(meds_dir / "triplet_tensors"),
        "paths.meds_cohort_dir": str(meds_dir / "MEDS_cohort"),
        "paths.output_dir": str(tmp_path / "train"),
        "data.task_name": SUPERVISED_TASK_NAME,
        "data.task_root_dir": str(meds_dir / "tasks"),
        "callbacks.model_checkpoint.save_top_k": -1,
        "trainer.max_epochs": 4,
        "trainer.accelerator": "gpu",
    }
    run_command(script=script, args=args, hydra_kwargs=kwargs, test_name="train")
    subdirectory_count = sum(1 for entry in os.scandir(tmp_path / "train" / "ray_tune") if entry.is_dir())
    assert subdirectory_count == 2
