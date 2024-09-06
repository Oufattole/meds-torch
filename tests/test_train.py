"""Tests the training configuration provided by the `cfg_train` pytest fixture.

TODO: finish the meds_torch.train function and setup tests for it
"""
import os
from pathlib import Path

import hydra
import pytest
from hydra.core.hydra_config import HydraConfig
from omegaconf import open_dict

from meds_torch.train import train
from tests.helpers.run_if import RunIf
from tests.test_configs import kwargs  # noqa: F401


def test_fast_dev_train(kwargs: dict, tmp_path):  # noqa: F811
    """Tests the training configuration provided by the `kwargs` pytest fixture.

    :param kwargs: A dictionary containing the configuration and a flag for expected ValueError.
    """
    cfg = kwargs["cfg"]
    raises_value_error = kwargs["raises_value_error"]

    with open_dict(cfg):
        cfg.trainer.fast_dev_run = True
        cfg.trainer.accelerator = "cpu"
        cfg.paths.output_dir = str(tmp_path)
    HydraConfig().set_config(cfg)

    if raises_value_error:
        with pytest.raises(hydra.errors.InstantiationException):
            train(cfg)
    else:
        train(cfg)


@RunIf(min_gpus=1)
def test_gpu_train(kwargs, tmp_path) -> None:  # noqa: F811
    """Tests the training configuration provided by the `cfg_train` pytest fixture.

    :param cfg_train: A DictConfig containing a valid training configuration.
    """
    cfg = kwargs["cfg"]
    raises_value_error = kwargs["raises_value_error"]

    with open_dict(cfg):
        cfg.trainer.fast_dev_run = True
        cfg.trainer.accelerator = "gpu"
        cfg.paths.output_dir = str(tmp_path)
    HydraConfig().set_config(cfg)
    if raises_value_error:
        with pytest.raises(hydra.errors.InstantiationException):
            train(cfg)
    else:
        train(cfg)


@RunIf(min_gpus=1)
@pytest.mark.slow
def test_train_epoch_gpu_amp(kwargs, tmp_path) -> None:  # noqa: F811
    """Train 1 epoch on GPU with mixed-precision.

    :param cfg_train: A DictConfig containing a valid training configuration.
    """
    cfg = kwargs["cfg"]
    raises_value_error = kwargs["raises_value_error"]
    with open_dict(cfg):
        cfg.trainer.max_epochs = 1
        cfg.trainer.accelerator = "gpu"
        cfg.trainer.precision = 16
        cfg.paths.output_dir = str(tmp_path)
    HydraConfig().set_config(cfg)
    if raises_value_error:
        with pytest.raises(hydra.errors.InstantiationException):
            train(cfg)
    else:
        train(cfg)


@pytest.mark.slow
def test_train_resume(tmp_path: Path, kwargs: dict) -> None:  # noqa: F811
    """Run 1 epoch, finish, and resume for another epoch.

    :param tmp_path: The temporary logging path.
    :param cfg_train: A DictConfig containing a valid training configuration.
    """
    cfg = kwargs["cfg"]
    raises_value_error = kwargs["raises_value_error"]
    HydraConfig().set_config(cfg)
    with open_dict(cfg):
        cfg.trainer.max_epochs = 1
        cfg.paths.output_dir = str(tmp_path)
        cfg.callbacks.model_checkpoint.save_top_k = -1

    if raises_value_error:
        with pytest.raises(hydra.errors.InstantiationException):
            train(cfg)
    else:
        metric_dict_1, _ = train(cfg)
        files = os.listdir(tmp_path / "checkpoints")
        assert "last.ckpt" in files
        assert "epoch_000.ckpt" in files
        assert len(files) == 2

        with open_dict(cfg):
            cfg.ckpt_path = str(tmp_path / "checkpoints" / "epoch_000.ckpt")
            cfg.trainer.max_epochs = 2

        metric_dict_2, _ = train(cfg)

        files = os.listdir(tmp_path / "checkpoints")
        assert "epoch_000.ckpt" in files
        assert "epoch_001.ckpt" in files
        assert "last.ckpt" in files
        assert "last-v1.ckpt" in files
        assert len(files) == 4

        assert "train/loss" in metric_dict_1 and "val/loss" in metric_dict_1
        assert "train/loss" in metric_dict_2 and "val/loss" in metric_dict_2
