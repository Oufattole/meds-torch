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
from tests.conftest import create_cfg
from tests.helpers.run_if import RunIf


def get_overrides_and_exceptions(data, model, early_fusion, input_encoder, backbone):
    overrides = [
        f"data={data}",
        f"model/input_encoder={input_encoder}",
        f"model/backbone={backbone}",
        f"model={model}",
    ]
    raises_value_error = False

    supervised = model == "supervised"
    if model == "token_forecasting" and backbone not in ["transformer_decoder", "lstm"]:
        raises_value_error = True

    if early_fusion is not None:
        overrides.append(f"model.early_fusion={early_fusion}")
    return overrides, raises_value_error, supervised


@pytest.fixture(
    params=[
        pytest.param(
            (data, model, early_fusion, input_encoder, backbone),
            id=f"{data}-{model}-earlyfusion{early_fusion}-{input_encoder}-{backbone}",
        )
        for data, model, early_fusion in [
            ("pytorch_dataset", "supervised", None),
            ("pytorch_dataset", "token_forecasting", None),
            ("multiwindow_pytorch_dataset", "ebcl", None),
            ("multiwindow_pytorch_dataset", "value_forecasting", None),
            ("multiwindow_pytorch_dataset", "ocp", "true"),
            ("multiwindow_pytorch_dataset", "ocp", "false"),
        ]
        for input_encoder in ["triplet_encoder", "triplet_prompt_encoder"]
        for backbone in ["transformer_decoder", "transformer_encoder", "transformer_encoder_attn_avg", "lstm"]
    ]
)
def kwargs(request, meds_dir) -> dict:
    data, model, early_fusion, input_encoder, backbone = request.param
    overrides, raises_value_error, supervised = get_overrides_and_exceptions(
        data, model, early_fusion, input_encoder, backbone
    )
    cfg = create_cfg(overrides=overrides, meds_dir=meds_dir, supervised=supervised)
    return dict(
        cfg=cfg,
        raises_value_error=raises_value_error,
        input_kwargs=dict(
            data=data, model=model, early_fusion=early_fusion, input_encoder=input_encoder, backbone=backbone
        ),
    )


def test_fast_dev_train(kwargs: dict, tmp_path):
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
def test_gpu_train(kwargs, tmp_path) -> None:  # cfg: DictConfig,
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
def test_train_epoch_gpu_amp(kwargs, tmp_path) -> None:
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
def test_train_resume(tmp_path: Path, kwargs: dict) -> None:
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
