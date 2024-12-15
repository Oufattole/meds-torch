"""Tests the training configuration provided by the `cfg_train` pytest fixture.

TODO: finish the meds_torch.train function and setup tests for it
"""
import os
import subprocess
from decimal import Decimal
from pathlib import Path

import hydra
import pytest
from hydra.core.hydra_config import HydraConfig
from omegaconf import open_dict

from meds_torch.train import main as train_main
from tests.helpers.run_if import RunIf
from tests.test_configs import create_cfg, get_overrides_and_exceptions  # noqa: F401


@pytest.fixture(
    params=[
        pytest.param(
            (data, model, early_fusion, input_encoder, backbone),
            id=f"{data}-{model}-earlyfusion{early_fusion}-{input_encoder}-{backbone}",
        )
        for data, model, early_fusion, input_encoder, backbone in [
            ("pytorch_dataset", "supervised", None, "textcode_encoder", "transformer_encoder"),
            ("pytorch_dataset", "supervised", None, "triplet_encoder", "transformer_encoder"),
            ("pytorch_dataset", "triplet_forecasting", None, "triplet_encoder", "transformer_decoder"),
            ("pytorch_dataset", "eic_forecasting", None, "eic_encoder", "transformer_decoder"),
            ("multiwindow_pytorch_dataset", "ebcl", None, "triplet_encoder", "transformer_encoder"),
            (
                "multiwindow_pytorch_dataset",
                "value_forecasting",
                None,
                "triplet_encoder",
                "transformer_encoder",
            ),
            ("multiwindow_pytorch_dataset", "ocp", "true", "triplet_encoder", "transformer_encoder"),
            ("multiwindow_pytorch_dataset", "ocp", "false", "triplet_encoder", "transformer_encoder"),
        ]
    ]
)
def get_kwargs(request, meds_dir) -> dict:
    def helper(extra_overrides: list[str] = []):
        data, model, early_fusion, input_encoder, backbone = request.param
        overrides, raises_value_error, supervised = get_overrides_and_exceptions(
            data, model, early_fusion, input_encoder, backbone
        )
        cfg = create_cfg(overrides=overrides + extra_overrides, meds_dir=meds_dir, supervised=supervised)
        return dict(
            cfg=cfg,
            raises_value_error=raises_value_error,
            input_kwargs=dict(
                data=data,
                model=model,
                early_fusion=early_fusion,
                input_encoder=input_encoder,
                backbone=backbone,
            ),
        )

    return helper


def test_fast_dev_train(get_kwargs: dict, tmp_path):  # noqa: F811
    """Tests the training configuration provided by the `kwargs` pytest fixture.

    :param kwargs: A dictionary containing the configuration and a flag for expected ValueError.
    """
    kwargs = get_kwargs()
    cfg = kwargs["cfg"]
    raises_value_error = kwargs["raises_value_error"]

    with open_dict(cfg):
        cfg.trainer.fast_dev_run = True
        cfg.trainer.accelerator = "cpu"
        cfg.paths.output_dir = str(tmp_path)
    HydraConfig().set_config(cfg)

    if raises_value_error:
        with pytest.raises(hydra.errors.InstantiationException):
            train_main(cfg)
    else:
        train_main(cfg)


def test_fast_dev_train_lr_scheduler(tmp_path, meds_dir):  # noqa: F811
    """Tests the training configuration provided by the `kwargs` pytest fixture.

    :param kwargs: A dictionary containing the configuration and a flag for expected ValueError.
    """
    data, model, early_fusion, input_encoder, backbone = (
        "pytorch_dataset",
        "supervised",
        None,
        "textcode_encoder",
        "transformer_encoder",
    )
    overrides, raises_value_error, supervised = get_overrides_and_exceptions(
        data, model, early_fusion, input_encoder, backbone
    )
    cfg = create_cfg(
        overrides=overrides + ["model/scheduler=reduce_lr_on_plateau"],
        meds_dir=meds_dir,
        supervised=supervised,
    )

    with open_dict(cfg):
        cfg.trainer.fast_dev_run = True
        cfg.trainer.accelerator = "cpu"
        cfg.paths.output_dir = str(tmp_path)
    HydraConfig().set_config(cfg)

    if raises_value_error:
        with pytest.raises(hydra.errors.InstantiationException):
            train_main(cfg)
    else:
        train_main(cfg)


@RunIf(min_gpus=1)
def test_gpu_train(get_kwargs, tmp_path) -> None:  # noqa: F811
    """Tests the training configuration provided by the `cfg_train` pytest fixture.

    :param cfg_train: A DictConfig containing a valid training configuration.
    """
    kwargs = get_kwargs()
    cfg = kwargs["cfg"]
    raises_value_error = kwargs["raises_value_error"]

    with open_dict(cfg):
        cfg.trainer.accelerator = "gpu"
        cfg.paths.output_dir = str(tmp_path)
    HydraConfig().set_config(cfg)
    if raises_value_error:
        with pytest.raises(hydra.errors.InstantiationException):
            train_main(cfg)
    else:
        train_main(cfg)
    time_output_dir = Path(
        subprocess.run(
            ["meds-torch-latest-dir", f"path={tmp_path}"], capture_output=True, text=True
        ).stdout.strip()
    )
    assert "best_model.ckpt" in os.listdir(time_output_dir / "checkpoints")


@RunIf(min_gpus=1)
@pytest.mark.slow
def test_train_epoch_gpu_amp(get_kwargs, tmp_path) -> None:  # noqa: F811
    """Train 1 epoch on GPU with mixed-precision.

    :param cfg_train: A DictConfig containing a valid training configuration.
    """
    kwargs = get_kwargs()
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
            train_main(cfg)
    else:
        train_main(cfg)


@pytest.mark.slow
def test_train_resume(tmp_path: Path, get_kwargs: dict) -> None:  # noqa: F811
    """Run 1 epoch, finish, and resume for another epoch.

    :param tmp_path: The temporary logging path.
    :param cfg_train: A DictConfig containing a valid training configuration.
    """
    kwargs = get_kwargs()
    cfg = kwargs["cfg"]
    raises_value_error = kwargs["raises_value_error"]
    HydraConfig().set_config(cfg)
    with open_dict(cfg):
        cfg.trainer.max_epochs = 1
        cfg.paths.output_dir = str(tmp_path)
        cfg.callbacks.model_checkpoint.save_top_k = -1

    if raises_value_error:
        with pytest.raises(hydra.errors.InstantiationException):
            train_main(cfg)
    else:
        train_main(cfg)
        time_output_dir = Path(
            subprocess.run(
                ["meds-torch-latest-dir", f"path={tmp_path}"], capture_output=True, text=True
            ).stdout.strip()
        )
        files = os.listdir(time_output_dir / "checkpoints")
        assert "last.ckpt" in files
        assert "epoch_000.ckpt" in files
        assert "best_model.ckpt" in files
        assert len(files) == 3

        with open_dict(cfg):
            cfg.ckpt_path = str(time_output_dir / "checkpoints" / "epoch_000.ckpt")
            cfg.trainer.max_epochs = 2

        train_main(cfg)
        time_output_dir = Path(
            subprocess.run(
                ["meds-torch-latest-dir", f"path={tmp_path}"], capture_output=True, text=True
            ).stdout.strip()
        )

        files = os.listdir(time_output_dir / "checkpoints")
        assert "epoch_000.ckpt" in files
        assert "epoch_001.ckpt" in files
        assert "last.ckpt" in files
        assert "last-v1.ckpt" in files
        assert "best_model.ckpt" in files
        assert len(files) == 5


@RunIf(min_gpus=1, wandb=True, do_log=True)
def test_model_memorization_train(get_kwargs, tmp_path) -> None:  # noqa: F811
    """Tests the training configuration provided by the `cfg_train` pytest fixture.

    :param cfg_train: A DictConfig containing a valid training configuration.
    """
    kwargs = get_kwargs()
    cfg = kwargs["cfg"]
    raises_value_error = kwargs["raises_value_error"]

    with open_dict(cfg):
        cfg.trainer.fast_dev_run = False
        cfg.trainer.accelerator = "gpu"
        cfg.paths.output_dir = str(tmp_path)
        cfg.data.dataloader.batch_size = 70
        cfg.trainer.limit_train_batches = None
        cfg.trainer.limit_val_batches = None
        cfg.trainer.limit_test_batches = None
        cfg.trainer.max_epochs = 1000
        cfg.trainer.max_steps = -1
        cfg.trainer.fast_dev_run = False
        cfg.model.backbone.dropout = 0
        lr = 1e-16
        lr_string = "%.2E" % Decimal(lr)
        cfg.model.optimizer.lr = lr
        del cfg.callbacks.early_stopping
        model = kwargs["input_kwargs"]["backbone"]
        cfg.logger.name = f"{model}--{lr_string}"

    HydraConfig().set_config(cfg)
    if raises_value_error:
        with pytest.raises(hydra.errors.InstantiationException):
            train_main(cfg)
    else:
        train_main(cfg)
