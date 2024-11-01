"""Tests the training configuration provided by the `cfg_train` pytest fixture.

TODO: finish the meds_torch.train function and setup tests for it
"""
import copy
import os
import subprocess
from pathlib import Path

import hydra
import polars as pl
import pytest
from hydra.core.hydra_config import HydraConfig
from omegaconf import open_dict

from meds_torch.finetune import main as finetune_main
from meds_torch.train import main as train_main
from meds_torch.tune import main as tune_main
from tests.conftest import create_cfg
from tests.helpers.run_if import RunIf
from tests.test_configs import get_overrides_and_exceptions  # noqa: F401


@pytest.fixture(
    params=[
        pytest.param(
            (data, model, early_fusion, input_encoder, backbone),
            id=f"{data}-{model}-earlyfusion{early_fusion}-{input_encoder}-{backbone}",
        )
        for data, model, early_fusion, input_encoder, backbone in [
            ("pytorch_dataset", "supervised", None, "triplet_encoder", "transformer_encoder"),
            ("pytorch_dataset", "triplet_forecasting", None, "triplet_encoder", "transformer_decoder"),
            ("pytorch_dataset", "eic_forecasting", None, "eic_encoder", "transformer_decoder"),
            ("random_windows_pytorch_dataset", "ebcl", None, "triplet_encoder", "transformer_encoder"),
            (
                "random_windows_pytorch_dataset",
                "value_forecasting",
                None,
                "triplet_encoder",
                "transformer_encoder",
            ),
            ("random_windows_pytorch_dataset", "ocp", "true", "triplet_encoder", "transformer_encoder"),
            ("random_windows_pytorch_dataset", "ocp", "false", "triplet_encoder", "transformer_encoder"),
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


@pytest.mark.slow
@RunIf(min_gpus=1)
def test_finetune(tmp_path: Path, get_kwargs, meds_dir) -> None:  # noqa: F811
    """Run 1 epoch, finish, and resume for another epoch.

    :param tmp_path: The temporary logging path.
    :param cfg_train: A DictConfig containing a valid training configuration.
    """
    kwargs = get_kwargs()
    cfg_pretrain = kwargs["cfg"]
    raises_value_error = kwargs["raises_value_error"]
    HydraConfig().set_config(cfg_pretrain)
    pretrain_path = tmp_path / "pretrain"
    finetune_path = tmp_path / "finetune"
    with open_dict(cfg_pretrain):
        cfg_pretrain.trainer.max_epochs = 1
        cfg_pretrain.paths.output_dir = str(pretrain_path)
        cfg_pretrain.callbacks.model_checkpoint.save_top_k = -1
        cfg_pretrain.trainer.accelerator = "gpu"

    if raises_value_error:
        with pytest.raises(hydra.errors.InstantiationException):
            train_main(cfg_pretrain)
    else:
        train_main(cfg_pretrain)
        latest_pretrain_path = Path(
            subprocess.run(
                ["meds-torch-latest-dir", f"path={pretrain_path}"], capture_output=True, text=True
            ).stdout.strip()
        )
        assert latest_pretrain_path.exists()
        assert "last.ckpt" in os.listdir(Path(latest_pretrain_path) / "checkpoints")
        assert "hydra_config.yaml" in os.listdir(latest_pretrain_path)

        supervised_input_kwargs = copy.deepcopy(kwargs["input_kwargs"])
        supervised_input_kwargs["model"] = "supervised"
        supervised_input_kwargs["data"] = "pytorch_dataset"
        supervised_input_kwargs["early_fusion"] = None
        overrides, _, supervised = get_overrides_and_exceptions(**supervised_input_kwargs)
        cfg_finetune = create_cfg(
            overrides=overrides,
            meds_dir=meds_dir,
            config_name="finetune.yaml",
            supervised=supervised,
        )

        with open_dict(cfg_finetune):
            cfg_finetune.pretrain_path = str(latest_pretrain_path.resolve())
            cfg_finetune.paths.output_dir = str(finetune_path)
            cfg_finetune.trainer.max_epochs = 2
            cfg_finetune.callbacks.model_checkpoint.save_top_k = -1
            cfg_finetune.trainer.accelerator = "gpu"

        HydraConfig().set_config(cfg_finetune)
        finetune_main(cfg_finetune)

        latest_finetune_path = Path(
            subprocess.run(
                ["meds-torch-latest-dir", f"path={finetune_path}"], capture_output=True, text=True
            ).stdout.strip()
        )
        files = os.listdir(latest_finetune_path / "checkpoints")
        assert "epoch_001.ckpt" in files
        assert "last.ckpt" in files


@pytest.mark.slow
@RunIf(min_gpus=1)
def test_finetune_multiseed(tmp_path: Path, get_kwargs, meds_dir) -> None:  # noqa: F811
    """Run 1 epoch, finish, and resume for another epoch. Tune the pretrained model Tune the finetuning model
    multiseed trained the finetuned model.

    :param tmp_path: The temporary logging path.
    :param cfg_train: A DictConfig containing a valid training configuration.
    """
    kwargs = get_kwargs(extra_overrides=["callbacks=tune_default", "hparams_search=ray_tune", "trainer=ray"])
    cfg_pretrain = kwargs["cfg"]
    raises_value_error = kwargs["raises_value_error"]
    HydraConfig().set_config(cfg_pretrain)
    pretrain_path = tmp_path / "tune/pretrain"
    finetune_path = tmp_path / "tune/finetune"
    multiseed_finetune_path = tmp_path / "multiseed/finetune"

    with open_dict(cfg_pretrain):
        cfg_pretrain.trainer.max_epochs = 1
        cfg_pretrain.paths.output_dir = str(pretrain_path)
        cfg_pretrain.hparams_search.ray.num_samples = 2
        cfg_pretrain.hparams_search.scheduler.grace_period = 1
        cfg_pretrain.trainer.accelerator = "gpu"

    if raises_value_error:
        with pytest.raises(hydra.errors.InstantiationException):
            tune_main(cfg_pretrain)
    else:
        # pretrain sweep
        tune_main(cfg_pretrain)
        latest_pretrain_path = Path(
            subprocess.run(
                ["meds-torch-latest-dir", f"path={pretrain_path}"], capture_output=True, text=True
            ).stdout.strip()
        )
        assert latest_pretrain_path.exists()
        assert "best_model.ckpt" in os.listdir(Path(latest_pretrain_path) / "checkpoints")
        assert "hydra_config.yaml" in os.listdir(latest_pretrain_path)

        # finetune sweep
        supervised_input_kwargs = copy.deepcopy(kwargs["input_kwargs"])
        supervised_input_kwargs["model"] = "supervised"
        supervised_input_kwargs["data"] = "pytorch_dataset"
        supervised_input_kwargs["early_fusion"] = None
        overrides, _, supervised = get_overrides_and_exceptions(**supervised_input_kwargs)
        cfg_finetune = create_cfg(
            overrides=overrides + ["callbacks=tune_default", "hparams_search=ray_tune", "trainer=ray"],
            meds_dir=meds_dir,
            config_name="finetune.yaml",
            supervised=supervised,
        )

        with open_dict(cfg_finetune):
            cfg_finetune.pretrain_path = str(latest_pretrain_path.resolve())
            cfg_finetune.paths.output_dir = str(finetune_path)
            cfg_finetune.trainer.max_epochs = 2
            cfg_finetune.hparams_search.ray.num_samples = 2
            cfg_finetune.hparams_search.scheduler.grace_period = 1
            cfg_finetune.trainer.accelerator = "gpu"

        HydraConfig().set_config(cfg_finetune)
        tune_main(cfg_finetune)

        latest_finetune_path = Path(
            subprocess.run(
                ["meds-torch-latest-dir", f"path={finetune_path}"], capture_output=True, text=True
            ).stdout.strip()
        )
        files = os.listdir(latest_finetune_path / "checkpoints")
        assert "best_model.ckpt" in files

        # multiseed finetune
        cfg_finetune = create_cfg(
            overrides=overrides + ["callbacks=tune_default", "hparams_search=ray_multiseed", "trainer=ray"],
            meds_dir=meds_dir,
            config_name="finetune.yaml",
            supervised=supervised,
        )
        with open_dict(cfg_finetune):
            cfg_finetune.pretrain_path = str(latest_finetune_path.resolve())
            cfg_finetune.paths.output_dir = str(multiseed_finetune_path)
            cfg_finetune.trainer.max_epochs = 2
            cfg_finetune.hparams_search.ray.num_samples = 2
            cfg_finetune.trainer.accelerator = "gpu"

        HydraConfig().set_config(cfg_finetune)
        tune_main(cfg_finetune)

        latest_multiseed_finetune_path = Path(
            subprocess.run(
                ["meds-torch-latest-dir", f"path={multiseed_finetune_path}"], capture_output=True, text=True
            ).stdout.strip()
        )
        files = os.listdir(latest_multiseed_finetune_path / "checkpoints")
        assert "best_model.ckpt" in files
        assert pl.read_parquet(latest_multiseed_finetune_path / "sweep_results_summary.parquet").shape[0] == 2
