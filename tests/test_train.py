"""Tests the training configuration provided by the `cfg_train` pytest fixture.

TODO: finish the meds_torch.train function and setup tests for it
"""

import hydra
import pytest
from hydra.core.hydra_config import HydraConfig
from omegaconf import open_dict

from meds_torch.train import train
from tests.conftest import SUPERVISED_TASK_NAME, create_cfg


@pytest.mark.parametrize("input_encoder", ["triplet_encoder", "triplet_prompt_encoder"])
@pytest.mark.parametrize(
    "backbone", ["transformer_decoder", "transformer_encoder", "transformer_encoder_attn_avg", "lstm"]
)
@pytest.mark.parametrize(
    "data, model, early_fusion",
    [
        # Supervised
        ("pytorch_dataset", "supervised", None),
        # Token Forecasting
        ("pytorch_dataset", "token_forecasting", None),
        # EBCL
        ("multiwindow_pytorch_dataset", "ebcl", None),
        # Value Forecasting
        ("multiwindow_pytorch_dataset", "value_forecasting", None),
        # OCP
        ("multiwindow_pytorch_dataset", "ocp", "true"),
        ("multiwindow_pytorch_dataset", "ocp", "false"),
    ],
)
def test_train(
    data: str, input_encoder: str, model: str, backbone: str, early_fusion: str, meds_dir
) -> None:  # cfg: DictConfig,
    """Tests the training configuration provided by the `cfg_train` pytest fixture.

    :param cfg_train: A DictConfig containing a valid training configuration.
    """
    # input_encoder=input_encoder
    overrides = [
        f"data={data}",
        f"model/input_encoder={input_encoder}",
        f"model/backbone={backbone}",
        f"model={model}",
    ]
    raises_value_error = False

    if model == "supervised":
        overrides.append(f"data.task_name={SUPERVISED_TASK_NAME}")
    elif model == "token_forecasting" and backbone not in ["transformer_decoder", "lstm"]:
        raises_value_error = True

    if early_fusion is not None:
        overrides.append(f"model.early_fusion={early_fusion}")

    cfg = create_cfg(overrides=overrides, meds_dir=meds_dir)
    with open_dict(cfg):
        cfg.trainer.fast_dev_run = True
        cfg.trainer.accelerator = "cpu"
    HydraConfig().set_config(cfg)
    if raises_value_error:
        with pytest.raises(hydra.errors.InstantiationException):
            train(cfg)
    else:
        train(cfg)


# @RunIf(min_gpus=1)
# def test_train_fast_dev_run_gpu(cfg_train: DictConfig) -> None:
#     """Run for 1 train, val and test step on GPU.

#     :param cfg_train: A DictConfig containing a valid training configuration.
#     """
#     HydraConfig().set_config(cfg_train)
#     with open_dict(cfg_train):
#         cfg_train.trainer.fast_dev_run = True
#         cfg_train.trainer.accelerator = "gpu"
#     train(cfg_train)


# @RunIf(min_gpus=1)
# @pytest.mark.slow
# def test_train_epoch_gpu_amp(cfg_train: DictConfig) -> None:
#     """Train 1 epoch on GPU with mixed-precision.

#     :param cfg_train: A DictConfig containing a valid training configuration.
#     """
#     HydraConfig().set_config(cfg_train)
#     with open_dict(cfg_train):
#         cfg_train.trainer.max_epochs = 1
#         cfg_train.trainer.accelerator = "gpu"
#         cfg_train.trainer.precision = 16
#     train(cfg_train)

# @pytest.mark.slow
# def test_train_resume(tmp_path: Path, cfg_train: DictConfig) -> None:
#     """Run 1 epoch, finish, and resume for another epoch.

#     :param tmp_path: The temporary logging path.
#     :param cfg_train: A DictConfig containing a valid training configuration.
#     """
#     with open_dict(cfg_train):
#         cfg_train.trainer.max_epochs = 1

#     HydraConfig().set_config(cfg_train)
#     metric_dict_1, _ = train(cfg_train)

#     files = os.listdir(tmp_path / "checkpoints")
#     assert "last.ckpt" in files
#     assert "epoch_000.ckpt" in files

#     with open_dict(cfg_train):
#         cfg_train.ckpt_path = str(tmp_path / "checkpoints" / "last.ckpt")
#         cfg_train.trainer.max_epochs = 2

#     metric_dict_2, _ = train(cfg_train)

#     files = os.listdir(tmp_path / "checkpoints")
#     assert "epoch_001.ckpt" in files
#     assert "epoch_002.ckpt" not in files

#     assert metric_dict_1["train/acc"] < metric_dict_2["train/acc"]
#     assert metric_dict_1["val/acc"] < metric_dict_2["val/acc"]
