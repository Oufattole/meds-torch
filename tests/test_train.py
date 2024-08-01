"""Tests the training configuration provided by the `cfg_train` pytest fixture.

TODO: finish the meds_torch.train function and setup tests for it
"""
from pathlib import Path

import hydra
import lightning
import pytest
import torch

from tests.conftest import SUPERVISED_TASK_NAME, create_cfg


@pytest.mark.parametrize("data", ["pytorch_dataset"])
@pytest.mark.parametrize("input_encoder", ["triplet_encoder"])
@pytest.mark.parametrize(
    "backbone",
    ["transformer_decoder"],
)
@pytest.mark.parametrize("model", ["supervised"])  # "token_masking"
def test_train_supervised(
    data: str, input_encoder: str, backbone: str, model: str, meds_dir
) -> None:  # cfg: DictConfig,
    """Tests the training configuration provided by the `cfg_train` pytest fixture.

    :param cfg_train: A DictConfig containing a valid training configuration.
    """
    overrides = [
        f"data={data}",
        f"model/input_encoder={input_encoder}",
        f"model/backbone={backbone}",
        f"model={model}",
        f"data.task_name={SUPERVISED_TASK_NAME}",
    ]
    cfg = create_cfg(overrides=overrides, meds_dir=meds_dir)
    assert Path(cfg.data.task_label_path).exists()
    dm = hydra.utils.instantiate(cfg.data)
    dm.setup()
    train_dataloader = dm.train_dataloader()
    lightning.Trainer(accelerator="cpu", fast_dev_run=True).fit(
        model=hydra.utils.instantiate(cfg.model),
        train_dataloaders=train_dataloader,
    )


@pytest.mark.parametrize("data", ["pytorch_dataset"])
@pytest.mark.parametrize("backbone", ["transformer_decoder"])
@pytest.mark.parametrize("model", ["token_forecasting"])  # "token_masking"
@pytest.mark.parametrize("input_encoder", ["triplet_encoder", "triplet_prompt_encoder"])  # "token_masking"
def test_train_token_forecasting(
    data: str, backbone: str, model: str, meds_dir, input_encoder: str
) -> None:  # cfg: DictConfig,
    """Tests the training configuration provided by the `cfg_train` pytest fixture.

    :param cfg_train: A DictConfig containing a valid training configuration.
    """
    if input_encoder == "triplet_prompt_encoder":
        tensorization_name = "prompt_expanded_observation"
    elif input_encoder == "triplet_encoder":
        tensorization_name = "default"
    else:
        raise ValueError(f"Invalid input_encoder: {input_encoder}")

    overrides = [
        f"data={data}",
        f"model/input_encoder={input_encoder}",
        f"model/backbone={backbone}",
        f"model={model}",
        f"data.task_name={SUPERVISED_TASK_NAME}",
    ]
    cfg = create_cfg(overrides=overrides, meds_dir=meds_dir)
    cfg.data.tensorization_name = tensorization_name
    assert Path(cfg.data.task_label_path).exists()
    dm = hydra.utils.instantiate(cfg.data)
    dm.setup()
    train_dataloader = dm.train_dataloader()
    model = hydra.utils.instantiate(cfg.model)
    lightning.Trainer(accelerator="cpu", fast_dev_run=True).fit(
        model=model,
        train_dataloaders=train_dataloader,
    )
    torch.manual_seed(0)
    batch = next(iter(train_dataloader))
    seq_len = 20
    output = model.generate(batch, seq_len)
    assert output.shape == torch.Size([2, seq_len, 4])


@pytest.mark.parametrize("data", ["multiwindow_pytorch_dataset"])
@pytest.mark.parametrize("input_encoder", ["triplet_encoder"])
@pytest.mark.parametrize(
    "backbone",
    ["transformer_decoder"],
)
@pytest.mark.parametrize("model", ["ebcl", "value_forecasting"])  # "token_forecasting"
def test_ebcl_train(
    data: str, input_encoder: str, backbone: str, model: str, meds_dir
) -> None:  # cfg: DictConfig,
    """Tests the training configuration provided by the `cfg_train` pytest fixture.

    :param cfg_train: A DictConfig containing a valid training configuration.
    """
    overrides = [
        f"data={data}",
        f"model/input_encoder={input_encoder}",
        f"model/backbone={backbone}",
        f"model={model}",
        f"data.task_name={SUPERVISED_TASK_NAME}",
    ]
    cfg = create_cfg(overrides=overrides, meds_dir=meds_dir)
    assert Path(cfg.data.task_label_path).exists()
    dm = hydra.utils.instantiate(cfg.data)
    dm.setup()
    train_dataloader = dm.train_dataloader()
    lightning.Trainer(accelerator="cpu", fast_dev_run=True).fit(
        model=hydra.utils.instantiate(cfg.model),
        train_dataloaders=train_dataloader,
    )


@pytest.mark.parametrize("data", ["multiwindow_pytorch_dataset"])
@pytest.mark.parametrize("input_encoder", ["triplet_encoder"])
@pytest.mark.parametrize(
    "backbone",
    ["transformer_decoder"],
)
@pytest.mark.parametrize("early_fusion", ["true", "false"])  # "token_forecasting", "value_forecasting"
def test_ocp_train(
    data: str, input_encoder: str, backbone: str, early_fusion: str, meds_dir
) -> None:  # cfg: DictConfig,
    """Tests the training configuration provided by the `cfg_train` pytest fixture.

    :param cfg_train: A DictConfig containing a valid training configuration.
    """
    # input_encoder=input_encoder
    overrides = [
        f"data={data}",
        f"model/input_encoder={input_encoder}",
        f"model/backbone={backbone}",
        "model=ocp",
        f"data.task_name={SUPERVISED_TASK_NAME}",
        f"model.early_fusion={early_fusion}",
    ]
    cfg = create_cfg(overrides=overrides, meds_dir=meds_dir)
    assert Path(cfg.data.task_label_path).exists()
    dm = hydra.utils.instantiate(cfg.data)
    dm.setup()
    train_dataloader = dm.train_dataloader()
    lightning.Trainer(accelerator="cpu", fast_dev_run=True).fit(
        model=hydra.utils.instantiate(cfg.model),
        train_dataloaders=train_dataloader,
    )


# def test_train_fast_dev_run(cfg_train: DictConfig) -> None:
#     """Run for 1 train, val and test step.

#     :param cfg_train: A DictConfig containing a valid training configuration.
#     """
#     HydraConfig().set_config(cfg_train)
#     with open_dict(cfg_train):
#         cfg_train.trainer.fast_dev_run = True
#         cfg_train.trainer.accelerator = "cpu"
#     train(cfg_train)


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
# def test_train_epoch_double_val_loop(cfg_train: DictConfig) -> None:
#     """Train 1 epoch with validation loop twice per epoch.

#     :param cfg_train: A DictConfig containing a valid training configuration.
#     """
#     HydraConfig().set_config(cfg_train)
#     with open_dict(cfg_train):
#         cfg_train.trainer.max_epochs = 1
#         cfg_train.trainer.val_check_interval = 0.5
#     train(cfg_train)


# @pytest.mark.slow
# def test_train_ddp_sim(cfg_train: DictConfig) -> None:
#     """Simulate DDP (Distributed Data Parallel) on 2 CPU processes.

#     :param cfg_train: A DictConfig containing a valid training configuration.
#     """
#     HydraConfig().set_config(cfg_train)
#     with open_dict(cfg_train):
#         cfg_train.trainer.max_epochs = 2
#         cfg_train.trainer.accelerator = "cpu"
#         cfg_train.trainer.devices = 2
#         cfg_train.trainer.strategy = "ddp_spawn"
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
