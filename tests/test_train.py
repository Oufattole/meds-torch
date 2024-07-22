"""Tests the training configuration provided by the `cfg_train` pytest fixture.

TODO: finish the meds_torch.train function and setup tests for it
"""
from pathlib import Path

import hydra
import lightning
import pytest

from tests.conftest import SUPERVISED_TASK_NAME, create_cfg


@pytest.mark.parametrize("data", ["meds_pytorch_dataset"])
@pytest.mark.parametrize("input_encoder", ["triplet_encoder"])
@pytest.mark.parametrize(
    "backbone",
    ["transformer_decoder"],
)
@pytest.mark.parametrize("model", ["supervised"])
def test_train(
    data: str, input_encoder: str, backbone: str, model: str, meds_dir
) -> None:  # cfg: DictConfig,
    """Tests the training configuration provided by the `cfg_train` pytest fixture.

    :param cfg_train: A DictConfig containing a valid training configuration.
    """
    # input_encoder=input_encoder
    overrides = [
        f"data={data}",
        f"sequence_model/input_encoder={input_encoder}",
        f"sequence_model/backbone={backbone}",
        f"sequence_model={model}",
        f"data.task_name={SUPERVISED_TASK_NAME}",
    ]
    cfg = create_cfg(overrides=overrides, meds_dir=meds_dir)
    assert Path(cfg.data.task_label_path).exists()
    dm = hydra.utils.instantiate(cfg.data)
    dm.setup(stage="train")
    train_dataloader = dm.train_dataloader()
    lightning.Trainer(accelerator="cpu", fast_dev_run=True).fit(
        model=hydra.utils.instantiate(cfg.sequence_model),
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
