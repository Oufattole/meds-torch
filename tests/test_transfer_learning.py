"""Tests the training configuration provided by the `cfg_train` pytest fixture.

TODO: finish the meds_torch.train function and setup tests for it
"""
import copy
import os
from pathlib import Path

import hydra
import pytest
from hydra.core.hydra_config import HydraConfig
from omegaconf import open_dict

from meds_torch.train import train
from meds_torch.transfer_learning import transfer_learning
from tests.conftest import create_cfg
from tests.test_train import get_overrides_and_exceptions


@pytest.mark.slow
def test_transfer_learning(tmp_path: Path, kwargs: dict, meds_dir) -> None:
    """Run 1 epoch, finish, and resume for another epoch.

    :param tmp_path: The temporary logging path.
    :param cfg_train: A DictConfig containing a valid training configuration.
    """
    cfg_pretrain = kwargs["cfg"]
    raises_value_error = kwargs["raises_value_error"]
    HydraConfig().set_config(cfg_pretrain)
    pretrain_path = tmp_path / "pretrain"
    finetune_path = tmp_path / "finetune"
    with open_dict(cfg_pretrain):
        cfg_pretrain.trainer.max_epochs = 1
        cfg_pretrain.paths.output_dir = str(pretrain_path)
        cfg_pretrain.callbacks.model_checkpoint.save_top_k = -1

    if raises_value_error:
        with pytest.raises(hydra.errors.InstantiationException):
            train(cfg_pretrain)
    else:
        metric_dict_1, _ = train(cfg_pretrain)
        files = os.listdir(pretrain_path / "checkpoints")
        assert "last.ckpt" in files
        assert "hydra_config.yaml" in os.listdir(pretrain_path)

        supervised_input_kwargs = copy.deepcopy(kwargs["input_kwargs"])
        supervised_input_kwargs["model"] = "supervised"
        supervised_input_kwargs["data"] = "pytorch_dataset"
        supervised_input_kwargs["early_fusion"] = None
        overrides, _, supervised = get_overrides_and_exceptions(**supervised_input_kwargs)
        cfg_finetune = create_cfg(
            overrides=overrides,
            meds_dir=meds_dir,
            config_name="transfer_learning.yaml",
            supervised=supervised,
        )

        with open_dict(cfg_finetune):
            cfg_finetune.pretrain_ckpt_path = str(pretrain_path / "checkpoints" / "last.ckpt")
            cfg_finetune.pretrain_yaml_path = str(pretrain_path / "hydra_config.yaml")
            cfg_finetune.paths.output_dir = str(finetune_path)
            cfg_finetune.trainer.max_epochs = 2
            cfg_finetune.callbacks.model_checkpoint.save_top_k = -1

        HydraConfig().set_config(cfg_finetune)
        metric_dict_2, _ = transfer_learning(cfg_finetune)

        files = os.listdir(finetune_path / "checkpoints")
        assert "epoch_001.ckpt" in files
        assert "last.ckpt" in files

        assert "train/loss" in metric_dict_1 and "val/loss" in metric_dict_1
        assert "train/loss" in metric_dict_2 and "val/loss" in metric_dict_2
