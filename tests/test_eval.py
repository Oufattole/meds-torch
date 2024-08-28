import os
from pathlib import Path

import hydra
import pytest
from hydra.core.hydra_config import HydraConfig
from omegaconf import open_dict

from meds_torch.eval import evaluate
from meds_torch.train import train
from tests.conftest import create_cfg
from tests.test_train import get_overrides_and_exceptions


@pytest.mark.slow
def test_train_eval(tmp_path: Path, kwargs: dict, meds_dir) -> None:
    """Tests training and evaluation by training for 1 epoch with `train.py` then evaluating with `eval.py`.

    :param tmp_path: The temporary logging path.
    :param cfg_train: A DictConfig containing a valid training configuration.
    :param cfg_eval: A DictConfig containing a valid evaluation configuration.
    """
    cfg_train = kwargs["cfg"]
    raises_value_error = kwargs["raises_value_error"]
    with open_dict(cfg_train):
        cfg_train.trainer.max_epochs = 1
        cfg_train.test = True
        cfg_train.paths.output_dir = str(tmp_path)
    HydraConfig().set_config(cfg_train)
    if raises_value_error:
        with pytest.raises(hydra.errors.InstantiationException):
            train(cfg_train)
    else:
        train_metric_dict, _ = train(cfg_train)
        assert "last.ckpt" in os.listdir(tmp_path / "checkpoints")
        overrides, _, supervised = get_overrides_and_exceptions(**kwargs["input_kwargs"])
        cfg_eval = create_cfg(
            overrides=overrides, meds_dir=meds_dir, config_name="eval.yaml", supervised=supervised
        )
        with open_dict(cfg_eval):
            cfg_eval.ckpt_path = str(tmp_path / "checkpoints" / "last.ckpt")

        HydraConfig().set_config(cfg_eval)
        test_metric_dict, _ = evaluate(cfg_eval)

        assert "train/loss" in train_metric_dict
        assert "test/loss" in test_metric_dict
