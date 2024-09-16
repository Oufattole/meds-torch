import os
import subprocess
from pathlib import Path

import hydra
import pytest
from hydra.core.hydra_config import HydraConfig
from omegaconf import open_dict

from meds_torch.eval import evaluate
from meds_torch.train import main as train_main
from tests.conftest import create_cfg
from tests.test_configs import get_overrides_and_exceptions


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


@pytest.mark.slow
def test_train_eval(tmp_path: Path, get_kwargs, meds_dir) -> None:  # noqa: F811
    """Tests training and evaluation by training for 1 epoch with `train.py` then evaluating with `eval.py`.

    :param tmp_path: The temporary logging path.
    :param cfg_train: A DictConfig containing a valid training configuration.
    :param cfg_eval: A DictConfig containing a valid evaluation configuration.
    """
    kwargs = get_kwargs()
    cfg_train = kwargs["cfg"]
    raises_value_error = kwargs["raises_value_error"]
    with open_dict(cfg_train):
        cfg_train.trainer.max_epochs = 1
        cfg_train.test = True
        cfg_train.paths.output_dir = str(tmp_path)
    HydraConfig().set_config(cfg_train)
    if raises_value_error:
        with pytest.raises(hydra.errors.InstantiationException):
            train_main(cfg_train)
    else:
        train_main(cfg_train)
        time_output_dir = Path(
            subprocess.run(
                ["meds-torch-latest-dir", f"path={tmp_path}"], capture_output=True, text=True
            ).stdout.strip()
        )
        assert "last.ckpt" in os.listdir(time_output_dir / "checkpoints")
        overrides, _, supervised = get_overrides_and_exceptions(**kwargs["input_kwargs"])
        cfg_eval = create_cfg(
            overrides=overrides, meds_dir=meds_dir, config_name="eval.yaml", supervised=supervised
        )
        with open_dict(cfg_eval):
            cfg_eval.ckpt_path = str(time_output_dir / "checkpoints" / "last.ckpt")
            cfg_eval.paths.output_dir = str(tmp_path)

        HydraConfig().set_config(cfg_eval)
        test_metric_dict, _ = evaluate(cfg_eval)

        assert "test/loss" in test_metric_dict
