import os
import subprocess
from pathlib import Path

import hydra
import polars as pl
import pytest
from hydra.core.hydra_config import HydraConfig
from omegaconf import open_dict

from meds_torch.predict import predict
from meds_torch.train import main as train_main
from tests.conftest import create_cfg
from tests.test_configs import get_overrides_and_exceptions


@pytest.fixture(
    params=[
        pytest.param(
            (data, model, model_early_fusion, data_early_fusion, input_encoder, backbone),
            id=(
                f"{data}-{model}-mearlyfusion{model_early_fusion}-dearlyfusion{data_early_fusion}"
                f"-{input_encoder}-{backbone}"
            ),
        )
        for data, model, model_early_fusion, data_early_fusion, input_encoder, backbone in [
            ("pytorch_dataset", "supervised", None, None, "triplet_encoder", "transformer_encoder"),
            ("pytorch_dataset", "triplet_forecasting", None, None, "triplet_encoder", "transformer_decoder"),
            ("pytorch_dataset", "eic_forecasting", None, None, "eic_encoder", "transformer_decoder"),
            (
                "multiwindow_pytorch_dataset",
                "eic_forecasting",
                None,
                None,
                "eic_encoder",
                "transformer_decoder",
            ),
            (
                "multiwindow_pytorch_dataset",
                "eic_forecasting",
                None,
                True,
                "eic_encoder",
                "transformer_decoder",
            ),
            ("multiwindow_pytorch_dataset", "ebcl", None, None, "triplet_encoder", "transformer_encoder"),
            (
                "multiwindow_pytorch_dataset",
                "value_forecasting",
                None,
                None,
                "triplet_encoder",
                "transformer_encoder",
            ),
            ("multiwindow_pytorch_dataset", "ocp", "false", None, "triplet_encoder", "transformer_encoder"),
        ]
    ]
)
def get_kwargs(request, meds_dir) -> dict:
    def helper(extra_overrides: list[str] = []):
        data, model, model_early_fusion, data_early_fusion, input_encoder, backbone = request.param
        overrides, raises_value_error, supervised = get_overrides_and_exceptions(
            data, model, model_early_fusion, input_encoder, backbone
        )
        if data_early_fusion:
            overrides += ["data.early_fusion_windows=['pre', 'post']"]
        cfg = create_cfg(overrides=overrides + extra_overrides, meds_dir=meds_dir, supervised=supervised)
        return dict(
            cfg=cfg,
            raises_value_error=raises_value_error,
            input_kwargs=dict(
                data=data,
                model=model,
                early_fusion=model_early_fusion,
                input_encoder=input_encoder,
                backbone=backbone,
            ),
        )

    return helper


@pytest.mark.slow
def test_train_predict(tmp_path: Path, get_kwargs, meds_dir) -> None:  # noqa: F811
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
        if "MultiWindowPytorchDataset" in cfg_train.data.dataset_cls:
            cfg_train.data.default_window_name = "pre"
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
        overrides += [
            "data.do_include_subject_id=true",
            "data.do_include_prediction_time=true",
        ]
        if kwargs["input_kwargs"]["model"] == "eic_forecasting":
            overrides += [
                "model.num_samples=2",
                "data.max_seq_len=10",
                "model.max_seq_len=20",
            ]
        cfg_pred = create_cfg(
            overrides=overrides,
            meds_dir=meds_dir,
            config_name="predict.yaml",
            supervised=supervised,
        )
        with open_dict(cfg_pred):
            cfg_pred.ckpt_path = str(time_output_dir / "checkpoints" / "last.ckpt")
            cfg_pred.paths.output_dir = str(tmp_path)
            if "MultiWindowPytorchDataset" in cfg_pred.data.dataset_cls:
                cfg_pred.data.default_window_name = "FUSED" if cfg_pred.data.early_fusion_windows else "pre"

        HydraConfig().set_config(cfg_pred)
        predict(cfg_pred)
        assert Path(cfg_pred.paths.predict_fp).exists()
        df = pl.read_parquet(cfg_pred.paths.predict_fp)
        assert "subject_id" in df.columns
        assert "prediction_time" in df.columns
        if kwargs["input_kwargs"]["model"] == "supervised":
            assert "boolean_value" in df.columns
            assert "predicted_boolean_value" in df.columns
            assert "predicted_boolean_probability" in df.columns
            assert "logits" in df.columns
        if kwargs["input_kwargs"]["model"] == "eic_forecasting":
            assert "logits_sequence" in df.columns
            assert "loss" in df.columns
        if kwargs["input_kwargs"]["model"] == "multiwindow_pytorch_dataset":
            assert "embeddings" in df.columns
            assert "logits_sequence" in df.columns
