import hydra
import pytest
from hydra.core.hydra_config import HydraConfig
from lightning import LightningModule
from omegaconf import DictConfig


@pytest.fixture
def cfg(request):
    # Use request.param to fetch the correct fixture dynamically
    return request.getfixturevalue(request.param)


from meds_torch.data.datamodule import MEDSDataModule
from meds_torch.input_encoder.eic_encoder import EicEncoder
from meds_torch.input_encoder.triplet_encoder import TripletEncoder
from meds_torch.models.components.lstm import LstmModel
from meds_torch.models.components.transformer_decoder import TransformerDecoderModel
from meds_torch.models.components.transformer_encoder import TransformerEncoderModel
from tests.conftest import create_cfg


def get_overrides_and_exceptions(data, model, early_fusion, input_encoder, backbone):
    token_type = input_encoder.split("_")[0]
    backbone_token_type = token_type

    if token_type == "triplet":
        collate_type_override = "data.collate_type=triplet"
    elif token_type == "eic":
        collate_type_override = "data.collate_type=eic"
    elif token_type == "textcode":
        collate_type_override = "data.collate_type=triplet"
    else:
        raise NotImplementedError(f"Unsupported token type {token_type}!")

    backbone = f"{backbone_token_type}_{backbone}"
    overrides = [
        f"data={data}",
        f"model/input_encoder={input_encoder}",
        f"model/backbone={backbone}",
        f"model={model}",
        collate_type_override,
        "logger=wandb",
    ]

    raises_value_error = False

    supervised = model == "supervised"
    if model in ["triplet_forecasting", "eic_forecasting"]:
        raises_value_error = "transformer_encoder" in backbone
        if model == "triplet_forecasting" and token_type == "eic":
            raises_value_error = True
        if model == "eic_forecasting" and token_type != "eic":
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
            ("pytorch_dataset", "triplet_forecasting", None),
            ("multiwindow_pytorch_dataset", "ebcl", None),
            ("multiwindow_pytorch_dataset", "value_forecasting", None),
            ("multiwindow_pytorch_dataset", "ocp", "true"),
            ("multiwindow_pytorch_dataset", "ocp", "false"),
        ]
        for input_encoder in ["triplet_encoder", "eic_encoder"]
        for backbone in ["transformer_decoder", "transformer_encoder", "lstm", "transformer_encoder_attn_avg"]
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


def test_train_config(get_kwargs) -> None:  # cfg: DictConfig,
    """Tests the training configuration provided by the `cfg_train` pytest fixture.

    :param cfg_train: A DictConfig containing a valid training configuration.
    """
    kwargs = get_kwargs()
    cfg = kwargs["cfg"]
    raises_value_error = kwargs["raises_value_error"]
    # input_encoder=input_encoder
    assert isinstance(cfg, DictConfig)
    assert isinstance(cfg.data, DictConfig)
    assert isinstance(cfg.model.input_encoder, DictConfig)
    assert isinstance(cfg.model.backbone, DictConfig)
    assert isinstance(cfg.model, DictConfig)

    HydraConfig().set_config(cfg)

    assert isinstance(hydra.utils.instantiate(cfg.data), MEDSDataModule)
    input_encoder = kwargs["input_kwargs"]["input_encoder"]
    if input_encoder == "triplet_encoder":
        assert isinstance(hydra.utils.instantiate(cfg.model.input_encoder), TripletEncoder)
    elif input_encoder == "eic_encoder":
        assert isinstance(hydra.utils.instantiate(cfg.model.input_encoder), EicEncoder)
    else:
        raise NotImplementedError(f"Unsupported input_encoder {input_encoder}!")

    backbone_model = hydra.utils.instantiate(cfg.model.backbone)
    backbone = kwargs["input_kwargs"]["backbone"]
    if backbone == "lstm":
        assert isinstance(backbone_model, LstmModel)
    elif backbone == "transformer_decoder":
        assert isinstance(backbone_model, TransformerDecoderModel)
    elif backbone == "transformer_encoder":
        assert isinstance(backbone_model, TransformerEncoderModel)
    elif backbone == "transformer_encoder_attn_avg":
        assert isinstance(backbone_model, TransformerEncoderModel)
    else:
        raise NotImplementedError(f"Unsupported backbone {backbone}!")

    if raises_value_error:
        with pytest.raises(hydra.errors.InstantiationException):
            hydra.utils.instantiate(cfg.model)
    else:
        assert isinstance(hydra.utils.instantiate(cfg.model), LightningModule)
