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
from meds_torch.input_encoder.triplet_encoder import TripletEncoder
from meds_torch.models.components.lstm import LstmModel
from meds_torch.models.components.mamba import MambaModel
from meds_torch.models.components.transformer_decoder import TransformerDecoderModel
from meds_torch.models.components.transformer_encoder import TransformerEncoderModel
from meds_torch.models.components.transformer_encoder_attn_avg import (
    AttentionAveragedTransformerEncoderModel,
)
from tests.conftest import create_cfg


@pytest.mark.parametrize("data", ["pytorch_dataset", "multiwindow_pytorch_dataset"])
@pytest.mark.parametrize("input_encoder", ["triplet_encoder"])
@pytest.mark.parametrize(
    "backbone",
    ["lstm", "transformer_decoder", "transformer_encoder", "transformer_encoder_attn_avg"],
)  # TODO: add mamba unittest with a runIF conditional
@pytest.mark.parametrize("model", ["supervised", "ebcl", "ocp", "token_forecasting", "value_forecasting"])
def test_train_config(
    data: str, input_encoder: str, backbone: str, model: str, meds_dir
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
    cfg = create_cfg(overrides=overrides, meds_dir=meds_dir, supervised=(model == "supervised"))
    assert isinstance(cfg, DictConfig)
    assert isinstance(cfg.data, DictConfig)
    assert isinstance(cfg.model.input_encoder, DictConfig)
    assert isinstance(cfg.model.backbone, DictConfig)
    assert isinstance(cfg.model, DictConfig)

    HydraConfig().set_config(cfg)

    assert isinstance(hydra.utils.instantiate(cfg.data), MEDSDataModule)
    assert isinstance(hydra.utils.instantiate(cfg.model.input_encoder), TripletEncoder)

    backbone_model = hydra.utils.instantiate(cfg.model.backbone)
    if backbone == "lstm":
        assert isinstance(backbone_model, LstmModel)
    elif backbone == "transformer_decoder":
        assert isinstance(backbone_model, TransformerDecoderModel)
    elif backbone == "mamba":
        assert isinstance(backbone_model, MambaModel)
    elif backbone == "transformer_encoder":
        assert isinstance(backbone_model, TransformerEncoderModel)
    elif backbone == "transformer_encoder_attn_avg":
        assert isinstance(backbone_model, AttentionAveragedTransformerEncoderModel)
    else:
        raise NotImplementedError(f"Unsupported backbone {backbone}!")
    if model == "token_forecasting" and backbone not in ["transformer_decoder", "lstm", "mamba"]:
        with pytest.raises(hydra.errors.InstantiationException):
            hydra.utils.instantiate(cfg.model)
    else:
        assert isinstance(hydra.utils.instantiate(cfg.model), LightningModule)
