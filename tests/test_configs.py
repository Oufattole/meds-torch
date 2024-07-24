import hydra
import pytest
from hydra.core.hydra_config import HydraConfig
from lightning import LightningModule
from omegaconf import DictConfig


@pytest.fixture
def cfg(request):
    # Use request.param to fetch the correct fixture dynamically
    return request.getfixturevalue(request.param)


from meds_torch.data.meds_datamodule import MEDSDataModule
from meds_torch.input_encoder.triplet_encoder import TripletEncoder
from meds_torch.sequence_models.components.lstm import LstmModel
from meds_torch.sequence_models.components.mamba import MambaModel
from meds_torch.sequence_models.components.transformer_decoder import (
    TransformerDecoderModel,
)
from meds_torch.sequence_models.components.transformer_encoder import (
    TransformerEncoderModel,
)
from meds_torch.sequence_models.components.transformer_encoder_attn_avg import (
    AttentionAveragedTransformerEncoderModel,
)
from tests.conftest import SUPERVISED_TASK_NAME, create_cfg


@pytest.mark.parametrize("data", ["meds_pytorch_dataset", "meds_multiwindow_pytorch_dataset"])
@pytest.mark.parametrize("input_encoder", ["triplet_encoder"])
@pytest.mark.parametrize(
    "backbone",
    ["lstm", "transformer_decoder", "transformer_encoder", "transformer_encoder_attn_avg"],
)  # TODO: add mamba unittest with a runIF conditional
@pytest.mark.parametrize("model", ["supervised"])
def test_train_config(
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
    ]
    if model == "supervised":
        overrides.append(f"data.task_name={SUPERVISED_TASK_NAME}")
    cfg = create_cfg(overrides=overrides, meds_dir=meds_dir)
    assert isinstance(cfg, DictConfig)
    assert isinstance(cfg.data, DictConfig)
    assert isinstance(cfg.sequence_model.input_encoder, DictConfig)
    assert isinstance(cfg.sequence_model.backbone, DictConfig)
    assert isinstance(cfg.sequence_model, DictConfig)

    HydraConfig().set_config(cfg)

    assert isinstance(hydra.utils.instantiate(cfg.data), MEDSDataModule)
    assert isinstance(hydra.utils.instantiate(cfg.sequence_model.input_encoder), TripletEncoder)
    backbone_model = hydra.utils.instantiate(cfg.sequence_model.backbone)
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
    assert isinstance(hydra.utils.instantiate(cfg.sequence_model), LightningModule)


# @pytest.mark.parametrize("sequence_model", ["supervised"]) # "transformer_decoder"
# def test_train_config(data: str, input_encoder: str, backbone: str, meds_dir) -> None: # cfg: DictConfig,
#     """Tests the training configuration provided by the `cfg_train` pytest fixture.

#     :param cfg_train: A DictConfig containing a valid training configuration.
#     """
#     # input_encoder=input_encoder
#     overrides = [f"data={data}", f"input_encoder={input_encoder}", f"sequence_model/backbone={backbone}"]
#     cfg = create_cfg(overrides=overrides, meds_dir=meds_dir)

#     HydraConfig().set_config(cfg)
#     backbone_model = hydra.utils.instantiate(cfg.sequence_model.backbone)
#     if backbone == "lstm":
#         assert isinstance(backbone_model, LstmModel)
#     elif backbone == "transformer_decoder":
#         assert isinstance(backbone_model, TransformerDecoderModel)
#     elif backbone == "mamba":
#         assert isinstance(backbone_model, MambaModel)
#     elif backbone == "transformer_encoder":
#         assert isinstance(backbone_model, TransformerEncoderModel)
#     elif backbone == "transformer_encoder_attn_avg":
#         assert isinstance(backbone_model, AttentionAveragedTransformerEncoderModel)
#     else:
#         raise NotImplementedError(f"Unsupported backbone {backbone}!")
