import hydra
import pytest
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig


@pytest.fixture
def cfg(request):
    # Use request.param to fetch the correct fixture dynamically
    return request.getfixturevalue(request.param)


@pytest.mark.parametrize("cfg", ["cfg_meds_train", "cfg_meds_multiwindow_train"], indirect=True)
def test_train_config(cfg: DictConfig) -> None:
    """Tests the training configuration provided by the `cfg_train` pytest fixture.

    :param cfg_train: A DictConfig containing a valid training configuration.
    """
    assert cfg
    assert cfg.data
    assert cfg.model
    assert cfg.trainer

    HydraConfig().set_config(cfg)

    hydra.utils.instantiate(cfg.data)
    # hydra.utils.instantiate(cfg.model)
    # hydra.utils.instantiate(cfg.trainer)
