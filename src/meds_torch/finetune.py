from importlib.resources import files
from typing import Any

import hydra
import lightning as L
import torch
from lightning import Callback, LightningDataModule, LightningModule, Trainer
from lightning.pytorch.loggers import Logger
from omegaconf import DictConfig, OmegaConf, open_dict

from meds_torch.utils import (
    RankedLogger,
    configure_logging,
    get_metric_value,
    instantiate_callbacks,
    instantiate_loggers,
    log_hyperparameters,
    task_wrapper,
)
from meds_torch.utils.resolvers import setup_resolvers

setup_resolvers()
log = RankedLogger(__name__, rank_zero_only=True)
config_yaml = files("meds_torch").joinpath("configs/finetune.yaml")


def initialize_finetune_objects(cfg: DictConfig, **kwargs) -> Trainer:
    """Instantiates a Lightning Trainer object.

    :param cfg: A DictConfig configuration composed by Hydra.
    :return: A Lightning Trainer object.
    """
    # load pretrained backbone and input encoder:
    # set seed for random number generators in pytorch, numpy and python.random
    pretrain_cfg = OmegaConf.load(cfg.pretrain_yaml_path)
    # TODO this just adds backwards compatibility find a better way for loading pretrain_cfg
    # see: https://github.com/Oufattole/meds-torch/issues/47
    with open_dict(pretrain_cfg):
        pretrain_cfg.data.vocab_size = cfg.data.vocab_size
        pretrain_cfg.model.vocab_size = cfg.data.vocab_size

    pretrain_model: LightningModule = hydra.utils.instantiate(pretrain_cfg.model)
    ckpt = torch.load(cfg.pretrain_ckpt_path)
    pretrain_model.load_state_dict(ckpt["state_dict"])

    # set seed for random number generators in pytorch, numpy and python.random
    if cfg.get("seed"):
        L.seed_everything(cfg.seed, workers=True)

    log.info(f"Instantiating datamodule <{cfg.data._target_}>")
    datamodule: LightningDataModule = hydra.utils.instantiate(cfg.data)

    log.info(f"Instantiating model <{cfg.model._target_}>")
    model: LightningModule = hydra.utils.instantiate(cfg.model)

    log.info(f"Loading backbone from {cfg.pretrain_yaml_path}")
    if not isinstance(model.model, pretrain_model.model.__class__):
        raise ValueError(
            f"Model {model.model.__class__} is not compatible with pretrained model"
            f" {pretrain_model.model.__class__}."
        )
    model.model = pretrain_model.model

    log.info(f"Loading input encoder from {cfg.pretrain_yaml_path}")
    if not isinstance(model.input_encoder, pretrain_model.input_encoder.__class__):
        raise ValueError(
            f"Input encoder {model.input_encoder.__class__} is not compatible with pretrained"
            f" input encoder {pretrain_model.input_encoder.__class__}."
        )
    model.input_encoder = pretrain_model.input_encoder

    log.info("Instantiating callbacks...")
    callbacks: list[Callback] = instantiate_callbacks(cfg.get("callbacks"))

    log.info("Instantiating loggers...")
    logger: list[Logger] = instantiate_loggers(cfg.get("logger"))

    log.info(f"Instantiating trainer <{cfg.trainer._target_}>")
    trainer_factory: Trainer = hydra.utils.instantiate(
        cfg.trainer, callbacks=callbacks, logger=logger, _partial_=True
    )
    trainer = trainer_factory(**kwargs)

    object_dict = {
        "cfg": cfg,
        "datamodule": datamodule,
        "model": model,
        "callbacks": callbacks,
        "logger": logger,
        "trainer": trainer,
    }

    return object_dict


@task_wrapper
def finetune(cfg: DictConfig) -> tuple[dict[str, Any], dict[str, Any]]:
    """Trains the model. Can additionally evaluate on a testset, using best weights
    obtained during training.

    This method is wrapped in optional @task_wrapper decorator, that controls the
    behavior during failure. Useful for multiruns, saving info about the crash, etc.

    Args:
        cfg: A DictConfig configuration composed by Hydra.
    Returns:
        A tuple with metrics and dict with all instantiated objects.
    """
    object_dict = initialize_finetune_objects(cfg)

    logger = object_dict["logger"]
    trainer = object_dict["trainer"]
    model = object_dict["model"]
    datamodule = object_dict["datamodule"]

    if logger:
        log.info("Logging hyperparameters!")
        log_hyperparameters(object_dict)

    if cfg.get("train"):
        log.info("Starting training!")
        trainer.fit(model=model, datamodule=datamodule, ckpt_path=cfg.get("ckpt_path"))

    train_metrics = trainer.callback_metrics

    if cfg.get("test"):
        log.info("Starting testing!")
        ckpt_path = trainer.checkpoint_callback.best_model_path
        if ckpt_path == "":
            log.warning("Best ckpt not found! Using current weights for testing...")
            ckpt_path = None
        trainer.test(model=model, datamodule=datamodule, ckpt_path=ckpt_path)
        log.info(f"Best ckpt path: {ckpt_path}")

    test_metrics = trainer.callback_metrics

    # merge train and test metrics
    metric_dict = {**train_metrics, **test_metrics}

    return metric_dict, object_dict


@hydra.main(version_base="1.3", config_path=str(config_yaml.parent.resolve()), config_name=config_yaml.stem)
def main(cfg: DictConfig) -> float | None:
    """Main entry point for training.

    Args:
        cfg: DictConfig configuration composed by Hydra.
    Returns:
        Optional[float] with optimized metric value.
    """
    # apply extra utilities
    # (e.g. ask for tags if none are provided in cfg, print cfg tree, etc.)
    configure_logging(cfg)

    # train the model
    metric_dict, _ = finetune(cfg)

    # safely retrieve metric value for hydra-based hyperparameter optimization
    metric_value = get_metric_value(metric_dict=metric_dict, metric_name=cfg.get("optimized_metric"))

    # return optimized metric
    return metric_value


if __name__ == "__main__":
    main()
