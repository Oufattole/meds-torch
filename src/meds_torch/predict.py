import os
from importlib.resources import files
from itertools import chain
from pathlib import Path
from typing import Any

import hydra
import loguru
import polars as pl
import torch
from lightning import LightningDataModule, LightningModule, Trainer
from lightning.pytorch.loggers import Logger
from omegaconf import DictConfig

from meds_torch.utils import (
    RankedLogger,
    extras,
    instantiate_loggers,
    log_hyperparameters,
    task_wrapper,
)
from meds_torch.utils.resolvers import setup_resolvers

setup_resolvers()
log = RankedLogger(__name__, rank_zero_only=True)
config_yaml = files("meds_torch").joinpath("configs/eval.yaml")


@task_wrapper
def predict(cfg: DictConfig, datamodule=None) -> tuple[dict[str, Any], dict[str, Any]]:
    """Evaluates given checkpoint on a datamodule testset.

    This method is wrapped in optional @task_wrapper decorator, that controls the
    behavior during failure. Useful for multiruns, saving info about the crash, etc.

    Args:
        cfg: DictConfig configuration composed by Hydra.
    Returns:
        Tuple[dict, dict] with metrics and dict with all instantiated objects.
    """
    assert cfg.ckpt_path
    if not cfg.data.do_include_subject_id:
        raise ValueError("Subject ID is required for generating trajectories")
    if not cfg.data.do_include_prediction_time:
        raise ValueError("Prediction time is required for generating trajectories")

    log.info(f"Instantiating datamodule <{cfg.data._target_}>")
    if not datamodule:
        datamodule: LightningDataModule = hydra.utils.instantiate(cfg.data)

    log.info(f"Instantiating model <{cfg.model._target_}>")
    model: LightningModule = hydra.utils.instantiate(cfg.model)
    model.load_state_dict(torch.load(cfg.ckpt_path)["state_dict"])

    log.info("Instantiating loggers...")
    logger: list[Logger] = instantiate_loggers(cfg.get("logger"))

    log.info(f"Instantiating trainer <{cfg.trainer._target_}>")
    trainer: Trainer = hydra.utils.instantiate(cfg.trainer, logger=logger)

    object_dict = {
        "cfg": cfg,
        "datamodule": datamodule,
        "model": model,
        "logger": logger,
        "trainer": trainer,
    }

    if logger:
        log.info("Logging hyperparameters!")
        log_hyperparameters(object_dict)

    log.info("Starting Generating Predictions!")
    predictions = trainer.predict(model=model, dataloaders=datamodule)

    # Extract input trajectory
    dfs = {}

    subject_ids = pl.Series(list(chain.from_iterable([batch["subject_id"] for batch in predictions])))
    prediction_times = pl.Series(
        list(chain.from_iterable([batch["prediction_time"] for batch in predictions]))
    )

    # Extract real trajectory
    predict_df = pl.DataFrame(
        {
            "subject_id": subject_ids,
            "prediction_time": prediction_times,
            **{name: df.to_struct() for name, df in dfs.items()},
        }
    )

    Path(cfg.paths.predict_fp).parent.mkdir(parents=True, exist_ok=True)
    predict_df.write_parquet(cfg.paths.predict_fp)

    loguru.logger.info(predict_df.head())


@hydra.main(version_base="1.3", config_path=str(config_yaml.parent.resolve()), config_name=config_yaml.stem)
def main(cfg: DictConfig) -> None:
    """Main entry point for evaluation.

    :param cfg: DictConfig configuration composed by Hydra.
    """
    # apply extra utilities
    # (e.g. ask for tags if none are provided in cfg, print cfg tree, etc.)
    os.makedirs(cfg.paths.time_output_dir, exist_ok=True)
    extras(cfg)

    predict(cfg)


if __name__ == "__main__":
    main()
