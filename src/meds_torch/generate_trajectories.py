import os
from importlib.resources import files
from itertools import chain
from pathlib import Path
from typing import Any

import hydra
import loguru
import polars as pl
import pyarrow as pa
import torch
from lightning import LightningDataModule, LightningModule, Trainer
from lightning.pytorch.loggers import Logger
from omegaconf import DictConfig

from meds_torch.models import ACTUAL_FUTURE, GENERATE_PREFIX, INPUT_DATA
from meds_torch.schemas.generate_analysis_schema import (
    reorder_struct_fields,
    validate_generated_data,
)
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


def convert_tensors_to_df(batches):
    keys = ["code", "numeric_value", "time_delta_days", "mask", "numeric_value_mask", "static_mask"]
    return pl.DataFrame(
        {
            key: pl.Series(
                [
                    batch[key][i][batch["mask"][i]].tolist()
                    for batch in batches
                    for i in range(batch[key].shape[0])
                ]
            )
            for key in keys
        }
    )


@task_wrapper
def generate_trajectories(cfg: DictConfig, datamodule=None) -> tuple[dict[str, Any], dict[str, Any]]:
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
    input_df = convert_tensors_to_df(predictions)
    dfs = {INPUT_DATA: input_df}
    if cfg.actual_future_name:
        actual_tensors = [batch[cfg.actual_future_name] for batch in predictions]
        actual_df = convert_tensors_to_df(actual_tensors)
        dfs[ACTUAL_FUTURE] = actual_df

    # Extract generated trajectories
    generated_trajectory_keys = [key for key in predictions[0].keys() if key.startswith(GENERATE_PREFIX)]
    for gen_key in generated_trajectory_keys:
        gen_traj = [pred[gen_key] for pred in predictions]
        gen_df = convert_tensors_to_df(gen_traj)
        dfs[gen_key] = gen_df
    subject_ids = pl.Series(list(chain.from_iterable([batch["subject_id"] for batch in predictions])))
    prediction_times = pl.Series(
        list(chain.from_iterable([batch["prediction_time"] for batch in predictions]))
    )

    # Extract real trajectory
    generate_trajectories_df = pl.DataFrame(
        {
            "subject_id": subject_ids,
            "prediction_time": prediction_times,
            **{name: df.to_struct() for name, df in dfs.items()},
        }
    )

    Path(cfg.paths.generated_trajectory_fp).parent.mkdir(parents=True, exist_ok=True)
    # Convert to arrow table and write to parquet
    for col in generate_trajectories_df.columns:
        if isinstance(generate_trajectories_df[col].dtype, pl.Struct):
            generate_trajectories_df = reorder_struct_fields(generate_trajectories_df, col)
    validated_table = validate_generated_data(generate_trajectories_df)
    pa.parquet.write_table(validated_table, cfg.paths.generated_trajectory_fp)
    loguru.logger.info(pl.from_arrow(validated_table).head())


@hydra.main(version_base="1.3", config_path=str(config_yaml.parent.resolve()), config_name=config_yaml.stem)
def main(cfg: DictConfig) -> None:
    """Main entry point for evaluation.

    :param cfg: DictConfig configuration composed by Hydra.
    """
    # apply extra utilities
    # (e.g. ask for tags if none are provided in cfg, print cfg tree, etc.)
    os.makedirs(cfg.paths.time_output_dir, exist_ok=True)
    extras(cfg)

    generate_trajectories(cfg)


if __name__ == "__main__":
    main()
