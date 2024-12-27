import os
from dataclasses import dataclass
from datetime import datetime
from importlib.resources import files
from pathlib import Path
from typing import Any

import hydra
import loguru
import polars as pl
import pyarrow.parquet as pq
import torch
from lightning import LightningDataModule, LightningModule, Trainer
from lightning.pytorch.loggers import Logger
from mixins.seedable import seed_everything
from omegaconf import DictConfig

from meds_torch.models import GENERATE_PREFIX
from meds_torch.predict import store_predictions
from meds_torch.schemas.generate_analysis_schema import validate_generated_data
from meds_torch.utils import (
    RankedLogger,
    configure_logging,
    instantiate_loggers,
    log_hyperparameters,
    task_wrapper,
)
from meds_torch.utils.resolvers import setup_resolvers

setup_resolvers()
log = RankedLogger(__name__, rank_zero_only=True)
config_yaml = files("meds_torch").joinpath("configs/generate_trajectories.yaml")


# Create dummy classes for testing
@dataclass
class DummyTrainer:
    logger: list[Logger] | None = None
    task_name: str = "test_task"

    def predict(self, model, dataloaders):
        base_df = pl.DataFrame(
            {
                "time": [
                    datetime(2024, 1, 1),
                    datetime(2024, 7, 1, 14, 54, 36),
                    datetime(2025, 12, 31, 11, 38, 24),
                    datetime(2024, 1, 1),
                    datetime(2026, 12, 31, 17, 27, 36),
                ],
                "code": ["A1", "A2", "A3", "A4", "A5"],
                "code/vocab_index": [1, 2, 3, 4, 5],
                "numeric_value": [0.5, 1.0, float("nan"), 2.0, float("nan")],
                "subject_id": [1, 1, 1, 2, 2],
                "prediction_time": [
                    datetime(2024, 1, 1),
                    datetime(2024, 1, 1),
                    datetime(2024, 1, 1),
                    datetime(2024, 1, 1),
                    datetime(2024, 1, 1),
                ],
            }
        ).with_columns(
            [
                pl.col("time").cast(pl.Datetime(time_unit="ns")),
                pl.col("code").cast(pl.String),
                pl.col("code/vocab_index").cast(pl.Int32),
                pl.col("numeric_value").cast(pl.Float32),
                pl.col("subject_id").cast(pl.Int32),
                pl.col("prediction_time").cast(pl.Datetime(time_unit="ns")),
            ]
        )

        # Return the same structure for input and generated trajectories
        return [
            {
                f"{GENERATE_PREFIX}1": base_df.with_columns(pl.col("numeric_value") * 2),
                "subject_id": [1, 2],
                "prediction_time": ["2020-01-01", "2020-01-02"],
                "boolean_value": [0, 1],
            }
        ]


@dataclass
class DummyModel(LightningModule):
    metadata_df: Any = None

    def load_state_dict(self, state_dict):
        pass

    def to_meds(self, predictions, metadata_df):
        # Return the input data from the first prediction
        return predictions[0]["input_data"]


@dataclass
class DummyDataModule(LightningDataModule):
    task_name: str
    do_include_subject_id: bool
    do_include_prediction_time: bool


@task_wrapper
def generate_trajectories(cfg: DictConfig, datamodule=None) -> tuple[dict[str, Any], dict[str, Any]]:
    """Evaluates given checkpoint on a datamodule testset.

    This method is wrapped in optional @task_wrapper decorator, that controls the
    behavior during failure. Useful for multiruns, saving info about the crash, etc.

    Args:
        cfg: DictConfig configuration composed by Hydra.
    Returns:
        Tuple[dict, dict] with metrics and dict with all instantiated objects.

    Examples:
    >>> import tempfile
    >>> from omegaconf import DictConfig
    >>> _ = pl.Config.set_tbl_width_chars(106)
    >>> # Create temporary checkpoint file
    >>> tmp_dir = tempfile.mkdtemp()
    >>> ckpt_path = Path(tmp_dir) / "model.ckpt"
    >>> torch.save({"state_dict": {}}, ckpt_path)
    >>> cfg = {
    ...     "seed": 0,
    ...     "ckpt_path": str(ckpt_path),
    ...     "model": {"_target_": "meds_torch.generate_trajectories.DummyModel"},
    ...     "data": {
    ...         "_target_": "meds_torch.generate_trajectories.DummyDataModule",
    ...         "task_name": "test_task",
    ...         "do_include_subject_id": True,
    ...         "do_include_prediction_time": True
    ...     },
    ...     "paths": {
    ...         "generated_trajectory_fp": str(Path(tmp_dir) / "trajectories.parquet"),
    ...         "predict_fp": str(Path(tmp_dir) / "prediction.parquet"),
    ...         "time_output_dir": tmp_dir
    ...     },
    ...     "trainer": {"_target_": "meds_torch.generate_trajectories.DummyTrainer"},
    ...     "logger": None
    ... }
    >>> cfg = DictConfig(cfg)
    >>> generate_trajectories(cfg)
    >>> assert Path(cfg.paths.generated_trajectory_fp).exists()
    >>> df = pl.read_parquet(cfg.paths.generated_trajectory_fp)
    >>> print(df.sort(["TRAJECTORY_TYPE", "subject_id"]).drop("TRAJECTORY_TYPE"))
    shape: (5, 6)
    ┌────────────┬─────────────────────┬─────────────────────┬──────┬──────────────────┬───────────────┐
    │ subject_id ┆ prediction_time     ┆ time                ┆ code ┆ code/vocab_index ┆ numeric_value │
    │ ---        ┆ ---                 ┆ ---                 ┆ ---  ┆ ---              ┆ ---           │
    │ i64        ┆ datetime[ns]        ┆ datetime[ns]        ┆ str  ┆ i64              ┆ f64           │
    ╞════════════╪═════════════════════╪═════════════════════╪══════╪══════════════════╪═══════════════╡
    │ 1          ┆ 2024-01-01 00:00:00 ┆ 2024-01-01 00:00:00 ┆ A1   ┆ 1                ┆ 1.0           │
    │ 1          ┆ 2024-01-01 00:00:00 ┆ 2024-07-01 14:54:36 ┆ A2   ┆ 2                ┆ 2.0           │
    │ 1          ┆ 2024-01-01 00:00:00 ┆ 2025-12-31 11:38:24 ┆ A3   ┆ 3                ┆ NaN           │
    │ 2          ┆ 2024-01-01 00:00:00 ┆ 2024-01-01 00:00:00 ┆ A4   ┆ 4                ┆ 4.0           │
    │ 2          ┆ 2024-01-01 00:00:00 ┆ 2026-12-31 17:27:36 ┆ A5   ┆ 5                ┆ NaN           │
    └────────────┴─────────────────────┴─────────────────────┴──────┴──────────────────┴───────────────┘
    """
    seed_everything(cfg.seed)
    loguru.logger.info(f"Set all seeds to {cfg.seed}")
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
    checkpoint = torch.load(cfg.ckpt_path, map_location="cpu")
    model.load_state_dict(checkpoint["state_dict"])

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

    # Extract generated trajectories
    generated_trajectory_keys = [key for key in predictions[0].keys() if key.startswith(GENERATE_PREFIX)]
    if not len(generated_trajectory_keys) == 1:
        raise ValueError(f"Exactly one trajectory key is expected, found: {generated_trajectory_keys}")

    gen_key = generated_trajectory_keys[0]
    generated_trajectories_df = pl.concat([pred[gen_key] for pred in predictions])
    generated_trajectories_df = generated_trajectories_df.with_columns(
        pl.lit(gen_key).alias("TRAJECTORY_TYPE")
    )

    Path(cfg.paths.generated_trajectory_fp).parent.mkdir(parents=True, exist_ok=True)
    # Convert to arrow table and write to parquet
    validated_table = validate_generated_data(generated_trajectories_df)
    pq.write_table(validated_table, cfg.paths.generated_trajectory_fp)
    loguru.logger.info(pl.from_arrow(validated_table).head())

    store_predictions(cfg.paths.predict_fp, cfg.data.task_name, predictions)


@hydra.main(version_base="1.3", config_path=str(config_yaml.parent.resolve()), config_name=config_yaml.stem)
def main(cfg: DictConfig) -> None:
    """Main entry point for evaluation.

    Args:
        cfg (DictConfig):  configuration composed by Hydra.
    """
    # apply extra utilities
    # (e.g. ask for tags if none are provided in cfg, print cfg tree, etc.)
    os.makedirs(cfg.paths.time_output_dir, exist_ok=True)
    configure_logging(cfg)

    generate_trajectories(cfg)


if __name__ == "__main__":
    main()
