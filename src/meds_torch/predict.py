import os
from dataclasses import dataclass
from importlib.resources import files
from itertools import chain
from pathlib import Path
from typing import Any

import hydra
import loguru
import numpy as np
import polars as pl
import pyarrow.parquet as pq
import torch
from lightning import LightningDataModule, LightningModule, Trainer
from lightning.pytorch.loggers import Logger
from mixins.seedable import seed_everything
from omegaconf import DictConfig

from meds_torch.models import (
    MODEL_EMBEDDINGS_KEY,
    MODEL_LOGITS_KEY,
    MODEL_LOGITS_SEQUENCE_KEY,
    MODEL_LOSS_KEY,
    MODEL_PRED_PROBA_KEY,
    MODEL_PRED_STATUS_KEY,
    MODEL_PREFIX,
    MODEL_TOKENS_KEY,
)
from meds_torch.schemas.predict_schema import validate_prediction_data
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
config_yaml = files("meds_torch").joinpath("configs/predict.yaml")

MODEL_KEY_TO_PREDICT_SCHEMA_NAME = {
    MODEL_EMBEDDINGS_KEY: "embeddings",
    MODEL_LOGITS_KEY: "logits",
    MODEL_LOGITS_SEQUENCE_KEY: "logits_sequence",
    MODEL_TOKENS_KEY: "tokens",
    MODEL_LOSS_KEY: "loss",
    MODEL_PRED_STATUS_KEY: "predictions",
}


# Create dummy config
@dataclass
class DummyTrainer:
    logger: list[Logger] | None = None
    task_name: str = "test_task"

    def predict(self, model, dataloaders):
        return [
            {
                "subject_id": [1, 2],
                "prediction_time": ["2020-01-01", "2020-01-02"],
                "boolean_value": [0, 1],
                MODEL_PRED_PROBA_KEY: [0.2, 0.8],
                self.task_name: [1, 1],
            }
        ]


@dataclass
class DummyModel(LightningModule):
    def load_state_dict(self, state_dict):
        pass


@dataclass
class DummyDataModule(LightningDataModule):
    task_name: str
    do_include_subject_id: bool
    do_include_prediction_time: bool


def process_tensor_batches(predictions: list[dict[str, Any]], key: str) -> list[torch.Tensor | np.ndarray]:
    """
    Process tensor batches of different dimensions into a list suitable for Polars DataFrame.

    Args:
        predictions: List of dictionaries containing tensor batches
        key: Key to access the tensor in each batch

    Returns:
        List where each element represents one row in the final DataFrame

    Examples:
    >>> # 1D tensor example
    >>> batch1 = {'values': torch.tensor([1., 2.])}
    >>> batch2 = {'values': torch.tensor([3., 4.])}
    >>> predictions = [batch1, batch2]
    >>> result = process_tensor_batches(predictions, 'values')
    >>> len(result)
    4
    >>> result[0]
    1.0
    >>> # 2D tensor example
    >>> batch1 = {'matrix': torch.tensor([[1., 2.], [3., 4.]])}
    >>> batch2 = {'matrix': torch.tensor([[5., 6.], [7., 8.]])}
    >>> predictions = [batch1, batch2]
    >>> result = process_tensor_batches(predictions, 'matrix')
    >>> len(result)
    4
    >>> result[0]
    [1.0, 2.0]
    >>> # 3D tensor example
    >>> batch1 = {'tokens': torch.tensor([[[1.], [3.]]])}
    >>> batch2 = {'tokens': torch.tensor([[[9., 10.], [11., 12.]], [[13., 14.], [15., 16.]]])}
    >>> predictions = [batch1, batch2]
    >>> result = process_tensor_batches(predictions, 'tokens')
    >>> len(result)
    3
    >>> result[0]
    [[1.0], [3.0]]
    """
    flattened_data = []

    for batch in predictions:
        tensor = batch[key]

        # Handle different tensor dimensions
        if len(tensor.shape) == 1:
            # 1D tensor: split into individual elements
            flattened_data.extend(tensor.tolist())
        elif len(tensor.shape) == 2:
            # 2D tensor: each row becomes an element
            flattened_data.extend(tensor.tolist())
        elif len(tensor.shape) == 3:
            # 3D tensor: first dimension is batch, store remaining 2D array
            for item in tensor:
                flattened_data.append(item.tolist())
        else:
            raise ValueError(f"Unsupported tensor dimension: {len(tensor.shape)}")

    return flattened_data


def process_predictions(predictions: list[dict[str, Any]], model_keys: dict[str, str]) -> pl.DataFrame:
    """
    Process predictions and create a Polars DataFrame handling tensors of different dimensions.

    Args:
        predictions: List of prediction batches
        model_keys: Dictionary mapping model keys to schema names

    Returns:
        Polars DataFrame with processed data

    Examples:
    >>> # Mixed dimension example with MODEL// prefixed keys
    >>> batch1 = {
    ...     '1d': torch.tensor([1., 2.]),
    ...     '2d': torch.tensor([[3., 4.], [5., 6.]]),
    ...     '3d': torch.tensor([[[7., 8.], [9., 10.]], [[11., 12.], [13., 14.]]]),
    ...     'MODEL//extra': torch.tensor([[100., 200.], [300., 400.]]),
    ...     'MODEL//another': torch.tensor([1000., 2000.]),
    ...     'not_model': torch.tensor([9999., 8888.])  # Should be ignored
    ... }
    >>> batch2 = {
    ...     '1d': torch.tensor([15., 16.]),
    ...     '2d': torch.tensor([[17., 18.], [19., 20.]]),
    ...     '3d': torch.tensor([[[21., 22.], [23., 24.]], [[25., 26.], [27., 28.]]]),
    ...     'MODEL//extra': torch.tensor([[500., 600.], [700., 800.]]),
    ...     'MODEL//another': torch.tensor([3000., 4000.]),
    ...     'not_model': torch.tensor([7777., 6666.])  # Should be ignored
    ... }
    >>> predictions = [batch1, batch2]
    >>> keys = {'1d': 'scalar', '2d': 'vector', '3d': 'matrix'}
    >>> df = process_predictions(predictions, keys)
    >>> df.shape[0]  # Number of rows
    4
    >>> sorted_df = df.sort("scalar")
    >>> sorted_df.columns
    ['scalar', 'vector', 'matrix', 'MODEL//extra', 'MODEL//another']
    >>> len(sorted_df.columns)  # Should include original columns plus MODEL_ columns
    5
    >>> sorted_df
    shape: (4, 5)
    ┌────────┬──────────────┬──────────────────────────────┬────────────────┬────────────────┐
    │ scalar ┆ vector       ┆ matrix                       ┆ MODEL//extra   ┆ MODEL//another │
    │ ---    ┆ ---          ┆ ---                          ┆ ---            ┆ ---            │
    │ f64    ┆ list[f64]    ┆ list[list[f64]]              ┆ list[f64]      ┆ f64            │
    ╞════════╪══════════════╪══════════════════════════════╪════════════════╪════════════════╡
    │ 1.0    ┆ [3.0, 4.0]   ┆ [[7.0, 8.0], [9.0, 10.0]]    ┆ [100.0, 200.0] ┆ 1000.0         │
    │ 2.0    ┆ [5.0, 6.0]   ┆ [[11.0, 12.0], [13.0, 14.0]] ┆ [300.0, 400.0] ┆ 2000.0         │
    │ 15.0   ┆ [17.0, 18.0] ┆ [[21.0, 22.0], [23.0, 24.0]] ┆ [500.0, 600.0] ┆ 3000.0         │
    │ 16.0   ┆ [19.0, 20.0] ┆ [[25.0, 26.0], [27.0, 28.0]] ┆ [700.0, 800.0] ┆ 4000.0         │
    └────────┴──────────────┴──────────────────────────────┴────────────────┴────────────────┘
    >>> # Test error handling
    >>> del predictions[1]['1d']
    >>> import pytest
    >>> with pytest.raises(RuntimeError):
    ...     pytest.raises(process_predictions(predictions, keys))
    """
    predict_df = pl.DataFrame()

    # Process explicitly defined model keys
    for key in model_keys:
        if key not in predictions[0]:
            continue

        if not isinstance(predictions[0][key], torch.Tensor):
            continue

        if len(predictions[0][key].shape) == 0:  # skip scalars
            continue

        key_name = model_keys[key]
        try:
            processed_data = process_tensor_batches(predictions, key)
            predict_df = predict_df.with_columns(pl.Series(processed_data).alias(key_name))
        except Exception as e:
            raise RuntimeError(f"Error processing key {key}: {str(e)}")

    # Process any additional MODEL_ prefixed keys not in model_keys
    for key in predictions[0].keys():
        if key.startswith(MODEL_PREFIX) and key not in model_keys:
            if not isinstance(predictions[0][key], torch.Tensor):
                continue

            if len(predictions[0][key].shape) == 0:  # skip scalars
                continue

            try:
                processed_data = process_tensor_batches(predictions, key)
                # Use the key itself as the column name for additional MODEL_ keys
                predict_df = predict_df.with_columns(pl.Series(processed_data).alias(key))
            except Exception as e:
                raise RuntimeError(f"Error processing {MODEL_PREFIX} prefixed key {key}: {str(e)}")

    return predict_df


def store_predictions(predict_fp, task_name, predictions):
    # Extract input trajectory
    dfs = {}

    subject_ids = pl.Series(list(chain.from_iterable([batch["subject_id"] for batch in predictions]))).cast(
        pl.Int64
    )
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
    if task_name:
        predict_df = predict_df.with_columns(
            pl.Series(list(chain.from_iterable([batch["boolean_value"] for batch in predictions])))
            .alias("boolean_value")
            .cast(pl.Boolean)
        )

    if MODEL_PRED_PROBA_KEY in predictions[0]:
        predict_df = predict_df.with_columns(
            pl.Series(
                list(chain.from_iterable([batch[MODEL_PRED_PROBA_KEY] for batch in predictions]))
            ).alias("predicted_boolean_probability")
        )
        predict_df = predict_df.with_columns(
            pl.col("predicted_boolean_probability").gt(0.5).alias("predicted_boolean_value"),
        )

    predict_df = predict_df.hstack(process_predictions(predictions, MODEL_KEY_TO_PREDICT_SCHEMA_NAME))

    Path(predict_fp).parent.mkdir(parents=True, exist_ok=True)
    try:
        validated_table = validate_prediction_data(predict_df)
        pq.write_table(validated_table, predict_fp)
    except:  # noqa: E722
        loguru.logger.warning("Could not validate, writing prediction table via polars instead of arrow")
        predict_df.write_parquet(predict_fp)


@task_wrapper
def predict(cfg: DictConfig, datamodule=None) -> tuple[dict[str, Any], dict[str, Any]]:
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
    >>> with tempfile.TemporaryDirectory() as tmp_dir:
    ...     ckpt_path = Path(tmp_dir) / "model.ckpt"
    ...     torch.save({"state_dict": {}}, ckpt_path)
    ...
    ...     # Create config
    ...     cfg = {
    ...         "seed": 0,
    ...         "ckpt_path": str(ckpt_path),
    ...         "model": {"_target_": "meds_torch.predict.DummyModel"},
    ...         "data": {
    ...             "_target_": "meds_torch.predict.DummyDataModule",
    ...             "task_name": "test_task",
    ...             "do_include_subject_id": True,
    ...             "do_include_prediction_time": True
    ...         },
    ...         "paths": {
    ...             "predict_fp": str(Path(tmp_dir) / "predictions.parquet"),
    ...             "time_output_dir": tmp_dir
    ...         },
    ...         "trainer": {"_target_": "meds_torch.predict.DummyTrainer"},
    ...         "logger": None
    ...     }
    ...     cfg = DictConfig(cfg)
    ...
    ...     # Run prediction
    ...     predict(cfg)
    ...
    ...     # Verify outputs
    ...     assert Path(cfg.paths.predict_fp).exists()
    ...     print(pl.read_parquet(cfg.paths.predict_fp))  # doctest: +NORMALIZE_WHITESPACE, +ELLIPSIS
    shape: (2, 5)
    ┌────────────┬─────────────────────┬───────────────┬─────────────────────────┬───────────────────────────┐
    │ subject_id ┆ prediction_time     ┆ boolean_value ┆ predicted_boolean_value ┆ predicted_boolean_probabi │
    │ ---        ┆ ---                 ┆ ---           ┆ ---                     ┆ lity                      │
    │ i64        ┆ datetime[ns]        ┆ bool          ┆ bool                    ┆ ---                       │
    │            ┆                     ┆               ┆                         ┆ f64                       │
    ╞════════════╪═════════════════════╪═══════════════╪═════════════════════════╪═══════════════════════════╡
    │ 1          ┆ 2020-01-01 00:00:00 ┆ ...           ┆ ...                     ┆ ...                       │
    │ 2          ┆ 2020-01-02 00:00:00 ┆ ...           ┆ ...                     ┆ ...                       │
    └────────────┴─────────────────────┴───────────────┴─────────────────────────┴───────────────────────────┘
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

    store_predictions(cfg.paths.predict_fp, cfg.data.task_name, predictions)


@hydra.main(version_base="1.3", config_path=str(config_yaml.parent.resolve()), config_name=config_yaml.stem)
def main(cfg: DictConfig) -> None:
    """Main entry point for evaluation.

    Args:
        cfg (DictConfig): configuration composed by Hydra.
    """
    # apply extra utilities
    # (e.g. ask for tags if none are provided in cfg, print cfg tree, etc.)
    os.makedirs(cfg.paths.time_output_dir, exist_ok=True)
    configure_logging(cfg)

    predict(cfg)


if __name__ == "__main__":
    main()
