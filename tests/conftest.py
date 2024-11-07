"""This file prepares config fixtures for other tests."""

import shutil
from pathlib import Path

import polars as pl
import pytest
from hydra import compose, initialize
from loguru import logger
from omegaconf import DictConfig, open_dict

from meds_torch.utils.resolvers import setup_resolvers

from .helpers.package_available import _DO_LOG

setup_resolvers()

SUPERVISED_TASK_NAME = "boolean_value"


@pytest.fixture(scope="package")
def meds_dir(tmp_path_factory) -> Path:
    meds_dir = tmp_path_factory.mktemp("meds_data")
    logger.info(meds_dir)
    # copy test data to temporary directory
    shutil.copytree(Path("./tests/test_data"), meds_dir, dirs_exist_ok=True)
    meds_df = pl.read_parquet(meds_dir / "MEDS_cohort" / "data/*/*.parquet")
    meds_df = meds_df.sort(["subject_id", "time"]).filter(pl.col("time").is_not_null())
    prediction_time_expr = pl.col("time").get(pl.len() // 2).alias("prediction_time")
    labels_df = meds_df.group_by("subject_id", maintain_order=True).agg(prediction_time_expr)
    filter_meds_df = meds_df.join(labels_df, on="subject_id", how="left").filter(
        pl.col("time") <= pl.col("prediction_time")
    )
    label_expr = (
        (pl.col("code").eq("ADMISSION//ONCOLOGY") | pl.col("code").eq("ADMISSION//CARDIAC"))
        .any()
        .alias(SUPERVISED_TASK_NAME)
    )
    labels_df = (
        filter_meds_df.group_by("subject_id").agg(label_expr).join(labels_df, on="subject_id", how="left")
    )
    tasks_dir = meds_dir / "tasks"
    tasks_dir.mkdir(parents=True, exist_ok=True)
    labels_df.write_parquet(tasks_dir / f"{SUPERVISED_TASK_NAME}.parquet")
    return meds_dir


def create_cfg(overrides, meds_dir: Path, config_name="train.yaml", supervised=False) -> DictConfig:
    """Helper function to create Hydra DictConfig with given overrides and common settings."""
    with initialize(version_base="1.3", config_path="../src/meds_torch/configs"):
        cfg = compose(config_name=config_name, return_hydra_config=True, overrides=overrides)

        with open_dict(cfg):
            if "data.collate_type=eic" in overrides:
                cfg.paths.data_dir = str(meds_dir / "eic_tensors")
            else:
                cfg.paths.data_dir = str(meds_dir / "triplet_tensors")
            if supervised:
                cfg.data.task_name = SUPERVISED_TASK_NAME
                cfg.data.task_root_dir = str(meds_dir / "tasks")
            cfg.paths.meds_cohort_dir = str(meds_dir / "MEDS_cohort")
            cfg.trainer.max_epochs = 1
            cfg.trainer.limit_train_batches = 0.1
            cfg.trainer.limit_val_batches = 0.25
            cfg.trainer.limit_test_batches = 0.25
            cfg.trainer.accelerator = "cpu"
            cfg.trainer.devices = 1
            cfg.data.num_workers = 0
            cfg.data.pin_memory = False
            cfg.extras.print_config = False
            cfg.extras.enforce_tags = False
            if not _DO_LOG:
                cfg.logger = None
            cfg.data.dataloader.batch_size = 2

            # Additional settings for specific fixtures
            if "data=multiwindow_pytorch_dataset" in overrides:
                cfg.data.raw_windows_fp = str(meds_dir / "windows" / "raw" / "random_windows.parquet")
                cfg.model.pre_window_name = "pre"
                cfg.model.post_window_name = "post"
                cfg.model.input_window_name = "pre"
                cfg.model.forecast_window_name = "post"

    return cfg
