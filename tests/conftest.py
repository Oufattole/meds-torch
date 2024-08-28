"""This file prepares config fixtures for other tests."""

import shutil
from pathlib import Path

import numpy as np
import polars as pl
import pytest
import rootutils
from hydra import compose, initialize
from loguru import logger
from omegaconf import DictConfig, open_dict

from meds_torch.utils.resolvers import setup_resolvers

setup_resolvers()

SUPERVISED_TASK_NAME = "supervised_task"


@pytest.fixture(scope="package")
def meds_dir(tmp_path_factory) -> Path:
    meds_dir = tmp_path_factory.mktemp("meds_data")
    logger.info(meds_dir)
    # copy test data to temporary directory
    shutil.copytree(Path("./tests/test_data"), meds_dir, dirs_exist_ok=True)
    label_df = pl.read_parquet(meds_dir / "MEDS_cohort" / "data/*/*.parquet")
    label_df = label_df.sort(["patient_id", "time"]).filter(pl.col("time").is_not_null())
    start_time_expr = pl.col("time").get(0).alias("start_time")
    end_time_expr = pl.col("time").get(pl.len() // 2).alias("end_time")
    label_df = label_df.group_by("patient_id", maintain_order=True).agg(start_time_expr, end_time_expr)
    rng = np.random.default_rng(0)
    label_df = label_df.with_columns(pl.lit(rng.integers(0, 2, label_df.height)).alias(SUPERVISED_TASK_NAME))
    tasks_dir = meds_dir / "tasks"
    tasks_dir.mkdir(parents=True, exist_ok=True)
    label_df.write_parquet(tasks_dir / f"{SUPERVISED_TASK_NAME}.parquet")
    return meds_dir


def create_cfg(overrides, meds_dir: Path, config_name="train.yaml", supervised=False) -> DictConfig:
    """Helper function to create Hydra DictConfig with given overrides and common settings."""
    with initialize(version_base="1.3", config_path="../src/meds_torch/configs"):
        cfg = compose(config_name=config_name, return_hydra_config=True, overrides=overrides)

        with open_dict(cfg):
            cfg.paths.root_dir = str(rootutils.find_root(indicator=".project-root"))
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
            cfg.logger = None
            cfg.data.collate_type = "triplet"
            cfg.data.dataloader.batch_size = 2

            # Additional settings for specific fixtures
            if "data=multiwindow_pytorch_dataset" in overrides:
                cfg.data.cached_windows_dir = str(meds_dir / "cached_windows")
                cfg.data.raw_windows_fp = str(meds_dir / "windows" / "raw_windows.parquet")

    return cfg
