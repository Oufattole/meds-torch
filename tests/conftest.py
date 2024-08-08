"""This file prepares config fixtures for other tests."""

import shutil
from pathlib import Path

import numpy as np
import polars as pl
import pytest
import rootutils
from hydra import compose, initialize
from hydra.core.global_hydra import GlobalHydra
from loguru import logger
from omegaconf import DictConfig, open_dict

from meds_torch.utils.resolvers import setup_resolvers

setup_resolvers()

SUPERVISED_TASK_NAME = "supervised_task"


@pytest.fixture(scope="package")
def cfg_eval_global() -> DictConfig:
    """A pytest fixture for setting up a default Hydra DictConfig for evaluation.

    :return: A DictConfig containing a default Hydra configuration for evaluation.
    """
    with initialize(version_base="1.3", config_path="../configs"):
        cfg = compose(config_name="eval.yaml", return_hydra_config=True, overrides=["ckpt_path=."])

        # set defaults for all tests
        with open_dict(cfg):
            cfg.paths.root_dir = str(rootutils.find_root(indicator=".project-root"))
            cfg.trainer.max_epochs = 1
            cfg.trainer.limit_test_batches = 0.1
            cfg.trainer.accelerator = "cpu"
            cfg.trainer.devices = 1
            cfg.data.num_workers = 0
            cfg.data.pin_memory = False
            cfg.extras.print_config = False
            cfg.extras.enforce_tags = False
            cfg.logger = None

    return cfg


@pytest.fixture(scope="package")
def meds_dir(tmp_path_factory) -> Path:
    meds_dir = tmp_path_factory.mktemp("meds_data")
    logger.info(meds_dir)
    # copy test data to temporary directory
    shutil.copytree(Path("./tests/test_data"), meds_dir, dirs_exist_ok=True)
    label_df = pl.read_parquet(meds_dir / "final_cohort" / "*/*.parquet")
    label_df = label_df.sort(["patient_id", "timestamp"]).filter(pl.col("timestamp").is_not_null())
    start_time_expr = pl.col("timestamp").get(0).alias("start_time")
    end_time_expr = pl.col("timestamp").get(pl.len() // 2).alias("end_time")
    label_df = label_df.group_by("patient_id", maintain_order=True).agg(start_time_expr, end_time_expr)
    rng = np.random.default_rng(0)
    label_df = label_df.with_columns(pl.lit(rng.integers(0, 2, label_df.height)).alias(SUPERVISED_TASK_NAME))
    tasks_dir = meds_dir / "tasks"
    tasks_dir.mkdir(parents=True, exist_ok=True)
    label_df.write_parquet(tasks_dir / f"{SUPERVISED_TASK_NAME}.parquet")
    return meds_dir


def create_cfg(overrides, meds_dir: Path) -> DictConfig:
    """Helper function to create Hydra DictConfig with given overrides and common settings."""
    with initialize(version_base="1.3", config_path="../configs"):
        cfg = compose(config_name="train.yaml", return_hydra_config=True, overrides=overrides)

        with open_dict(cfg):
            cfg.paths.root_dir = str(rootutils.find_root(indicator=".project-root"))
            cfg.paths.data_dir = str(meds_dir)
            cfg.paths.meds_dir = str(meds_dir / "final_cohort")
            cfg.trainer.max_epochs = 1
            cfg.trainer.limit_train_batches = 0.05
            cfg.trainer.limit_val_batches = 0.1
            cfg.trainer.limit_test_batches = 0.1
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
                cfg.data.raw_windows_fp = str(meds_dir / "raw_windows.parquet")

    return cfg


@pytest.fixture(scope="package")
def cfg_multiwindow_train_global(meds_dir: Path) -> DictConfig:
    """A pytest fixture for setting up a default Hydra DictConfig for training. All tests share the test data
    directory.

    :return: A DictConfig object containing a default Hydra configuration for training.
    """
    return create_cfg(overrides=["data=multiwindow_pytorch_dataset"], meds_dir=meds_dir)


@pytest.fixture(scope="package")
def cfg_train_global(meds_dir: Path) -> DictConfig:
    """A pytest fixture for setting up a default Hydra DictConfig for training. All tests share the test data
    directory.

    :return: A DictConfig object containing a default Hydra configuration for training.
    """
    return create_cfg(overrides=[], meds_dir=meds_dir)


@pytest.fixture(scope="function")
def cfg_train(cfg_train_global: DictConfig, tmp_path: Path) -> DictConfig:
    """A pytest fixture built on top of the `cfg_train_global()` fixture, which accepts a temporary logging
    path `tmp_path` for generating a temporary logging path.

    This is called by each test which uses the `cfg_train` arg. Each test generates its own temporary logging
    path.

    :param cfg_train_global: The input DictConfig object to be modified.
    :param tmp_path: The temporary logging path.
    :return: A DictConfig with updated output and log directories corresponding to `tmp_path`.
    """
    cfg = cfg_train_global.copy()

    with open_dict(cfg):
        cfg.paths.output_dir = str(tmp_path)
        cfg.paths.log_dir = str(tmp_path)

    yield cfg

    GlobalHydra.instance().clear()


@pytest.fixture(scope="function")
def cfg_multiwindow_train(cfg_multiwindow_train_global: DictConfig, tmp_path: Path) -> DictConfig:
    """A pytest fixture built on top of the `cfg_train_global()` fixture, which accepts a temporary logging
    path `tmp_path` for generating a temporary logging path.

    This is called by each test which uses the `cfg_train` arg. Each test generates its own temporary logging
    path.

    :param cfg_train_global: The input DictConfig object to be modified.
    :param tmp_path: The temporary logging path.
    :return: A DictConfig with updated output and log directories corresponding to `tmp_path`.
    """
    cfg = cfg_multiwindow_train_global.copy()

    with open_dict(cfg):
        cfg.paths.output_dir = str(tmp_path)
        cfg.paths.log_dir = str(tmp_path)

    yield cfg

    GlobalHydra.instance().clear()


@pytest.fixture(scope="function")
def cfg_eval(cfg_eval_global: DictConfig, tmp_path: Path) -> DictConfig:
    """A pytest fixture built on top of the `cfg_eval_global()` fixture, which accepts a temporary logging
    path `tmp_path` for generating a temporary logging path.

    This is called by each test which uses the `cfg_eval` arg. Each test generates its own temporary logging
    path.

    :param cfg_train_global: The input DictConfig object to be modified.
    :param tmp_path: The temporary logging path.
    :return: A DictConfig with updated output and log directories corresponding to `tmp_path`.
    """
    cfg = cfg_eval_global.copy()

    with open_dict(cfg):
        cfg.paths.output_dir = str(tmp_path)
        cfg.paths.log_dir = str(tmp_path)

    yield cfg

    GlobalHydra.instance().clear()
