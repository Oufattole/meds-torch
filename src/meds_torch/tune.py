import copy
import glob
import json
import os
import shutil
from collections.abc import Callable
from importlib.resources import files
from pathlib import Path

import hydra
import polars as pl
import ray
from loguru import logger
from omegaconf import DictConfig, OmegaConf, open_dict
from ray import tune
from ray.train import CheckpointConfig, RunConfig, ScalingConfig
from ray.train.lightning import RayLightningEnvironment, prepare_trainer
from ray.train.torch import TorchTrainer

from meds_torch.eval import evaluate
from meds_torch.finetune import initialize_finetune_objects
from meds_torch.train import initialize_train_objects
from meds_torch.utils import RankedLogger, configure_logging, task_wrapper
from meds_torch.utils.resolvers import setup_resolvers

log = RankedLogger(__name__, rank_zero_only=True)

setup_resolvers()
config_yaml = files("meds_torch").joinpath("configs/train.yaml")


@task_wrapper
def ray_tune_runner(cfg: DictConfig, train_func: Callable):
    def objective(config):
        setup_resolvers()
        # Create a new config for this trial
        trial_cfg = copy.deepcopy(cfg)
        # Update the config with the current trial's hyperparameters
        for key, value in config.items():
            OmegaConf.update(trial_cfg, key, value, merge=True)

        # Run the training function
        _ = train_func(trial_cfg)

    # Set up the Ray Tune search space
    search_space = {}
    for key, value in cfg.hparams_search.search_space.items():
        search_space[key] = hydra.utils.instantiate(value)

    scaling_config = ScalingConfig(
        use_gpu=True, resources_per_worker=OmegaConf.to_container(cfg.hparams_search.ray.resources_per_trial)
    )

    run_config = RunConfig(
        checkpoint_config=CheckpointConfig(
            num_to_keep=1,
            checkpoint_score_attribute=cfg.hparams_search.optimized_metric,
            checkpoint_score_order=cfg.hparams_search.direction,
        ),
        name="ray_tune",
        storage_path=cfg.paths.time_output_dir,
    )

    # Define a TorchTrainer without hyper-parameters for Tuner
    ray_trainer = TorchTrainer(
        objective,
        scaling_config=scaling_config,
        run_config=run_config,
    )

    tuner = tune.Tuner(
        ray_trainer,
        param_space={"train_loop_config": search_space},
        tune_config=tune.TuneConfig(
            metric=cfg.hparams_search.optimized_metric,
            mode=cfg.hparams_search.direction,
            num_samples=cfg.hparams_search.ray.num_samples,
            scheduler=hydra.utils.instantiate(cfg.hparams_search.scheduler),
        ),
    )
    analysis = tuner.fit()

    # Return the best trial results
    best_trial = analysis.get_best_result(
        cfg.hparams_search.optimized_metric, cfg.hparams_search.direction, scope="all"
    )
    return analysis, best_trial


def get_checkpoint_path(log_dir, checkpoint_dir_name, time_output_path):
    # Construct the glob pattern
    glob_pattern = os.path.join(
        time_output_path, "ray_tune", f"*{log_dir}*", checkpoint_dir_name, "checkpoint.ckpt"
    )

    # Find the checkpoint file
    checkpoint_files = glob.glob(glob_pattern)

    if len(checkpoint_files) != 1:
        raise ValueError(
            "Expected to find exactly one checkpoint file matching the pattern "
            f"{glob_pattern}, but found {len(checkpoint_files)} files."
        )
    else:
        return checkpoint_files[0]


def train_func(cfg):
    if cfg.hparams_search.train_fn == "train":
        initialize_objects = initialize_train_objects
    elif cfg.hparams_search.train_fn == "finetune":
        initialize_objects = initialize_finetune_objects
    else:
        raise ValueError(f"Invalid train_fn: {cfg.hparams_search.train_fn}, should be 'train' or 'finetune'")
    plugins = [RayLightningEnvironment()]

    object_dict = initialize_objects(cfg, plugins=plugins)
    dm = object_dict["datamodule"]
    model = object_dict["model"]
    trainer = object_dict["trainer"]

    trainer = prepare_trainer(trainer)
    trainer.fit(model, datamodule=dm)


@hydra.main(version_base="1.3", config_path=str(config_yaml.parent.resolve()), config_name=config_yaml.stem)
def main(cfg: DictConfig) -> float | None:
    """Main entry point for training.

    Args:
        cfg: DictConfig configuration composed by Hydra.
    Returns:
        Optional[float] with optimized metric value.
    """
    os.environ["RAY_memory_monitor_refresh_ms"] = cfg.ray_memory_monitor_refresh_ms
    # apply extra utilities
    configure_logging(cfg)

    if cfg.best_config_path:
        if not Path(cfg.best_config_path).exists():
            raise FileNotFoundError(f"Best config file not found at {cfg.best_config_path}")
        logger.info(f"Loading best tuning config from {cfg.best_config_path}")
        with open(Path(cfg.best_config_path)) as config_json:
            best_config = json.load(config_json)["train_loop_config"]
        for key, value in best_config.items():
            OmegaConf.update(cfg, key, value, merge=False)
    else:
        logger.info("No best config provided")

    analysis, best_trial = ray_tune_runner(cfg, train_func=train_func)
    ray.shutdown()

    # return tune results
    results_df = pl.from_dataframe(
        analysis.get_dataframe(
            filter_metric=cfg.hparams_search.optimized_metric, filter_mode=cfg.hparams_search.direction
        )
    )
    results_df.write_parquet(cfg.paths.time_output_dir / "sweep_results.parquet")

    best_model_path = (
        Path(
            best_trial.get_best_checkpoint(
                metric=cfg.hparams_search.optimized_metric, mode=cfg.hparams_search.direction
            ).path
        )
        / "checkpoint.ckpt"
    )

    checkpoint_dir = cfg.paths.time_output_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy(best_model_path, checkpoint_dir / "best_model.ckpt")

    with open(cfg.paths.time_output_dir / "best_config.json", "w") as outfile:
        json.dump(best_trial.config, outfile)

    # Generate summary of results
    summary_results_df_cols = [
        each
        for each in results_df.columns
        if each.startswith("config/")
        or each.startswith("train")
        or each.startswith("val")
        or each.startswith("test")
    ] + ["logdir", "checkpoint_dir_name"]
    summary_df = results_df[summary_results_df_cols]

    # Create a new column 'best_checkpoint_path' using the get_checkpoint_path function
    log_dir_index = summary_df.columns.index("logdir")
    ckpt_dir_name_index = summary_df.columns.index("checkpoint_dir_name")
    checkpoint_paths = summary_df.map_rows(
        lambda x: get_checkpoint_path(x[log_dir_index], x[ckpt_dir_name_index], cfg.paths.time_output_dir)
    )
    summary_df = summary_df.with_columns(best_checkpoint_path=checkpoint_paths.to_series())

    if cfg.get("test"):
        logger.info("Computing Test Results")
        test_results = []
        with open_dict(cfg):
            del cfg.trainer.strategy
            del cfg.callbacks
            cfg.trainer.devices = cfg.test_devices
        datamodule = hydra.utils.instantiate(cfg.data)
        for ckpt_path in summary_df["best_checkpoint_path"].to_list():
            with open_dict(cfg):
                cfg.ckpt_path = ckpt_path
            result, _ = evaluate(cfg, datamodule=datamodule)
            test_results.append(result)
        results = {key: [result[key] for result in test_results] for key in test_results[0].keys()}
        for key, values in results.items():
            summary_df = summary_df.with_columns(pl.Series(values).alias(key))

    logger.info(summary_df)
    summary_df.write_parquet(cfg.paths.time_output_dir / "sweep_results_summary.parquet")

    return best_trial


if __name__ == "__main__":
    main()
