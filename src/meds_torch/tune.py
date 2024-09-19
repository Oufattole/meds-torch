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

from meds_torch.eval import evaluate
from meds_torch.finetune import finetune
from meds_torch.train import train
from meds_torch.utils import RankedLogger, extras
from meds_torch.utils.resolvers import setup_resolvers

log = RankedLogger(__name__, rank_zero_only=True)

setup_resolvers()
config_yaml = files("meds_torch").joinpath("configs/train.yaml")


def ray_tune_runner(cfg: DictConfig, train_fn: Callable):
    def objective(config):
        setup_resolvers()
        # Create a new config for this trial
        trial_cfg = copy.deepcopy(cfg)
        # Update the config with the current trial's hyperparameters
        for key, value in config.items():
            OmegaConf.update(trial_cfg, key, value, merge=True)

        # Run the training function
        _ = train_fn(trial_cfg)

    # Set up the Ray Tune search space
    search_space = {}
    for key, value in cfg.hparams_search.search_space.items():
        search_space[key] = hydra.utils.instantiate(value)

    # Run the Ray Tune optimization
    analysis = tune.run(
        objective,
        config=search_space,
        num_samples=cfg.hparams_search.ray.num_samples,
        scheduler=hydra.utils.instantiate(cfg.hparams_search.scheduler),
        resources_per_trial=OmegaConf.to_container(cfg.hparams_search.ray.resources_per_trial),
        name="ray_tune",
        mode=cfg.hparams_search.direction,
        metric=cfg.hparams_search.optimized_metric,
        storage_path=cfg.paths.time_output_dir,
    )

    # Return the best trial results
    best_trial = analysis.get_best_trial(
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


@hydra.main(version_base="1.3", config_path=str(config_yaml.parent.resolve()), config_name=config_yaml.stem)
def main(cfg: DictConfig) -> float | None:
    """Main entry point for training.

    :param cfg: DictConfig configuration composed by Hydra.
    :return: Optional[float] with optimized metric value.
    """
    # apply extra utilities
    os.makedirs(cfg.paths.time_output_dir, exist_ok=True)
    extras(cfg)

    # Choose the training function based on the configuration
    if cfg.hparams_search.train_fn == "train":
        train_fn = train
    elif cfg.hparams_search.train_fn == "finetune":
        train_fn = finetune
    else:
        raise ValueError(f"Invalid train_fn: {cfg.hparams_search.train_fn}, should be 'train' or 'finetune'")

    if cfg.best_config_path:
        if not Path(cfg.best_config_path).exists():
            raise FileNotFoundError(f"Best config file not found at {cfg.best_config_path}")
        logger.info(f"Loading best tuning config from {cfg.best_config_path}")
        with open(Path(cfg.best_config_path)) as config_json:
            best_config = json.load(config_json)
        for key, value in best_config.items():
            OmegaConf.update(cfg, key, value, merge=False)
    else:
        logger.info("No best config provided")

    # Run Ray Tune optimization
    with open_dict(cfg):
        # Manually resolve the path to avoid issues with Ray Tune
        # see https://github.com/Oufattole/meds-torch/issues/41
        cfg.paths.time_output_dir = str(Path(cfg.paths.time_output_dir))
    analysis, best_trial = ray_tune_runner(cfg, train_fn=train_fn)
    ray.shutdown()

    # return tune results
    results_df = pl.from_dataframe(
        analysis.dataframe(metric=cfg.hparams_search.optimized_metric, mode=cfg.hparams_search.direction)
    )
    results_df.write_parquet(Path(cfg.paths.time_output_dir) / "sweep_results.parquet")

    result_value = best_trial.last_result[cfg.hparams_search.optimized_metric]
    analysis.get_best_checkpoint(best_trial)
    best_model_path = Path(analysis.get_best_checkpoint(best_trial).path) / "checkpoint.ckpt"

    checkpoint_dir = Path(cfg.paths.time_output_dir) / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy(best_model_path, checkpoint_dir / "best_model.ckpt")

    with open(Path(cfg.paths.time_output_dir) / "best_config.json", "w") as outfile:
        json.dump(analysis.get_best_config(), outfile)

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
        datamodule = hydra.utils.instantiate(cfg.data)
        test_results = []
        for ckpt_path in summary_df["best_checkpoint_path"].to_list():
            cfg.ckpt_path = ckpt_path
            result, _ = evaluate(cfg, datamodule=datamodule)
            test_results.append(result)
        results = {key: [result[key] for result in test_results] for key in test_results[0].keys()}
        for key, values in results.items():
            summary_df = summary_df.with_columns(pl.Series(values).alias(key))

    summary_df.write_parquet(Path(cfg.paths.time_output_dir) / "sweep_results_summary.parquet")

    return best_trial, result_value


if __name__ == "__main__":
    main()
