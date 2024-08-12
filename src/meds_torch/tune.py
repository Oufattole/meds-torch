import copy
import json
import os
from collections.abc import Callable
from importlib.resources import files
from pathlib import Path

import hydra
import ray
from omegaconf import DictConfig, OmegaConf
from ray import tune

from meds_torch.train import train
from meds_torch.transfer_learning import transfer_learning
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
        storage_path=cfg.paths.output_dir,
    )

    # Return the best trial results
    best_trial = analysis.get_best_trial(cfg.hparams_search.optimized_metric, "min", "last")
    return analysis, best_trial


@hydra.main(version_base="1.3", config_path=str(config_yaml.parent.resolve()), config_name=config_yaml.stem)
def main(cfg: DictConfig) -> float | None:
    """Main entry point for training.

    :param cfg: DictConfig configuration composed by Hydra.
    :return: Optional[float] with optimized metric value.
    """
    # apply extra utilities
    os.makedirs(cfg.paths.output_dir, exist_ok=True)
    extras(cfg)

    # Choose the training function based on the configuration
    if cfg.hparams_search.train_fn == "train":
        train_fn = train
    elif cfg.hparams_search.train_fn == "transfer_learning":
        train_fn = transfer_learning
    else:
        raise ValueError(f"Invalid train_fn: {cfg.hparams_search.train_fn}")

    # Run Ray Tune optimization
    analysis, best_trial = ray_tune_runner(cfg, train_fn=train_fn)

    # return tune results
    analysis.dataframe().to_csv(Path(cfg.paths.output_dir) / "tune_results.csv")
    result_value = best_trial.last_result[cfg.hparams_search.optimized_metric]

    ray.shutdown()

    with open(Path(cfg.paths.output_dir) / "best_result.json", "w") as outfile:
        json.dump(best_trial.last_result, outfile)

    return result_value


if __name__ == "__main__":
    main()
