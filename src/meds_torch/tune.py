import copy
import json
import os
from collections.abc import Callable
from pathlib import Path

import hydra
import ray
import rootutils
from omegaconf import DictConfig, OmegaConf
from ray import tune

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)
from meds_torch.train import train
from meds_torch.transfer_learning import transfer_learning

# ------------------------------------------------------------------------------------ #
# the setup_root above is equivalent to:
# - adding project root dir to PYTHONPATH
#       (so you don't need to force user to install project as a package)
#       (necessary before importing any local modules e.g. `from meds_torch import utils`)
# - setting up PROJECT_ROOT environment variable
#       (which is used as a base for paths in "configs/paths/default.yaml")
#       (this way all filepaths are the same no matter where you run the code)
# - loading environment variables from ".env" in root dir
#
# you can remove it if you:
# 1. either install project as a package or move entry files to project root dir
# 2. set `root_dir` to "." in "configs/paths/default.yaml"
#
# more info: https://github.com/ashleve/rootutils
# ------------------------------------------------------------------------------------ #
from meds_torch.utils import RankedLogger, extras
from meds_torch.utils.resolvers import setup_resolvers

log = RankedLogger(__name__, rank_zero_only=True)

setup_resolvers()


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


@hydra.main(version_base="1.3", config_path="../configs", config_name="train.yaml")
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
