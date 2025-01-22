import os
import re
from importlib.resources import files
from pathlib import Path
from typing import Any

import hydra
import lightning as L
import loguru
import polars as pl
from lightning import LightningDataModule
from omegaconf import DictConfig, OmegaConf
from tqdm.auto import tqdm

from meds_torch.data.datamodule import get_dataset
from meds_torch.utils import RankedLogger, configure_logging, task_wrapper
from meds_torch.utils.resolvers import setup_resolvers

setup_resolvers()
log = RankedLogger(__name__, rank_zero_only=True)
config_yaml = files("meds_torch").joinpath("configs/train.yaml")


def extract_data(input_string):
    # Split the string into lines
    lines = input_string.strip().split("\n")

    # Regular expression pattern to match the data
    pattern = r"^(\w+):\s+([\d.]+)\s*±\s*([\d.]+)\s*(\w+)\s*\(x(\d+)\)$"

    # Lists to store extracted data
    names = []
    mean_times = []
    std_devs = []
    units = []
    iterations = []

    # Extract data from each line
    for line in lines:
        match = re.match(pattern, line)
        if match:
            names.append(match.group(1))
            mean_time = float(match.group(2))
            std_dev = float(match.group(3))
            unit = match.group(4)
            iteration = int(match.group(5))

            # Convert to milliseconds
            if unit == "μs":
                mean_time /= 1000
                std_dev /= 1000
                unit = "ms"

            mean_times.append(mean_time)
            std_devs.append(std_dev)
            units.append(unit)  # All units are now ms
            iterations.append(iteration)

    # Create a Polars dataframe
    df = pl.DataFrame(
        {
            "Name": names,
            "Mean Time (ms)": mean_times,
            "Std Dev (ms)": std_devs,
            "Unit": units,
            "Iterations": iterations,
        }
    )

    # Add Total Time column
    df = df.with_columns((pl.col("Mean Time (ms)") * pl.col("Iterations")).alias("Total Time (ms)"))

    return df


@task_wrapper
def benchmark(cfg: DictConfig) -> tuple[dict[str, Any], dict[str, Any]]:
    """Trains the model. Can additionally evaluate on a testset, using best weights obtained during training.

    This method is wrapped in optional @task_wrapper decorator, that controls the behavior during failure.
    Useful for multiruns, saving info about the crash, etc.

    :param cfg: A DictConfig configuration composed by Hydra.
    :return: A tuple with metrics and dict with all instantiated objects.
    """
    # cache hydra config
    os.makedirs(cfg.paths.time_output_dir, exist_ok=True)
    OmegaConf.save(config=cfg, f=Path(cfg.paths.time_output_dir) / "hydra_config.yaml")

    # set seed for random number generators in pytorch, numpy and python.random
    if cfg.get("seed"):
        L.seed_everything(cfg.seed, workers=True)

    log.info(f"Instantiating datamodule <{cfg.data._target_}>")
    datamodule: LightningDataModule = hydra.utils.instantiate(cfg.data)

    datamodule.data_train = get_dataset(datamodule.cfg, split="train")

    model = hydra.utils.instantiate(cfg.model)

    loguru.logger.info(f"Benchmarking Over: {len(datamodule.data_train)} patients")
    from itertools import islice

    import torch

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    model.to(device)

    import time

    load_to_device_time = 0
    forward_pass_time = 0
    backward_pass_time = 0
    optimizer_time = 0

    num_batches = 10
    for batch in islice(tqdm(datamodule.train_dataloader()), num_batches):
        optimizer.zero_grad()

        start = time.time()
        batch = {k: v.to(device) for k, v in batch.items()}
        load_to_device_time += time.time() - start

        start = time.time()
        output = model(batch)
        loss = output["MODEL//BATCH_LOSS"]
        forward_pass_time += time.time() - start

        start = time.time()
        loss.backward()
        backward_pass_time += time.time() - start

        start = time.time()

        optimizer.step()
        optimizer_time += time.time() - start

    loguru.logger.info(f"Load to device time: {load_to_device_time}")
    loguru.logger.info(f"Forward pass time: {forward_pass_time}")
    loguru.logger.info(f"Backward pass time: {backward_pass_time}")
    loguru.logger.info(f"Optimizer time: {optimizer_time}")

    # log time profiles: https://github.com/Oufattole/meds-torch/issues/44
    if hasattr(datamodule.data_train, "_timings"):
        benchmark_results_df = extract_data(datamodule.data_train._profile_durations())
        loguru.logger.info(f"Train Time:\n{benchmark_results_df}")
        benchmark_results_df.write_parquet(Path(cfg.paths.time_output_dir) / "benchmark_results.parquet")
    else:
        loguru.logger.warning("No _timings attribute found in datamodule.data_train")

    if hasattr(model, "_timings"):
        benchmark_results_df = extract_data(model._profile_durations())
        loguru.logger.info(f"Train Time:\n{benchmark_results_df}")
        benchmark_results_df.write_parquet(Path(cfg.paths.time_output_dir) / "benchmark_results.parquet")
    else:
        loguru.logger.warning("No _timings attribute found in model")


@hydra.main(version_base="1.3", config_path=str(config_yaml.parent.resolve()), config_name=config_yaml.stem)
def main(cfg: DictConfig) -> float | None:
    """Main entry point for training.

    :param cfg: DictConfig configuration composed by Hydra.
    :return: Optional[float] with optimized metric value.
    """
    # apply extra utilities
    # (e.g. ask for tags if none are provided in cfg, print cfg tree, etc.)
    os.makedirs(cfg.paths.time_output_dir, exist_ok=True)
    configure_logging(cfg)

    # train the model
    benchmark(cfg)


if __name__ == "__main__":
    main()
