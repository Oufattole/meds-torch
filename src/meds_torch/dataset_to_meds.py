import os
from importlib.resources import files
from pathlib import Path
from typing import Any

import hydra
import numpy as np
import polars as pl
import pyarrow.parquet as pq
import torch
from lightning import LightningDataModule
from mixins.seedable import seed_everything
from omegaconf import DictConfig
from tqdm.auto import tqdm

from meds_torch.models.components.utils import TrajectoryBatch, get_time_days_delta
from meds_torch.schemas.generate_analysis_schema import validate_generated_data
from meds_torch.utils import RankedLogger, configure_logging, task_wrapper
from meds_torch.utils.resolvers import setup_resolvers

setup_resolvers()
log = RankedLogger(__name__, rank_zero_only=True)
config_yaml = files("meds_torch").joinpath("configs/dataset_to_meds.yaml")


def convert_to_meds(dataloader: torch.utils.data.DataLoader, metadata_df: pl.DataFrame) -> pl.DataFrame:
    """Converts a list of batches to a Polars DataFrame.

    Args:
        batches: List of dictionaries with batch data.
    Returns:
        Polars DataFrame with all batches.
    """
    dfs = []
    for batch in tqdm(dataloader):
        time = (-batch["time_delta_days"].flip(1)).cumsum(dim=1).flip(1)  # cumulative sum time deltas back
        code = batch["code"]
        mask = batch["mask"]
        numeric_value = batch["numeric_value"]
        numeric_value_mask = batch["numeric_value_mask"]
        metadata_df = metadata_df
        prediction_time = batch["prediction_time"]
        subject_id = batch["subject_id"]
        end_time = batch["end_time"]

        prediction_time_offset_days = (-get_time_days_delta(prediction_time, end_time)).numpy()
        # add delta from end time to task prediction time
        time += prediction_time_offset_days[:, np.newaxis]
        df = TrajectoryBatch(
            time,
            code,
            mask,
            numeric_value,
            numeric_value_mask,
            metadata_df,
            time_scale="D",
        ).to_meds(prediction_time, subject_id)
        if "boolean_value" in batch:
            label_df = pl.DataFrame(
                {
                    "subject_id": subject_id.numpy(),
                    "prediction_time": prediction_time,
                    "boolean_value": pl.Series(batch["boolean_value"].to(torch.bool).numpy()),
                },
                schema={
                    "subject_id": df.schema["subject_id"],
                    "prediction_time": df.schema["prediction_time"],
                    "boolean_value": pl.Boolean,
                },
            )
            df = df.join(label_df, on=["subject_id", "prediction_time"], how="left")
        dfs.append(df)

    meds_df = pl.concat(dfs)
    meds_df = meds_df.with_columns(pl.lit("INPUT_DATA").alias("TRAJECTORY_TYPE"))
    # Convert to arrow table and write to parquet
    validated_table = validate_generated_data(meds_df)
    return validated_table


@task_wrapper
def input_to_meds(cfg: DictConfig, datamodule=None) -> tuple[dict[str, Any], dict[str, Any]]:
    """Evaluates given checkpoint on a datamodule testset.

    This method is wrapped in optional @task_wrapper decorator, that controls the
    behavior during failure. Useful for multiruns, saving info about the crash, etc.

    Args:
        cfg: DictConfig configuration composed by Hydra.
    Returns:
        Tuple[dict, dict] with metrics and dict with all instantiated objects.
    """
    seed_everything(cfg.seed)
    log.info(f"Set all seeds to {cfg.seed}")
    if not cfg.data.do_include_subject_id:
        raise ValueError("Subject ID is required for getting input data")
    if not cfg.data.do_include_prediction_time:
        raise ValueError("Prediction time is required for getting input data")
    if not cfg.data.do_include_end_time:
        raise ValueError("End time is required for getting input data")

    log.info(f"Instantiating datamodule <{cfg.data._target_}>")
    if not datamodule:
        datamodule: LightningDataModule = hydra.utils.instantiate(cfg.data)
    datamodule.setup()

    metadata_df = pl.read_parquet(cfg.data.code_metadata_fp)
    if "[EOS]" not in metadata_df["code"].unique():
        metadata_df = pl.concat(
            [
                metadata_df,
                pl.DataFrame(
                    {
                        "code": ["[EOS]"],
                        "code/vocab_index": [cfg.data.EOS_TOKEN_ID],
                    },
                    schema={k: metadata_df.schema[k] for k in ["code", "code/vocab_index"]},
                ),
            ],
            how="diagonal",
        )
    meds_data = convert_to_meds(datamodule.predict_dataloader(), metadata_df)
    split = cfg.data.predict_dataset
    pq.write_table(meds_data, str(Path(cfg.paths.output_dir) / f"{split}.parquet"))


@hydra.main(version_base="1.3", config_path=str(config_yaml.parent.resolve()), config_name=config_yaml.stem)
def main(cfg: DictConfig) -> None:
    """Main entry point for evaluation.

    Args:
        cfg (DictConfig):  configuration composed by Hydra.
    """
    # apply extra utilities
    # (e.g. ask for tags if none are provided in cfg, print cfg tree, etc.)
    os.makedirs(cfg.paths.output_dir, exist_ok=True)
    configure_logging(cfg)
    input_to_meds(cfg)


if __name__ == "__main__":
    main()
