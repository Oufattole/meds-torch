import hydra
import polars as pl
from hydra.core.config_store import ConfigStore
from loguru import logger
from omegaconf import DictConfig, OmegaConf

from meds_torch.EIToken import (
    assign_quantiles,
    calculate_time_intervals,
    create_event_tokens,
    generate_patient_timeline,
    load_data,
)


# Functions corresponding to each command
def joint_quantile_timeline(df: pl.DataFrame, output_path):
    result = df[["patient_id", "timestamp", "joint_quantile_timeline"]]
    result.rename(columns={"joint_quantile_timeline": "timeline"}, inplace=True)
    result.write_parquet(output_path)
    logger.info("Joint Quantile Timeline saved to:", output_path)


def split_quantile_timeline(df: pl.DataFrame, output_path):
    result = df[["patient_id", "timestamp", "split_quantile_timeline"]]
    result.rename(columns={"split_quantile_timeline": "timeline"}, inplace=True)
    result.write_parquet(output_path)
    logger.info("Split Quantile Timeline saved to:", output_path)


cs = ConfigStore.instance()
cs.store(name="config", node=DictConfig)


@hydra.main(config_path="../configs", config_name="config")
def main(cfg: DictConfig):
    logger.info(f"Running with config: {OmegaConf.to_yaml(cfg)}")
    data_path = cfg.data_path
    output_path = cfg.output_path

    if not data_path.exists():
        raise FileNotFoundError(f"The specified data path {data_path} does not exist.")

    df = load_data(cfg.data_path)
    df = assign_quantiles(df)
    df = create_event_tokens(df)
    df = calculate_time_intervals(df)
    df = generate_patient_timeline(df)

    if cfg.timeline_type == "joint":
        joint_quantile_timeline(df, output_path)
    elif cfg.timeline_type == "split":
        split_quantile_timeline(df, output_path)
    else:
        logger.error(f"Unknown timeline type: {cfg.timeline_type}")


if __name__ == "__main__":
    main()


# Script
# scripts/EIToken_tokenizer.py //
# data_path='/path/to/data' output_path='/path/to/output.parquet' timeline_type=joint
