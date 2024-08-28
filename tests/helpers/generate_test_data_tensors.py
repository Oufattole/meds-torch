"""Generates processed test data using the MEDS_transform package."""

import rootutils

root = rootutils.setup_root(__file__, dotenv=True, pythonpath=True, cwd=True)

import shutil
import tempfile
from pathlib import Path

import polars as pl
from loguru import logger
from omegaconf import OmegaConf

from meds_torch.utils.custom_time_token import TIME_DELTA_TOKEN
from tests.helpers.run_sh_command import run_command

AGGREGATIONS = [
    "code/n_occurrences",
    "code/n_patients",
    "values/n_occurrences",
    "values/n_patients",
    "values/sum",
    "values/sum_sqd",
    "values/n_ints",
    "values/min",
    "values/max",
    {"name": "values/quantiles", "quantiles": [0.25, 0.5, 0.75]},
]


# Aggregate metadata from MEDS_OUTPUTS
def aggregate_metadata(meds_outputs, unique_codes):
    metadata_entries = []
    for code in unique_codes:
        if code is None:
            continue

        filtered_dfs = [df.filter(pl.col("code") == code) for df in meds_outputs.values()]

        # Summarize data
        code_occurrences = sum(df.shape[0] for df in filtered_dfs)
        patient_ids = set()
        values_sum = 0.0
        values_sum_sqd = 0.0
        values_occurrences = 0

        for df in filtered_dfs:
            patient_ids.update(df.select(pl.col("patient_id")).to_series().to_list())

            valid_values_df = df.filter(pl.col("numeric_value").is_not_null())
            values_occurrences += valid_values_df.shape[0]

            if "numeric_value" in df.columns and not df["numeric_value"].is_null().all():
                valid_values = df["numeric_value"].fill_null(0)
                values_sum += valid_values.sum()
                values_sum_sqd += (valid_values**2).sum()

        metadata_entries.append(
            {
                "code": code,
                "code_occurrences": code_occurrences,
                "patient_counts": len(patient_ids),
                "values_occurrences": values_occurrences,
                "values_sum": values_sum,
                "values_sum_sqd": values_sum_sqd,
            }
        )

    metadata_str = "code,code/n_occurrences,code/n_patients,values/n_occurrences,values/sum,values/sum_sqd\n"
    for entry in metadata_entries:
        metadata_str += (
            f"{entry['code']},{entry['code_occurrences']},"
            f"{entry['patient_counts']},{entry['values_occurrences']},"
            f"{entry['values_sum']},{entry['values_sum_sqd']}\n"
        )
    return metadata_str


def list_subdir_files(root: Path | str, ext: str) -> list[Path]:
    """List files in subdirectories of a directory with a given extension."""
    return sorted(list(Path(root).glob(f"**/*.{ext}")))


def generate_test_triplet_tokenize(tmp_path):
    input_dir = tmp_path / "MEDS_cohort"
    cohort_dir = tmp_path / "triplet_tensors"

    # Create the directories
    input_dir.mkdir(parents=True, exist_ok=True)
    cohort_dir.mkdir(parents=True, exist_ok=True)

    config_kwargs = {
        "input_dir": str(input_dir.resolve()),
        "cohort_dir": str(cohort_dir.resolve()),
        "stage_configs": {
            "aggregate_code_metadata": {"aggregations": AGGREGATIONS, "do_summarize_over_all_codes": True}
        },
        "stages": [
            "aggregate_code_metadata",
            "filter_patients",
            "add_time_derived_measurements",
            "filter_measurements",
            "occlude_outliers",
            "fit_vocabulary_indices",
            "normalization",
            "tokenization",
            "tensorization",
        ],
    }
    # config = OmegaConf.create(config_kwargs)
    config_name = "preprocess"
    if config_name is None:
        raise ValueError("config_name must be provided if do_use_config_yaml is True.")

    conf = OmegaConf.create(
        {
            "defaults": [config_name],
            **config_kwargs,
        }
    )

    conf_dir = tempfile.TemporaryDirectory()
    conf_path = Path(conf_dir.name) / "config.yaml"
    OmegaConf.save(conf, conf_path)

    args = [
        f"--config-path={str(conf_path.parent.resolve())}",
        "--config-name=config",
        "'hydra.searchpath=[pkg://MEDS_transforms.configs]'",
    ]
    run_command("MEDS_transform-aggregate_code_metadata", args, {}, "aggregate code metadata")

    logger.info("Converting to code metadata...")

    logger.info("Filtering patients...")
    stderr, stdout = run_command("MEDS_transform-filter_patients", args, {}, "filter patients")

    logger.info("Generating time derived measurements...")
    stderr, stdout = run_command(
        "MEDS_transform-add_time_derived_measurements", args, {}, "time derived measurements"
    )

    logger.info("Filtering measurements...")
    stderr, stdout = run_command("MEDS_transform-filter_measurements", args, {}, "filter_codes")

    logger.info("Occluding outliers...")
    stderr, stdout = run_command("MEDS_transform-occlude_outliers", args, {}, "filter_outliers")

    logger.info("Fitting vocabulary indices...")
    stderr, stdout = run_command("MEDS_transform-fit_vocabulary_indices", args, {}, "fit_vocabulary_indices")

    logger.info("Normalizing data (converting codes to use integer encodings)...")
    stderr, stdout = run_command("MEDS_transform-normalization", args, {}, "normalize")

    logger.info("Converting to tokenization...")
    stderr, stdout = run_command("MEDS_transform-tokenization", args, {}, "tokenize")

    logger.info("Converting to tensor...")
    stderr, stdout = run_command("MEDS_transform-tensorization", args, {}, "tensorize")


def generate_test_eic_tokenize(tmp_path):
    input_dir = tmp_path / "MEDS_cohort"
    cohort_dir = tmp_path / "eic_tensors"

    # Create the directories
    input_dir.mkdir(parents=True, exist_ok=True)
    cohort_dir.mkdir(parents=True, exist_ok=True)

    config_kwargs = {
        "input_dir": str(input_dir.resolve()),
        "cohort_dir": str(cohort_dir.resolve()),
        "stage_configs": {
            "aggregate_code_metadata": {"aggregations": AGGREGATIONS, "do_summarize_over_all_codes": True},
            "custom_time_token": {"time_delta": {"time_unit": "years"}},
            "custom_normalization": {
                "custom_quantiles": {
                    TIME_DELTA_TOKEN: {
                        "values/quantile/0.2": 0.1,
                        "values/quantile/0.4": 1,
                        "values/quantile/0.6": 30,
                        "values/quantile/0.8": 365,
                    }
                }
            },
        },
        "stages": [
            "custom_time_token",
            "aggregate_code_metadata",
            "filter_patients",
            "filter_measurements",
            "fit_vocabulary_indices",
            "custom_normalization",
            "tokenization",
            "tensorization",
        ],
    }
    config_name = "preprocess"
    if config_name is None:
        raise ValueError("config_name must be provided if do_use_config_yaml is True.")

    conf = OmegaConf.create(
        {
            "defaults": [config_name],
            **config_kwargs,
        }
    )

    conf_dir = tempfile.TemporaryDirectory()
    conf_path = Path(conf_dir.name) / "config.yaml"
    OmegaConf.save(conf, conf_path)

    args = [
        f"--config-path={str(conf_path.parent.resolve())}",
        "--config-name=config",
        "'hydra.searchpath=[pkg://MEDS_transforms.configs]'",
    ]
    stderr, stdout = run_command("python -m meds_torch.utils.custom_time_token", args, {}, "add time deltas")

    logger.info("Aggregating code metadata...")
    stderr, stdout = run_command(
        "MEDS_transform-aggregate_code_metadata", args, {}, "aggregate code metadata"
    )

    logger.info("Filtering patients...")
    stderr, stdout = run_command("MEDS_transform-filter_patients", args, {}, "filter patients")

    logger.info("Filtering measurements...")
    stderr, stdout = run_command("MEDS_transform-filter_measurements", args, {}, "filter_codes")

    logger.info("Fitting vocabulary indices...")
    stderr, stdout = run_command("MEDS_transform-fit_vocabulary_indices", args, {}, "fit_vocabulary_indices")

    logger.info("Normalizing data (converting codes to use integer encodings)...")
    stderr, stdout = run_command("python -m meds_torch.utils.custom_normalization", args, {}, "normalize")

    logger.info("Converting to tokenization...")
    stderr, stdout = run_command("MEDS_transform-tokenization", args, {}, "tokenize")

    logger.info("Converting to tensor...")
    stderr, stdout = run_command("MEDS_transform-tensorization", args, {}, "tensorize")


if __name__ == "__main__":
    output_dir = Path(__file__).parent.parent / "test_data"

    generate_test_triplet_tokenize(output_dir)
    generate_test_eic_tokenize(output_dir)

    logger.info("Deleting log files and moving data from temporary directory to project...")
    for log_dir in output_dir.rglob(".logs"):
        shutil.rmtree(log_dir)
    logger.info("All done. Exiting...")
