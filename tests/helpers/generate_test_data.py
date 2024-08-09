"""Generates processed test data using the MEDS_transform package."""

import rootutils

root = rootutils.setup_root(__file__, dotenv=True, pythonpath=True, cwd=True)

import json
import shutil
from io import StringIO
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import polars as pl
from loguru import logger

from tests.helpers.run_sh_command import run_command

# Data split configuration
splits_config = {"train": 84, "tuning": 18, "held_out": 18}

# Example data blocks
example_data = {
    "1": """
        patient_id,code,timestamp,numerical_value
        239684,HEIGHT,,175.271115221764
        239684,EYE_COLOR//BROWN,,
        239684,DOB,1980-12-28T00:00:00.000000,
        239684,TEMP,2010-05-11T17:41:51.000000,96.0
        239684,ADMISSION//CARDIAC,2010-05-11T17:41:51.000000,
        239684,HR,2010-05-11T17:41:51.000000,102.6
        239684,TEMP,2010-05-11T17:48:48.000000,96.2
        239684,HR,2010-05-11T17:48:48.000000,105.1
        239684,TEMP,2010-05-11T18:25:35.000000,95.8
        239684,HR,2010-05-11T18:25:35.000000,113.4
        239684,HR,2010-05-11T18:57:18.000000,112.6
        239684,TEMP,2010-05-11T18:57:18.000000,95.5
        239684,DISCHARGE,2010-05-11T19:27:19.000000,
        1195293,HEIGHT,,164.6868838269085
        1195293,EYE_COLOR//BLUE,,
        1195293,DOB,1978-06-20T00:00:00.000000,
        1195293,TEMP,2010-06-20T19:23:52.000000,100.0
        1195293,ADMISSION//CARDIAC,2010-06-20T19:23:52.000000,
        1195293,HR,2010-06-20T19:23:52.000000,109.0
        1195293,TEMP,2010-06-20T19:25:32.000000,100.0
        1195293,HR,2010-06-20T19:25:32.000000,114.1
        1195293,HR,2010-06-20T19:45:19.000000,119.8
        1195293,TEMP,2010-06-20T19:45:19.000000,99.9
        1195293,HR,2010-06-20T20:12:31.000000,112.5
        1195293,TEMP,2010-06-20T20:12:31.000000,99.8
        1195293,HR,2010-06-20T20:24:44.000000,107.7
        1195293,TEMP,2010-06-20T20:24:44.000000,100.0
        1195293,TEMP,2010-06-20T20:41:33.000000,100.4
        1195293,HR,2010-06-20T20:41:33.000000,107.5
        1195293,DISCHARGE,2010-06-20T20:50:04.000000,
    """,
    "2": """
        patient_id,code,timestamp,numerical_value
        68729,EYE_COLOR//HAZEL,,
        68729,HEIGHT,,160.3953106166676
        68729,DOB,1978-03-09T00:00:00.000000,
        68729,HR,2010-05-26T02:30:56.000000,86.0
        68729,ADMISSION//PULMONARY,2010-05-26T02:30:56.000000,
        68729,TEMP,2010-05-26T02:30:56.000000,97.8
        68729,DISCHARGE,2010-05-26T04:51:52.000000,
        814703,EYE_COLOR//HAZEL,,
        814703,HEIGHT,,156.48559093209357
        814703,DOB,1976-03-28T00:00:00.000000,
        814703,TEMP,2010-02-05T05:55:39.000000,100.1
        814703,HR,2010-02-05T05:55:39.000000,170.2
        814703,ADMISSION//ORTHOPEDIC,2010-02-05T05:55:39.000000,
        814703,DISCHARGE,2010-02-05T07:02:30.000000,
    """,
    "3": """
        patient_id,code,timestamp,numerical_value
        1500733,HEIGHT,,158.60131573580904
        1500733,EYE_COLOR//BROWN,,
        1500733,DOB,1986-07-20T00:00:00.000000,
        1500733,TEMP,2010-06-03T14:54:38.000000,100.0
        1500733,HR,2010-06-03T14:54:38.000000,91.4
        1500733,ADMISSION//ORTHOPEDIC,2010-06-03T14:54:38.000000,
        1500733,HR,2010-06-03T15:39:49.000000,84.4
        1500733,TEMP,2010-06-03T15:39:49.000000,100.3
        1500733,HR,2010-06-03T16:20:49.000000,90.1
        1500733,TEMP,2010-06-03T16:20:49.000000,100.1
        1500733,DISCHARGE,2010-06-03T16:44:26.000000,
    """,
    "4": """
        patient_id,code,timestamp,numerical_value
        754281,EYE_COLOR//BROWN,,
        754281,HEIGHT,,166.22261567137025
        754281,DOB,1988-12-19T00:00:00.000000,
        754281,ADMISSION//PULMONARY,2010-01-03T06:27:59.000000,
        754281,TEMP,2010-01-03T06:27:59.000000,99.8
        754281,HR,2010-01-03T06:27:59.000000,142.0
        754281,DEATH,2010-01-03T08:22:13.000000,
    """,
}


def process_data(data_string, current_patient_id):
    data_io = StringIO(data_string.strip())
    df = pl.read_csv(data_io)
    df = df.with_columns(pl.col("patient_id").str.replace_all(" ", "").cast(pl.Int64).alias("patient_id"))
    unique_ids = df["patient_id"].unique().to_list()
    id_replacement = {old_id: current_patient_id + i for i, old_id in enumerate(unique_ids)}

    try:
        df = df.with_columns(pl.col("patient_id").replace_strict(id_replacement))
    except Exception as e:
        logger.error(f"Error during ID replacement: {e}")
        raise

    return df, current_patient_id + len(unique_ids)


def generate_data(example_data, config, initial_patient_id):
    """Generates Test Meds Dataset."""
    current_patient_id = initial_patient_id
    meds_outputs = {}
    all_splits = {}
    all_codes = set()
    rng = np.random.default_rng(0)

    for split_name, count in config.items():
        for i in range(count):
            key = f"{split_name}/{i}"
            chosen_block = rng.choice(list(example_data.keys()))
            df, current_patient_id = process_data(example_data[chosen_block], current_patient_id)
            meds_outputs[key] = df
            all_splits[key] = df.select(pl.col("patient_id").unique()).to_numpy().flatten().tolist()
            all_codes.update(df.get_column("code").to_numpy().flatten())

    return meds_outputs, all_splits, all_codes


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

            valid_values_df = df.filter(pl.col("numerical_value").is_not_null())
            values_occurrences += valid_values_df.shape[0]

            if "numerical_value" in df.columns and not df["numerical_value"].is_null().all():
                valid_values = df["numerical_value"].fill_null(0)
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


def test_tokenize(tmp_path):
    MEDS_cohort_dir = tmp_path / "processed" / "final_cohort"

    # Create the directories
    MEDS_cohort_dir.mkdir(parents=True, exist_ok=True)

    # Store MEDS outputs
    for split, df in MEDS_OUTPUTS.items():
        file_path = MEDS_cohort_dir / f"{split}.parquet"
        file_path.parent.mkdir(exist_ok=True)
        df.with_columns(pl.col("timestamp").str.to_datetime("%Y-%m-%dT%H:%M:%S%.f")).write_parquet(file_path)
    meds_files = list_subdir_files(Path(MEDS_cohort_dir).parent, "parquet")
    assert len(meds_files) == len(MEDS_OUTPUTS), "Mismatch in number of expected data files!"
    for f in meds_files:
        assert pl.read_parquet(f).shape[0] > 0, "MEDS Data Tabular Dataframe Should not be Empty!"

    splits_fp = MEDS_cohort_dir.parent / "splits.json"
    json.dump(SPLITS_JSON, splits_fp.open("w"))

    pl.read_csv(source=StringIO(MEDS_OUTPUT_CODE_METADATA_FILE)).write_parquet(
        MEDS_cohort_dir.parent / "code_metadata.parquet"
    )
    # TODO: Eventually we want to use meds-transform to aggregate code metadata
    # logger.info("Converting to code metadata...")
    # code_kwargs = {
    # "input_dir": str(MEDS_cohort_dir.resolve()),
    # "cohort_dir": str(MEDS_cohort_dir.resolve()), "--config-name": "extract"}
    # stderr, stdout = run_command(
    # "MEDS_transform-aggregate_code_metadata", code_kwargs, "aggregate code metadata")

    config_kwargs = {
        "input_dir": str(MEDS_cohort_dir.parent.resolve()),
        "cohort_dir": str(MEDS_cohort_dir.resolve()),
        "stage_configs.add_time_derived_measurements.age.DOB_code": "DOB",
        "stages": "["
        + ",".join(
            [
                "filter_patients",
                "add_time_derived_measurements",
                "filter_measurements",
                "occlude_outliers",
                "fit_vocabulary_indices",
                "normalization",
                "tokenization",
                "tensorization",
            ]
        )
        + "]",
    }

    logger.info("Filtering patients...")
    stderr, stdout = run_command("MEDS_transform-filter_patients", [], config_kwargs, "filter patients")
    logger.info("Generating time derived measurements...")
    stderr, stdout = run_command(
        "MEDS_transform-add_time_derived_measurements", config_kwargs, "time derived measurements"
    )
    logger.info("Filtering measurements...")
    stderr, stdout = run_command("MEDS_transform-filter_measurements", [], config_kwargs, "filter_codes")
    logger.info("Occluding outliers...")
    stderr, stdout = run_command("MEDS_transform-occlude_outliers", [], config_kwargs, "filter_outliers")
    logger.info("Fitting vocabulary indices...")
    stderr, stdout = run_command(
        "MEDS_transform-fit_vocabulary_indices", [], config_kwargs, "fit_vocabulary_indices"
    )
    logger.info("Normalizing data (converting codes to use integer encodings)...")
    stderr, stdout = run_command("MEDS_transform-normalization", [], config_kwargs, "normalize")
    logger.info("Converting to tokenization...")
    stderr, stdout = run_command("MEDS_transform-tokenization", [], config_kwargs, "tokenize")
    logger.info("Converting to tensor...")
    stderr, stdout = run_command("MEDS_transform-tensorization", [], config_kwargs, "tensorize")
    shutil.move(MEDS_cohort_dir / "tensorization", MEDS_cohort_dir / "default_tensorization")
    logger.info("Converting to tensor...")
    config_kwargs["stage_configs.tensorization.strategy"] = "prompt_expanded_observation"
    stderr, stdout = run_command("MEDS_transform-tensorization", [], config_kwargs, "tensorize")
    shutil.move(
        MEDS_cohort_dir / "tensorization", MEDS_cohort_dir / "prompt_expanded_observation_tensorization"
    )


if __name__ == "__main__":
    output_dir = Path(__file__).parent.parent / "test_data"
    shutil.rmtree(output_dir, ignore_errors=True)
    with TemporaryDirectory() as d:
        tmp_path = Path(d)
        initial_patient_id = 10000

        MEDS_OUTPUTS, SPLITS_JSON, unique_codes = generate_data(
            example_data, splits_config, initial_patient_id
        )

        MEDS_OUTPUT_CODE_METADATA_FILE = aggregate_metadata(MEDS_OUTPUTS, unique_codes)
        logger.info("Data generation complete and files saved. Converting to Nested ragged tensor...")

        test_tokenize(tmp_path)

        logger.info("Deleting log files and moving data from temporary directory to project...")
        for log_dir in tmp_path.rglob(".logs"):
            shutil.rmtree(log_dir)
        shutil.copytree(tmp_path / "processed", output_dir)

    logger.info("All done. Exiting...")
