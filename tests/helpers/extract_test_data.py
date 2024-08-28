"""Tests the full end-to-end extraction process.

Set the bash env variable `DO_USE_LOCAL_SCRIPTS=1` to use the local py files, rather than the installed
scripts.
"""

import os
import shutil

import rootutils
from loguru import logger

root = rootutils.setup_root(__file__, dotenv=True, pythonpath=True, cwd=True)

code_root = root / "src" / "MEDS_transforms"
extraction_root = code_root / "extract"

if os.environ.get("DO_USE_LOCAL_SCRIPTS", "0") == "1":
    SHARD_EVENTS_SCRIPT = extraction_root / "shard_events.py"
    SPLIT_AND_SHARD_SCRIPT = extraction_root / "split_and_shard_patients.py"
    CONVERT_TO_SHARDED_EVENTS_SCRIPT = extraction_root / "convert_to_sharded_events.py"
    MERGE_TO_MEDS_COHORT_SCRIPT = extraction_root / "merge_to_MEDS_cohort.py"
    AGGREGATE_CODE_METADATA_SCRIPT = code_root / "aggregate_code_metadata.py"
    EXTRACT_CODE_METADATA_SCRIPT = extraction_root / "extract_code_metadata.py"
    FINALIZE_DATA_SCRIPT = extraction_root / "finalize_MEDS_data.py"
    FINALIZE_METADATA_SCRIPT = extraction_root / "finalize_MEDS_metadata.py"
else:
    SHARD_EVENTS_SCRIPT = "MEDS_extract-shard_events"
    SPLIT_AND_SHARD_SCRIPT = "MEDS_extract-split_and_shard_patients"
    CONVERT_TO_SHARDED_EVENTS_SCRIPT = "MEDS_extract-convert_to_sharded_events"
    MERGE_TO_MEDS_COHORT_SCRIPT = "MEDS_extract-merge_to_MEDS_cohort"
    AGGREGATE_CODE_METADATA_SCRIPT = "MEDS_transform-aggregate_code_metadata"
    EXTRACT_CODE_METADATA_SCRIPT = "MEDS_extract-extract_code_metadata"
    FINALIZE_DATA_SCRIPT = "MEDS_extract-finalize_MEDS_data"
    FINALIZE_METADATA_SCRIPT = "MEDS_extract-finalize_MEDS_metadata"

import json

# Test data (inputs)
import random
import tempfile
from datetime import datetime, timedelta
from pathlib import Path

import polars as pl

from tests.helpers.utils import run_command

INPUT_METADATA_FILE = """
lab_code,title,loinc
HR,Heart Rate,8867-4
temp,Body Temperature,8310-5
"""


DEMO_METADATA_FILE = """
eye_color,description
BROWN,"Brown Eyes. The most common eye color."
BLUE,"Blue Eyes. Less common than brown."
HAZEL,"Hazel eyes. These are uncommon"
GREEN,"Green eyes. These are rare."
"""


EVENT_CFGS_YAML = """
subjects:
  patient_id_col: MRN
  eye_color:
    code:
      - EYE_COLOR
      - col(eye_color)
    time: null
    _metadata:
      demo_metadata:
        description: description
  height:
    code: HEIGHT
    time: null
    numeric_value: height
  dob:
    code: DOB
    time: col(dob)
    time_format: "%m/%d/%Y"
admit_vitals:
  admissions:
    code:
      - ADMISSION
      - col(department)
    time: col(admit_date)
    time_format: "%m/%d/%Y, %H:%M:%S"
  discharge:
    code: DISCHARGE
    time: col(disch_date)
    time_format: "%m/%d/%Y, %H:%M:%S"
  HR:
    code: HR
    time: col(vitals_date)
    time_format: "%m/%d/%Y, %H:%M:%S"
    numeric_value: HR
    _metadata:
      input_metadata:
        description: {"title": {"lab_code": "HR"}}
        parent_codes: {"LOINC/{loinc}": {"lab_code": "HR"}}
  temp:
    code: TEMP
    time: col(vitals_date)
    time_format: "%m/%d/%Y, %H:%M:%S"
    numeric_value: temp
    _metadata:
      input_metadata:
        description: {"title": {"lab_code": "temp"}}
        parent_codes: {"LOINC/{loinc}": {"lab_code": "temp"}}
"""


def generate_patient_data(rng, num_patients=100):
    patients = []
    eye_colors = ["BLUE", "BROWN", "HAZEL", "GREEN"]
    departments = ["CARDIAC", "PULMONARY", "ORTHOPEDIC", "NEUROLOGY", "ONCOLOGY"]

    for _i in range(1, num_patients + 1):
        mrn = f"{rng.randint(100000, 999999)}"
        dob = (datetime.now() - timedelta(days=rng.randint(365 * 18, 365 * 80))).strftime("%m/%d/%Y")
        eye_color = rng.choice(eye_colors)
        height = round(rng.uniform(150, 190), 2)
        department = rng.choice(departments)

        patients.append(f"{mrn},{dob},{eye_color},{height},{department}")

    return "\n".join(["MRN,dob,eye_color,height,department"] + patients)


def generate_admit_vitals(rng, patients, num_visits_per_patient=2):
    admit_vitals = []

    for patient in patients.split("\n")[1:]:  # Skip header
        mrn = patient.split(",")[0]
        for _ in range(num_visits_per_patient):
            admit_date = datetime.now() - timedelta(days=rng.randint(1, 365))
            discharge_date = admit_date + timedelta(hours=rng.randint(1, 48))
            department = rng.choice(["CARDIAC", "PULMONARY", "ORTHOPEDIC", "NEUROLOGY", "ONCOLOGY"])

            for _ in range(rng.randint(1, 5)):  # Generate 1-5 vital readings per visit
                vitals_date = admit_date + timedelta(
                    minutes=rng.randint(0, int((discharge_date - admit_date).total_seconds() / 60))
                )
                hr = round(rng.uniform(60, 100), 1)
                temp = round(rng.uniform(97, 99), 1)

                admit_vitals.append(
                    f"{mrn},\"{admit_date.strftime('%m/%d/%Y, %H:%M:%S')}\",\""
                    f"{discharge_date.strftime('%m/%d/%Y, %H:%M:%S')}\",{department},\""
                    f"{vitals_date.strftime('%m/%d/%Y, %H:%M:%S')}\",{hr},{temp}"
                )

    return "\n".join(["patient_id,admit_date,disch_date,department,vitals_date,HR,temp"] + admit_vitals)


def test_extraction(output_dir: Path):
    with tempfile.TemporaryDirectory() as d:
        raw_cohort_dir = Path(d) / "raw_cohort"
        MEDS_cohort_dir = Path(d) / "MEDS_cohort"

        # Create the directories
        raw_cohort_dir.mkdir()
        MEDS_cohort_dir.mkdir()

        subjects_csv = raw_cohort_dir / "subjects.csv"
        admit_vitals_csv = raw_cohort_dir / "admit_vitals.csv"
        event_cfgs_yaml = raw_cohort_dir / "event_cfgs.yaml"

        demo_metadata_csv = raw_cohort_dir / "demo_metadata.csv"
        input_metadata_csv = raw_cohort_dir / "input_metadata.csv"

        # Generate data for 100 patients
        rng = random.Random(42)
        SUBJECTS_CSV = generate_patient_data(rng, 100)
        ADMIT_VITALS_CSV = generate_admit_vitals(rng, SUBJECTS_CSV)

        # Write the CSV files
        subjects_csv.write_text(SUBJECTS_CSV.strip())
        admit_vitals_csv.write_text(ADMIT_VITALS_CSV.strip())
        demo_metadata_csv.write_text(DEMO_METADATA_FILE.strip())
        input_metadata_csv.write_text(INPUT_METADATA_FILE.strip())

        # Mix things up -- have one CSV be also in parquet format.
        admit_vitals_parquet = raw_cohort_dir / "admit_vitals.parquet"
        df = pl.read_csv(admit_vitals_csv)
        df.write_parquet(admit_vitals_parquet, use_pyarrow=True)

        # Write the event config YAML
        event_cfgs_yaml.write_text(EVENT_CFGS_YAML)

        # Run the extraction script
        extraction_config_kwargs = {
            "input_dir": str(raw_cohort_dir.resolve()),
            "cohort_dir": str(MEDS_cohort_dir.resolve()),
            "event_conversion_config_fp": str(event_cfgs_yaml.resolve()),
            "stage_configs.split_and_shard_patients.split_fracs.train": 0.7,
            "stage_configs.split_and_shard_patients.split_fracs.tuning": 0.15,
            "stage_configs.split_and_shard_patients.split_fracs.held_out": 0.15,
            "stage_configs.shard_events.row_chunksize": 100,
            "stage_configs.split_and_shard_patients.n_patients_per_shard": 20,
            "hydra.verbose": True,
            "etl_metadata.dataset_name": "TEST",
            "etl_metadata.dataset_version": "1.0",
        }

        # ... (rest of the function remains the same)

        all_stderrs = []
        all_stdouts = []

        # Stage 1: Sub-shard the data
        stderr, stdout = run_command(SHARD_EVENTS_SCRIPT, extraction_config_kwargs, "shard_events")

        all_stderrs.append(stderr)
        all_stdouts.append(stdout)

        # Stage 2: Collect the patient splits
        stderr, stdout = run_command(
            SPLIT_AND_SHARD_SCRIPT,
            extraction_config_kwargs,
            "split_and_shard_patients",
        )

        all_stderrs.append(stderr)
        all_stdouts.append(stdout)

        try:
            shards_fp = MEDS_cohort_dir / "metadata" / ".shards.json"
            assert shards_fp.is_file(), f"Expected splits @ {str(shards_fp.resolve())} to exist."

        except AssertionError as e:
            print("Failed to split patients")
            print(f"stderr:\n{stderr}")
            print(f"stdout:\n{stdout}")
            raise e

        # Stage 3: Extract the events and sub-shard by patient
        stderr, stdout = run_command(
            CONVERT_TO_SHARDED_EVENTS_SCRIPT,
            extraction_config_kwargs,
            "convert_events",
        )
        all_stderrs.append(stderr)
        all_stdouts.append(stdout)

        patient_subsharded_folder = MEDS_cohort_dir / "convert_to_sharded_events"
        assert patient_subsharded_folder.is_dir(), f"Expected {patient_subsharded_folder} to be a directory."

        # Stage 4: Merge to the final output
        stderr, stdout = run_command(
            MERGE_TO_MEDS_COHORT_SCRIPT,
            extraction_config_kwargs,
            "merge_to_MEDS_cohort",
        )
        all_stderrs.append(stderr)
        all_stdouts.append(stdout)

        # Stage 6: Extract code metadata
        stderr, stdout = run_command(
            EXTRACT_CODE_METADATA_SCRIPT,
            extraction_config_kwargs,
            "extract_code_metadata",
        )
        all_stderrs.append(stderr)
        all_stdouts.append(stdout)

        output_file = MEDS_cohort_dir / "extract_code_metadata" / "codes.parquet"
        assert output_file.is_file(), f"Expected {output_file} to exist: stderr:\n{stderr}\nstdout:\n{stdout}"

        got_df = pl.read_parquet(output_file, glob=False)

        # We collapse the list type as it throws an error in the assert_df_equal otherwise
        got_df = got_df.with_columns(pl.col("parent_codes").list.join("||"))

        # Stage 7: Finalize the MEDS data
        stderr, stdout = run_command(
            FINALIZE_DATA_SCRIPT,
            extraction_config_kwargs,
            "finalize_MEDS_data",
        )
        all_stderrs.append(stderr)
        all_stdouts.append(stdout)

        # Stage 8: Finalize the metadata
        stderr, stdout = run_command(
            FINALIZE_METADATA_SCRIPT,
            extraction_config_kwargs,
            "finalize_metadata",
        )
        all_stderrs.append(stderr)
        all_stdouts.append(stdout)

        # Check code metadata
        output_file = MEDS_cohort_dir / "metadata" / "codes.parquet"
        assert output_file.is_file(), f"Expected {output_file} to exist: stderr:\n{stderr}\nstdout:\n{stdout}"

        got_df = pl.read_parquet(output_file, glob=False, use_pyarrow=True)

        # We collapse the list type as it throws an error in the assert_df_equal otherwise
        got_df = got_df.with_columns(pl.col("parent_codes").list.join("||"))

        # Check dataset metadata
        output_file = MEDS_cohort_dir / "metadata" / "dataset.json"
        assert output_file.is_file(), f"Expected {output_file} to exist: stderr:\n{stderr}\nstdout:\n{stdout}"

        got_json = json.loads(output_file.read_text())
        assert "etl_version" in got_json, "Expected 'etl_version' to be in the dataset metadata."
        got_json.pop("etl_version")  # We don't test this as it changes with the commits.

        # Check the splits parquet
        output_file = MEDS_cohort_dir / "metadata" / "patient_splits.parquet"
        assert output_file.is_file(), f"Expected {output_file} to exist: stderr:\n{stderr}\nstdout:\n{stdout}"

        got_df = pl.read_parquet(output_file, glob=False, use_pyarrow=True)
        shutil.copytree(d, output_dir)
    logger.info("All extraction tests passed!")


if __name__ == "__main__":
    output_dir = Path(__file__).parent.parent / "test_data"
    shutil.rmtree(output_dir, ignore_errors=True)
    test_extraction(output_dir)
