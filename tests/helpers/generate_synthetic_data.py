"""Tests the full end-to-end extraction process.

Set the bash env variable `DO_USE_LOCAL_SCRIPTS=1` to use the local py files, rather than the installed
scripts.
"""

import shutil

import rootutils

root = rootutils.setup_root(__file__, dotenv=True, pythonpath=True, cwd=True)

# Test data (inputs)
import random
import tempfile
from datetime import datetime, timedelta
from pathlib import Path

import polars as pl

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


def generate_subject_data(rng, num_subjects=100):
    subjects = []
    eye_colors = ["BLUE", "BROWN", "HAZEL", "GREEN"]
    departments = ["CARDIAC", "PULMONARY", "ORTHOPEDIC", "NEUROLOGY", "ONCOLOGY"]

    for _i in range(1, num_subjects + 1):
        mrn = f"{rng.randint(100000, 999999)}"
        dob = (datetime.now() - timedelta(days=rng.randint(365 * 18, 365 * 80))).strftime("%m/%d/%Y")
        eye_color = rng.choice(eye_colors)
        height = round(rng.uniform(150, 190), 2)
        department = rng.choice(departments)

        subjects.append(f"{mrn},{dob},{eye_color},{height},{department}")

    return "\n".join(["MRN,dob,eye_color,height,department"] + subjects)


def generate_admit_vitals(rng, subjects, num_visits_per_subject=2):
    admit_vitals = []

    for subject in subjects.split("\n")[1:]:  # Skip header
        mrn = subject.split(",")[0]
        for _ in range(num_visits_per_subject):
            admit_date = datetime.now() - timedelta(days=rng.randint(1, 365))
            discharge_date = admit_date + timedelta(hours=rng.randint(1, 48))
            department = rng.choice(["CARDIAC", "PULMONARY", "ORTHOPEDIC", "NEUROLOGY", "ONCOLOGY"])

            for _ in range(rng.randint(1, 5)):  # Generate 1-5 vital readings per visit
                vitals_date = admit_date + timedelta(
                    minutes=rng.randint(0, int((discharge_date - admit_date).total_seconds() / 60))
                )
                hr = round(rng.uniform(60, 100), 1)
                temp = round(rng.uniform(97, 99), 1)
                hr_text = rng.choice(["normal", "abnormal"])

                admit_vitals.append(
                    f"{mrn},\"{admit_date.strftime('%m/%d/%Y, %H:%M:%S')}\",\""
                    f"{discharge_date.strftime('%m/%d/%Y, %H:%M:%S')}\",{department},\""
                    f"{vitals_date.strftime('%m/%d/%Y, %H:%M:%S')}\",{hr},{temp},{hr_text}"
                )

    return "\n".join(
        ["subject_id,admit_date,disch_date,department,vitals_date,HR,temp,HR_text"] + admit_vitals
    )


def generate_synthetic_data(output_dir: Path):
    with tempfile.TemporaryDirectory() as d:
        raw_cohort_dir = Path(d) / "raw_cohort"

        # Create the directories
        raw_cohort_dir.mkdir()

        subjects_csv = raw_cohort_dir / "subjects.csv"
        admit_vitals_csv = raw_cohort_dir / "admit_vitals.csv"

        demo_metadata_csv = raw_cohort_dir / "demo_metadata.csv"
        input_metadata_csv = raw_cohort_dir / "input_metadata.csv"

        # Generate data for 100 subjects
        rng = random.Random(42)
        SUBJECTS_CSV = generate_subject_data(rng, 100)
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
        shutil.copytree(d, output_dir)


if __name__ == "__main__":
    output_dir = Path(__file__).parent.parent / "test_data"
    shutil.rmtree(output_dir, ignore_errors=True)
    generate_synthetic_data(output_dir)
