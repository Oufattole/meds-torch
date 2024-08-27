"""Generates Random Local Windows for testing multiwindow.

You must run `pip install es-aces` first.
"""
from pathlib import Path
from tempfile import TemporaryDirectory

import polars as pl
import rootutils

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)
from tests.helpers.run_sh_command import run_command

RANDOM_LOCAL_WINDOWS = """
predicates:
    event:
        code: _IGNORE

trigger: _ANY_EVENT

windows:
    pre:
        start: null
        end: trigger
        start_inclusive: True
        end_inclusive: False
    post:
        start: pre.end
        end: null
        start_inclusive: True
        end_inclusive: True
"""

with TemporaryDirectory() as tmp_path:
    meds_dir = "tests/test_data/MEDS_cohort"
    raw_windows_path = "tests/test_data/raw_windows.parquet"
    aces_task_cfg_path = Path(tmp_path) / "aces_config.yaml"

    aces_task_cfg_path.write_text(RANDOM_LOCAL_WINDOWS)
    aces_kwargs = {
        "data.path": str((Path(meds_dir) / "*/*.parquet").resolve()),
        "data.standard": "meds",
        "cohort_dir": str(aces_task_cfg_path.parent.resolve()),
        "cohort_name": "aces_config",
        "output_filepath": raw_windows_path,
        "hydra.verbose": True,
    }

    run_command("aces-cli", aces_kwargs, "aces-cli")

    meds_df = pl.read_parquet(str((Path(meds_dir) / "*/*.parquet").resolve()))
    # TODO: ACES is duplicating these outputs should be fixed
    aces_df = pl.read_parquet(raw_windows_path).unique()

    number_of_unique_event_times = meds_df.unique(["patient_id", "time"]).drop_nulls("time").shape[0]
    assert number_of_unique_event_times == aces_df.shape[0]
    aces_df.write_parquet(raw_windows_path)
