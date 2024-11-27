#!/usr/bin/env python

"""Prep MEDS-ready CSV File

This script stores a CSV file for MIMIC-IV ECGs that can be read directly by MEDS.
We read in record_list.csv provided by MIMIC-IV then convert the ECG record path
to an absolute path on stultzlab03. Additionally, we provide a function to read the
raw ECG data to a numpy array. MIMIC-IV ECGs are 12-lead, 10 seconds long, and sampled
at 500Hz; the resulting array is of shape (12, 10*500).

Important caveats to MIMIC-IV ECG data:

1. "Many of the provided diagnostic ECGs overlap with a MIMIC-IV hospital or emergency department stay
    but a number of them do not overlap. Approximately 55% of the ECGs overlap with a hospital admission
    and 25% overlap with an emergency department visit."

2. "The date and time for each ECG were recorded by the machine's internal clock, which in most cases
    was not synchronized with any external time source. As a result, the ECG time stamps could be
    significantly out of sync with the corresponding time stamps in the MIMIC-IV Clinical Database,
    MIMIC-IV Waveform Database, or other modules in MIMIC-IV."

"""

from pathlib import Path

import hydra
import polars as pl
import torch
import wfdb
from MEDS_transforms.utils import hydra_loguru_init
from omegaconf import DictConfig


class WaveformReader:
    def read_modality(self, relative_modality_fp: str) -> torch.Tensor:
        """
        Convert a WFDB (Waveform Database) record to a NumPy array.

        Parameters:
        ----------
        relative_modality_fp : str
            relative Path to the WFDB record file. For example:
            'files/p1000/p10000635/s45386375/45386375'

        Returns:
        -------
        numpy.ndarray
            A 2D NumPy array containing the signal data from the WFDB record,
            shape (T, 12). Each row corresponds to a sample point in time and
            each column corresponds to a different ECG lead.
        """
        data = wfdb.rdrecord(Path(self.base_path) / relative_modality_fp)
        return data.p_signal


@hydra.main(version_base=None, config_path="configs", config_name="pre_MEDS")
def main(cfg: DictConfig):
    """Performs pre-MEDS data wrangling for MIMIC-IV.

    Inputs are the raw MIMIC files, read from the `input_dir` config parameter. Output files are either
    symlinked (if they are not modified) or written in processed form to the `MEDS_input_dir` config
    parameter. Hydra is used to manage configuration parameters and logging.
    """

    hydra_loguru_init()

    input_dir = Path(cfg.input_dir)
    MEDS_input_dir = Path(cfg.cohort_dir)
    records = pl.read_csv(input_dir / "record_list.csv")["subject_id", "ecg_time", "path"]
    records.write_parquet(MEDS_input_dir / "record_list.parquet")


if __name__ == "__main__":
    main()
