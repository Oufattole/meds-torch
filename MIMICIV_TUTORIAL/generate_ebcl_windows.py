from dataclasses import dataclass
from pathlib import Path
from tempfile import TemporaryDirectory

import hydra
import rootutils
from hydra.core.config_store import ConfigStore

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


@dataclass
class Config:
    meds_dir: str
    output_path: str
    subsample_fraction: float


cs = ConfigStore.instance()
cs.store(name="config", node=Config)


@hydra.main(config_path=None, config_name="config")
def main(cfg: Config):
    with TemporaryDirectory() as tmp_path:
        meds_dir = cfg.meds_dir
        output_path = cfg.output_path
        raw_windows_path = output_path
        aces_task_cfg_path = Path(tmp_path) / "aces_config.yaml"

        aces_task_cfg_path.write_text(RANDOM_LOCAL_WINDOWS)
        aces_kwargs = {
            "data.path": str((Path(meds_dir) / "**" / "[0-9]*.parquet").resolve()),
            "data.standard": "meds",
            "cohort_dir": str(aces_task_cfg_path.parent.resolve()),
            "cohort_name": "aces_config",
            "output_filepath": raw_windows_path,
            "hydra.verbose": True,
        }

        run_command("aces-cli", [], aces_kwargs, "aces-cli")

        # meds_df = pl.read_parquet(str((Path(meds_dir) / "*/*.parquet").resolve()))

        # # Perform group-based subsampling
        # if cfg.subsample_fraction < 1.0:
        #     sampled_patients = (
        #         meds_df.select("patient_id")
        #         .unique()
        #         .sample(fraction=cfg.subsample_fraction)
        #     )
        #     meds_df = meds_df.join(sampled_patients, on="patient_id")

        # # TODO: It is a known issue that ACES duplicated windows
        # # See https://github.com/justin13601/ACES/issues/73
        # aces_df = pl.read_parquet(raw_windows_path).unique()

        # number_of_unique_event_times = (
        #     meds_df.unique(["patient_id", "time"]).drop_nulls("time").shape[0]
        # )
        # assert number_of_unique_event_times == aces_df.shape[0]
        # aces_df.write_parquet(raw_windows_path)


if __name__ == "__main__":
    main()
