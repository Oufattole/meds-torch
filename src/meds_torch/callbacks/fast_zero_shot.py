import subprocess
import tempfile
from pathlib import Path
import gc

import polars as pl
import torch
from lightning.pytorch.callbacks import Callback

from meds_torch.utils import RankedLogger

log = RankedLogger(__name__, rank_zero_only=True)


def launch_eval(BEST_CHECKPOINT, OUTPUT_DIR):
    ROOT_DIR = "/storage/shared/mimic-iv/meds_v0.3.2/"
    MEDS_DIR = f"{ROOT_DIR}/meds/"
    TENSOR_DIR = f"{ROOT_DIR}/eic_tensors"
    script = "notebooks/window_pretrain_eval/window_metric.py"
    overrides = [
        "experiment=v2_64_gptneox_histogram_forecast_auto_6L",
        f"paths.window_results_dir={OUTPUT_DIR}",
        "every_n_patients=40",
        "data.subsequence_sampling_strategy=from_start",
        "data.max_seq_len=1152",
        "data.dataloader.batch_size=512",
        "data.predict_dataset=val",
        "data.do_include_subject_id=true",
        "data.do_include_prediction_time=true",
        "data.do_include_end_time=true",
        f"paths.meds_cohort_dir={MEDS_DIR}",
        f"ckpt_path={BEST_CHECKPOINT}",
        f"paths.data_dir={TENSOR_DIR}",
        f"paths.output_dir={OUTPUT_DIR}",
        f"paths.generated_trajectory_fp={OUTPUT_DIR}/baseline_pretrain_eval.parquet",
        "hydra.searchpath=[pkg://meds_torch.configs,/home/nassim/projects/histogram_forecasting/ZERO_SHOT_TUTORIAL/configs/]",
    ]
    cmd = ["python", script] + overrides
    log.debug("Running:\n" + " ".join(cmd))
    command_out = subprocess.run(" ".join(cmd), shell=True, capture_output=True)
    if command_out.returncode != 0:
        log.error(f"Command failed with return code {command_out.returncode}.")
        log.error(f"Command stdout:\n{command_out.stdout.decode()}")
        log.error(f"Command stderr:\n{command_out.stderr.decode()}")
        raise ValueError(f"Command failed with return code {command_out.returncode}.")
    else:
        log.debug(f"Command stdout:\n{command_out.stdout.decode()}")


class FastZeroShot(Callback):
    def on_validation_epoch_end(self, trainer, pl_module):
        # 1) remember where we were
        device = pl_module.device

        # 2) move the model off GPU
        pl_module.cpu()

        # 3) free up PyTorch's cache
        torch.cuda.empty_cache()
        gc.collect()   # optionally force a Python GC

        # 5) Compute Zero shot metric
        with tempfile.TemporaryDirectory() as tmp_dir:
            ckpt_fp = Path(tmp_dir) / "model.ckpt"
            torch.save(pl_module, str(ckpt_fp))
            launch_eval(str(ckpt_fp), Path(tmp_dir) / "output")
            df = pl.read_parquet(
                Path(tmp_dir) / "output/eval/baseline/window_lag128_input_length1024_40.parquet"
            )
            df = df.with_columns(
                pl.mean_horizontal(
                    [
                        "abnormal_creatinine/auroc",
                        "abnormal_hematocrit/auroc",
                        "abnormal_hemoglobin/auroc",
                        "abnormal_leukocytes/auroc",
                        "abnormal_platets/auroc",
                        "mortality/auroc",
                        "hospital_admission/auroc",
                        "hospital_discharge/auroc",
                    ]
                ).alias("mean/task/auroc")
            )
            metrics = {f"FAST_ZERO_SHOT/val/{col}": df[col][0] for col in df.columns}
            for key, value in metrics.items():
                pl_module.log(
                    key,
                    value,
                    on_step=False,  # must be False in epoch-end hooks
                    on_epoch=True,  # emit once per validation run
                    logger=True,  # ensure it goes to your WandBLogger
                )
        # 6) (if you plan to continue training) move it back
        pl_module.to(device)
