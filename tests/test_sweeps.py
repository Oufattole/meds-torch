# TODO: finish the sweep tests
import os
from pathlib import Path

import pytest
import wandb

from tests.conftest import SUPERVISED_TASK_NAME
from tests.helpers.run_if import RunIf
from tests.helpers.run_sh_command import run_command

startfile = "src/meds_torch/train.py"
os.environ["WANDB_MODE"] = "offline"
wandb.init(mode="offline")


# @RunIf(sh=True)
# @pytest.mark.slow
# def test_experiments(tmp_path: Path) -> None:
#     """Test running all available experiment configs with `fast_dev_run=True.`

#     :param tmp_path: The temporary logging path.
#     """
#     overrides = {
#         "hydra.sweep.dir": str(tmp_path),
#         "trainer.fast_dev_run": "true",
#         "data.task_name": SUPERVISED_TASK_NAME,
#         "logger": "[]",
#         "paths.data_dir": "tests/test_data",
#         "data.dataloader.batch_size": 32
#     }

#     script = startfile

#     # Run the command
#     run_command(script, hydra_kwargs=overrides, test_name="experiment")


# @RunIf(sh=True)
# @pytest.mark.slow
# @pytest.mark.parametrize("lr", ["0.005", "0.01"])
# @pytest.mark.parametrize("model", ["supervised", "ebcl", "ocp", "token_forecasting", "value_forecasting"])
# def test_hydra_sweep(tmp_path: Path, model: str, lr: str) -> None:
#     """Test default hydra sweep.

#     :param tmp_path: The temporary logging path.
#     """
#     overrides = {
#         "hydra.sweep.dir": str(tmp_path),
#         "trainer.fast_dev_run": "true",
#         "data.task_name": SUPERVISED_TASK_NAME,
#         "logger": "[]",
#         "paths.data_dir": "tests/test_data",
#         "data.dataloader.batch_size": "16",
#         "model": model,
#         "model.optimizer.lr": lr
#     }

#     script = startfile + ' -m'

#     # Run the command
#     run_command(script, hydra_kwargs=overrides, test_name="Hydra sweep")


# @RunIf(sh=True)
# @pytest.mark.slow
# @pytest.mark.parametrize("lr", ["0.005", "0.01", "0.02"])
# @pytest.mark.parametrize("model", ["supervised", "ebcl", "ocp", "token_forecasting", "value_forecasting"])
# def test_hydra_sweep_ddp_sim(tmp_path: Path, model: str, lr: str) -> None:
#     """Test default hydra sweep with ddp sim.

#     :param tmp_path: The temporary logging path.
#     """
#     overrides = {
#         "hydra.sweep.dir": str(tmp_path),
#         "trainer": "ddp_sim",
#         "trainer.max_epochs": "3",
#         "+trainer.limit_train_batches": "0.01",
#         "+trainer.limit_val_batches": "0.1",
#         "+trainer.limit_test_batches": "0.1",
#         "data.task_name": SUPERVISED_TASK_NAME,
#         "logger": "[]",
#         "paths.data_dir": "tests/test_data",
#         "model": model,
#         "model.optimizer.lr": lr
#     }

#     script = startfile + ' -m'

#     # Run the command
#     run_command(script, hydra_kwargs=overrides, test_name="Hydra sweep with ddp sim")


# @RunIf(sh=True)
# @pytest.mark.slow
# @pytest.mark.parametrize("model", ["supervised", "ebcl", "ocp", "token_forecasting", "value_forecasting"])
# def test_optuna_sweep(tmp_path: Path, model: str) -> None:
#     """Test Optuna hyperparam sweeping.

#     :param tmp_path: The temporary logging path.
#     """
#     overrides = {
#         "hydra.sweep.dir": str(tmp_path),
#         "hparams_search": "optuna",
#         "hydra.sweeper.n_trials": "10",
#         "hydra.sweeper.sampler.n_startup_trials": "5",
#         "trainer.fast_dev_run": "true",
#         "data.task_name": SUPERVISED_TASK_NAME,
#         "logger": "[]",
#         "paths.data_dir": "tests/test_data",
#         "model": model
#     }

#     # Run the command
#     run_command(startfile, hydra_kwargs=overrides, test_name="Optuna sweep")


@RunIf(wandb=True, sh=True)
@pytest.mark.slow
@pytest.mark.parametrize("model", ["supervised", "ebcl", "ocp", "token_forecasting", "value_forecasting"])
def test_optuna_sweep_ddp_sim_wandb(tmp_path: Path, model: str) -> None:
    """Test Optuna sweep with wandb logging and ddp sim.

    :param tmp_path: The temporary logging path.
    """
    overrides = {
        "hydra.sweep.dir": str(tmp_path),
        "hparams_search": "optuna",
        "hydra.sweeper.n_trials": "5",
        "trainer": "ddp_sim",
        "trainer.max_epochs": "3",
        "+trainer.limit_train_batches": "0.01",
        "+trainer.limit_val_batches": "0.1",
        "+trainer.limit_test_batches": "0.1",
        "data.task_name": SUPERVISED_TASK_NAME,
        "logger": "wandb",
        "paths.data_dir": "tests/test_data",
        "model": model,
        "model.net.lin1_size": "256",
    }

    # Run the command
    run_command(startfile, hydra_kwargs=overrides, test_name="Optuna sweep with wandb and ddp sim")
