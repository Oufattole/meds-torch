import rootutils

root = rootutils.setup_root(__file__, dotenv=True, pythonpath=True, cwd=True)

import shutil
from datetime import datetime
from pathlib import Path

import polars as pl
import torch
from hydra import compose, initialize

from meds_torch import embedder
from meds_torch.pytorch_dataset import PytorchDataset


def test_event_stream(tmp_path):
    MEDS_cohort_dir = tmp_path / "processed" / "final_cohort"
    shutil.copytree(Path("./tests/test_data"), MEDS_cohort_dir.parent)

    kwargs = {
        "raw_MEDS_cohort_dir": str(MEDS_cohort_dir.parent.resolve()),
        "MEDS_cohort_dir": str(MEDS_cohort_dir.resolve()),
        "max_seq_len": 512,
        "model.embedder.token_dim": 4,
        "collate_type": "event_stream",
    }

    with initialize(version_base=None, config_path="../src/meds_torch/configs"):  # path to config.yaml
        overrides = [f"{k}={v}" for k, v in kwargs.items()]
        cfg = compose(config_name="pytorch_dataset", overrides=overrides)  # config.yaml

    pyd = PytorchDataset(cfg, split="train")
    item = pyd[0]
    assert item.keys() == {"static_indices", "static_values", "dynamic"}
    batch = pyd.collate([pyd[i] for i in range(2)])
    assert batch.keys() == {
        "event_mask",
        "dynamic_values_mask",
        "time_delta_days",
        "dynamic_indices",
        "dynamic_values",
        "static_indices",
        "static_values",
    }


def test_triplet(tmp_path):
    MEDS_cohort_dir = tmp_path / "processed" / "final_cohort"
    shutil.copytree(Path("./tests/test_data"), MEDS_cohort_dir.parent)

    kwargs = {
        "raw_MEDS_cohort_dir": str(MEDS_cohort_dir.parent.resolve()),
        "MEDS_cohort_dir": str(MEDS_cohort_dir.resolve()),
        "max_seq_len": 512,
        "model.embedder.token_dim": 4,
        "collate_type": "triplet",
    }

    with initialize(version_base=None, config_path="../src/meds_torch/configs"):  # path to config.yaml
        overrides = [f"{k}={v}" for k, v in kwargs.items()]
        cfg = compose(config_name="pytorch_dataset", overrides=overrides)  # config.yaml

    # triplet collating
    pyd = PytorchDataset(cfg, split="train")
    batch = pyd.collate([pyd[i] for i in range(2)])
    assert batch.keys() == {
        "mask",
        "static_mask",
        "code",
        "numerical_value",
        "time_delta_days",
        "numerical_value_mask",
    }
    for key in batch.keys():
        assert batch[key].shape == torch.Size([2, 17])

    # check the continuous value embedder works
    data = torch.ones((2, 3), dtype=torch.float32)
    cve_embedding = embedder.CVE(cfg).forward(data[None, :].T).permute(1, 2, 0)
    # Output should have shape B x D x T
    assert cve_embedding.shape == torch.Size([2, 4, 3])
    model = embedder.ObservationEmbedder(cfg)
    embedding, mask = model.embed(batch)

    # Check that the embedding is shape batch size x embedding dimension size x sequence length
    assert embedding.shape == torch.Size([2, 4, 17])
    # Check that the mask is shape batch size x sequence length
    assert mask.shape == torch.Size([2, 17])


def test_task(tmp_path):
    MEDS_cohort_dir = tmp_path / "processed" / "final_cohort"
    shutil.copytree(Path("./tests/test_data"), MEDS_cohort_dir.parent)

    task_name = "bababooey"
    kwargs = {
        "raw_MEDS_cohort_dir": str(MEDS_cohort_dir.parent.resolve()),
        "MEDS_cohort_dir": str(MEDS_cohort_dir.resolve()),
        "max_seq_len": 512,
        "model.embedder.token_dim": 4,
        "collate_type": "triplet",
        "task_name": task_name,
    }

    task_df = pl.DataFrame(
        {
            "patient_id": [239684, 1195293, 68729, 814703],
            "start_time": [
                datetime(1980, 12, 28),
                datetime(1978, 6, 20),
                datetime(1978, 3, 9),
                datetime(1976, 3, 28),
            ],
            "end_time": [
                datetime(2010, 5, 11, 18, 25, 35),
                datetime(2010, 6, 20, 20, 12, 31),
                datetime(2010, 5, 26, 2, 30, 56),
                datetime(2010, 2, 5, 5, 55, 39),
            ],
            task_name: [0, 1, 0, 1],
        }
    )
    tasks_dir = MEDS_cohort_dir / "tasks"
    tasks_dir.mkdir(parents=True, exist_ok=True)
    task_df.write_parquet(tasks_dir / f"{task_name}.parquet")

    with initialize(version_base=None, config_path="../src/meds_torch/configs"):  # path to config.yaml
        overrides = [f"{k}={v}" for k, v in kwargs.items()]
        cfg = compose(config_name="pytorch_dataset", overrides=overrides)  # config.yaml

    # triplet collating
    pyd = PytorchDataset(cfg, split="train")
    batch = pyd.collate([pyd[i] for i in range(2)])
    assert batch.keys() == {
        "mask",
        "static_mask",
        "code",
        "numerical_value",
        "time_delta_days",
        "numerical_value_mask",
        task_name,
    }
    for key in batch.keys() - {task_name}:
        assert batch[key].shape == torch.Size([2, 10])
    assert batch[task_name].shape == torch.Size([2])
