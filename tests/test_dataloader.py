import rootutils

root = rootutils.setup_root(__file__, dotenv=True, pythonpath=True, cwd=True)

import shutil
from pathlib import Path

from hydra import compose, initialize

from MEDS_torch.pytorch_dataset import PytorchDataset


def test_event_stream(tmp_path):
    MEDS_cohort_dir = tmp_path / "processed" / "final_cohort"
    shutil.copytree(Path("./tests/test_data"), MEDS_cohort_dir.parent)

    kwargs = {
        "raw_MEDS_cohort_dir": str(MEDS_cohort_dir.parent.resolve()),
        "MEDS_cohort_dir": str(MEDS_cohort_dir.resolve()),
        "max_seq_len": 512,
        "embedder.token_dim": 4,
        "collate_type": "event_stream",
    }

    with initialize(version_base=None, config_path="../src/MEDS_torch/configs"):  # path to config.yaml
        overrides = [f"{k}={v}" for k, v in kwargs.items()]
        cfg = compose(config_name="pytorch_dataset", overrides=overrides)  # config.yaml

    pyd = PytorchDataset(cfg, split="train")
    item = pyd[0]
    batch = pyd.collate([pyd[i] for i in range(2)])


def test_triplet(tmp_path):
    MEDS_cohort_dir = tmp_path / "processed" / "final_cohort"
    shutil.copytree(Path("./tests/test_data"), MEDS_cohort_dir.parent)

    kwargs = {
        "raw_MEDS_cohort_dir": str(MEDS_cohort_dir.parent.resolve()),
        "MEDS_cohort_dir": str(MEDS_cohort_dir.resolve()),
        "max_seq_len": 512,
        "embedder.token_dim": 4,
        "collate_type": "triplet",
    }

    with initialize(version_base=None, config_path="../src/MEDS_torch/configs"):  # path to config.yaml
        overrides = [f"{k}={v}" for k, v in kwargs.items()]
        cfg = compose(config_name="pytorch_dataset", overrides=overrides)  # config.yaml

    # event stream collating
    pyd = PytorchDataset(cfg, split="train")
    item = pyd[0]
    batch = pyd.collate([pyd[i] for i in range(2)])
    # observation level collating

    # from MEDS_torch import embedder
    # data = torch.ones((2, 3), dtype=torch.float32)
    # cve_embedding = embedder.CVE(cfg).forward(data[None, :].T).permute(1, 2, 0)
    # # Output should have shape B x D x T
    # assert cve_embedding.shape == torch.Size([2, 4, 3])
    # model = embedder.ObservationEmbedder(cfg)
    # # embedding = model.embed(batch)
