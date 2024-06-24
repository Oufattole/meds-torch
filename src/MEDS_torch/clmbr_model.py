from pathlib import Path
from MEDS_pytorch_dataset.pytorch_dataset import PytorchDataset
from omegaconf import OmegaConf

cfg_fp = Path("configs/pytorch_dataset.yaml")
cfg = OmegaConf.load(cfg_fp)

cfg.raw_MEDS_cohort_dir = (
    "/n/data1/hms/dbmi/zaklab/inovalon_mbm47/processed/06-12-24_cohorts/t2d/MEDS"
)
cfg.MEDS_cohort_dir = (
    "/n/data1/hms/dbmi/zaklab/inovalon_mbm47/processed/06-12-24_cohorts/t2d/CITPP"
)
cfg.max_seq_len = 512

pyd = PytorchDataset(cfg, split="tuning")

# Get an item:
item = pyd[0]

# Get a batch:
batch = pyd.collate([pyd[i] for i in range(5)])