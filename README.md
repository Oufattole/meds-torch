# MEDS PyTorch Dataset

A template PyTorch dataset for structured data in the MEDS format. Best paired with
[MEDS polars functions](https://github.com/mmcdermott/MEDS_polars_functions) to generate tensorized and
tokenized data.

## Installation

```bash
git clone https://github.com/mmcdermott/MEDS_pytorch_dataset.git
pip install .
```

## Usage

TODO: details

Example:

```python
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
```

## Design Philosophy

This repo currently supports a "direct retireval" style of data ingestion for deep-learning training. This
mode has the data needed for any given PyTorch Dataset `__getitem__` call retrieved from disk on an as-needed
basis. This mode is extremely scalable, because the entire dataset never need be loaded or stored in memory in
its entirety. When done properly, retrieving data from disk can be done in a manner that is independent of the
total dataset size as well, thereby rendering the load time similarly unconstrained by total dataset size.
This mode is also extremely flexible, because different cohorts can be loaded from the same base dataset
simply by changing which patients and what offsets within patient data are read on any given cohort, all
without changing the base files or underlying code. However, this mode does require ragged dataset collation
which can be more resource intensive than pre-batched iteration, so it is slower than the "Fixed-batch
retrieval" approach. This mode is what is currently supported by this repository.
