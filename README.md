# MEDS-torch: Advanced Machine Learning for Electronic Health Records

<a href="https://pytorch.org/get-started/locally/"><img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-ee4c2c?logo=pytorch&logoColor=white"></a>
<a href="https://pytorchlightning.ai/"><img alt="Lightning" src="https://img.shields.io/badge/-Lightning-792ee5?logo=pytorchlightning&logoColor=white"></a>
<a href="https://hydra.cc/"><img alt="Config: Hydra" src="https://img.shields.io/badge/Config-Hydra-89b8cd"></a>
<a href="https://github.com/ashleve/lightning-hydra-template"><img alt="Template" src="https://img.shields.io/badge/-Lightning--Hydra--Template-017F2F?style=flat&logo=github&labelColor=gray"></a>
<a href="https://www.python.org/downloads/release/python-3100/"><img alt="Python" src="https://img.shields.io/badge/-Python_3.12+-blue?logo=python&logoColor=white"></a>
<a href="https://pypi.org/project/meds-torch/"><img alt="PyPI" src="https://img.shields.io/badge/PyPI-v0.0.1a1-blue?logoColor=blue"></a>
<a href="https://hydra.cc/"><img alt="Hydra" src="https://img.shields.io/badge/Config-Hydra_1.3-89b8cd"></a>
<a href="https://codecov.io/github/Oufattole/meds-torch"><img src="https://codecov.io/github/Oufattole/meds-torch/graph/badge.svg?token=BV119L5JQJ"/></a>
<a href="https://github.com/Oufattole/meds-torch/actions/workflows/tests.yaml"><img alt="Tests" src="https://github.com/Oufattole/meds-torch/actions/workflows/tests.yaml/badge.svg"></a>
<a href="https://github.com/Oufattole/meds-torch/actions/workflows/code-quality-main.yaml"><img alt="Code Quality" src="https://github.com/Oufattole/meds-torch/actions/workflows/code-quality-main.yaml/badge.svg"></a>
<a href="https://github.com/Oufattole/meds-torch/graphs/contributors"><img alt="Contributors" src="https://img.shields.io/github/contributors/oufattole/meds-torch.svg"></a>
<a href="https://github.com/Oufattole/meds-torch/pulls"><img alt="Pull Requests" src="https://img.shields.io/badge/PRs-welcome-brightgreen.svg"></a>
<a href="https://github.com/Oufattole/meds-torch#license"><img alt="License" src="https://img.shields.io/badge/License-MIT-green.svg?labelColor=gray"></a>
<a href='https://meds-torch.readthedocs.io/en/latest/?badge=latest'><img src='https://readthedocs.org/projects/meds-torch/badge/?version=latest' alt='Documentation Status' /></a>

## 🚀 Quick Start

### Installation

```bash
pip install meds-torch
```

### Set up environment variables

```bash
# Define data paths
PATHS_KWARGS="paths.data_dir=/CACHED/NESTED/RAGGED/TENSORS/DIR paths.meds_cohort_dir=/PATH/TO/MEDS/DATA/ paths.output_dir=/OUTPUT/RESULTS/DIRECTORY"

# Define task parameters (for supervised learning)
TASK_KWARGS="data.task_name=NAME_OF_TASK data.task_root_dir=/PATH/TO/TASK/LABELS/"
```

### Basic Usage

1. Train a supervised model (GPU)

```bash
meds-torch-train trainer=gpu $PATHS_KWARGS $TASK_KWARGS
```

2. Pretrain an autoregressive forecasting model (GPU)

```bash
meds-torch-train trainer=gpu $PATHS_KWARGS model=eic_forecasting
```

3. Train with a specific experiment configuration

```bash
meds-torch-train experiment=experiment.yaml $PATHS_KWARGS $TASK_KWARGS hydra.searchpath=[pkg://meds_torch.configs,/PATH/TO/CUSTOM/CONFIGS]
```

4. Override parameters

```bash
meds-torch-train trainer.max_epochs=20 data.batch_size=64 $PATHS_KWARGS $TASK_KWARGS
```

5. Hyperparameter search

```bash
meds-torch-tune trainer=ray callbacks=tune_default hparams_search=ray_tune experiment=triplet_mtr $PATHS_KWARGS $TASK_KWARGS hydra.searchpath=[pkg://meds_torch.configs,/PATH/TO/CUSTOM/CONFIGS/WITH/experiment/triplet_mtr]
```

### Advanced Examples

For detailed examples and tutorials:

- Check `MIMICIV_INDUCTIVE_EXPERIMENTS/README.md` for a comprehensive guide to using MEDS-torch with MIMIC-IV data, including data preparation, task extraction, and running experiments with different tokenization and transfer learning methods.
- See `ZERO_SHOT_TUTORIAL/README.md` for a rough WIP walkthrough of zero-shot prediction (and please share feedback on improving this! 🙂)

### Example Experiment Configuration

Here's a sample `experiment.yaml`:

```yaml
# @package _global_

defaults:
  - override /data: pytorch_dataset
  - override /logger: wandb
  - override /model/backbone: triplet_transformer_encoder
  - override /model/input_encoder: triplet_encoder
  - override /model: supervised
  - override /trainer: gpu

tags: [mimiciv, triplet, transformer_encoder]

seed: 0

trainer:
  min_epochs: 1
  max_epochs: 10
  gradient_clip_val: 1.0

data:
  dataloader:
    batch_size: 64
    num_workers: 6
  max_seq_len: 128
  collate_type: triplet
  subsequence_sampling_strategy: to_end

model:
  token_dim: 128
  optimizer:
    lr: 0.001
  backbone:
    n_layers: 2
    nheads: 4
    dropout: 0

logger:
  wandb:
    tags: ${tags}
    group: mimiciv_tokenization
```

This configuration sets up a supervised learning experiment using a triplet transformer encoder on MIMIC-IV data. Modify this file to suit your specific needs.

## 🌟 Key Features

- **Flexible ML Pipeline**: Utilizes Hydra for dynamic configuration and PyTorch Lightning for scalable training.
- **Advanced Tokenization**: Supports multiple strategies for embedding EHR data (Triplet, Text Code, Everything In Code).
- **Supervised Learning**: Train models on arbitrary tasks defined in MEDS format data.
- **Transfer Learning**: Pretrain models using contrastive learning, forecasting, and other methods, then finetune for specific tasks.
- **Multiple Pretraining Methods**: Supports EBCL, OCP, STraTS Value Forecasting, and Autoregressive Observation Forecasting.

## 🛠 Installation

### PyPI

```bash
pip install meds-torch
```

### From Source

```bash
git clone git@github.com:Oufattole/meds-torch.git
cd meds-torch
pip install -e .
```

## 📚 Documentation

For detailed usage instructions, API reference, and examples, visit our [documentation](https://meds-torch.readthedocs.io/).

For a comprehensive demo of our pipeline and to see results from a suite of inductive experiments comparing different tokenization methods and learning approaches, please refer to the `MIMICIV_INDUCTIVE_EXPERIMENTS/README.MD` file. This document provides detailed scripts and performance metrics.

## 🧪 Running Experiments

### Supervised Learning

```bash
bash MIMICIV_INDUCTIVE_EXPERIMENTS/launch_supervised.sh $MIMICIV_ROOT_DIR meds-torch
```

### Transfer Learning

```bash
# Pretraining
bash MIMICIV_INDUCTIVE_EXPERIMENTS/launch_multi_window_pretrain.sh $MIMICIV_ROOT_DIR meds-torch [METHOD]
bash MIMICIV_INDUCTIVE_EXPERIMENTS/launch_ar_pretrain.sh $MIMICIV_ROOT_DIR meds-torch [AR_METHOD]

# Finetuning
bash MIMICIV_INDUCTIVE_EXPERIMENTS/launch_finetune.sh $MIMICIV_ROOT_DIR meds-torch [METHOD]
bash MIMICIV_INDUCTIVE_EXPERIMENTS/launch_ar_finetune.sh $MIMICIV_ROOT_DIR meds-torch [AR_METHOD]
```

Replace `[METHOD]` with one of the following:

- `ocp` (Observation Contrastive Pretraining)
- `ebcl` (Event-Based Contrastive Learning)
- `value_forecasting` (STraTS Value Forecasting)

Replace `[AR_METHOD]` with one of the following:

- `eic_forecasting` (Everything In Code Forecasting)
- `triplet_forecasting` (Triplet Forecasting)

These scripts allow you to run various experiments, including supervised learning, different pretraining methods, and finetuning for both standard and autoregressive models.

## 📞 Support

For questions, issues, or feature requests, please open an issue on our [GitHub repository](https://github.com/Oufattole/meds-torch/issues).

______________________________________________________________________

MEDS-torch: Advancing healthcare machine learning through flexible, robust, and scalable sequence modeling tools.
