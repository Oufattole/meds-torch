# MEDS-torch

<p align="center">
  <a href="https://pytorch.org/get-started/locally/"><img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-ee4c2c?logo=pytorch&logoColor=white"></a>
  <a href="https://pytorchlightning.ai/"><img alt="Lightning" src="https://img.shields.io/badge/-Lightning-792ee5?logo=pytorchlightning&logoColor=white"></a>
  <a href="https://hydra.cc/"><img alt="Config: Hydra" src="https://img.shields.io/badge/Config-Hydra-89b8cd"></a>
  <a href="https://github.com/ashleve/lightning-hydra-template"><img alt="Template" src="https://img.shields.io/badge/-Lightning--Hydra--Template-017F2F?style=flat&logo=github&labelColor=gray"></a><br>
  <a href="https://www.python.org/downloads/release/python-3100/"><img alt="Python" src="https://img.shields.io/badge/-Python_3.12+-blue?logo=python&logoColor=white"></a>
  <a href="https://pypi.org/project/meds-torch/"><img alt="PyPI" src="https://img.shields.io/badge/PyPI-v0.0.1a1-blue?logoColor=blue"></a>
  <a href="https://hydra.cc/"><img alt="Hydra" src="https://img.shields.io/badge/Config-Hydra_1.3-89b8cd"></a>
  <a href="https://codecov.io/github/Oufattole/meds-torch"><img src="https://codecov.io/github/Oufattole/meds-torch/graph/badge.svg?token=BV119L5JQJ"/></a>
  <a href="https://github.com/Oufattole/meds-torch/actions/workflows/tests.yaml"><img alt="Tests" src="https://github.com/Oufattole/meds-torch/actions/workflows/tests.yaml/badge.svg"></a>
  <a href="https://github.com/Oufattole/meds-torch/actions/workflows/code-quality-main.yaml"><img alt="Code Quality" src="https://github.com/Oufattole/meds-torch/actions/workflows/code-quality-main.yaml/badge.svg"></a>
  <a href="https://github.com/Oufattole/meds-torch/graphs/contributors"><img alt="Contributors" src="https://img.shields.io/github/contributors/oufattole/meds-torch.svg"></a>
  <a href="https://github.com/Oufattole/meds-torch/pulls"><img alt="Pull Requests" src="https://img.shields.io/badge/PRs-welcome-brightgreen.svg"></a>
  <a href="https://github.com/Oufattole/meds-torch#license"><img alt="License" src="https://img.shields.io/badge/License-MIT-green.svg?labelColor=gray"></a>
</p>

<!--   <a href='https://meds-torch.readthedocs.io/en/latest/?badge=latest'><img src='https://readthedocs.org/projects/meds-torch/badge/?version=latest' alt='Documentation Status' /></a> -->

## Description

This repository provides a flexible suite for advanced machine learning over Electronic Health Records (EHR) using PyTorch, PyTorch Lightning, and Hydra for configuration management. The project ingests tensorized data from the [MEDS_transforms](<>) repository, a robust system for transforming EHR data into ML ready sequence data. By employing a variety of tokenization strategies and sequence model architectures, this framework facilitates the development and testing of models that can perform.

Key features include:

- **Configurable ML Pipeline**: Utilize Hydra to dynamically adjust configurations and seamlessly integrate with PyTorch Lightning for scalable training across multiple environments.
- **Advanced Tokenization Techniques**: Explore different approaches to embedding EHR data in tokens that sequence model can reason over.
- **Supervised Models**: Support for supervised training on arbitrary tasks defined on MEDS format data.
- **Transfer Learning**: Pretrain via contrastive learning, forecasting, and other pre-training methods, and finetune to supervised tasks.

The goal of this project is to push the boundaries of what's possible in healthcare machine learning by providing a flexible, robust, and scalable sequence model tools that accommodate a wide range of research and operational needs. Whether you're conducting academic research or developing clinical applications with MEDS format EHR data, this repository offers tools and flexibility to develop deep sequence models.

## Installation

#### Pip

**PyPi**

```bash
pip install meds-torch
```

**git**

```bash
# clone project
git clone git@github.com:Oufattole/meds-torch.git
cd meds-torch

# [OPTIONAL] create conda environment
conda create -n meds-torch python=3.12
conda activate meds-torch

# install pytorch according to instructions
# https://pytorch.org/get-started/

# install requirements
pip install -e .
```

## How to run

Train model with default configuration

```bash
# train on CPU
python -m meds_torch.train trainer=cpu

# train on GPU
python -m meds_torch.train trainer=gpu
```

Train model with chosen experiment configuration from [configs/experiment/](configs/experiment/)

```bash
python -m meds_torch.train experiment=experiment_name.yaml
```

You can override any parameter from command line like this

```bash
python -m meds_torch.train trainer.max_epochs=20 data.batch_size=64
```

## 📌  Introduction

**Why you might want to use it:**

✅ Support different tokenization methods for EHR data <br>

- Triplet
- Everything Is text
- Everything Is a code

✅ MEDS data Supervised Learning and Transfer Learning Support <br>

- randomly initialize a model and train it in a supervised maner on your MEDS format medical data.
- General Contrastive window Pretraining
- Random [EBCL](https://arxiv.org/abs/2312.10308) Example
- [OCP](https://arxiv.org/abs/2111.02599) Example
- [STraTS](https://arxiv.org/abs/2107.14293) Value Forecasting

✅ Ease of Use and Reusability <br>
Collection of useful EHR sequence modeling tools, configs, and code snippets. You can use this repo as a reference for developing your own models. Additionally you can easily add new models, datasets, tasks, experiments, and train on different accelerators, like multi-GPU.

## Loggers

By default `wandb` logger is installed with the repo. Please install a different logger below if you wish to use it:

```console
pip install neptune-client
pip install mlflow
pip install comet-ml
pip install aim>=3.16.2  # no lower than 3.16.2, see https://github.com/aimhubio/aim/issues/2550
```

## Development Help

To run tests on 8 parallel workers run:

```bash
pytest -n 8
```
