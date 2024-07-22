# MEDS-torch

<p align="center">
  <a href="https://pytorch.org/get-started/locally/"><img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-ee4c2c?logo=pytorch&logoColor=white"></a>
  <a href="https://pytorchlightning.ai/"><img alt="Lightning" src="https://img.shields.io/badge/-Lightning-792ee5?logo=pytorchlightning&logoColor=white"></a>
  <a href="https://hydra.cc/"><img alt="Config: Hydra" src="https://img.shields.io/badge/Config-Hydra-89b8cd"></a>
  <a href="https://github.com/ashleve/lightning-hydra-template"><img alt="Template" src="https://img.shields.io/badge/-Lightning--Hydra--Template-017F2F?style=flat&logo=github&labelColor=gray"></a><br>
  <a href="https://www.python.org/downloads/release/python-3100/"><img alt="Python" src="https://img.shields.io/badge/-Python_3.12+-blue?logo=python&logoColor=white"></a>
  <a href="https://pypi.org/project/meds-torch/"><img alt="PyPI" src="https://img.shields.io/badge/PyPI-v0.2.5-orange?logoColor=orange"></a>
  <a href="https://hydra.cc/"><img alt="Hydra" src="https://img.shields.io/badge/Config-Hydra_1.3-89b8cd"></a>
  <a href="https://codecov.io/gh/oufattole/meds-torch"><img alt="Codecov" src="https://codecov.io/gh/mmcdermott/MEDS_Tabular_AutoML/graph/badge.svg?token=6GD05EDQ39"></a>
  <a href="https://github.com/Oufattole/meds-torch/actions/workflows/tests.yaml"><img alt="Tests" src="https://github.com/Oufattole/meds-torch/actions/workflows/tests.yaml/badge.svg"></a>
  <a href="https://github.com/Oufattole/meds-torch/actions/workflows/code-quality-main.yaml"><img alt="Code Quality" src="https://github.com/Oufattole/meds-torch/actions/workflows/code-quality-main.yaml/badge.svg"></a>
  <a href='https://meds-torch.readthedocs.io/en/latest/?badge=latest'><img src='https://readthedocs.org/projects/meds-torch/badge/?version=latest' alt='Documentation Status' /></a>
  <a href="https://github.com/Oufattole/meds-torch/graphs/contributors"><img alt="Contributors" src="https://img.shields.io/github/contributors/mmcdermott/MEDS_Tabular_AutoML.svg"></a>
  <a href="https://github.com/Oufattole/meds-torch/pulls"><img alt="Pull Requests" src="https://img.shields.io/badge/PRs-welcome-brightgreen.svg"></a>
  <a href="https://github.com/Oufattole/meds-torch#license"><img alt="License" src="https://img.shields.io/badge/License-MIT-green.svg?labelColor=gray"></a>
</p>
<!-- [![Paper](http://img.shields.io/badge/paper-arxiv.1001.2234-B31B1B.svg)](https://www.nature.com/articles/nature14539)
[![Conference](http://img.shields.io/badge/AnyConference-year-4b44ce.svg)](https://papers.nips.cc/paper/2020) -->

</div>

## Description

This repository provides a comprehensive suite for advanced machine learning over Electronic Health Records (EHR) using PyTorch, PyTorch Lightning, and Hydra for configuration management. The project leverages MEDS-Tab, a robust system for transforming EHR data into a structured, tabular format that enhances the accessibility and analyzability of medical datasets. By employing a variety of tokenization strategies and neural network architectures, this framework facilitates the development and testing of models that can predict, generate, and understand complex medical trajectories.

Key features include:

- **Configurable ML Pipeline**: Utilize Hydra to dynamically adjust configurations and seamlessly integrate with PyTorch Lightning for scalable training across multiple environments.
- **Advanced Tokenization Techniques**: Explore different approaches to processing EHR data, such as triplet tokenization and code-specific embeddings, to capture the nuances of medical information.
- **Pre-training Strategies**: Leverage contrastive learning, autoregressive token forecasting, and other pre-training techniques to boost model performance with MEDS data.
- **Transfer Learning**: Implement and test transfer learning scenarios to adapt pre-trained models to new tasks or datasets effectively.
- **Generative and Supervised Models**: Support for zero-shot generative models and supervised training allows for a broad application of the framework in predictive and generative tasks within healthcare.

The goal of this project is to push the boundaries of what's possible in healthcare machine learning by providing a flexible, robust, and scalable platform that accommodates a wide range of research and operational needs. Whether you're conducting academic research, developing clinical applications, or exploring new machine learning methodologies, this repository offers the tools and flexibility needed to innovate and excel in the field of medical data analysis.

## Installation

#### Pip

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

✅ Save on boilerplate <br>
Easily add new models, datasets, tasks, experiments, and train on different accelerators, like multi-GPU, TPU or SLURM clusters.

✅ Support different tokenization methods for EHR data <br>

- [ ] [Triplet Tokenization](https://github.com/Oufattole/meds-torch/issues/1) -- add to read the docs explanations of each subtype
- [ ] [Everything is text](https://github.com/Oufattole/meds-torch/issues/12) -- add to read the docs explanations of each subtype
- [ ] Everything is a code TODO -- add to read the docs explanations of each subtype

✅ MEDS data pretraining (and Transfer Learning Support) <br>

- [ ] General Contrastive window Pretraining
  - [ ] Random [EBCL](https://arxiv.org/abs/2312.10308) Example
  - [ ] [OCP](https://arxiv.org/abs/2111.02599) Example
- [ ] [STraTS](https://arxiv.org/abs/2107.14293) Value Forecasting
- [ ] Autoregressive Token Forecasting
- [ ] Token Masked Imputation

✅ Zero shot Generative Model Support <br>

- [ ] Allow support for generating meds format future trajectories for patients using the Autoregressive Token Forecasting.

✅ Supervised Model Support <br>
Load pretrained model weights (TODO) or [randomly initialize](https://github.com/Oufattole/meds-torch/issues/2) a model and train it in a supervised maner on you MEDS format medical data.

✅ Education <br>
Thoroughly commented. You can use this repo as a learning resource.

✅ Reusability <br>
Collection of useful MLOps tools, configs, and code snippets. You can use this repo as a reference for various utilities.

**Why you might not want to use it:**

❌ Things break from time to time <br>
Lightning and Hydra are still evolving and integrate many libraries, which means sometimes things break. For the list of currently known problems visit [this page](https://github.com/ashleve/lightning-hydra-template/labels/bug).

❌ Not adjusted for data engineering <br>
Template is not really adjusted for building data pipelines that depend on each other. It's more efficient to use it for model prototyping on ready-to-use data.

❌ Overfitted to simple use case <br>
The configuration setup is built with simple lightning training in mind. You might need to put some effort to adjust it for different use cases, e.g. lightning fabric.

❌ Might not support your workflow <br>
For example, you can't resume hydra-based multirun or hyperparameter search.

## Loggers

By default `wandb` logger is installed with the repo. Please install a different logger below if you wish to use it:

```python
# neptune-client
# mlflow
# comet-ml
# aim>=3.16.2  # no lower than 3.16.2, see https://github.com/aimhubio/aim/issues/2550
```

## Development Help

pytest-instafail shows failures and errors instantly instead of waiting until the end of test session, run it with:

```bash
pytest --instafail
```

To run failing tests continuously each time you edit code until they pass:

```bash
pytest --looponfail
```

To run tests on 8 parallel workers run:

```bash
pytest -n 8
```
