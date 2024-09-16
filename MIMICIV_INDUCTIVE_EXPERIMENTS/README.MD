# MIMIC-IV Example

This is an example of how to extract a MEDS dataset from MIMIC-IV. All scripts in this README are assumed to
be run **not** from this directory but from the root directory of this entire repository (e.g., one directory
up from this one).

## Extract MIMIC-IV MEDS Data

Take the raw MIMIC-IV data and perform the following extraction yourself. Follow the MEDS-Transforms tutorial
[here](https://github.com/mmcdermott/MEDS_transforms/blob/main/MIMIC-IV_Example/README.md):

## Pre-Processing for Sequence Modeling

We support two tensorization scripts corresponding to `triplet` and `eic` tokenization.

Run the following end to end script to generate triplet tensors

```console
export MIMICIV_MEDS_DIR=??? # set to the directory in which you want to store the MIMIC-IV MEDS data
export MIMICIV_TRIPLET_TENSOR_DIR=??? # set to the directory in which you want to output the tensorized MIMIC-IV data
export N_PARALLEL_WORKERS=8 # set to the number of parallel workers you want to use
export PIPELINE_CONFIG_PATH="$(pwd)/MIMICIV_TUTORIAL/configs/triplet_config.yaml" # set to the directory in which the config file is stored, must be an absolute path.
export JOBLIB_RUNNER_CONFIG_PATH="$(pwd)/MIMICIV_TUTORIAL/configs/joblib_runner.yaml" # set to the directory in which the config file is stored, must be an absolute path.


bash MIMICIV_TUTORIAL/tokenize.sh $MIMICIV_MEDS_DIR $MIMICIV_TRIPLET_TENSOR_DIR $N_PARALLEL_WORKERS $PIPELINE_CONFIG_PATH stage_runner_fp=$JOBLIB_RUNNER_CONFIG_PATH
```

Run the following end to end script to generate eic tensors

```console
export MIMICIV_MEDS_DIR=??? # set to the directory in which you want to store the raw MIMIC-IV data
export MIMICIV_EIC_DIR=??? # set to the directory in which you want to output the tensorized MIMIC-IV data
export N_PARALLEL_WORKERS=8 # set to the number of parallel workers you want to use
export PIPELINE_CONFIG_PATH="$(pwd)/MIMICIV_TUTORIAL/configs/eic_config.yaml" # set to the directory in which the config file is stored, must be an absolute path.
export JOBLIB_RUNNER_CONFIG_PATH="$(pwd)/MIMICIV_TUTORIAL/configs/joblib_runner.yaml" # set to the directory in which the config file is stored, must be an absolute path.

bash MIMICIV_TUTORIAL/tokenize.sh $MIMICIV_MEDS_DIR $MIMICIV_EIC_DIR $N_PARALLEL_WORKERS $PIPELINE_CONFIG_PATH stage_runner_fp=$JOBLIB_RUNNER_CONFIG_PATH
```

## Task Extraction

Next we need to get some labels for our tasks. We will use the `long_los` task as an example.

There are two options:

### Use ACES to extract labels using a task config definition:

We can manually extract the supervised task labels from our meds dataset using [aces](https://github.com/justin13601/ACES/tree/main). First install aces:

```console
conda create -n aces python=3.12
conda activate aces
pip install es-aces==0.5.0
pip install hydra-joblib-launcher
```

Second, run the following command to extract the supervised task labels:

```console
TASKS=(
    "mortality/in_hospital/first_24h"
    "mortality/in_icu/first_24h"
    "mortality/post_hospital_discharge/1y"
    "readmission/30d"
)
for TASK_NAME in "${TASKS[@]}"; do
    SINGLE_TASK_DIR="${MIMICIV_MEDS_DIR}/tasks/${TASK_NAME}"
    mkdir -p $SINGLE_TASK_DIR # create a directory for the task
    cp MIMICIV_INDUCTIVE_EXPERIMENTS/configs/tasks/${TASK_NAME}.yaml "${SINGLE_TASK_DIR}.yaml"
    aces-cli --multirun hydra/launcher=joblib data=sharded data.standard=meds data.root="$MIMICIV_MEDS_DIR/data" "data.shard=$(expand_shards $MIMICIV_MEDS_DIR/data)" cohort_dir="$TASKS_DIR" cohort_name="$TASK_NAME"
done
```

## Launching Tokenization Inductive Experiments

Run Supervised Learning Experiments:

```console
export MIMICIV_ROOT_DIR=??? # set to the parent directory of the meds folder
export CONDA_ENV_NAME=meds-torch # set to the name of the conda environment you want to use
bash MIMICIV_INDUCTIVE_EXPERIMENTS/launch_supervised.sh $MIMICIV_ROOT_DIR meds-torch $CONDA_ENV_NAME
```

Or run Transfer Learning Experiments:

```console
TRANSFER_LEARNING_METHOD=??? # set to the transfer learning method you want to use, choose one from "ebcl" "ocp" "value_forecasting"

bash MIMICIV_INDUCTIVE_EXPERIMENTS/launch_multi_window_pretrain.sh $MIMICIV_ROOT_DIR $CONDA_ENV_NAME $TRANSFER_LEARNING_METHOD

bash MIMICIV_INDUCTIVE_EXPERIMENTS/launch_finetune.sh $MIMICIV_ROOT_DIR $CONDA_ENV_NAME $TRANSFER_LEARNING_METHOD
```

```console
TRANSFER_LEARNING_METHOD=??? # set to the transfer learning method you want to use, choose one from "token_forecasting" "triplet_forecasting"

bash MIMICIV_INDUCTIVE_EXPERIMENTS/launch_pretrain.sh $MIMICIV_ROOT_DIR $CONDA_ENV_NAME $TRANSFER_LEARNING_METHOD

bash MIMICIV_INDUCTIVE_EXPERIMENTS/launch_finetune.sh $MIMICIV_ROOT_DIR $CONDA_ENV_NAME $TRANSFER_LEARNING_METHOD
```

Results will be stored in the `$MIMICIV_ROOT_DIR/results` directory. Specifically the \`
