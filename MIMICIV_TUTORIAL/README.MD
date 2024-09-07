# MIMIC-IV Example

This is an example of how to extract a MEDS dataset from MIMIC-IV. All scripts in this README are assumed to
be run **not** from this directory but from the root directory of this entire repository (e.g., one directory
up from this one).

## Extract MIMIC-IV MEDS Data

### Option 1: Download pre-extracted data from gpc

Run the following command to download the MEDS data from the gcp bucket:

```console
export MIMICIV_MEDS_DIR=??? # set to the directory in which you want to store the raw MIMIC-IV data

cd $MIMICIV_MEDS_DIR
gcloud storage cp gs://ehr_standardization_schema/MEDS_Extract_v0.0.7_test.zip meds_extract_0.0.7_data.zip
unzip meds_extract_0.0.7_data.zip
rm meds_extract_0.0.7_data.zip
```

```console
conda create -n meds-torch python=3.12
conda activate meds-torch
pip install "meds-torch==0.0.3"
```

### Option 2: Download MIMIC-IV and locally extract via MEDS-Transform

Alternatively take the raw MIMIC-IV data and perform the following extraction yourself. Follow the MEDS-Transforms tutorial
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

Now extract supervised task labels uses aces. First install aces:

```console
conda create -n aces python=3.12
conda activate aces
pip install es-aces==0.5.0
pip install hydra-joblib-launcher
```

Second, run the following command to extract the supervised task labels:

```console
TASK_NAME="mortality-in_hospital-first_48h" # set name of the task
TASKS_DIR="$MIMICIV_MEDS_DIR/tasks/" # set to the directory in which you want to store all tasks
mkdir -p "$TASKS_DIR" # create a directory for the task
cp MIMICIV_TUTORIAL/configs/${TASK_NAME}.yaml $TASKS_DIR/${TASK_NAME}.yaml
aces-cli --multirun hydra/launcher=joblib data=sharded data.standard=meds data.root="$MIMICIV_MEDS_DIR/data" "data.shard=$(expand_shards $MIMICIV_MEDS_DIR/data)" cohort_dir="$TASKS_DIR" cohort_name="$TASK_NAME"
```

Now we can launch a meds-torch experiment. The `eic_mtr.yaml` experiment will require around `26GB` of GPU memory and takes ~30min per epoch.

```console
export MIMICIV_MEDS_DIR=??? # set to the directory in which you want to store the raw MIMIC-IV data
export MIMICIV_EIC_DIR=??? # set to the directory in which you want to output the tensorized MIMIC-IV data
export OUTPUT_DIR=??? # set to the directory in which you want to output checkpoints and results
TASK_NAME="mortality-in_hospital-first_48h" # set name of the task
TASKS_DIR="$MIMICIV_MEDS_DIR/tasks/" # set to the directory in which you want to store all tasks
export CUDA_VISIBLE_DEVICES=0 # set to the GPU you want to use
conda activate meds-torch

MAX_POLARS_THREADS=4 meds-torch-train experiment=eic_mtr paths.data_dir="${MIMICIV_EIC_DIR}" \
    paths.meds_cohort_dir="${MIMICIV_MEDS_DIR}" paths.output_dir="${OUTPUT_DIR}" \
    data.task_name="$TASK_NAME" data.task_root_dir="$TASKS_DIR" \
    data.task_label_path="$TASKS_DIR/${TASK_NAME}/**/*.parquet" \
    hydra.searchpath=[pkg://meds_torch.configs,"$(pwd)/MIMICIV_TUTORIAL/configs/meds-torch-configs"]
```

Now we can launch a meds-torch experiment. The `triplet_mtr.yaml` experiment will require around `26GB` of GPU memory and takes ~30min per epoch on a single V100 GPU.

```console
export MIMICIV_MEDS_DIR=??? # set to the directory in which you want to store the raw MIMIC-IV data
export MIMICIV_TRIPLET_DIR=??? # set to the directory in which you want to output the tensorized MIMIC-IV data
export OUTPUT_DIR=??? # set to the directory in which you want to output checkpoints and results
TASK_NAME="mortality-in_hospital-first_48h" # set name of the task
TASKS_DIR="$MIMICIV_MEDS_DIR/tasks/" # set to the directory in which you want to store all tasks
export CUDA_VISIBLE_DEVICES=1 # set to the GPU you want to use
conda activate meds-torch

MAX_POLARS_THREADS=4 meds-torch-train experiment=triplet_mtr paths.data_dir="${MIMICIV_TRIPLET_DIR}" \
    paths.meds_cohort_dir="${MIMICIV_MEDS_DIR}" paths.output_dir="${OUTPUT_DIR}" \
    data.task_name="$TASK_NAME" data.task_root_dir="$TASKS_DIR" \
    data.task_label_path="$TASKS_DIR/${TASK_NAME}/**/*.parquet" \
    hydra.searchpath=[pkg://meds_torch.configs,"$(pwd)/MIMICIV_TUTORIAL/configs/meds-torch-configs"]
```

Now we can launch a meds-torch experiment. The `text_code_mtr.yaml` experiment will require around `26GB` of GPU memory and takes ~47min per epoch.

```console
export MIMICIV_MEDS_DIR=??? # set to the directory in which you want to store the raw MIMIC-IV data
export MIMICIV_TRIPLET_DIR=??? # set to the directory in which you want to output the tensorized MIMIC-IV data
export OUTPUT_DIR=??? # set to the directory in which you want to output checkpoints and results
TASK_NAME="mortality-in_hospital-first_48h" # set name of the task
TASKS_DIR="$MIMICIV_MEDS_DIR/tasks/" # set to the directory in which you want to store all tasks
export CUDA_VISIBLE_DEVICES=2 # set to the GPU you want to use
conda activate meds-torch

MAX_POLARS_THREADS=4 meds-torch-train experiment=text_code_mtr paths.data_dir="${MIMICIV_TRIPLET_DIR}" \
    paths.meds_cohort_dir="${MIMICIV_MEDS_DIR}" paths.output_dir="${OUTPUT_DIR}" \
    data.task_name="$TASK_NAME" data.task_root_dir="$TASKS_DIR" \
    data.task_label_path="$TASKS_DIR/${TASK_NAME}/**/*.parquet" \
    hydra.searchpath=[pkg://meds_torch.configs,"$(pwd)/MIMICIV_TUTORIAL/configs/meds-torch-configs"]
```
