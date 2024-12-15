# MIMIC-IV Example

This is an example of how to extract a MEDS dataset from MIMIC-IV. To see a faster demo on smaller public medical datasets with meds-torch and other meds community models, please go to this [MEDS-DEV demo page](https://github.com/mmcdermott/MEDS-DEV/blob/demo_independent_data_generation/demo/README.MD).

All scripts in this README are assumed to
be run **not** from this directory but from the root directory of this entire repository (e.g., one directory
up from this one).

## Extract MIMIC-IV MEDS Data

Take the raw MIMIC-IV data and perform the following extraction yourself. Follow the MEDS-Transforms tutorial
[here](https://github.com/mmcdermott/MEDS_transforms/blob/main/MIMIC-IV_Example/README.md).

## Install MEDS-Torch

Either install the package from PyPI:

```console
pip install meds-torch
```

or clone the repository and install the package from source:

```console
git clone git@github.com:Oufattole/meds-torch.git
cd meds-torch
pip install -e .
```

## Pre-Processing for Sequence Modeling

We support two tensorization scripts corresponding to `triplet` and `eic` tokenization.

Run the following end to end script to generate triplet tensors

```console
export MIMICIV_MEDS_DIR=??? # set to the directory in which you want to store the MIMIC-IV MEDS data
export MIMICIV_TRIPLET_TENSOR_DIR=??? # set to the directory in which you want to output the tensorized MIMIC-IV data
export N_PARALLEL_WORKERS=8 # set to the number of parallel workers you want to use
export PIPELINE_CONFIG_PATH="$(pwd)/MIMICIV_INDUCTIVE_EXPERIMENTS/configs/triplet_config.yaml" # set to the directory in which the config file is stored, must be an absolute path.
export JOBLIB_RUNNER_CONFIG_PATH="$(pwd)/MIMICIV_INDUCTIVE_EXPERIMENTS/configs/joblib_runner.yaml" # set to the directory in which the config file is stored, must be an absolute path.


bash MIMICIV_INDUCTIVE_EXPERIMENTS/tokenize.sh $MIMICIV_MEDS_DIR $MIMICIV_TRIPLET_TENSOR_DIR $N_PARALLEL_WORKERS $PIPELINE_CONFIG_PATH stage_runner_fp=$JOBLIB_RUNNER_CONFIG_PATH
```

Run the following end to end script to generate eic tensors

```console
export MIMICIV_MEDS_DIR=??? # set to the directory in which you want to store the raw MIMIC-IV data
export MIMICIV_EIC_DIR=??? # set to the directory in which you want to output the tensorized MIMIC-IV data
export N_PARALLEL_WORKERS=8 # set to the number of parallel workers you want to use
export PIPELINE_CONFIG_PATH="$(pwd)/MIMICIV_INDUCTIVE_EXPERIMENTS/configs/eic_config.yaml" # set to the directory in which the config file is stored, must be an absolute path.
export JOBLIB_RUNNER_CONFIG_PATH="$(pwd)/MIMICIV_INDUCTIVE_EXPERIMENTS/configs/joblib_runner.yaml" # set to the directory in which the config file is stored, must be an absolute path.

bash MIMICIV_INDUCTIVE_EXPERIMENTS/tokenize.sh $MIMICIV_MEDS_DIR $MIMICIV_EIC_DIR $N_PARALLEL_WORKERS $PIPELINE_CONFIG_PATH stage_runner_fp=$JOBLIB_RUNNER_CONFIG_PATH
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
    aces-cli --multirun hydra/launcher=joblib data=sharded data.standard=meds data.root="$MIMICIV_MEDS_DIR/data" "data.shard=$(expand_shards $MIMICIV_MEDS_DIR/data)" cohort_dir="${MIMICIV_MEDS_DIR}/tasks/" cohort_name="$TASK_NAME"
done
```

## Running a Single Task with MEDS-Torch

Training a model is simple with MEDS-Torch. Here's an example of how to train a supervised model for the `mortality/in_hospital/first_24h` task using the triplet tokenization method. All a user must do is input the paths to the tensorized data (cached patient time series data), the cohort directory (directory of the raw MEDS data), and the output directory (where results are stored). The evaluation is automatically performed at the end of the training process.

First, let's set up our environment variables:

```console
ROOT_DIR="/path/to/your/root/directory"  # Replace with your actual root directory
TENSOR_DIR="${ROOT_DIR}/triplet_tensors/"
MEDS_DIR="${ROOT_DIR}/meds/"
TASKS_DIR="${MEDS_DIR}/tasks/"
OUTPUT_DIR="${ROOT_DIR}/results/triplet_mtr/mortality/in_hospital/first_24h/"
TRAIN_DIR="${OUTPUT_DIR}/supervised/train/"
```

These variables define the key directories for our MEDS-Torch workflow:

- `ROOT_DIR`: The base directory for your MEDS-Torch project
- `TENSOR_DIR`: Where the tensorized (preprocessed) patient data is stored
- `MEDS_DIR`: Location of the raw MEDS data
- `TASKS_DIR`: Directory containing task parquet files following the [MEDS label_schema](https://github.com/Medical-Event-Data-Standard/meds).
- `OUTPUT_DIR`: Where the results will be stored
- `TRAIN_DIR`: Specific directory for training output

Next, activate your conda environment:

```console
conda activate meds-torch
```

Now you're ready to train your model. The `meds-torch-train` command takes care of the entire training process, including evaluation:

```console
meds-torch-train \
    experiment=triplet_mtr \
    paths.data_dir=${TENSOR_DIR} \
    paths.meds_cohort_dir=${MEDS_DIR} \
    paths.output_dir=${TRAIN_DIR} \
    data.task_name=mortality/in_hospital/first_24h \
    data.task_root_dir=${TASKS_DIR} \
    hydra.searchpath=[pkg://meds_torch.configs,./MIMICIV_INDUCTIVE_EXPERIMENTS/configs/meds-torch-configs]
```

This command specifies:

- The experiment type (`triplet_mtr`): This loads the `triplet_mtr.yaml` configuration file, which overrides default arguments (in the `train.yaml`) with settings specific to the triplet model training recipe.
- Paths to the tensorized data, MEDS cohort, and output directory
- The specific task name and its root directory
- The search path for configuration files

The `meds-torch-train` command will train the model and automatically evaluate it at the end of the training process. The results, including evaluation metrics, will be saved in the specified output directory.

With these steps, you can easily train a MEDS-Torch model for a single task, with evaluation included as part of the process.

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
TRANSFER_LEARNING_METHOD=??? # set to the transfer learning method you want to use, choose one from "eic_forecasting" "triplet_forecasting"

bash MIMICIV_INDUCTIVE_EXPERIMENTS/launch_ar_pretrain.sh $MIMICIV_ROOT_DIR $CONDA_ENV_NAME $TRANSFER_LEARNING_METHOD

bash MIMICIV_INDUCTIVE_EXPERIMENTS/launch_ar_finetune.sh $MIMICIV_ROOT_DIR $CONDA_ENV_NAME $TRANSFER_LEARNING_METHOD
```

To run all experiments in one go:

The following script will train a supervised model for all three tokenization methods (Triplet, EIC, and Text Code) on all four tasks.

- `mortality/in_hospital/first_24h`
- `mortality/in_icu/first_24h`
- `mortality/post_hospital_discharge/1y`
- `readmission/30d`

Note this will be 12 experiments in total.

```console
# Launch Supervised experiments
bash MIMICIV_INDUCTIVE_EXPERIMENTS/launch_supervised.sh $MIMICIV_ROOT_DIR meds-torch
```

We then pretrain all transfer learning models (EBCL, OCP, Value Forecasting, and Autoregressive Forecasting) for all three tokenization methods (Triplet, EIC, and Text Code) on the full dataset.
Note that this will be 12 experiments in total.

```console
# Launch pretraining
bash MIMICIV_INDUCTIVE_EXPERIMENTS/launch_multi_window_pretrain.sh $MIMICIV_ROOT_DIR meds-torch ocp
bash MIMICIV_INDUCTIVE_EXPERIMENTS/launch_multi_window_pretrain.sh $MIMICIV_ROOT_DIR meds-torch ebcl
bash MIMICIV_INDUCTIVE_EXPERIMENTS/launch_multi_window_pretrain.sh $MIMICIV_ROOT_DIR meds-torch value_forecasting
bash MIMICIV_INDUCTIVE_EXPERIMENTS/launch_ar_pretrain.sh $MIMICIV_ROOT_DIR meds-torch
```

Finally we finetune all transfer learning models (EBCL, OCP, Value Forecasting, and Autoregressive Forecasting) on the four tasks.
Note that this will be 16 experiments in total.

```console
#Launch Finetuning
bash MIMICIV_INDUCTIVE_EXPERIMENTS/launch_finetune.sh $MIMICIV_ROOT_DIR meds-torch ocp
bash MIMICIV_INDUCTIVE_EXPERIMENTS/launch_finetune.sh $MIMICIV_ROOT_DIR meds-torch ebcl
bash MIMICIV_INDUCTIVE_EXPERIMENTS/launch_finetune.sh $MIMICIV_ROOT_DIR meds-torch value_forecasting

bash MIMICIV_INDUCTIVE_EXPERIMENTS/launch_ar_finetune.sh $MIMICIV_ROOT_DIR meds-torch meds-torch
```

Results will be stored in the `${MIMICIV_ROOT_DIR}/results` directory. You can view the latest results using the following python code snippet:

```python
import os
import pandas as pd
import polars as pl
from datetime import datetime

# Specify the directory to search
search_dir = ??? # set to the ${MIMICIV_ROOT_DIR}/results directory

def find_files(directory, filename):
    found_files = []
    for root, dirs, files in os.walk(directory):
        if filename in files:
            found_files.append(os.path.join(root, filename))
    return found_files

def extract_info_from_path(file_path):
    parts = file_path.split('/')
    task_name = "/".join(parts[8:-4])
    method = parts[6]

    # Extract the full timestamp
    timestamp_str = parts[-2]
    date = datetime.strptime(timestamp_str, '%Y-%m-%d_%H-%M-%S_%f')
    print(file_path)
    match file_path:
        case _ if "eic_mtr" in file_path:
            tokenization_strategy = "eic"
        case _ if "triplet_mtr" in file_path:
            tokenization_strategy = "triplet"
        case _ if "triplet_forecast_mtr" in file_path:
            tokenization_strategy = "triplet"
        case _ if "eic_forecast_mtr" in file_path:
            tokenization_strategy = "eic"
        case _:
            tokenization_strategy = "unknown"

    return task_name, method, date, tokenization_strategy

def process_files(search_dir, target_file):
    results = find_files(search_dir, target_file)
    data = []

    for file_path in results:
        df = pl.read_parquet(file_path)
        if 'test/auc' in df.columns and "multiseed" in file_path:
            task_name, method, date, tokenization_strategy = extract_info_from_path(file_path)
            auc_mean, auc_std = df['test/auc'].mean()*100, df['test/auc'].std()*100
            result = f"{auc_mean:.1f} ± {auc_std:.1f}"
            data.append({
                'task_name': task_name,
                'method': method,
                'date': date,
                'tokenization_strategy': tokenization_strategy,
                'result': result
            })

    return pd.DataFrame(data)

# Specify the filename to search for
target_file = 'sweep_results_summary.parquet'

# Process the files and create the DataFrame
results_df = process_files(search_dir, target_file)

# filter to tasks of interest
tasks_of_interest = [
    "mortality/in_hospital/first_24h",
    "mortality/in_icu/first_24h",
    "mortality/post_hospital_discharge/1y",
    "readmission/30d",
]
results_df = results_df[results_df['task_name'].isin(tasks_of_interest)]

# Rename the methods
method_mapping = {
    'eic_forecasting': 'AR forecasting',
    'triplet_forecasting': 'AR forecasting'
}
results_df['method'] = results_df['method'].replace(method_mapping)

# Display the results
results_df.sort_values(["task_name", "date"]).groupby(['task_name', "method", "tokenization_strategy"]).last()
```

This table presents the results of various tokenization methods across different tasks for the MIMIC-IV dataset. The results are reported as Mean ± Standard Deviation of the Area Under the Curve (AUC) over five trials. The table compares different methods (Contrastive, Forecasting, and Supervised) with various types and tokenization strategies (EIC, Text Code, and Triplet) across four key tasks in medical risk prediction.

- **hosp-Mortality** (`mortality/in_hospital/first_24h`): Prediction of in-hospital mortality given data up to the end of the first 24 hours of hospital observations.
- **ICU-Mortality** (`mortality/in_icu/first_24h`): Prediction of in-ICU mortality given data up to the end of the first 24 hours of ICU observations.
- **Discharge-Mortality** (`mortality/post_hospital_discharge/1y`): Prediction of whether a subject will survive 1 year after hospital discharge.
- **Readmission** (`readmission/30d`): Prediction of whether a subject is readmitted within 30 days after hospital discharge.

| Method      | Type           | Tokenization | hosp-Mortality | ICU-Mortality | Discharge-Mortality | Readmission |
| ----------- | -------------- | ------------ | -------------- | ------------- | ------------------- | ----------- |
| Contrastive | OCP            | EIC          | 86.1 ± 0.3     | 77.2 ± 0.4    | 84.1 ± 0.1          | 70.3 ± 0.3  |
|             |                | Text Code    | 80.0 ± 0.5     | 69.3 ± 1.6    | 81.6 ± 0.1          | 68.3 ± 0.3  |
|             |                | Triplet      | 87.3 ± 0.4     | 75.8 ± 0.4    | 85.6 ± 0.1          | 71.5 ± 0.2  |
|             | LCL            | EIC          | 86.1 ± 0.7     | 76.6 ± 0.6    | 84.0 ± 0.0          | 70.6 ± 0.2  |
|             |                | Text Code    | 85.2 ± 0.3     | 74.9 ± 0.9    | 83.9 ± 0.3          | 70.1 ± 0.3  |
|             |                | Triplet      | 86.5 ± 0.2     | 77.4 ± 0.7    | 85.6 ± 0.1          | 71.8 ± 0.1  |
| Forecasting | Autoregressive | EIC          | 84.8 ± 0.1     | 76.9 ± 0.5    | 83.6 ± 0.3          | 69.9 ± 0.2  |
|             |                | Text Code    | 81.9 ± 0.4     | 71.6 ± 1.3    | 81.9 ± 0.4          | 69.1 ± 0.3  |
|             |                | Triplet      | 74.0 ± 0.6     | 65.1 ± 2.2    | 76.0 ± 1.8          | 64.4 ± 0.5  |
|             | Value          | EIC          | 86.1 ± 0.6     | 77.8 ± 1.0    | 84.0 ± 0.2          | 70.2 ± 0.6  |
|             |                | Text Code    | 85.2 ± 0.2     | 75.8 ± 0.6    | 85.8 ± 0.1          | 72.1 ± 0.1  |
|             |                | Triplet      | 88.2 ± 0.1     | 77.3 ± 0.8    | 86.2 ± 0.1          | 72.1 ± 0.1  |
| Supervised  | -              | EIC          | 86.4 ± 0.2     | 77.9 ± 0.4    | 84.0 ± 0.1          | 70.0 ± 0.1  |
|             |                | Text Code    | 85.3 ± 0.3     | 75.5 ± 0.7    | 85.0 ± 0.3          | 70.8 ± 0.3  |
|             |                | Triplet      | 87.1 ± 0.9     | 77.4 ± 1.1    | 85.4 ± 0.2          | 71.5 ± 0.2  |

We analyze the performance of different tokenization strategies across all four tasks and all methods:

<img src="https://github.com/Oufattole/meds-torch/blob/182e282e87aa75b1523f0526f62635a36ce92593/docs/assets/overall_win_rate.png?raw=true" alt="Tokenization Strategy Bar Plot" style="height: 300px;">

This bar plot illustrates the overall win rates for each tokenization strategy, aggregated across all tasks and methods. The Triplet tokenization demonstrates superior performance, consistently outperforming both EIC and Text Code. This suggests that Triplet tokenization may be more effective in capturing and representing the underlying structure of the data across a wide range of scenarios.

To gain deeper insights, we also examine the performance of each tokenization strategy across different methods:

![Method Bar Plot](https://github.com/Oufattole/meds-torch/blob/182e282e87aa75b1523f0526f62635a36ce92593/docs/assets/grouped_win_rate.png?raw=true)

This grouped bar plot breaks down the win rates by method, aggregated across tasks and normalized for each method. We observe that triplet is more effective for contrastive and supervised learning methods, but EIC performs better for forecasting methods. This suggests that the choice of tokenization strategy should be tailored to the specific learning method and task at hand.
