# Evaluating an Autoregressive EHR Model

```console
ROOT_DIR="/storage/shared/mimic-iv/meds_v0.3.2/"  # Replace with your actual root directory
```

We first need to tensorize your meds data (cache it in a format such that meds-torch can efficiently train models on the dataset). We can tensorize a low vocabulary dataset centered around 5 common lab codes in MIMIC IV by running the tokenize script with the `eic_top_10.yaml`, which performs EIC tokenization on these common codes:

```console
export MIMICIV_MEDS_DIR=${ROOT_DIR}/meds/ # set to the directory in which you want to store the raw MIMIC-IV data
export MIMICIV_EIC_DIR=${ROOT_DIR}/eic_tensors/ # set to the directory in which you want to output the tensorized MIMIC-IV data
export N_PARALLEL_WORKERS=8 # set to the number of parallel workers you want to use
export PIPELINE_CONFIG_PATH="$(pwd)/ZERO_SHOT_TUTORIAL/configs/eic_config.yaml" # set to the directory in which the config file is stored, must be an absolute path.
export JOBLIB_RUNNER_CONFIG_PATH="$(pwd)/ZERO_SHOT_TUTORIAL/configs/joblib_runner.yaml" # set to the directory in which the config file is stored, must be an absolute path.

bash ZERO_SHOT_TUTORIAL/tokenize.sh $MIMICIV_MEDS_DIR $MIMICIV_EIC_DIR $N_PARALLEL_WORKERS $PIPELINE_CONFIG_PATH stage_runner_fp=$JOBLIB_RUNNER_CONFIG_PATH
```

Alternatively we can tensorize the full dataset:

```console
export MIMICIV_MEDS_DIR=${ROOT_DIR}/meds/ # set to the directory in which you want to store the raw MIMIC-IV data
export MIMICIV_EIC_DIR=${ROOT_DIR}/eic_top10_tensors/ # set to the directory in which you want to output the tensorized MIMIC-IV data
export N_PARALLEL_WORKERS=8 # set to the number of parallel workers you want to use
export PIPELINE_CONFIG_PATH="$(pwd)/ZERO_SHOT_TUTORIAL/configs/eic_top_10.yaml" # set to the directory in which the config file is stored, must be an absolute path.
export JOBLIB_RUNNER_CONFIG_PATH="$(pwd)/ZERO_SHOT_TUTORIAL/configs/joblib_runner.yaml" # set to the directory in which the config file is stored, must be an absolute path.

bash ZERO_SHOT_TUTORIAL/tokenize.sh $MIMICIV_MEDS_DIR $MIMICIV_EIC_DIR $N_PARALLEL_WORKERS $PIPELINE_CONFIG_PATH stage_runner_fp=$JOBLIB_RUNNER_CONFIG_PATH
```

Use the following code to pull a suite of abnormal lab value tasks:

```python
import polars as pl
from pathlib import Path

ROOT_DIR = (
    "/storage/shared/mimic-iv/meds_v0.3.2/"  # Replace with your actual root directory
)

lab_to_codes = {}
pl.Config.set_fmt_str_lengths(100)
df = pl.read_parquet(f"{ROOT_DIR}/meds/metadata/codes.parquet")
creatinine_codes = df.filter(
    pl.col("description").str.contains(
        "Creatinine [Mass/volume] in Blood", literal=True
    )
    | pl.col("description").str.contains(
        "Creatinine [Mass/volume] in Serum or Plasma", literal=True
    )
)["code"].to_list()
lab_to_codes["creatinine"] = "|".join(creatinine_codes)

hemoglobin_codes = df.filter(
    pl.col("description").str.contains(
        "Hemoglobin [Mass/volume] in Blood by calculation", literal=True
    )
    | pl.col("description").str.contains(
        "Hemoglobin [Mass/volume] in Blood", literal=True
    )
)["code"].to_list()
lab_to_codes["hemoglobin"] = "|".join(hemoglobin_codes)

hematocrit_codes = df.filter(
    pl.col("description").str.contains(
        "Hematocrit [Volume Fraction] of Blood by Automated count", literal=True
    )
    | pl.col("description").str.contains(
        "Hematocrit [Volume Fraction] of Blood by Estimated", literal=True
    )
)["code"].to_list()
lab_to_codes["hematocrit"] = "|".join(hematocrit_codes)


leukocytes_codes = df.filter(
    pl.col("description").str.contains(
        "Leukocytes [#/volume] in Blood by Automated count", literal=True
    )
)["code"].to_list()
lab_to_codes["leukocytes"] = "|".join(leukocytes_codes)


platets_codes = df.filter(
    pl.col("description").str.contains(
        "Platelets [#/volume] in Blood by Automated count", literal=True
    )
)["code"].to_list()
lab_to_codes["platets"] = "|".join(platets_codes)


def get_aces_config(location, lab, time_interval, extrema):
    lab_codes = lab_to_codes[lab]
    min_val, max_val = extrema
    if min_val is None and max_val is None:
        raise ValueError("Can't define both min and max")
    if not (min_val is None or max_val is None):
        raise ValueError("Defined neither min nor max")
    if min_val is not None:
        # YES MIN SHOULD BE NAMED MAX!!! The min value cutoff is the max value for aces to search for when defining this predicate.
        extrema_type = "max"
        value = min_val
    else:
        extrema_type = "min"
        value = max_val
    lab_requirement = f"""  abnormal_lab:
    code: {{regex: "{lab_codes}"}}
    value_{extrema_type}: {value}
    value_{extrema_type}_inclusive: True
    """
    return f"""#This config checks for an abnormal {lab} lab within {time_interval} after {location}
predicates:
  trigger_event:
    code: {{regex: "{location}//.*"}}
  lab:
    code: {{regex: "{lab_codes}"}}
{lab_requirement}

trigger: trigger_event

windows:
  input:
    start: NULL
    end: trigger
    start_inclusive: True
    end_inclusive: True
    index_timestamp: end
  target:
    start: input.end
    end: start + {time_interval}
    start_inclusive: False
    end_inclusive: True
    has:
      lab: (1, None)
    label: abnormal_lab

"""


results = {}
tasks = []
extrema = {
    "creatinine": (None, 2.0),
    "hemoglobin": (None, 2.0),
    "hematocrit": (24, None),
    "leukocytes": (5, None),
    "platets": (20, None),
}
for location in [
    "HOSPITAL_ADMISSION",
    "HOSPITAL_DISCHARGE",
    "ICU_ADMISSION",
    "ICU_DISCHARGE",
]:
    for lab in ["creatinine", "hemoglobin", "hematocrit", "leukocytes", "platets"]:
        for time_interval in ["30d", "60d", "90d"]:
            config = get_aces_config(location, lab, time_interval, extrema[lab])
            task_name = f"abnormal_lab/{location.lower()}/{lab}/{time_interval}"
            fp = Path(f"../ZERO_SHOT_TUTORIAL/configs/tasks/{task_name}.yaml")
            fp.parent.mkdir(parents=True, exist_ok=True)
            fp.write_text(config)
            tasks.append('"' + task_name + '"')
print("\n".join(tasks))
```

```
ROOT_DIR="/storage/shared/mimic-iv/meds_v0.3.2/"  # Replace with your actual root directory

MEDS_DIR=${ROOT_DIR}/meds/
TASKS=(
    "abnormal_lab/hospital_admission/creatinine/30d"
)
for TASK_NAME in "${TASKS[@]}"; do
    SINGLE_TASK_DIR="${MEDS_DIR}/tasks/${TASK_NAME}"
    mkdir -p $SINGLE_TASK_DIR # create a directory for the task
    CONFIG_PATH=ZERO_SHOT_TUTORIAL/configs/tasks/${TASK_NAME}.yaml
    aces-cli --multirun hydra/launcher=joblib data=sharded data.standard=meds data.root="$MEDS_DIR/data" "data.shard=$(expand_shards $MEDS_DIR/data)" cohort_dir="${MEDS_DIR}/tasks/" cohort_name="$TASK_NAME" config_path=$CONFIG_PATH
done
```

Extract aces tasks

```console
MEDS_DIR=${ROOT_DIR}/meds/
TASKS=(
    "mortality/in_hospital/first_24h"
    "mortality/in_icu/first_24h"
    "mortality/post_hospital_discharge/1y"
    "readmission/30d"
)
for TASK_NAME in "${TASKS[@]}"; do
    SINGLE_TASK_DIR="${MEDS_DIR}/tasks/${TASK_NAME}"
    mkdir -p $SINGLE_TASK_DIR # create a directory for the task
    CONFIG_PATH=ZERO_SHOT_TUTORIAL/configs/tasks/${TASK_NAME}.yaml
    aces-cli --multirun hydra/launcher=joblib data=sharded data.standard=meds data.root="$MEDS_DIR/data" "data.shard=$(expand_shards $MEDS_DIR/data)" cohort_dir="${MEDS_DIR}/tasks/" cohort_name="$TASK_NAME" config_path=$CONFIG_PATH
    cp $CONFIG_PATH ${MEDS_DIR}/tasks/${TASK_NAME}.yaml
done

ROOT_DIR="/storage/shared/mimic-iv/meds_v0.3.2/"  # Replace with your actual root directory
MEDS_DIR=${ROOT_DIR}/meds/
TASKS=(
    "abnormal_lab/hospital_admission/creatinine/30d"
    "abnormal_lab/hospital_admission/creatinine/60d"
    "abnormal_lab/hospital_admission/creatinine/90d"
    "abnormal_lab/hospital_discharge/creatinine/30d"
    "abnormal_lab/hospital_discharge/creatinine/60d"
    "abnormal_lab/hospital_discharge/creatinine/90d"
    "abnormal_lab/icu_admission/creatinine/30d"
    "abnormal_lab/icu_admission/creatinine/60d"
    "abnormal_lab/icu_admission/creatinine/90d"
    "abnormal_lab/icu_discharge/creatinine/30d"
    "abnormal_lab/icu_discharge/creatinine/60d"
    "abnormal_lab/icu_discharge/creatinine/90d"
    "abnormal_lab/hospital_admission/hemoglobin/30d"
    "abnormal_lab/hospital_admission/hemoglobin/60d"
    "abnormal_lab/hospital_admission/hemoglobin/90d"
    "abnormal_lab/hospital_discharge/hemoglobin/30d"
    "abnormal_lab/hospital_discharge/hemoglobin/60d"
    "abnormal_lab/hospital_discharge/hemoglobin/90d"
    "abnormal_lab/icu_admission/hemoglobin/30d"
    "abnormal_lab/icu_admission/hemoglobin/60d"
    "abnormal_lab/icu_admission/hemoglobin/90d"
    "abnormal_lab/icu_discharge/hemoglobin/30d"
    "abnormal_lab/icu_discharge/hemoglobin/60d"
    "abnormal_lab/icu_discharge/hemoglobin/90d"
    "abnormal_lab/hospital_admission/hematocrit/30d"
    "abnormal_lab/hospital_admission/hematocrit/60d"
    "abnormal_lab/hospital_admission/hematocrit/90d"
    "abnormal_lab/hospital_discharge/hematocrit/30d"
    "abnormal_lab/hospital_discharge/hematocrit/60d"
    "abnormal_lab/hospital_discharge/hematocrit/90d"
    "abnormal_lab/icu_admission/hematocrit/30d"
    "abnormal_lab/icu_admission/hematocrit/60d"
    "abnormal_lab/icu_admission/hematocrit/90d"
    "abnormal_lab/icu_discharge/hematocrit/30d"
    "abnormal_lab/icu_discharge/hematocrit/60d"
    "abnormal_lab/icu_discharge/hematocrit/90d"
    "abnormal_lab/hospital_admission/leukocytes/30d"
    "abnormal_lab/hospital_admission/leukocytes/60d"
    "abnormal_lab/hospital_admission/leukocytes/90d"
    "abnormal_lab/hospital_discharge/leukocytes/30d"
    "abnormal_lab/hospital_discharge/leukocytes/60d"
    "abnormal_lab/hospital_discharge/leukocytes/90d"
    "abnormal_lab/icu_admission/leukocytes/30d"
    "abnormal_lab/icu_admission/leukocytes/60d"
    "abnormal_lab/icu_admission/leukocytes/90d"
    "abnormal_lab/icu_discharge/leukocytes/30d"
    "abnormal_lab/icu_discharge/leukocytes/60d"
    "abnormal_lab/icu_discharge/leukocytes/90d"
    "abnormal_lab/hospital_admission/platets/30d"
    "abnormal_lab/hospital_admission/platets/60d"
    "abnormal_lab/hospital_admission/platets/90d"
    "abnormal_lab/hospital_discharge/platets/30d"
    "abnormal_lab/hospital_discharge/platets/60d"
    "abnormal_lab/hospital_discharge/platets/90d"
    "abnormal_lab/icu_admission/platets/30d"
    "abnormal_lab/icu_admission/platets/60d"
    "abnormal_lab/icu_admission/platets/90d"
    "abnormal_lab/icu_discharge/platets/30d"
    "abnormal_lab/icu_discharge/platets/60d"
    "abnormal_lab/icu_discharge/platets/90d"
)
for TASK_NAME in "${TASKS[@]}"; do
    SINGLE_TASK_DIR="${MEDS_DIR}/tasks/${TASK_NAME}"
    mkdir -p $SINGLE_TASK_DIR # create a directory for the task
    CONFIG_PATH=ZERO_SHOT_TUTORIAL/configs/tasks/${TASK_NAME}.yaml
    aces-cli --multirun hydra/launcher=joblib data=sharded data.standard=meds data.root="$MEDS_DIR/data" "data.shard=$(expand_shards $MEDS_DIR/data)" cohort_dir="${MEDS_DIR}/tasks/" cohort_name="$TASK_NAME" config_path=$CONFIG_PATH
    cp $CONFIG_PATH ${MEDS_DIR}/tasks/${TASK_NAME}.yaml
done
```

### Training Models

**Let's first train a supervised model which we will use as a baseline.**

Checkout (and modify if you wish) the experiment file `$(pwd)/ZERO_SHOT_TUTORIAL/configs/eic_top10_forecast_mtr.yaml` which defines most of the input args for training this model:

```console
CUDA_VISIBLE_DEVICES=1

MEDS_DIR=${ROOT_DIR}/meds/
EIC_DIR=${ROOT_DIR}/eic_top10_tensors # set to the directory in which you want to output the tensorized MIMIC-IV data
TASKS_DIR=${MEDS_DIR}/tasks/
TASK_NAME="mortality/in_icu/first_24h"
OUTPUT_DIR=/storage/nassim/tmp/supervised/

meds-torch-train \
    experiment="eic_top10_forecast_mtr" paths.data_dir=${EIC_DIR} \
    paths.meds_cohort_dir=${MEDS_DIR} paths.output_dir=${OUTPUT_DIR} \
    data.task_name=$TASK_NAME data.task_root_dir=$TASKS_DIR \
    hydra.searchpath=[pkg://meds_torch.configs,$(pwd)/ZERO_SHOT_TUTORIAL/configs/]

```

**Let's train an autoregressive generative model**

The experiment file uses a supervised model by default, let's override it by setting `model=eic_forecasting` to train on the autoregressive next token prediction task. Notice that this doesn't require labels, so we drop the task input args.

```console
CUDA_VISIBLE_DEVICES=1

MEDS_DIR=${ROOT_DIR}/meds/
EIC_DIR=${ROOT_DIR}/eic_top10_tensors # set to the directory in which you want to output the tensorized MIMIC-IV data
OUTPUT_DIR=/storage/nassim/tmp/autoregressive/

meds-torch-train model=eic_forecasting trainer=gpu \
    experiment=eic_top10_forecast_mtr paths.data_dir=${EIC_DIR} \
    data.subsequence_sampling_strategy=random \
    paths.meds_cohort_dir=${MEDS_DIR} paths.output_dir=${OUTPUT_DIR} \
    hydra.searchpath=[pkg://meds_torch.configs,$(pwd)/ZERO_SHOT_TUTORIAL/configs/]
```

**Next let's do a distributed hyperparameter tuning of this autoregressive model.**

We have to add the defaults:

- use the cli endpoint `meds-torch-tune` instead of `meds-torch-train` as this will launch the tune script which provides code to leverage
- `trainer=ray` which adds ray logging support to the pytorch lightning trainer
- `hparams_search=ray_tune` which adds a default learning rate and dropout hparam search space
- `callbacks=tune_default` which adds ray callbacks for storing the top_k checkpoints while training
- `hparams_search.ray.resources_per_trial.gpu=1` will make ray launch jobs in parallel assigning one job to each gpu (set this to a fraction to having multiple jobs on each gpu)
- ` hparams_search.ray.num_samples=8` will randomly sample 8 hyperparameter draws, so ray will run a total of 8 jobs

```console
# unset CUDA_VISIBLE_DEVICES # remove setting of CUDA_VISIBLE_DEVICES, so all gpus can be used
# or set some specific devices
export CUDA_VISIBLE_DEVICES=0,1

MEDS_DIR=${ROOT_DIR}/meds/
EIC_DIR=${ROOT_DIR}/eic_top10_tensors # set to the directory in which you want to output the tensorized MIMIC-IV data
OUTPUT_DIR=${ROOT_DIR}/results/zero_shot/eic_top_10_hparam_sweep/

meds-torch-tune model=eic_forecasting trainer=gpu \
    callbacks=tune_default trainer=ray hparams_search=ray_tune \
    hparams_search.ray.resources_per_trial.GPU=1  hparams_search.ray.num_samples=8 \
    experiment=eic_top10_forecast_mtr paths.data_dir=${EIC_DIR} \
    data.subsequence_sampling_strategy=random \
    paths.meds_cohort_dir=${MEDS_DIR} paths.output_dir=${OUTPUT_DIR} \
    hydra.searchpath=[pkg://meds_torch.configs,$(pwd)/ZERO_SHOT_TUTORIAL/configs/]

# Train a full vocab model:
export CUDA_VISIBLE_DEVICES=0,2,4,5

MEDS_DIR=${ROOT_DIR}/meds/
EIC_DIR=${ROOT_DIR}/eic_tensors # set to the directory in which you want to output the tensorized MIMIC-IV data
OUTPUT_DIR=${ROOT_DIR}/results/zero_shot/eic_hparam_sweep/

meds-torch-tune model=eic_forecasting trainer=gpu \
    callbacks=tune_default trainer=ray hparams_search=ray_tune \
    hparams_search.ray.resources_per_trial.GPU=1  hparams_search.ray.num_samples=8 \
    experiment=eic_forecast_mtr paths.data_dir=${EIC_DIR} \
    data.subsequence_sampling_strategy=random \
    paths.meds_cohort_dir=${MEDS_DIR} paths.output_dir=${OUTPUT_DIR} \
    hydra.searchpath=[pkg://meds_torch.configs,$(pwd)/ZERO_SHOT_TUTORIAL/configs/]
```

```
PRETRAIN_OUTPUT_DIR=${ROOT_DIR}/results/zero_shot/eic_top_10_hparam_sweep
MODEL_SWEEP_DIR=$(meds-torch-latest-dir path=${PRETRAIN_OUTPUT_DIR})
BEST_CHECKPOINT=${MODEL_SWEEP_DIR}/checkpoints/best_model.ckpt
BEST_CONFIG=${MODEL_SWEEP_DIR}/best_config.json
MEDS_DIR=${ROOT_DIR}/meds/
TENSOR_DIR=${ROOT_DIR}/eic_top10_tensors/
TASKS_DIR=${MEDS_DIR}/tasks/
TASK_NAME="abnormal_lab/hospital_discharge/creatinine/60d"
OUTPUT_DIR=${ROOT_DIR}/results/zero_shot/inference/eic_top_10/${TASK_NAME}
# Let's generate 20 trajectories
NUM_SAMPLES=20
TASK_CONFIG_PATH=${TASKS_DIR}/${TASK_NAME}.yaml


meds-torch-generate model=eic_forecasting experiment=eic_top10_forecast_mtr \
    model/trajectory_labeler=aces_schema_labeler model.trajectory_labeler.yaml_path=$TASK_CONFIG_PATH \
    data.dataloader.batch_size=512 model.generate_id=0 trainer.devices=[0] data.predict_dataset=test \
	data.do_include_subject_id=true data.do_include_prediction_time=true data.do_include_end_time=true \
    data.task_name=${TASK_NAME} data.task_root_dir=${TASKS_DIR} \
    paths.meds_cohort_dir=${MEDS_DIR} ckpt_path=${BEST_CHECKPOINT} \
    paths.data_dir=${TENSOR_DIR} paths.output_dir=${OUTPUT_DIR} \
    "hydra.searchpath=[pkg://meds_torch.configs,$(pwd)/ZERO_SHOT_TUTORIAL/configs/]"


meds-torch-generate --multirun model=eic_forecasting experiment=eic_top10_forecast_mtr \
    model/trajectory_labeler=aces_schema_labeler model.trajectory_labeler.yaml_path=$TASK_CONFIG_PATH \
    data.dataloader.batch_size=64 model.generate_id="range(0,$NUM_SAMPLES)" trainer.devices=[0] data.predict_dataset=test\
	data.do_include_subject_id=true data.do_include_prediction_time=true \
    data.task_name=${TASK_NAME} data.task_root_dir=${TASKS_DIR} \
    paths.meds_cohort_dir=${MEDS_DIR} ckpt_path=${BEST_CHECKPOINT} \
    paths.data_dir=${TENSOR_DIR} paths.output_dir=${OUTPUT_DIR} \
    "hydra.searchpath=[pkg://meds_torch.configs,$(pwd)/ZERO_SHOT_TUTORIAL/configs/]"
```

We can also run this on the full vocabulary

```console
PRETRAIN_OUTPUT_DIR=${ROOT_DIR}/results/zero_shot/eic_hparam_sweep
MODEL_SWEEP_DIR=$(meds-torch-latest-dir path=${PRETRAIN_OUTPUT_DIR})
BEST_CHECKPOINT=${MODEL_SWEEP_DIR}/checkpoints/best_model.ckpt
BEST_CONFIG=${MODEL_SWEEP_DIR}/best_config.json
MEDS_DIR=${ROOT_DIR}/meds/
TENSOR_DIR=${ROOT_DIR}/eic_tensors/
TASKS_DIR=${MEDS_DIR}/tasks/
TASK_NAME="mortality/in_icu/first_24h"
OUTPUT_DIR=${ROOT_DIR}/results/zero_shot/inference/eic/${TASK_NAME}
# Let's generate 20 trajectories
NUM_SAMPLES=20
TASK_CONFIG_PATH=${TASKS_DIR}/${TASK_NAME}.yaml


meds-torch-generate model=eic_forecasting experiment=eic_forecast_mtr \
    model/trajectory_labeler=aces_schema_labeler model.trajectory_labeler.yaml_path=$TASK_CONFIG_PATH \
    data.dataloader.batch_size=512 model.generate_id=0 trainer.devices=[0] data.predict_dataset=test \
	data.do_include_subject_id=true data.do_include_prediction_time=true data.do_include_end_time=true \
    data.task_name=${TASK_NAME} data.task_root_dir=${TASKS_DIR} \
    paths.meds_cohort_dir=${MEDS_DIR} ckpt_path=${BEST_CHECKPOINT} \
    paths.data_dir=${TENSOR_DIR} paths.output_dir=${OUTPUT_DIR} \
    "hydra.searchpath=[pkg://meds_torch.configs,$(pwd)/ZERO_SHOT_TUTORIAL/configs/]"

```

**Checkout the `abnormal_lab_inference.sh` script for how to loop through a suite of tasks.**

## Controlled Analysis of Trajectories

**If you just want to generate N tokens given the first M tokens from a patient trajectory, run the following command:**

- `data.subsequence_sampling_strategy=from_start` makes the model start at the beginning of the patient trajectory
- `data.predict_dataset=test` makes generation run on the `test` set of patients, but you can also use the `val` or `train` set.
- `trainer.devices=[0]` uses gpu id 0 for inference
- `model.generate_id="range(0,$NUM_SAMPLES)"` paired with the `--multirun` flag will make this run i a for loop 20 times with the generate_id ranging from 0 to 19 -- i.e. generating 20 samples for each patient. You can remove the `--multirun` flag and set `model.generate_id=0` to just generate a single sample.

```console
PRETRAIN_OUTPUT_DIR=${ROOT_DIR}/results/zero_shot/eic_hparam_sweep
MODEL_SWEEP_DIR=$(meds-torch-latest-dir path=${PRETRAIN_OUTPUT_DIR})
BEST_CHECKPOINT=${MODEL_SWEEP_DIR}/checkpoints/best_model.ckpt
BEST_CONFIG=${MODEL_SWEEP_DIR}/best_config.json
MEDS_DIR=${ROOT_DIR}/meds/
TENSOR_DIR=${ROOT_DIR}/eic_tensors/
N=10
M=10
OUTPUT_DIR=${ROOT_DIR}/results/zero_shot/inference/eic/FIRST${N}_GENERATE${M}/
# Let's generate 20 trajectories
NUM_SAMPLES=20

meds-torch-generate --multirun model=eic_forecasting experiment=eic_forecast_mtr \
    model.max_tokens_budget=${M} data.subsequence_sampling_strategy=from_start \
    data.dataloader.batch_size=512 model.generate_id="range(0,$NUM_SAMPLES)" trainer.devices=[0] data.predict_dataset=test \
	data.do_include_subject_id=true data.do_include_prediction_time=true data.do_include_end_time=true \
    paths.meds_cohort_dir=${MEDS_DIR} ckpt_path=${BEST_CHECKPOINT} \
    paths.data_dir=${TENSOR_DIR} paths.output_dir=${OUTPUT_DIR} \
    "hydra.searchpath=[pkg://meds_torch.configs,$(pwd)/ZERO_SHOT_TUTORIAL/configs/]"
```

Note that you can find the generated patient trajectories in
`${OUTPUT_DIR}/<DATE>/generated_trajectory_*.parquet`
Note that in the actual directory `${OUTPUT_DIR}/<DATE>/` you will find a `generated_trajectory_.parquet`, `generated_trajectory_.parquet`, ..., `generated_trajectory_19.parquet`

If you used an aces_schema_labeler, there will be zero-shot prediction results in

`${OUTPUT_DIR}/<DATE>/predictions_*.parquet`

You can load all the samples simultaneously in a jupyter notebook with:

```python
# pip install polars
import polars as pl

prediction_df = pl.read_parquet(f"{OUTPUT_DIR}/<DATE>/predictions_*.parquet")
generate_df = pl.read_parquet(f"{OUTPUT_DIR}/<DATE>/generated_trajectory_*.parquet")
```

See the `inference_analysis.ipynb` notebook to learn

1. how to checkout what the actual input tokens for patient data are and interpret them using the metadata.
2. how to prompt the generative model with a custom sequence of codes.
