# Evaluating an Autoregressive EHR Model

Train via meds-torch
ROOT_DIR on MGH servers:
/storage/shared/mimiciv/meds_v0.3.2/
training_stultz:
/storage/shared/mimic-iv/meds_v0.3.2/
quadro:
/home/shared/mimiciv2.2/meds_v0.3.2/

```console
ROOT_DIR="/storage/shared/mimic-iv/meds_v0.3.2/"  # Replace with your actual root directory
```

We first need to tensorize your meds data (cache it in a format such that meds-torch can efficiently train models on the dataset). We can tensorize 10 commone lab codes in MIMIC IV by running the tokenize script with the `eic_top_10.yaml`, which performs EIC tokenization on these common codes:

```console
export MEDS_DIR=${ROOT_DIR}/meds/
export EIC_DIR=${ROOT_DIR}/eic_top10_tensors # set to the directory in which you want to output the tensorized MIMIC-IV data
export N_WORKERS=8 # set to the number of parallel workers you want to use
export PIPELINE_CONFIG_PATH="$(pwd)/TUTORIAL/configs/eic_top_10.yaml" # set to the directory in which the config file is stored, must be an absolute path.
export JOBLIB_RUNNER_CONFIG_PATH="$(pwd)/TUTORIAL/configs/local_parallelism_runner.yaml" # set to the directory in which the config file is stored, must be an absolute path.

bash TUTORIAL/tokenize.sh $MEDS_DIR $EIC_DIR $N_WORKERS $PIPELINE_CONFIG_PATH stage_runner_fp=$JOBLIB_RUNNER_CONFIG_PATH
```

Let's first train a supervised model which we will use as a baseline.

Checkout (and modify if you wish) the experiment file `$(pwd)/TUTORIAL/configs/eic_top10_forecast_mtr.yaml` which defines most of the input args for training this model:

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
    hydra.searchpath=[pkg://meds_torch.configs,$(pwd)/TUTORIAL/configs/]

```

The experiment file uses a supervised model by default, let's override it by setting `model=eic_forecasting` to train on the autoregressive next token prediction task. Notice that this doesn't require labels, so we drop the task input args.

```console
CUDA_VISIBLE_DEVICES=1

MEDS_DIR=${ROOT_DIR}/meds/
EIC_DIR=${ROOT_DIR}/eic_top10_tensors # set to the directory in which you want to output the tensorized MIMIC-IV data
TASKS_DIR=${MEDS_DIR}/tasks/
TASK_NAME="mortality/in_icu/first_24h"
OUTPUT_DIR=/storage/nassim/tmp/autoregressive/

meds-torch-train model=eic_forecasting trainer=gpu \
    experiment=eic_top10_forecast_mtr paths.data_dir=${EIC_DIR} \
    data.subsequence_sampling_strategy=random \
    paths.meds_cohort_dir=${MEDS_DIR} paths.output_dir=${OUTPUT_DIR} \
    hydra.searchpath=[pkg://meds_torch.configs,$(pwd)/TUTORIAL/configs/]

```

Next let's do a distributed hyperparameter tuning of this autoregressive model. We have to add the defaults

- use the cli endpoint `meds-torch-tune` instead of `meds-torch-train` as this will launch the tune script which provides code to leverage
- `trainer=ray` which adds ray logging support to the pytorch lightning trainer
- `hparams_search=ray_tune` which adds a default learning rate and dropout hparam search space
- `callbacks=tune_default` which adds ray callbacks for storing the top_k checkpoints while training
- `hparams_search.ray.resources_per_trial.gpu=1` will make ray launch jobs in parallel assigning one job to each gpu (set this to a fraction to having multiple jobs on each gpu)
- ` hparams_search.ray.num_samples=8` will randomly sample 8 hyperparameter draws, so ray will run a total of 8 jobs

```console
# unset CUDA_VISIBLE_DEVICES # remove setting of CUDA_VISIBLE_DEVICES, so all gpus can be used
# or set some specific devices
export CUDA_VISIBLE_DEVICES=0,1,5

MEDS_DIR=${ROOT_DIR}/meds/
EIC_DIR=${ROOT_DIR}/eic_top10_tensors # set to the directory in which you want to output the tensorized MIMIC-IV data
OUTPUT_DIR=/storage/nassim/tmp/eic_top_10_hparam_sweep/

meds-torch-tune model=eic_forecasting trainer=gpu \
    callbacks=tune_default trainer=ray hparams_search=ray_tune \
    hparams_search.ray.resources_per_trial.GPU=1  hparams_search.ray.num_samples=8 \
    experiment=eic_top10_forecast_mtr paths.data_dir=${EIC_DIR} \
    data.subsequence_sampling_strategy=random \
    paths.meds_cohort_dir=${MEDS_DIR} paths.output_dir=${OUTPUT_DIR} \
    hydra.searchpath=[pkg://meds_torch.configs,$(pwd)/TUTORIAL/configs/]

# Train a full vocab model:
export CUDA_VISIBLE_DEVICES=0,2,4,5

MEDS_DIR=${ROOT_DIR}/meds/
EIC_DIR=${ROOT_DIR}/eic_tensors # set to the directory in which you want to output the tensorized MIMIC-IV data
OUTPUT_DIR=/storage/nassim/tmp/eic_hparam_sweep/

meds-torch-tune model=eic_forecasting trainer=gpu \
    callbacks=tune_default trainer=ray hparams_search=ray_tune \
    hparams_search.ray.resources_per_trial.GPU=1  hparams_search.ray.num_samples=8 \
    experiment=eic_forecast_mtr paths.data_dir=${EIC_DIR} \
    data.subsequence_sampling_strategy=random \
    paths.meds_cohort_dir=${MEDS_DIR} paths.output_dir=${OUTPUT_DIR} \
    hydra.searchpath=[pkg://meds_torch.configs,$(pwd)/TUTORIAL/configs/]


# Let's also train a supervised model

MEDS_DIR=${ROOT_DIR}/meds/
EIC_DIR=${ROOT_DIR}/eic_top10_tensors # set to the directory in which you want to output the tensorized MIMIC-IV data
OUTPUT_DIR=/storage/nassim/tmp/eic_top_10_hparam_sweep_supervised/
TASKS_DIR=${MEDS_DIR}/tasks/
TASK_NAME="mortality/in_icu/first_24h"

meds-torch-tune model=supervised trainer=gpu \
    callbacks=tune_default trainer=ray hparams_search=ray_tune \
    hparams_search.ray.resources_per_trial.GPU=1  hparams_search.ray.num_samples=8 \
    experiment=eic_top10_forecast_mtr paths.data_dir=${EIC_DIR} \
    data.do_include_subject_id=true data.do_include_prediction_time=true \
    data.task_name=${TASK_NAME} data.task_root_dir=${TASKS_DIR} \
    paths.meds_cohort_dir=${MEDS_DIR} paths.output_dir=${OUTPUT_DIR} \
    hydra.searchpath=[pkg://meds_torch.configs,$(pwd)/TUTORIAL/configs/]
```

We can now see the logits over the input data:

```console
PRETRAIN_OUTPUT_DIR=/storage/nassim/tmp/eic_top_10_hparam_sweep/
MODEL_SWEEP_DIR=$(meds-torch-latest-dir path=${PRETRAIN_OUTPUT_DIR})
BEST_CHECKPOINT=${MODEL_SWEEP_DIR}/checkpoints/best_model.ckpt
BEST_CONFIG=${MODEL_SWEEP_DIR}/best_config.json
MEDS_DIR=${ROOT_DIR}/meds/
TENSOR_DIR=${ROOT_DIR}/eic_top10_tensors/
OUTPUT_DIR=/storage/nassim/tmp/predict/
TASKS_DIR=${MEDS_DIR}/tasks/
TASK_NAME="mortality/in_icu/first_24h"
# Let's generate 20 trajectories
NUM_SAMPLES=20


meds-torch-predict --multirun model=eic_forecasting \
    model.generate_id="range(0,$NUM_SAMPLES)" experiment=eic_top10_forecast_mtr \
    data.do_include_subject_id=true data.do_include_prediction_time=true \
    data.task_name=${TASK_NAME} data.task_root_dir=${TASKS_DIR} \
    paths.meds_cohort_dir=${MEDS_DIR} ckpt_path=${BEST_CHECKPOINT} \
    paths.data_dir=${TENSOR_DIR} paths.output_dir=${OUTPUT_DIR} \
    "hydra.searchpath=[$(pwd)/TUTORIAL/configs/]"
```

## Run zero-shot by:

- increasing the `NUM_SAMPLES`
- setting a zero-shot labeler for a task
  - for the task of predicting mortality within 48 hours given the first 24 hours of an icu admission, we add the zero-shot labeler args  `model/zero_shot_labeler=time_to_event_labeler model.zero_shot_labeler.target_codes=[203] model.zero_shot_labeler.time_length=0.0054757`. the Death code is code 203, check the \$\{TENSOR_DIR}/metadata/codes.parquet file to see the MEDS_DEATH code/vocab_index when you run this. The times are in years so 2/365.25 gives us 48 hours (2 days).
- Optionally, select a time based budget, rather than a max_seq_len budget by setting `model/backbone/generation_budget=years model.backbone.generation_budget.value=0.0054757 and increase the batch size for more speed with data.dataloader.batch_size=1024`

```console
CUDA_VISIBLE_DEVICES=0
PRETRAIN_OUTPUT_DIR=/storage/nassim/tmp/eic_top_10_hparam_sweep/
MODEL_SWEEP_DIR=$(meds-torch-latest-dir path=${PRETRAIN_OUTPUT_DIR})
BEST_CHECKPOINT=${MODEL_SWEEP_DIR}/checkpoints/best_model.ckpt
BEST_CONFIG=${MODEL_SWEEP_DIR}/best_config.json
MEDS_DIR=${ROOT_DIR}/meds/
TENSOR_DIR=${ROOT_DIR}/eic_top10_tensors/
OUTPUT_DIR=/storage/nassim/tmp/zero-shot/
TASKS_DIR=${MEDS_DIR}/tasks/
TASK_NAME="mortality/in_icu/first_24h"

# for max_seq_len budget
meds-torch-predict --multirun model/zero_shot_labeler=time_to_event_labeler model=eic_forecasting \
    model.generate_id="range(0,100)" experiment=eic_top10_forecast_mtr \
    model.zero_shot_labeler.target_codes=[203] model.zero_shot_labeler.time_length=0.0054757 \
    data.do_include_subject_id=true data.do_include_prediction_time=true \
    data.task_name=${TASK_NAME} data.task_root_dir=${TASKS_DIR} \
    paths.meds_cohort_dir=${MEDS_DIR} ckpt_path=${BEST_CHECKPOINT} \
    paths.data_dir=${TENSOR_DIR} paths.output_dir=${OUTPUT_DIR} \
    "hydra.searchpath=[$(pwd)/TUTORIAL/configs/]"

# for time budget
meds-torch-predict --multirun model/zero_shot_labeler=time_to_event_labeler model/backbone/generation_budget=years model=eic_forecasting \
    model.backbone.generation_budget.value=0.0054757 data.dataloader.batch_size=512 \
    model.generate_id="range(0,10)" experiment=eic_top10_forecast_mtr \
    model.zero_shot_labeler.target_codes=[203] model.zero_shot_labeler.time_length=0.0054757 \
    data.do_include_subject_id=true data.do_include_prediction_time=true \
    data.task_name=${TASK_NAME} data.task_root_dir=${TASKS_DIR} \
    paths.meds_cohort_dir=${MEDS_DIR} ckpt_path=${BEST_CHECKPOINT} \
    paths.data_dir=${TENSOR_DIR} paths.output_dir=${OUTPUT_DIR} \
    "hydra.searchpath=[$(pwd)/TUTORIAL/configs/]"

CUDA_VISIBLE_DEVICES=0
PRETRAIN_OUTPUT_DIR=/storage/nassim/tmp/eic_top_10_hparam_sweep_6layer_8nhead/
MODEL_SWEEP_DIR=$(meds-torch-latest-dir path=${PRETRAIN_OUTPUT_DIR})
BEST_CHECKPOINT=${MODEL_SWEEP_DIR}/checkpoints/best_model.ckpt
BEST_CONFIG=${MODEL_SWEEP_DIR}/best_config.json
MEDS_DIR=${ROOT_DIR}/meds/
TENSOR_DIR=${ROOT_DIR}/eic_top10_tensors/
OUTPUT_DIR=/storage/nassim/tmp/zero-shot/
TASKS_DIR=${MEDS_DIR}/tasks/
TASK_NAME="mortality/in_icu/first_24h"

# for time budget -- /storage/nassim/tmp/zero-shot/2024-11-13_01-42-22_821171/
meds-torch-predict --multirun model/zero_shot_labeler=time_to_event_labeler \
    model/backbone/generation_budget=years model=eic_forecasting \
    model.backbone.generation_budget.value=0.0054757 data.dataloader.batch_size=64 \
    model.generate_id="range(0,10)" experiment=eic_top10_forecast_large_mtr \
    model.zero_shot_labeler.target_codes=[203] model.zero_shot_labeler.time_length=0.0054757 \
    data.do_include_subject_id=true data.do_include_prediction_time=true \
    data.task_name=${TASK_NAME} data.task_root_dir=${TASKS_DIR} \
    paths.meds_cohort_dir=${MEDS_DIR} ckpt_path=${BEST_CHECKPOINT} \
    paths.data_dir=${TENSOR_DIR} paths.output_dir=${OUTPUT_DIR} \
    "hydra.searchpath=[$(pwd)/TUTORIAL/configs/]"

CUDA_VISIBLE_DEVICES=6
PRETRAIN_OUTPUT_DIR=/storage/nassim/tmp/eic_top_10_hparam_sweep_16tokendim/
MODEL_SWEEP_DIR=$(meds-torch-latest-dir path=${PRETRAIN_OUTPUT_DIR})
BEST_CHECKPOINT=${MODEL_SWEEP_DIR}/checkpoints/best_model.ckpt
BEST_CONFIG=${MODEL_SWEEP_DIR}/best_config.json
MEDS_DIR=${ROOT_DIR}/meds/
TENSOR_DIR=${ROOT_DIR}/eic_top10_tensors/
OUTPUT_DIR=/storage/nassim/tmp/zero-shot/
TASKS_DIR=${MEDS_DIR}/tasks/
TASK_NAME="mortality/in_icu/first_24h"

# for time budget --
meds-torch-predict --multirun model/zero_shot_labeler=time_to_event_labeler \
    model/backbone/generation_budget=years model=eic_forecasting model.token_dim=16 \
    model.backbone.generation_budget.value=0.0054757 data.dataloader.batch_size=64 \
    model.generate_id="range(0,10)" experiment=eic_top10_forecast_mtr \
    model.zero_shot_labeler.target_codes=[203] model.zero_shot_labeler.time_length=0.0054757 \
    data.do_include_subject_id=true data.do_include_prediction_time=true \
    data.task_name=${TASK_NAME} data.task_root_dir=${TASKS_DIR} \
    paths.meds_cohort_dir=${MEDS_DIR} ckpt_path=${BEST_CHECKPOINT} \
    paths.data_dir=${TENSOR_DIR} paths.output_dir=${OUTPUT_DIR} \
    "hydra.searchpath=[$(pwd)/TUTORIAL/configs/]"

```

Results for 1 sample:

- 1 sample (max_seq_len budget): /storage/nassim/tmp/zero-shot/2024-11-12_12-04-53_670984/predict.parquet
- 1 sample (time budget): /storage/nassim/tmp/zero-shot/2024-11-12_13-16-06_535430
- 100 sample (max_seq_len budget): /storage/nassim/tmp/predict/2024-11-12_12-37-34_557897/predict.parquet
- 10 sample time budget: /storage/nassim/tmp/zero-shot/2024-11-13_01-15-39_138118/
- 10 sample time budget (large): /storage/nassim/tmp/zero-shot/2024-11-13_01-48-26_351350
-

## Generate trajectories for analysis

```console

CUDA_VISIBLE_DEVICES=2
PRETRAIN_OUTPUT_DIR=/storage/nassim/tmp/eic_top_10_hparam_sweep/
MODEL_SWEEP_DIR=$(meds-torch-latest-dir path=${PRETRAIN_OUTPUT_DIR})
BEST_CHECKPOINT=${MODEL_SWEEP_DIR}/checkpoints/best_model.ckpt
BEST_CONFIG=${MODEL_SWEEP_DIR}/best_config.json
MEDS_DIR=${ROOT_DIR}/meds/
TENSOR_DIR=${ROOT_DIR}/eic_top10_tensors/
OUTPUT_DIR=/storage/nassim/tmp/generate/
TASKS_DIR=${MEDS_DIR}/tasks/
TASK_NAME="mortality/in_icu/first_24h"


meds-torch-generate --multirun model/backbone/generation_budget=years model=eic_forecasting \
    model.backbone.generation_budget.value=0.0054757 data.dataloader.batch_size=64 \
    model.generate_id="range(0,100)" experiment=eic_top10_forecast_mtr \
	data.do_include_subject_id=true data.do_include_prediction_time=true \
    data.task_name=${TASK_NAME} data.task_root_dir=${TASKS_DIR} \
    paths.meds_cohort_dir=${MEDS_DIR} ckpt_path=${BEST_CHECKPOINT} \
    paths.data_dir=${TENSOR_DIR} paths.output_dir=${OUTPUT_DIR} \
    "hydra.searchpath=[$(pwd)/TUTORIAL/configs/]"


CUDA_VISIBLE_DEVICES=3
PRETRAIN_OUTPUT_DIR=/storage/nassim/tmp/eic_top_10_hparam_sweep_6layer_8nhead/
MODEL_SWEEP_DIR=$(meds-torch-latest-dir path=${PRETRAIN_OUTPUT_DIR})
BEST_CHECKPOINT=${MODEL_SWEEP_DIR}/checkpoints/best_model.ckpt
BEST_CONFIG=${MODEL_SWEEP_DIR}/best_config.json
MEDS_DIR=${ROOT_DIR}/meds/
TENSOR_DIR=${ROOT_DIR}/eic_top10_tensors/
OUTPUT_DIR=/storage/nassim/tmp/generate/
TASKS_DIR=${MEDS_DIR}/tasks/
TASK_NAME="mortality/in_icu/first_24h"


meds-torch-generate --multirun model/backbone/generation_budget=years model=eic_forecasting \
    model.backbone.generation_budget.value=0.0054757 data.dataloader.batch_size=64 \
    model.generate_id="range(0,100)" experiment=eic_top10_forecast_large_mtr \
    data.do_include_subject_id=true data.do_include_prediction_time=true \
    data.task_name=${TASK_NAME} data.task_root_dir=${TASKS_DIR} \
    paths.meds_cohort_dir=${MEDS_DIR} ckpt_path=${BEST_CHECKPOINT} \
    paths.data_dir=${TENSOR_DIR} paths.output_dir=${OUTPUT_DIR} \
    "hydra.searchpath=[$(pwd)/TUTORIAL/configs/]"

CUDA_VISIBLE_DEVICES=1
PRETRAIN_OUTPUT_DIR=/storage/nassim/tmp/eic_top_10_hparam_sweep_16tokendim/
MODEL_SWEEP_DIR=$(meds-torch-latest-dir path=${PRETRAIN_OUTPUT_DIR})
BEST_CHECKPOINT=${MODEL_SWEEP_DIR}/checkpoints/best_model.ckpt
BEST_CONFIG=${MODEL_SWEEP_DIR}/best_config.json
MEDS_DIR=${ROOT_DIR}/meds/
TENSOR_DIR=${ROOT_DIR}/eic_top10_tensors/
OUTPUT_DIR=/storage/nassim/tmp/generate/
TASKS_DIR=${MEDS_DIR}/tasks/
TASK_NAME="mortality/in_icu/first_24h"


meds-torch-generate --multirun model/backbone/generation_budget=years model=eic_forecasting model.token_dim=16 \
    model.backbone.generation_budget.value=0.0054757 data.dataloader.batch_size=64 \
     model.generate_id="range(0,100)" experiment=eic_top10_forecast_mtr \
	data.do_include_subject_id=true data.do_include_prediction_time=true \
    data.task_name=${TASK_NAME} data.task_root_dir=${TASKS_DIR} \
    paths.meds_cohort_dir=${MEDS_DIR} ckpt_path=${BEST_CHECKPOINT} \
    paths.data_dir=${TENSOR_DIR} paths.output_dir=${OUTPUT_DIR} \
    "hydra.searchpath=[$(pwd)/TUTORIAL/configs/]"

```

Let's train a much larger model now, we add the args:

- `model.backbone.n_layers=6 model.backbone.nheads=8`
- TODO: consider modifying `model.token_dim=128`
- - 100 samples small: /storage/nassim/tmp/generate/2024-11-13_02-17-06_228003/generated_trajectory.parquet
  - 100 samples big: ???

```console
# unset CUDA_VISIBLE_DEVICES # remove setting of CUDA_VISIBLE_DEVICES, so all gpus can be used
# or set some specific devices
export CUDA_VISIBLE_DEVICES=2,3,4,5



MEDS_DIR=${ROOT_DIR}/meds/
EIC_DIR=${ROOT_DIR}/eic2_top10_tensors # set to the directory in which you want to output the tensorized MIMIC-IV data
OUTPUT_DIR=/storage/nassim/tmp/eic_top_10_hparam_sweep_12layer_12nhead/

meds-torch-tune model=eic_forecasting trainer=gpu model.backbone.n_layers=12 model.backbone.nheads=12 \
    callbacks=tune_default trainer=ray hparams_search=ray_tune \
    hparams_search.ray.resources_per_trial.GPU=1  hparams_search.ray.num_samples=4 \
    experiment=eic_top10_forecast_mtr paths.data_dir=${EIC_DIR} \
    data.subsequence_sampling_strategy=random \
    paths.meds_cohort_dir=${MEDS_DIR} paths.output_dir=${OUTPUT_DIR} \
    hydra.searchpath=[pkg://meds_torch.configs,$(pwd)/TUTORIAL/configs/]

MEDS_DIR=${ROOT_DIR}/meds/
EIC_DIR=${ROOT_DIR}/eic_top10_tensors # set to the directory in which you want to output the tensorized MIMIC-IV data
OUTPUT_DIR=/storage/nassim/tmp/eic_top_10_hparam_sweep_16tokendim/

meds-torch-tune model=eic_forecasting trainer=gpu model.token_dim=16 \
    callbacks=tune_default trainer=ray hparams_search=ray_tune \
    hparams_search.ray.resources_per_trial.GPU=1  hparams_search.ray.num_samples=8 \
    experiment=eic_top10_forecast_mtr paths.data_dir=${EIC_DIR} \
    data.subsequence_sampling_strategy=random \
    paths.meds_cohort_dir=${MEDS_DIR} paths.output_dir=${OUTPUT_DIR} \
    hydra.searchpath=[pkg://meds_torch.configs,$(pwd)/TUTORIAL/configs/]
```
