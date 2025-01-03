#!/bin/bash
set -e

# Define the tasks array


# Define other required variables
ROOT_DIR="/storage/shared/mimic-iv/meds_v0.3.2/"  # Replace with your actual root directory
PRETRAIN_OUTPUT_DIR="${ROOT_DIR}/results/zero_shot/eic_hparam_sweep"
MODEL_SWEEP_DIR=$(meds-torch-latest-dir path=${PRETRAIN_OUTPUT_DIR})
BEST_CHECKPOINT="${MODEL_SWEEP_DIR}/checkpoints/best_model.ckpt"
BEST_CONFIG="${MODEL_SWEEP_DIR}/best_config.json"
MEDS_DIR="${ROOT_DIR}/meds/"
TENSOR_DIR="${ROOT_DIR}/eic_tensors/"
TASKS_DIR="${MEDS_DIR}/tasks/"
NUM_SAMPLES=18
N_WORKERS=6

# Loop over tasks
TASK_NAME="time/hospital_admission/90d"
echo "Processing task: ${TASK_NAME}"

OUTPUT_DIR="${ROOT_DIR}/results/zero_shot/inference/eic/${TASK_NAME}"
TASK_CONFIG_PATH="$(pwd)/ZERO_SHOT_TUTORIAL/configs/tasks/eic/${TASK_NAME}.yaml"
# trainer.devices=[0,1,2,3,5,7]
# ^ this arg in conjunction with do_manual_gpu_scheduling=true maps jobs to a single gpu in the list

meds-torch-generate --multirun hydra/launcher=joblib experiment=eic_inference_mtr \
    model/trajectory_labeler=aces_schema_labeler model.trajectory_labeler.yaml_path=$TASK_CONFIG_PATH \
    "paths.time_output_dir=/storage/shared/mimic-iv/meds_v0.3.2/results/zero_shot/inference/eic/time/hospital_admission/90d/2024-12-31_13-52-19_129584/" \
    do_manual_gpu_scheduling=true hydra.launcher.n_jobs=$N_WORKERS worker_num="range(0,$N_WORKERS)" num_samples=$NUM_SAMPLES \
    data.dataloader.batch_size=512 trainer.devices=[0,1,2,3,5,7] data.predict_dataset=test \
    data.do_include_subject_id=true data.do_include_prediction_time=true data.do_include_end_time=true \
    data.task_name=${TASK_NAME} data.task_root_dir=${TASKS_DIR} \
    paths.meds_cohort_dir=${MEDS_DIR} ckpt_path=${BEST_CHECKPOINT} \
    paths.data_dir=${TENSOR_DIR} paths.output_dir=${OUTPUT_DIR} \
    "hydra.searchpath=[pkg://meds_torch.configs,$(pwd)/ZERO_SHOT_TUTORIAL/configs/]"