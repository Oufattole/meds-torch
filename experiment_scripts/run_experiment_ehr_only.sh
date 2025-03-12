#!/bin/bash

# Set CUDA GPUs
export CUDA_VISIBLE_DEVICES=0,1,2,3

# Set experiment environment variables
METHOD=ehr_only
TASK_NAME=mortality/in_icu/ecg

# Set MEDS environment variables
ROOT_DIR=/storage/shared/mimic-iv/multimodal_ecg/
MEDS_DIR=${ROOT_DIR}/meds/
TENSOR_DIR=${ROOT_DIR}/ecg_triplet_tensors/
OUTPUT_DIR=${ROOT_DIR}/results/sweep/${METHOD}/${TASK_NAME}/
TASKS_DIR=${MEDS_DIR}/tasks/
CONFIGS_FOLDER=MULTIMODAL_TUTORIAL

# Run experiment
# meds-torch-train \
# 		experiment=ecg_triplet_mtr \
#     	paths.data_dir=${TENSOR_DIR} \
#     	paths.meds_cohort_dir=${MEDS_DIR} \
#     	paths.output_dir=${OUTPUT_DIR} \
#     	data.task_name=${TASK_NAME} \
#     	data.task_root_dir=${TASKS_DIR} \
#     	hydra.searchpath=[pkg://meds_torch.configs,./MIMICIV_INDUCTIVE_EXPERIMENTS/configs/meds-torch-configs] \
# 		model=multimodal_supervised \
#         model.input_encoder.early_fusion=false \
# 		model.isolate_ehr=true

# Tune it
# meds-torch-tune \
#     	callbacks=tune_default \
# 		trainer=ray \
# 		hparams_search=ray_tune \
# 		experiment=ecg_triplet_mtr \
#     	hparams_search.ray.resources_per_trial.gpu=1  \
# 		hparams_search.ray.num_samples=8 \
#     	paths.data_dir=${TENSOR_DIR} \
#     	paths.meds_cohort_dir=${MEDS_DIR} \
#     	paths.output_dir=${OUTPUT_DIR} \
#     	data.task_name=${TASK_NAME} \
#     	data.task_root_dir=${TASKS_DIR} \
#     	hydra.searchpath=[pkg://meds_torch.configs,./MIMICIV_INDUCTIVE_EXPERIMENTS/configs/meds-torch-configs] \
# 		model=multimodal_supervised \
# 		model.input_encoder.early_fusion=false \
# 		model.isolate_ehr=true

DATETIME=2024-12-05_16-02-12_348871
OUTPUT_DIR_FROM_TRAINING=${ROOT_DIR}/results/sweep/${METHOD}/${TASK_NAME}/${DATETIME}
BEST_CONFIG_PATH=${OUTPUT_DIR_FROM_TRAINING}/best_config.json
FINETUNE_MULTISEED_DIR=${OUTPUT_DIR}/${DATETIME}/finetune/multiseed/

# Multiseed Sweep
meds-torch-tune \
		callbacks=tune_default \
		trainer=ray \
		best_config_path=${BEST_CONFIG_PATH} \
        hparams_search=ray_multiseed \
		experiment=ecg_triplet_mtr \
		hparams_search.ray.resources_per_trial.gpu=1  \
 		hparams_search.ray.num_samples=8 \
		paths.data_dir=${TENSOR_DIR} \
        paths.meds_cohort_dir=${MEDS_DIR} \
		paths.output_dir=${FINETUNE_MULTISEED_DIR} \
        data.task_name=${TASK_NAME} \
		data.task_root_dir=${TASKS_DIR} \
        hydra.searchpath=[pkg://meds_torch.configs,./MIMICIV_INDUCTIVE_EXPERIMENTS/configs/meds-torch-configs] \
		model=multimodal_supervised \
		model.input_encoder.early_fusion=false \
		model.isolate_ehr=true
