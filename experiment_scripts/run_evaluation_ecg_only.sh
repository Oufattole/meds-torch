#!/bin/bash

# Set CUDA GPUs
export CUDA_VISIBLE_DEVICES=5

# Set experiment environment variables
METHOD=ecg_only
TASK_NAME=mortality/in_icu/ecg

# Set MEDS environment variables
ROOT_DIR=/storage/shared/mimic-iv/multimodal_ecg/
MEDS_DIR=${ROOT_DIR}/meds/
TENSOR_DIR=${ROOT_DIR}/ecg_triplet_tensors/
OUTPUT_DIR=${ROOT_DIR}/results/predict-test/${METHOD}/${TASK_NAME}/
DATETIME=2024-12-05_23-03-05_461322
OUTPUT_DIR_FROM_TRAINING=${ROOT_DIR}/results/sweep/${METHOD}/${TASK_NAME}/${DATETIME}
BEST_CHECKPOINT=${OUTPUT_DIR_FROM_TRAINING}/checkpoints/best_model.ckpt
TASKS_DIR=${MEDS_DIR}/tasks/
CONFIGS_FOLDER=MULTIMODAL_TUTORIAL

# Run experiment
python -m meds_torch.predict \
        +experiment=ecg_triplet_mtr \
	    ckpt_path=${BEST_CHECKPOINT} \
    	paths.data_dir=${TENSOR_DIR} \
    	paths.meds_cohort_dir=${MEDS_DIR} \
    	paths.output_dir=${OUTPUT_DIR} \
		data.predict_dataset=test \
    	data.task_name=${TASK_NAME} \
    	data.task_root_dir=${TASKS_DIR} \
        data.do_include_subject_id=true \
        data.do_include_prediction_time=true \
    	hydra.searchpath=[pkg://meds_torch.configs,./MIMICIV_INDUCTIVE_EXPERIMENTS/configs/meds-torch-configs] \
		model=multimodal_supervised \
        model.input_encoder.early_fusion=false \
		model.input_encoder.ecg_embedder.embedding_dim=128 \
		model.isolate_ecg=true
