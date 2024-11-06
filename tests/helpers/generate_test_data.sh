#!/usr/bin/env bash

# This makes the script fail if any internal script fails
set -e

# MEDS Cohort Variables
export RAW_COHORT=$(pwd)/tests/test_data/raw_cohort
export MEDS_DIR=$(pwd)/tests/test_data/MEDS_cohort

# Tensorize MEDS Variables
export EIC_DIR=$(pwd)/tests/test_data/eic_tensors
export MULTIMODAL_DIR=$(pwd)/tests/test_data/multimodal_triplet_tensors

# ACES Task Variables
export TASK_PATH=$(pwd)/tests/helpers/configs/aces_random_windows.yaml
export TASK_OUTPUT_FP=$(pwd)/tests/test_data/windows/raw_windows.parquet
export TASK_WINDOW_STATS_DIR=$(pwd)/tests/test_data/windows/

# Raw synthetic data CSVs to meds data using meds-transforms
python tests/helpers/generate_synthetic_data.py

# Extract synthetic data in MEDS format
export EVENT_CONVERSION_CONFIG_FP=$(pwd)/tests/helpers/configs/events_config.yaml
export PIPELINE_CONFIG_FP=$(pwd)/tests/helpers/configs/extraction_pipeline_config.yaml
MEDS_transform-runner "pipeline_config_fp=$PIPELINE_CONFIG_FP"


# Generate test data tensors (mostly uses meds-transforms)

export PIPELINE_CONFIG_FP=$(pwd)/tests/helpers/configs/triplet_config.yaml
export MODEL_DIR=$(pwd)/tests/test_data/triplet_tensors
MEDS_transform-runner "pipeline_config_fp=$PIPELINE_CONFIG_FP"

export PIPELINE_CONFIG_FP=$(pwd)/tests/helpers/configs/eic_config.yaml
export MODEL_DIR=$(pwd)/tests/test_data/eic_tensors
MEDS_transform-runner "pipeline_config_fp=$PIPELINE_CONFIG_FP"

export PIPELINE_CONFIG_FP=$(pwd)/tests/helpers/configs/multimodal_triplet_config.yaml
export MODEL_DIR=$(pwd)/tests/test_data/multimodal_triplet_tensors
MEDS_transform-runner "pipeline_config_fp=$PIPELINE_CONFIG_FP"

# Extract Pre event and post event windows using ACES -- allowing temporal contrastive learning
# methods via the multiwindow_pytorch_dataset class
# aces-cli \
#     data.path="${MEDS_DIR}/**/*.parquet" \
#     data.standard="meds" \
#     cohort_dir=$MEDS_DIR \
#     config_path=raw_windows \
#     config_path=$TASK_PATH \
#     output_filepath=$TASK_OUTPUT_FP
#     window_stats_dir=$TASK_WINDOW_STATS_DIR
