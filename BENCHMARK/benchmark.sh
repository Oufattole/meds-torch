#!/bin/bash

ROOT_DIR="$1"
CONDA_ENV="$2"

shift 2

set -e  # Exit immediately if a command exits with a non-zero status.

# Function to run a job
run_job() {
    local task_name=$1
    local experiment=$2
    local tensor_dir=$3
    local root_dir=$4
    local conda_env=$5

    echo "Running job for ${task_name}..."

    export METHOD=supervised
    export CONFIGS_FOLDER="MIMICIV_INDUCTIVE_EXPERIMENTS"
    export ROOT_DIR=${root_dir}
    export MEDS_DIR="${ROOT_DIR}/meds/"
    export TASKS_DIR=${MEDS_DIR}/tasks/
    export TENSOR_DIR=${ROOT_DIR}/${tensor_dir}_tensors/
    export OUTPUT_DIR=${ROOT_DIR}/benchmark/${METHOD}/${experiment}/${task_name}/
    BENCHMARK_DIR=${OUTPUT_DIR}/

    # Activate conda environment
    source $(conda info --base)/etc/profile.d/conda.sh
    conda activate ${conda_env}

    MAX_POLARS_THREADS=4 meds-torch-benchmark \
        data.dataloader.num_workers=0 \
        experiment=$experiment paths.data_dir=${TENSOR_DIR} \
        paths.meds_cohort_dir=${MEDS_DIR} paths.output_dir=${BENCHMARK_DIR} \
        data.task_name=$task_name data.task_root_dir=$TASKS_DIR \
        hydra.searchpath=[pkg://meds_torch.configs,$(pwd)/${CONFIGS_FOLDER}/configs/meds-torch-configs]

    echo "Job for ${task_name} completed."
}

TASKS=(
    "mortality/in_hospital/first_24h"
    # "mortality/in_icu/first_24h"
    # "mortality/post_hospital_discharge/1y"
    # "readmission/30d"
)

# Run jobs sequentially
for TASK_NAME in "${TASKS[@]}"; do
    run_job ${TASK_NAME} "triplet_mtr" "triplet" "$ROOT_DIR" "$CONDA_ENV"
done

echo "All jobs completed sequentially."
