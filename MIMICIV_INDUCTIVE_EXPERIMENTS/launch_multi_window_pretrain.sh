#!/bin/bash

ROOT_DIR="$1"
CONDA_ENV="$2"
METHOD="$3"

shift 3

set -e  # Exit immediately if a command exits with a non-zero status.

# Function to run a job
run_job() {
    local method=$1
    local experiment=$2
    local tensor_dir=$3
    local root_dir=$4
    local conda_env=$5

    echo "Running job for ${method}..."

    export METHOD=${method}
    export CONFIGS_FOLDER="MIMICIV_INDUCTIVE_EXPERIMENTS"
    export ROOT_DIR=${root_dir}
    export MEDS_DIR="${ROOT_DIR}/meds/"
    export TENSOR_DIR=${ROOT_DIR}/${tensor_dir}_tensors/
    export OUTPUT_DIR=${ROOT_DIR}/results/${METHOD}/${experiment}/
    PRETRAIN_SWEEP_DIR=${OUTPUT_DIR}/pretrain/sweep/

    # Activate conda environment
    source $(conda info --base)/etc/profile.d/conda.sh
    conda activate ${conda_env}
    echo $OUTPUT_DIR


    CHECK_FILE=$(meds-torch-latest-dir path=${PRETRAIN_SWEEP_DIR})/sweep_results_summary.parquet 2>/dev/null || CHECK_FILE=""

    if [ -z "$CHECK_FILE" ] || [ ! -f "$CHECK_FILE" ]; then
        MAX_POLARS_THREADS=4 meds-torch-tune callbacks=tune_default trainer=ray \
            hparams_search.ray.resources_per_trial.GPU=1 data.dataloader.num_workers=16 \
            hparams_search=ray_tune experiment=$experiment paths.data_dir=${TENSOR_DIR} \
            paths.meds_cohort_dir=${MEDS_DIR} paths.output_dir=${PRETRAIN_SWEEP_DIR} \
            model=$METHOD data=random_windows_pytorch_dataset \
            hydra.searchpath=[pkg://meds_torch.configs,$(pwd)/${CONFIGS_FOLDER}/configs/meds-torch-configs]
    else
        echo "${CHECK_FILE} already exists. Skipping the job execution."
    fi

    echo "Job for ${method} completed."
}

# Run jobs sequentially
run_job "$METHOD" "eic_mtr" "eic" "$ROOT_DIR" "$CONDA_ENV"
run_job "$METHOD" "triplet_mtr" "triplet" "$ROOT_DIR" "$CONDA_ENV"

echo "All jobs completed sequentially."
