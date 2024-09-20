#!/bin/bash

ROOT_DIR="$1"
CONDA_ENV="$2"

shift 2

set -e  # Exit immediately if a command exits with a non-zero status.

# Function to launch a job
launch_job() {
    local gpu=$1
    local task_name=$2
    local experiment=$3
    local tensor_dir=$4
    local root_dir=$5
    local conda_env=$6

    echo "Attempting to create tmux session for job_${gpu}..."

    # Create a new tmux session
    tmux new-session -d -s "job_${gpu}" || {
        echo "Failed to create tmux session for job_${gpu}"
        return 1
    }

    # Send commands to the tmux session
    # export CUDA_VISIBLE_DEVICES=$gpu
    tmux send-keys -t "job_${gpu}" "
    export ROOT_DIR=${root_dir}
    export MIMICIV_MEDS_DIR='${ROOT_DIR}/meds/'
    export TASKS_DIR=\${MIMICIV_MEDS_DIR}/tasks/
    export MIMICIV_${tensor_dir}_DIR=${ROOT_DIR}/${tensor_dir}_tensors/
    export OUTPUT_DIR=${ROOT_DIR}/results/test/${experiment}_exp/${task_name}/
    conda activate ${conda_env}

    MAX_POLARS_THREADS=4 meds-torch-tune callbacks=tune_default model.backbone.n_layers=2 model.backbone.nheads=4 model.token_dim=32 \
        hparams_search=ray_tune experiment=$experiment paths.data_dir=\${MIMICIV_${tensor_dir}_DIR} \
        paths.meds_cohort_dir=\${MIMICIV_MEDS_DIR} paths.output_dir=\${OUTPUT_DIR} \
        data.task_name=$task_name data.task_root_dir=\$TASKS_DIR \
        hydra.searchpath=[pkg://meds_torch.configs,\$(pwd)/MIMICIV_TUTORIAL/configs/meds-torch-configs]
    " Enter

    echo "Tmux session job_${gpu} created and command sent."
}

# Try to kill all existing tmux sessions, but don't error if there are none
tmux kill-server 2>/dev/null || true

# Launch jobs
# launch_job 0 "icu_mortality" "eic_mtr" "eic" "$ROOT_DIR" "$CONDA_ENV"
# launch_job 1 "long_los" "eic_mtr" "eic" "$ROOT_DIR" "$CONDA_ENV"
launch_job 2 "icu_mortality" "triplet_mtr" "triplet" "$ROOT_DIR" "$CONDA_ENV"
# launch_job 3 "long_los" "triplet_mtr" "triplet" "$ROOT_DIR" "$CONDA_ENV"
# launch_job 4 "icu_mortality" "text_code_mtr" "triplet" "$ROOT_DIR" "$CONDA_ENV"
# launch_job 5 "long_los" "text_code_mtr" "triplet" "$ROOT_DIR" "$CONDA_ENV"

echo "All jobs launched. Checking tmux sessions..."
tmux ls || echo "No tmux sessions found."

echo "Use 'tmux attach-session -t job_X' to view a specific job, where X is the GPU number (0-5)."
