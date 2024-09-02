#!/usr/bin/env bash

# This makes the script fail if any internal script fails
set -e

# Function to display help message
function display_help() {
    echo "Usage: $0 <MEDS_DIR> <N_PARALLEL_WORKERS>"
    echo
    echo "This script processes MIMIC-IV data through several steps, handling raw data conversion,"
    echo "sharding events, splitting subjects, converting to sharded events, and merging into a MEDS cohort."
    echo
    echo "Arguments:"
    echo "  MEDS_DIR                              Input directory for processed MEDS data."
    echo "  MODEL_DIR                             Output directory for processed model data."
    echo "  N_PARALLEL_WORKERS                    Number of parallel workers for processing."
    echo "  CONFIG_DIR                            Directory for config files."
    echo "  CONFIG_NAME                           Name of config file."
    echo
    echo "Options:"
    echo "  -h, --help          Display this help message and exit."
    exit 1
}

# Check if the first parameter is '-h' or '--help'
if [[ "$1" == "-h" || "$1" == "--help" ]]; then
    display_help
fi

# Check for mandatory parameters
if [ "$#" -lt 3 ]; then
    echo "Error: Incorrect number of arguments provided."
    display_help
fi

MEDS_DIR="$1"
MODEL_DIR="$2"
N_PARALLEL_WORKERS="$3"
CONFIG_DIR="$4"
CONFIG_NAME="$5"

echo "Converting to code metadata..."
MEDS_transform-aggregate_code_metadata --multirun \
    --config-path="$CONFIG_DIR" \
    --config-name="$CONFIG_NAME" \
    worker="range(0,${N_PARALLEL_WORKERS})" \
    input_dir="$MEDS_DIR" \
    cohort_dir="$MODEL_DIR" \
    stage="aggregate_code_metadata" \
    hydra.searchpath="[pkg://MEDS_transforms.configs]" \
    polling_time=5

echo "Filtering subjects..."
MEDS_transform-filter_subjects --multirun \
    --config-path="$CONFIG_DIR" \
    --config-name="$CONFIG_NAME" \
    worker="range(0,${N_PARALLEL_WORKERS})" \
    input_dir="$MEDS_DIR" \
    cohort_dir="$MODEL_DIR" \
    stage="filter_subjects" \
    hydra.searchpath="[pkg://MEDS_transforms.configs]"

echo "Generating time derived measurements..."
MEDS_transform-add_time_derived_measurements --multirun \
    --config-path="$CONFIG_DIR" \
    --config-name="$CONFIG_NAME" \
    worker="range(0,${N_PARALLEL_WORKERS})" \
    input_dir="$MEDS_DIR" \
    cohort_dir="$MODEL_DIR" \
    stage="add_time_derived_measurements" \
    hydra.searchpath="[pkg://MEDS_transforms.configs]"

echo "Filtering measurements..."
MEDS_transform-filter_measurements --multirun \
    --config-path="$CONFIG_DIR" \
    --config-name="$CONFIG_NAME" \
    worker="range(0,${N_PARALLEL_WORKERS})" \
    input_dir="$MEDS_DIR" \
    cohort_dir="$MODEL_DIR" \
    stage="filter_measurements" \
    hydra.searchpath="[pkg://MEDS_transforms.configs]"

echo "Occluding outliers..."
MEDS_transform-occlude_outliers --multirun \
    --config-path="$CONFIG_DIR" \
    --config-name="$CONFIG_NAME" \
    worker="range(0,${N_PARALLEL_WORKERS})" \
    input_dir="$MEDS_DIR" \
    cohort_dir="$MODEL_DIR" \
    stage="occlude_outliers" \
    hydra.searchpath="[pkg://MEDS_transforms.configs]"

# We rerun aggregate_code_metadata to remove the filtered out codes, it seems the filter_measurements stage does not do that in meds v0.0.6
rm -rf "${MODEL_DIR}/aggregate_code_metadata"
rm -f "${MODEL_DIR}/metadata/codes.parquet"
echo "Converting to code metadata..."
MEDS_transform-aggregate_code_metadata \
    --config-path="$CONFIG_DIR" \
    --config-name="$CONFIG_NAME" \
    input_dir="$MODEL_DIR" \
    cohort_dir="$MODEL_DIR" \
    stage="aggregate_code_metadata" \
    hydra.searchpath="[pkg://MEDS_transforms.configs]" \
    ++stage_configs.aggregate_code_metadata.data_input_dir="${MODEL_DIR}/occlude_outliers/" \
    ++stage_configs.aggregate_code_metadata.metadata_input_dir="${MODEL_DIR}/metadata/" \
    polling_time=5

mkdir -p "${MODEL_DIR}/metadata/"
cp "${MODEL_DIR}/aggregate_code_metadata/codes.parquet" "${MODEL_DIR}/metadata/codes.parquet"

echo "Fitting vocabulary indices..."
MEDS_transform-fit_vocabulary_indices --multirun \
    --config-path="$CONFIG_DIR" \
    --config-name="$CONFIG_NAME" \
    worker="range(0,${N_PARALLEL_WORKERS})" \
    input_dir="$MEDS_DIR" \
    cohort_dir="$MODEL_DIR" \
    stage="fit_vocabulary_indices" \
    hydra.searchpath="[pkg://MEDS_transforms.configs]"

echo "Normalizing data (converting codes to use integer encodings)..."
MEDS_transform-normalization --multirun \
    --config-path="$CONFIG_DIR" \
    --config-name="$CONFIG_NAME" \
    worker="range(0,${N_PARALLEL_WORKERS})" \
    input_dir="$MEDS_DIR" \
    cohort_dir="$MODEL_DIR" \
    stage="normalization" \
    hydra.searchpath="[pkg://MEDS_transforms.configs]"


echo "Converting to tokenization..."
python -m meds_torch.utils.custom_tokenization \
    --config-path="$CONFIG_DIR" \
    --config-name="$CONFIG_NAME" \
    input_dir="$MEDS_DIR" \
    cohort_dir="$MODEL_DIR" \
    stage="tokenization" \
    hydra.searchpath="[pkg://MEDS_transforms.configs]"

echo "Converting to tensor..."
MEDS_transform-tensorization  --multirun \
    --config-path="$CONFIG_DIR" \
    --config-name="$CONFIG_NAME" \
    worker="range(0,${N_PARALLEL_WORKERS})" \
    input_dir="$MEDS_DIR" \
    cohort_dir="$MODEL_DIR" \
    stage="tensorization" \
    hydra.searchpath="[pkg://MEDS_transforms.configs]"
