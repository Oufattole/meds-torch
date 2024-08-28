#!/usr/bin/env bash

# This makes the script fail if any internal script fails
set -e

# Function to display help message
function display_help() {
    echo "Usage: $0 <MIMICIV_MEDS_DIR> <N_PARALLEL_WORKERS>"
    echo
    echo "This script processes MIMIC-IV data through several steps, handling raw data conversion,"
    echo "sharding events, splitting patients, converting to sharded events, and merging into a MEDS cohort."
    echo
    echo "Arguments:"
    echo "  MIMICIV_MEDS_DIR                              Output directory for processed MEDS data."
    echo "  N_PARALLEL_WORKERS                            Number of parallel workers for processing."
    echo "  (OPTIONAL) do_unzip=true OR do_unzip=false    Optional flag to unzip csv files before processing."
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
if [ "$#" -lt 2 ]; then
    echo "Error: Incorrect number of arguments provided."
    display_help
fi

MIMICIV_MEDS_DIR="$1"
N_PARALLEL_WORKERS="$2"


echo "Filtering patients..."
MEDS_transform-filter_patients \
    --multirun \
    worker="range(0,$N_PARALLEL_WORKERS)" \
    hydra/launcher=joblib \
    "input_dir=${MIMICIV_MEDS_DIR}" \
    "cohort_dir=${MIMICIV_MEDS_DIR}/data" \
    "stage_configs.add_time_derived_measurements.age.DOB_code=DOB" \
    "stages=[filter_patients,filter_measurements,occlude_outliers,fit_vocabulary_indices,normalization,tokenization,tensorization]" \
    "stage=filter_patients" \


echo "Filtering measurements..."
MEDS_transform-filter_measurements \
    --multirun \
    worker="range(0,$N_PARALLEL_WORKERS)" \
    hydra/launcher=joblib \
    "input_dir=${MIMICIV_MEDS_DIR}" \
    "cohort_dir=${MIMICIV_MEDS_DIR}/data" \
    "stage_configs.add_time_derived_measurements.age.DOB_code=DOB" \
    "stages=[filter_patients,filter_measurements,occlude_outliers,fit_vocabulary_indices,normalization,tokenization,tensorization]" \
    "stage=filter_measurements"


echo "Occluding outliers..."
MEDS_transform-occlude_outliers \
    --multirun \
    worker="range(0,$N_PARALLEL_WORKERS)" \
    hydra/launcher=joblib \
    "input_dir=${MIMICIV_MEDS_DIR}" \
    "cohort_dir=${MIMICIV_MEDS_DIR}/data" \
    "stage_configs.add_time_derived_measurements.age.DOB_code=DOB" \
    "stages=[filter_patients,filter_measurements,occlude_outliers,fit_vocabulary_indices,normalization,tokenization,tensorization]" \
    "stage=occlude_outliers"

echo "Fitting vocabulary indices..."
MEDS_transform-fit_vocabulary_indices \
    --multirun \
    worker="range(0,$N_PARALLEL_WORKERS)" \
    hydra/launcher=joblib \
    "input_dir=${MIMICIV_MEDS_DIR}" \
    "cohort_dir=${MIMICIV_MEDS_DIR}/data" \
    "stage_configs.add_time_derived_measurements.age.DOB_code=DOB" \
    "stages=[filter_patients,filter_measurements,occlude_outliers,fit_vocabulary_indices,normalization,tokenization,tensorization]" \
    "stage=fit_vocabulary_indices"

cp "${MIMICIV_MEDS_DIR}/data/metadata/codes.parquet" "${MIMICIV_MEDS_DIR}/data/codes.parquet"

echo "Normalizing data (converting codes to use integer encodings)..."
MEDS_transform-normalization \
    --multirun \
    worker="range(0,$N_PARALLEL_WORKERS)" \
    hydra/launcher=joblib \
    "input_dir=${MIMICIV_MEDS_DIR}" \
    "cohort_dir=${MIMICIV_MEDS_DIR}/data" \
    "stage_configs.add_time_derived_measurements.age.DOB_code=DOB" \
    "stages=[filter_patients,filter_measurements,occlude_outliers,fit_vocabulary_indices,normalization,tokenization,tensorization]" \
    "stage=normalization"

echo "Converting to tokenization..."
MEDS_transform-tokenization \
    --multirun \
    worker="range(0,$N_PARALLEL_WORKERS)" \
    hydra/launcher=joblib \
    "input_dir=${MIMICIV_MEDS_DIR}" \
    "cohort_dir=${MIMICIV_MEDS_DIR}/data" \
    "stage_configs.add_time_derived_measurements.age.DOB_code=DOB" \
    "stages=[filter_patients,filter_measurements,occlude_outliers,fit_vocabulary_indices,normalization,tokenization,tensorization]" \
    "stage=tokenization"

echo "Converting to tensor..."
MEDS_transform-tensorization \
    --multirun \
    worker="range(0,$N_PARALLEL_WORKERS)" \
    hydra/launcher=joblib \
    "input_dir=${MIMICIV_MEDS_DIR}" \
    "cohort_dir=${MIMICIV_MEDS_DIR}/data" \
    "stage_configs.add_time_derived_measurements.age.DOB_code=DOB" \
    "stages=[filter_patients,filter_measurements,occlude_outliers,fit_vocabulary_indices,normalization,tokenization,tensorization]" \
    "stage=tensorization"
