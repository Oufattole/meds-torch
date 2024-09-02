#!/usr/bin/env bash

# This makes the script fail if any internal script fails
set -e

# add environment variables
# shellcheck disable=SC1091
. scripts/hf_cohort/.env

# echo "Cleaning RPDR data"
# python scripts/clean_rpdr.py output_dir="${PREMEDS_DIR}" input_dir="${RAW_RPDR_DIR}"

# echo "Merging parquets"
# python scripts/merge_parquets.py pre_meds_dir="${PREMEDS_DIR}"

# echo "Processing Encounters"
# python scripts/process_encounters.py pre_meds_dir="${MERGED_PREMEDS_DIR}"


echo "Running shard_events.py with $N_PARALLEL_WORKERS workers in parallel"
"MEDS_extract-shard_events" \
    --multirun \
    worker="range(0,$N_PARALLEL_WORKERS)" \
    hydra/launcher=joblib \
    input_dir="$MERGED_PREMEDS_DIR" \
    cohort_dir="$MEDS_DIR" \
    stage="shard_events" \
    stage_configs.shard_events.infer_schema_length=999999999 \
    etl_metadata.dataset_name="hf_cohort" \
    etl_metadata.dataset_version="1.0" \
    event_conversion_config_fp=scripts/hf_cohort/hf_cohort.yaml "$@"


echo "Splitting patients in serial"
"MEDS_extract-split_and_shard_patients" \
    input_dir="$MERGED_PREMEDS_DIR" \
    cohort_dir="$MEDS_DIR" \
    stage="split_and_shard_patients" \
    etl_metadata.dataset_name="hf_cohort" \
    etl_metadata.dataset_version="1.0" \
    event_conversion_config_fp=scripts/hf_cohort/hf_cohort.yaml "$@"

echo "Converting to sharded events with $N_PARALLEL_WORKERS workers in parallel"
"MEDS_extract-convert_to_sharded_events" \
    --multirun \
    worker="range(0,$N_PARALLEL_WORKERS)" \
    hydra/launcher=joblib \
    input_dir="$MERGED_PREMEDS_DIR" \
    cohort_dir="$MEDS_DIR" \
    stage="convert_to_sharded_events" \
    etl_metadata.dataset_name="hf_cohort" \
    etl_metadata.dataset_version="1.0" \
    event_conversion_config_fp=scripts/hf_cohort/hf_cohort.yaml "$@"

echo "Merging to a MEDS cohort with $N_PARALLEL_WORKERS workers in parallel"
"MEDS_extract-merge_to_MEDS_cohort" \
    input_dir="$MERGED_PREMEDS_DIR" \
    cohort_dir="$MEDS_DIR" \
    stage="merge_to_MEDS_cohort" \
    etl_metadata.dataset_name="hf_cohort" \
    etl_metadata.dataset_version="1.0" \
    event_conversion_config_fp=scripts/hf_cohort/hf_cohort.yaml "$@"

echo "Extracting Metadata in parallel"
"MEDS_extract-extract_code_metadata" \
    input_dir="$MERGED_PREMEDS_DIR" \
    cohort_dir="$MEDS_DIR" \
    stage="merge_to_MEDS_cohort" \
    etl_metadata.dataset_name="hf_cohort" \
    etl_metadata.dataset_version="1.0" \
    event_conversion_config_fp=scripts/hf_cohort/hf_cohort.yaml "$@"

echo "Finalizing MEDS metadata in serial."
MEDS_extract-finalize_MEDS_data \
    input_dir="$MERGED_PREMEDS_DIR" \
    cohort_dir="$MEDS_DIR" \
    stage="finalize_MEDS_metadata" \
    etl_metadata.dataset_name="hf_cohort" \
    etl_metadata.dataset_version="1.0" \
    event_conversion_config_fp=scripts/hf_cohort/hf_cohort.yaml "$@"

echo "Finalizing MEDS metadata in serial."
MEDS_extract-finalize_MEDS_metadata \
    input_dir="$MERGED_PREMEDS_DIR" \
    cohort_dir="$MEDS_DIR" \
    stage="finalize_MEDS_metadata" \
    etl_metadata.dataset_name="hf_cohort" \
    etl_metadata.dataset_version="1.0" \
    event_conversion_config_fp=scripts/hf_cohort/hf_cohort.yaml "$@"
