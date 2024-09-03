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
    echo "  MEDS_DIR                              Output directory for processed MEDS data."
    echo "  N_PARALLEL_WORKERS                    Number of parallel workers for processing."
    echo "  PIPELINE_CONFIG_PATH                  Pipeline configuration file."
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
if [ "$#" -lt 4 ]; then
    echo "Error: Incorrect number of arguments provided."
    display_help
fi

export MEDS_DIR="$1"
export MODEL_DIR="$2"
export N_WORKERS="$3"
export PIPELINE_CONFIG_PATH="$4"

shift 4

echo "Running extraction pipeline."
MEDS_transform-runner "pipeline_config_fp=$PIPELINE_CONFIG_PATH" "$@"
