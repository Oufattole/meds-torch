#!/usr/bin/env bash

# This makes the script fail if any internal script fails
set -e

python tests/helpers/extract_test_data.py

python tests/helpers/generate_test_data_tensors.py
