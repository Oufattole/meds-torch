#!/usr/bin/env bash

# This makes the script fail if any internal script fails
set -e

# Raw synthetic data CSVs to meds data using meds-transforms
# TODO: add text synthetic data
# python tests/helpers/extract_test_data.py

# Generate test data tensors (mostly uses meds-transforms)
# TODO: add tokenized text to the nested ragged tensor in a final stage "custom_text_tensorization"
python tests/helpers/generate_test_data_tensors.py

# python tests/helpers/generate_test_windows.py
