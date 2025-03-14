#!/bin/bash
set -e

ROOT_DIR="/storage/shared/hf_subtype/bryan_mgb_cohort/"
MEDS_DIR=${ROOT_DIR}/meds/
EIC_DIR=${ROOT_DIR}/model/
OUTPUT_DIR=${ROOT_DIR}/results/debug/multimodal_early_fusion/
meds-torch-train \
    experiment=multimodal_supervised paths.data_dir=${EIC_DIR} \
    paths.meds_cohort_dir=${MEDS_DIR} paths.output_dir=${OUTPUT_DIR} \
    hydra.searchpath=[pkg://meds_torch.configs,$(pwd)/MIMICIV_INDUCTIVE_EXPERIMENTS/configs/meds-torch-configs]