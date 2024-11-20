# Extract MEDS format from MIMIC-IV
export TEST_DIR="${HOME}/meds_transforms_extraction"
export MIMICIV_MEDS_COHORT_DIR=/ssd-shared/meds_test/cohort # set to the directory in which you want to store the raw MIMIC-IV data

cd $TEST_DIR

export MIMICIV_TRIPLET_TENSOR_DIR=/ssd-shared/meds_test/triplets # set to the directory in which you want to output the tensorized MIMIC-IV data
export TASK_NAME="readmission/30d"
export TASKS_DIR="${MIMICIV_MEDS_COHORT_DIR}/tasks/"
export OUTPUT_DIR="$(pwd)/results" # set to the directory in which you want to output checkpoints and results
export CUDA_VISIBLE_DEVICES=6 # set to the GPU you want to use

export METHOD=bicl
MAX_POLARS_THREADS=4 meds-torch-tune callbacks=tune_default \
    hparams_search.ray.resources_per_trial.gpu=1 data.dataloader.num_workers=16 \
    hparams_search=ray_tune experiment=triplet_mtr paths.data_dir=${MIMICIV_TRIPLET_TENSOR_DIR} \
    paths.meds_cohort_dir=${MIMICIV_MEDS_COHORT_DIR} paths.output_dir=${OUTPUT_DIR} \
    model=$METHOD data=bimodal_pytorch_dataset \
    hydra.searchpath=[pkg://meds_torch.configs,"$(pwd)/meds-torch/MIMICIV_INDUCTIVE_EXPERIMENTS/configs/meds-torch-configs"]
