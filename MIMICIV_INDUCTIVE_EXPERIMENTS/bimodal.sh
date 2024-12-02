export MIMICIV_MEDS_COHORT_DIR=/ssd-shared/meds_test/cohort 
export MIMICIV_TRIPLET_TENSOR_DIR=/ssd-shared/meds_test/triplets 
export TASK_NAME="readmission/30d"
export TASKS_DIR="${MIMICIV_MEDS_COHORT_DIR}/tasks/"
export OUTPUT_DIR="$(pwd)/results" 
export TEXT_ROOT="/ssd-shared/physionet.org/files/mimic-iv-note/2.2/note/discharge.csv.gz"
export CUDA_VISIBLE_DEVICES=4,5,6,7

export METHOD=bicl
MAX_POLARS_THREADS=4 meds-torch-tune callbacks=tune_default \
    hparams_search.ray.resources_per_trial.gpu=1 data.dataloader.num_workers=16 \
    hparams_search=ray_tune experiment=triplet_bimodal_mtr paths.data_dir=${MIMICIV_TRIPLET_TENSOR_DIR} \
    paths.meds_cohort_dir=${MIMICIV_MEDS_COHORT_DIR} paths.output_dir=${OUTPUT_DIR} \
    model=$METHOD data=bimodal_pytorch_dataset data.task_root_dir=${TASKS_DIR} data.text_root=$TEXT_ROOT\
    hydra.searchpath=[pkg://meds_torch.configs,"$(pwd)/MIMICIV_INDUCTIVE_EXPERIMENTS/configs/meds-torch-configs"]
