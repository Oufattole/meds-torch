# MIMIC-IV Example

This is an example of how to extract a MEDS dataset from MIMIC-IV. All scripts in this README are assumed to
be run **not** from this directory but from the root directory of this entire repository (e.g., one directory
up from this one).

For this tutorial make sure you are cd'd into the root directory of the repository.

## Extract MIMIC-IV MEDS Data

To get MEDS data, we can either download pre-extracted data and tasks from gcp (Option 1) or extract the data from scratch (Option 2).

Next we need to get some labels for our tasks. We will use the `long_los` and `icu_mortality` tasks as examples.

### Extract MEDS data from scratch

We can manually extract the supervised task labels from our meds dataset using [aces](https://github.com/justin13601/ACES/tree/main). First install aces:

```console
conda create -n aces python=3.12
conda activate aces
pip install es-aces==0.5.0
pip install hydra-joblib-launcher
```

Second, run the following command to extract the supervised task labels:

```console
TASKS=(
    "mortality/in_hospital/first_24h"
    "mortality/in_icu/first_24h"
    "mortality/post_hospital_discharge/1y"
    "readmission/30d"
)
TASKS_DIR="$MIMICIV_MEDS_DIR/tasks/" # set to the directory in which you want to store all tasks

for TASK_NAME in "${TASKS[@]}"; do
    SINGLE_TASK_DIR="${MIMICIV_MEDS_DIR}/tasks/${TASK_NAME}"
    mkdir -p $SINGLE_TASK_DIR # create a directory for the task
    cp MIMICIV_TUTORIAL/tasks/${TASK_NAME}.yaml "${SINGLE_TASK_DIR}.yaml"
    aces-cli --multirun hydra/launcher=joblib data=sharded data.standard=meds data.root="$MIMICIV_MEDS_DIR/data" "data.shard=$(expand_shards $MIMICIV_MEDS_DIR/data)" cohort_dir="$TASKS_DIR" cohort_name="$TASK_NAME"
done
```

## Run Tabularization and XGBoost Baseline

```console
MIMICIV_MEDS_DIR=/storage/shared/mimic-iv/meds_v0.3.2/meds/
MIMICIV_MEDS_RESHARD_DIR=/storage/shared/mimic-iv/meds_v0.3.2/reshareded_meds/
OUTPUT_TABULARIZATION_DIR=/storage/shared/mimic-iv/meds_v0.3.2/meds_tab/
TASKS=(
    "abnormal_lab/hospital_admission/creatinine/30d"
    "abnormal_lab/hospital_admission/creatinine/60d"
    "abnormal_lab/hospital_admission/creatinine/90d"
    "abnormal_lab/hospital_discharge/creatinine/30d"
    "abnormal_lab/hospital_discharge/creatinine/60d"
    "abnormal_lab/hospital_discharge/creatinine/90d"
    "abnormal_lab/icu_admission/creatinine/30d"
    "abnormal_lab/icu_admission/creatinine/60d"
    "abnormal_lab/icu_admission/creatinine/90d"
    "abnormal_lab/icu_discharge/creatinine/30d"
    "abnormal_lab/icu_discharge/creatinine/60d"
    "abnormal_lab/icu_discharge/creatinine/90d"
    "abnormal_lab/hospital_admission/hemoglobin/30d"
    "abnormal_lab/hospital_admission/hemoglobin/60d"
    "abnormal_lab/hospital_admission/hemoglobin/90d"
    "abnormal_lab/hospital_discharge/hemoglobin/30d"
    "abnormal_lab/hospital_discharge/hemoglobin/60d"
    "abnormal_lab/hospital_discharge/hemoglobin/90d"
    "abnormal_lab/icu_admission/hemoglobin/30d"
    "abnormal_lab/icu_admission/hemoglobin/60d"
    "abnormal_lab/icu_admission/hemoglobin/90d"
    "abnormal_lab/icu_discharge/hemoglobin/30d"
    "abnormal_lab/icu_discharge/hemoglobin/60d"
    "abnormal_lab/icu_discharge/hemoglobin/90d"
    "abnormal_lab/hospital_admission/hematocrit/30d"
    "abnormal_lab/hospital_admission/hematocrit/60d"
    "abnormal_lab/hospital_admission/hematocrit/90d"
    "abnormal_lab/hospital_discharge/hematocrit/30d"
    "abnormal_lab/hospital_discharge/hematocrit/60d"
    "abnormal_lab/hospital_discharge/hematocrit/90d"
    "abnormal_lab/icu_admission/hematocrit/30d"
    "abnormal_lab/icu_admission/hematocrit/60d"
    "abnormal_lab/icu_admission/hematocrit/90d"
    "abnormal_lab/icu_discharge/hematocrit/30d"
    "abnormal_lab/icu_discharge/hematocrit/60d"
    "abnormal_lab/icu_discharge/hematocrit/90d"
    "abnormal_lab/hospital_admission/leukocytes/30d"
    "abnormal_lab/hospital_admission/leukocytes/60d"
    "abnormal_lab/hospital_admission/leukocytes/90d"
    "abnormal_lab/hospital_discharge/leukocytes/30d"
    "abnormal_lab/hospital_discharge/leukocytes/60d"
    "abnormal_lab/hospital_discharge/leukocytes/90d"
    "abnormal_lab/icu_admission/leukocytes/30d"
    "abnormal_lab/icu_admission/leukocytes/60d"
    "abnormal_lab/icu_admission/leukocytes/90d"
    "abnormal_lab/icu_discharge/leukocytes/30d"
    "abnormal_lab/icu_discharge/leukocytes/60d"
    "abnormal_lab/icu_discharge/leukocytes/90d"
    "abnormal_lab/hospital_admission/platets/30d"
    "abnormal_lab/hospital_admission/platets/60d"
    "abnormal_lab/hospital_admission/platets/90d"
    "abnormal_lab/hospital_discharge/platets/30d"
    "abnormal_lab/hospital_discharge/platets/60d"
    "abnormal_lab/hospital_discharge/platets/90d"
    "abnormal_lab/icu_admission/platets/30d"
    "abnormal_lab/icu_admission/platets/60d"
    "abnormal_lab/icu_admission/platets/90d"
    "abnormal_lab/icu_discharge/platets/30d"
    "abnormal_lab/icu_discharge/platets/60d"
    "abnormal_lab/icu_discharge/platets/90d"
)
TASKS_DIR=/storage/shared/mimic-iv/meds_v0.3.2/meds/tasks/
OUTPUT_MODEL_DIR=/storage/shared/mimic-iv/meds_v0.3.2/results/meds_tab/
N_PARALLEL_WORKERS=48


export N_PARALLEL_WORKERS=48 # Set number of workers
TASKS_STR=$(echo ${TASKS[@]} | tr ' ' ',')
bash tabularize_meds.sh "${MIMICIV_MEDS_DIR}" "$MIMICIV_MEDS_RESHARD_DIR" $OUTPUT_TABULARIZATION_DIR \
    "${TASKS_STR}" $TASKS_DIR $OUTPUT_MODEL_DIR $N_PARALLEL_WORKERS \
    "tabularization.aggs=[static/present,code/count,value/count,value/sum,value/sum_sqd,value/min,value/max]" \
    "tabularization.window_sizes=[2h,12h,1d,7d,30d,365d,full]"
```

```console
MIMICIV_MEDS_DIR=/storage/shared/mimic-iv/meds_v0.3.2/meds/
OUTPUT_TABULARIZATION_DIR=/storage/shared/mimic-iv/meds_v0.3.2/task_specific_meds_tab/
TASKS=(
    "abnormal_lab/hospital_admission/creatinine/30d"
)
TASKS_DIR=/storage/shared/mimic-iv/meds_v0.3.2/meds/tasks/
OUTPUT_MODEL_DIR=/storage/shared/mimic-iv/meds_v0.3.2/results/task_specific_meds_tab_model/


export N_PARALLEL_WORKERS=48 # Set number of workers
TASKS_STR=$(echo ${TASKS[@]} | tr ' ' ',')
bash task_tabularize_meds.sh "${MIMICIV_MEDS_DIR}" $OUTPUT_TABULARIZATION_DIR \
    "${TASKS_STR}" $TASKS_DIR $OUTPUT_MODEL_DIR $N_PARALLEL_WORKERS \
    "tabularization.aggs=[static/present,code/count,value/count,value/sum,value/sum_sqd,value/min,value/max]" \
    "tabularization.window_sizes=[2h,12h,1d,7d,30d,365d,full]"
```

This does resharding correctly:

```console
TASKS=(
    "abnormal_lab/hospital_admission/creatinine/30d"
)
export ROOT_DIR=/storage/shared/mimic-iv/meds_v0.3.2/
export RESHARD_DIR=${ROOT_DIR}/reshareded_meds/ # set to directory to output reshareded meds data
export MIMICIV_MEDS_DIR=${ROOT_DIR}/meds/ # set to the directory in which you want to store the raw MIMIC-IV data
export OUTPUT_TABULARIZATION_DIR=${ROOT_DIR}/task_specific_meds_tab/ # set to the output directory for the tabularized data
export OUTPUT_MODEL_DIR=${OUTPUT_TABULARIZATION_DIR}/results/ # set to the base results directory
TASKS_DIR="${RESHARD_DIR}/tasks/" # set to the directory in which you want to store all tasks
TASKS_STR=$(echo ${TASKS[@]} | tr ' ' ',')
N_PARALLEL_WORKERS=48

export WINDOW_SIZES="2h,12h,1d,7d,30d,365d,full"
export AGGREGATIONS="static/present,code/count,value/count,value/sum,value/sum_sqd,value/min,value/max"

bash task_tabularize_meds.sh "${MIMICIV_MEDS_DIR}" "$RESHARD_DIR" $OUTPUT_TABULARIZATION_DIR \
    "${TASKS_STR}" $TASKS_DIR $OUTPUT_MODEL_DIR $N_PARALLEL_WORKERS \
    ${WINDOW_SIZES} ${AGGREGATIONS}
```

```console
MIMICIV_MEDS_DIR=/storage/shared/mimic-iv/meds_v0.3.2/meds/
OUTPUT_TABULARIZATION_DIR=/storage/shared/mimic-iv/meds_v0.3.2/task_specific_meds_tab/
TASKS=(
    "mortality/in_hospital/first_24h"
)
TASKS_DIR=/storage/shared/mimic-iv/meds_v0.3.2/meds/tasks/
OUTPUT_MODEL_DIR=/storage/shared/mimic-iv/meds_v0.3.2/results/task_specific_meds_tab_model/


export N_PARALLEL_WORKERS=48 # Set number of workers
TASKS_STR=$(echo ${TASKS[@]} | tr ' ' ',')
bash task_tabularize_meds.sh "${MIMICIV_MEDS_DIR}" $OUTPUT_TABULARIZATION_DIR \
    "${TASKS_STR}" $TASKS_DIR $OUTPUT_MODEL_DIR $N_PARALLEL_WORKERS \
    "tabularization.aggs=[static/present,code/count,value/count,value/sum,value/sum_sqd,value/min,value/max]" \
    "tabularization.window_sizes=[2h,12h,1d,7d,30d,365d,full]"
```
