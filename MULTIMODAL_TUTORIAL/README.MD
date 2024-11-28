# Extract MIMIC-IV MEDS Data

## Step 0.5: Set-up

Set some environment variables and download the necessary files:

```bash
conda create -n ecg python=3.12
conda activate ecg
# Get the latest version of MEDS_transforms from pypi
LATEST_VERSION=$(pip index versions "meds-transforms" 2>/dev/null | egrep -o '([0-9]+\.){2}[0-9]+' | head -n 1)
export VERSION=$LATEST_VERSION # or whatever version you want, at the time of writing this is "0.0.8"
pip install "MEDS_transforms[local_parallelism,slurm_parallelism]==${VERSION} wfdb"


export MIMICIV_RAW_DIR=/storage/shared/mimic-iv/raw/ehr/2.2 # set to the directory in which you want to store the raw MIMIC-IV data
export MIMICIV_ECG_RAW_DIR=/storage/shared/mimic-iv/raw/ecgs/1.0
export MIMICIV_PRE_MEDS_DIR=/storage/shared/mimic-iv/multimodal_ecg/pre_meds/ # set to the directory in which you want to store the intermediate MEDS MIMIC-IV data
export MIMICIV_MEDS_DIR=/storage/shared/mimic-iv/multimodal_ecg/meds/ # set to the directory in which you want to store the final MEDS MIMIC-IV data
```

## Step 1: Download MIMIC-IV

Download the MIMIC-IV dataset from https://physionet.org/content/mimiciv/2.2/ following the instructions on
that page. You will need the raw `.csv.gz` files for this example. We will use `$MIMICIV_RAW_DIR` to denote
the root directory of where the resulting _core data files_ are stored -- e.g., there should be a `hosp` and
`icu` subdirectory of `$MIMICIV_RAW_DIR`.

## Step 1.5: Download MIMIC-IV Metadata files

```bash
cd $MIMICIV_RAW_DIR
export MIMICIV_RAW_DIR=https://raw.githubusercontent.com/MIT-LCP/mimic-code/v2.4.0/mimic-iv/concepts/concept_map
wget $MIMICIV_RAW_DIR/d_labitems_to_loinc.csv
wget $MIMICIV_RAW_DIR/inputevents_to_rxnorm.csv
wget $MIMICIV_RAW_DIR/lab_itemid_to_loinc.csv
wget $MIMICIV_RAW_DIR/meas_chartevents_main.csv
wget $MIMICIV_RAW_DIR/meas_chartevents_value.csv
wget $MIMICIV_RAW_DIR/numerics-summary.csv
wget $MIMICIV_RAW_DIR/outputevents_to_loinc.csv
wget $MIMICIV_RAW_DIR/proc_datetimeevents.csv
wget $MIMICIV_RAW_DIR/proc_itemid.csv
wget $MIMICIV_RAW_DIR/waveforms-summary.csv
```

## Step 2: Run the MEDS ETL

To run the MEDS ETL, run the following command:

```bash
./run.sh $MIMICIV_RAW_DIR $MIMICIV_PRE_MEDS_DIR $MIMICIV_MEDS_COHORT_DIR do_unzip=true
```

> \[!NOTE\]
> This can take up large amounts of memory if not parallelized. You can reduce the shard size to reduce memory usage by setting the `shard_size` parameter in the `extract_MIMIC.yaml` file.
> Check that your environment variables are set correctly.
> To not unzip the `.csv.gz` files, set `do_unzip=false` instead of `do_unzip=true`.

To use a specific stage runner file (e.g., to set different parallelism options), you can specify it as an
additional argument

```bash
export N_WORKERS=5
./meds_extract.sh $MIMICIV_RAW_DIR $MIMICIV_ECG_RAW_DIR $MIMICIV_PRE_MEDS_DIR $MIMICIV_MEDS_DIR do_unzip=true stage_runner_fp=configs/local_parallelism_runner.yaml
```

The `N_WORKERS` environment variable set before the command controls how many parallel workers should be used
at maximum.

The `local_parallelism_runner.yaml` file (downloaded above) runs each stage via separate processes on the
launching machine. There are no additional arguments needed for this stage beyond the `N_WORKERS` environment
variable and there is nothing to customize in this file.

# MEDS Multimodal Tensorization

To tensorize the ecgs and ehr data run:

```bash
ROOT_DIR=/storage/shared/mimic-iv/multimodal_ecg/
export MIMICIV_MEDS_DIR=$ROOT_DIR/meds # set to the directory in which you want to store the MIMIC-IV MEDS data
export MIMICIV_TRIPLET_TENSOR_DIR=$ROOT_DIR/ecg_triplet_tensors/ # set to the directory in which you want to output the tensorized MIMIC-IV data
export N_PARALLEL_WORKERS=5 # set to the number of parallel workers you want to use
export PIPELINE_CONFIG_PATH="$(pwd)/MULTIMODAL_TUTORIAL/configs/ecg_triplet_config.yaml" # set to the directory in which the config file is stored, must be an absolute path.
export JOBLIB_RUNNER_CONFIG_PATH="$(pwd)/MULTIMODAL_TUTORIAL/configs/joblib_runner.yaml" # set to the directory in which the config file is stored, must be an absolute path.

bash MULTIMODAL_TUTORIAL/meds_tensorize.sh $MIMICIV_MEDS_DIR $MIMICIV_TRIPLET_TENSOR_DIR $N_PARALLEL_WORKERS $PIPELINE_CONFIG_PATH stage_runner_fp=$JOBLIB_RUNNER_CONFIG_PATH
```
