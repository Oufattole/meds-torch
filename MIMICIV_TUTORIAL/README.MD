# MIMIC-IV Example

This is an example of how to extract a MEDS dataset from MIMIC-IV. All scripts in this README are assumed to
be run **not** from this directory but from the root directory of this entire repository (e.g., one directory
up from this one).

## Extract MIMIC-IV MEDS Data

### Option 1: Download pre-extracted data from gpc

Run the following command to download the MEDS data from the gcp bucket:

```console
export MIMICIV_MEDS_DIR=??? # set to the directory in which you want to store the raw MIMIC-IV data

cd $MIMICIV_MEDS_DIR
gcloud storage cp gs://ehr_standardization_schema/MEDS_Extract_v0.0.7_test.zip meds_extract_0.0.7_data.zip
unzip meds_extract_0.0.7_data.zip
rm meds_extract_0.0.7_data.zip
```

```console
conda create -n meds-torch python=3.12
conda activate meds-torch
pip install "meds-torch==0.0.3"
```

### Option 2: Download MIMIC-IV and locally extract via MEDS-Transform

Alternatively take the raw MIMIC-IV data and perform the following extraction yourself. Follow the MEDS-Transforms tutorial
[here](https://github.com/mmcdermott/MEDS_transforms/blob/main/MIMIC-IV_Example/README.md):

## Pre-Processing for Sequence Modeling

We support two tensorization scripts corresponding to `triplet` and `eic` tokenization.

Run the following end to end script to generate triplet tensors

```console
export MIMICIV_MEDS_DIR=??? # set to the directory in which you want to store the MIMIC-IV MEDS data
export MIMICIV_TRIPLET_TENSOR_DIR=??? # set to the directory in which you want to output the tensorized MIMIC-IV data
export N_PARALLEL_WORKERS=1 # set to the number of parallel workers you want to use
export CONFIG_DIR="$(pwd)/MIMICIV_TUTORIAL/" # set to the directory in which the config file is stored, must be an absolute path.
export CONFIG_NAME="triplet_config" # set to the name of the config file

bash MIMICIV_TUTORIAL/triplet_tokenize.sh $MIMICIV_MEDS_DIR $MIMICIV_TRIPLET_TENSOR_DIR $N_PARALLEL_WORKERS $CONFIG_DIR $CONFIG_NAME
```

Run the following end to end script to generate eic tensors

```console
export MIMICIV_MEDS_DIR=??? # set to the directory in which you want to store the raw MIMIC-IV data
export MIMICIV_EIC_DIR=??? # set to the directory in which you want to output the tensorized MIMIC-IV data
export N_PARALLEL_WORKERS=1 # set to the number of parallel workers you want to use
export CONFIG_DIR="$(pwd)/MIMICIV_TUTORIAL/" # set to the directory in which the config file is stored, must be an absolute path.
export CONFIG_NAME="eic_config" # set to the name of the config file

bash MIMICIV_TUTORIAL/eic_tokenize.sh $MIMICIV_MEDS_DIR $MIMICIV_EIC_DIR $N_PARALLEL_WORKERS $CONFIG_DIR $CONFIG_NAME
```
