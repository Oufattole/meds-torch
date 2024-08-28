`pip install -e .` in this repo to get setup.

Follow this MIMICIV `MEDS_Transform` tutorial [here](https://github.com/mmcdermott/MEDS_transforms/tree/main/MIMIC-IV_Example), to convert MIMIC IV to MEDS format.

Now run the `post_meds_extract.sh` script to extract machine learning ready tensorized data:

```bash
MIMICIV_MEDS_DIR=/path/to/meds/folder/ # DO NOT INCLUDE THE `/data` SUBDIRECTORY
N_PARALLEL_WORKERS=16
bash MIMICIV_TUTORIAL/post_meds_extract.sh $MIMICIV_MEDS_DIR $N_PARALLEL_WORKERS
```

We now generate ebcl pre- and post-windows around random events -- we subsample the events to make a reasonable number of windows, as making one window per event is too many:

```bash
MIMICIV_MEDS_DIR=/path/to/meds/folder/ # DO NOT INCLUDE THE `/data` SUBDIRECTORY
python MIMICIV_TUTORIAL/generate_ebcl_windows.py meds_dir=${MIMICIV_MEDS_DIR} output_path=${MIMICIV_MEDS_DIR}/ebcl_windows.parquet subsample_fraction=0.01
```
