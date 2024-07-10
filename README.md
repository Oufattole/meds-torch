# MEDS torch models

This branch is meant for trying our clmbr and femr

Make a python 3.12 environment
```bash
conda create -n meds_torch python=3.12
```

Install the repo and it's dependancies with

```bash
pip install .
```

Run clmbr on some dummy meds data processed using the [meds_polars_repo]([url](https://github.com/mmcdermott/MEDS_polars_functions/tree/main)):
```bash
python scripts/launch_clmbr.py
```

Run motor with:
```bash
python scripts/launch_motor.py
```
