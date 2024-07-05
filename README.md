# MEDS torch models \[WIP\]

Pipeline:

-\[x\] Takes a Nested Ragged Tensor file path (with it's corresponding schema file_path) as input
-\[x\] `PytorchDatset` class That can generate batches
-\[x\] Generate Triplet style batches
-\[x\] Generate Eventstream style batches
-\[ \] Supervised Models
-\[ \] LSTM
-\[ \] Transformer
-\[ \] Mamba
-\[ \] Transfer Learning Models
-\[ \] EBCL
-\[ \] Forecasting
-\[ \] Masked Imputation
-\[ \] OCP
-\[ \] Generate Embeddings

Development Help

pytest-instafail shows failures and errors instantly instead of waiting until the end of test session, run it with:

```bash
pytest --instafail
```

To run failing tests continuously each time you edit code until they pass:

```bash
pytest --looponfail
```

To run tests on 8 parallel workers run:

```bash
pytest -n 8
```
