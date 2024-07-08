# MEDS torch models  \[WIP\]

Several Ideas are laid out in this [planning doc](https://docs.google.com/document/d/1SjWP6RyHJC9eU5y0Dy7LTjoGiKgF_J5bIF6juab1X3g/edit)

Pipeline:

- [x] Takes a Nested Ragged Tensor file path (with it's corresponding schema file_path) as input
- [x] `PytorchDatset` class That can generate batches
- [x] Generate Triplet style batches
- [x] Generate Eventstream style batches
- [x] LSTM
- [x] Transformer Encoder (to get a fixed size representation— a representation token is used in the `TransformerEncoderModel` class and an attention-based token average is used in the `AttentionAveragedTransformerEncoderModel` class)
- [x] Transformer Decoder Model (GPT-Style)
- [x] Mamba
- [x] Supervised Training
- [ ] Transfer Learning Models
  - [ ] EBCL
  - [ ] Forecasting
  - [ ] Masked Imputation
  - [ ] OCP
- [ ] Generate Embeddings

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
