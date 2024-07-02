# MEDS torch models [WIP]

Pipeline:
- Takes a Nested Ragged Tensor file path (with it's corresponding schema file_path) as input
- Creates a `PytorchDatset` class
- iterates through batches and trains a supervised LSTM or Transformer on it
- Evalutes this supervised model
