import os

import rootutils

root = rootutils.setup_root(os.path.abspath(""), dotenv=True, pythonpath=True, cwd=True)

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import numpy as np
import polars as pl
import torch
from dateutil.relativedelta import relativedelta
from nested_ragged_tensors.ragged_numpy import JointNestedRaggedTensorDict
from omegaconf import DictConfig

from meds_torch.latest_dir import get_latest_directory


@dataclass
class DummyConfig:
    """Dummy configuration for testing MEDS dataset"""

    schema_files_root: str
    task_label_path: str
    data_dir: str
    task_name: str = "dummy_task"
    max_seq_len: int = 64
    do_prepend_static_data: bool = False
    postpend_eos_token: bool = False
    do_flatten_tensors: bool = True
    EOS_TOKEN_ID: int = 5
    do_include_subject_id: bool = True
    do_include_subsequence_indices: bool = True
    do_include_start_time_min: bool = True
    do_include_end_time: bool = True
    do_include_prediction_time: bool = True
    subsequence_sampling_strategy: str = "from_start"
    code_metadata_fp: str = field(init=False)
    augmented_code_metadata_fp: str = field(init=False)
    token_bin_size: int = 4
    token_insertion_strategy: str = "token_count"
    vocab_size: int = 6
    augmented_vocab_size: int = 8

    def __post_init__(self):
        self.code_metadata_fp = self.data_dir + "/metadata.parquet"
        self.augmented_code_metadata_fp = self.data_dir + "/augmented_codes.parquet"


def create_dummy_dataset(
    base_dir: str | Path, n_subjects: int = 3, split: str = "train", seed: int | None = 42, n_repeats: int = 3
) -> DummyConfig:
    if seed is not None:
        np.random.seed(seed)

    base_dir = Path(base_dir)

    # Create directories
    schema_dir = base_dir / "schema" / split
    schema_dir.mkdir(parents=True, exist_ok=True)
    base_dir.joinpath("data").mkdir(exist_ok=True)

    # Create static data
    base_datetime = datetime(1995, 1, 1)
    static_data = []
    for subject_id in range(n_subjects):
        static_data.append(
            {
                "subject_id": subject_id,
                "start_time": base_datetime,
                "time": [base_datetime + relativedelta(days=i) for i in range(8 * n_repeats)],
                "code": [1, 2, 3],
                "numeric_value": [0.1, 0.2, 0.3],
            }
        )
    static_df = pl.DataFrame(static_data)
    static_df.write_parquet(schema_dir / "shard_0.parquet", use_pyarrow=True)

    # Create dynamic data with consistent sequence lengths
    subject_dynamic_data = []
    for subject_id in range(n_subjects):
        rand_n_repeats = np.random.randint(8, 8 * n_repeats)
        dynamic_data = JointNestedRaggedTensorDict(
            raw_tensors={
                "code": ([[1], [2], [1], [2], [1], [2], [1], [3]] * rand_n_repeats),
                "numeric_value": (
                    [
                        [np.nan],
                        [np.nan],
                        [np.nan],
                        [np.nan],
                        [np.nan],
                        [np.nan],
                        [np.nan],
                        [np.nan],
                    ]
                    * rand_n_repeats
                ),
                "time_delta_days": ([1, 1, 1, 1, 1, 1, 1, 1] * rand_n_repeats),
            }
        )
        subject_dynamic_data.append(dynamic_data)
    dynamic_data = JointNestedRaggedTensorDict.vstack(subject_dynamic_data)

    nrt_output_dir = base_dir / "data" / split
    nrt_output_dir.mkdir(parents=True, exist_ok=True)
    dynamic_data.save(nrt_output_dir / "shard_0.nrt")

    # Create task labels
    task_df = pl.DataFrame(
        {
            "subject_id": list(range(n_subjects)),
            "prediction_time": [base_datetime + relativedelta(years=3)] * n_subjects,
            "boolean_value": [i % 2 for i in range(n_subjects)],
        }
    )

    task_fp = base_dir / "task_labels.parquet"
    task_df.write_parquet(task_fp, use_pyarrow=True)

    metadata_df = pl.DataFrame(
        {
            "code": ["a", "b", "c", "[H]", "[NTP]"],
            "code/vocab_index": [1, 2, 3, 4, 5],
            "values/min": [None, None, None, None, None],
            "values/max": [None, None, None, None, None],
            "values/sum": [None, None, None, None, None],
            "values/n_occurrences": [None, None, None, None, None],
            "values/quantiles": [
                {"values/quantile/0.25": None, "values/quantile/0.5": None, "values/quantile/0.75": None},
                {"values/quantile/0.25": None, "values/quantile/0.5": None, "values/quantile/0.75": None},
                {"values/quantile/0.25": None, "values/quantile/0.5": None, "values/quantile/0.75": None},
                {"values/quantile/0.25": None, "values/quantile/0.5": None, "values/quantile/0.75": None},
                {"values/quantile/0.25": None, "values/quantile/0.5": None, "values/quantile/0.75": None},
            ],
        }
    )

    config = DummyConfig(
        schema_files_root=str(base_dir / "schema"),
        task_label_path=str(task_fp),
        data_dir=str(base_dir),
    )
    assert config.vocab_size == len(metadata_df) + 1

    metadata_df.write_parquet(config.code_metadata_fp)

    return config


import os
import tempfile

from meds_torch.data.components.histogram_pytorch_dataset import HistogramPytorchDataset

os.environ["CUDA_VISIBLE_DEVICES"] = "0"


tmp_dir = tempfile.TemporaryDirectory()
data_config = create_dummy_dataset(tmp_dir.name, n_subjects=64)
dataset = HistogramPytorchDataset(data_config, split="train")

dynamic_data, subject_id, st, end = dataset.load_subject_dynamic_data(0)
print("every patient has identical data:")
print(dynamic_data.flatten().to_dense()["code"])
print("the length of the data is:", str(len(dynamic_data.flatten().to_dense()["code"])))
"""This file prepares config fixtures for other tests."""

from pathlib import Path

import hydra
from hydra import compose, initialize

from meds_torch.utils.resolvers import setup_resolvers

setup_resolvers()


def create_cfg(overrides, config_name="train.yaml") -> DictConfig:
    """Helper function to create Hydra DictConfig with given overrides and common settings."""
    with initialize(version_base="1.3", config_path="../src/meds_torch/configs"):
        cfg = compose(config_name=config_name, return_hydra_config=True, overrides=overrides)
    return cfg


output_dir = Path(tmp_dir.name) / "output"
overrides = [
    "experiment=histogram_eic_forecast_mtr",
    "model=histogram_forecasting",
    "model/backbone=histogram_transformer_decoder",
    "model/input_encoder=histogram_encoder",
    "data=histogram_pytorch_dataset",
    f"data.vocab_size={data_config.vocab_size}",
    "trainer=gpu",
    "data.subsequence_sampling_strategy=random",
    "data.token_insertion_strategy=token_count",
    "data.token_bin_size=8",
    f"data.code_metadata_fp={data_config.code_metadata_fp}",
    f"data.augmented_code_metadata_fp={data_config.augmented_code_metadata_fp}",
    "model.optimizer.lr=0.001",
    "trainer.max_epochs=10",
    f"paths.output_dir={output_dir}",
    "model.top_k_acc=[1]",
    f"hydra.searchpath=[pkg://meds_torch.configs,{root}/ZERO_SHOT_TUTORIAL/configs/]",
]
cfg = create_cfg(overrides)
model = hydra.utils.instantiate(cfg.model)
type(model)
from torch.utils.data.dataloader import DataLoader

train_dataloader = DataLoader(dataset, batch_size=8, shuffle=True, collate_fn=dataset.collate)
val_dataloader = DataLoader(dataset, batch_size=8, shuffle=False, collate_fn=dataset.collate)
trainer = hydra.utils.instantiate(cfg.trainer)

trainer.fit(
    model=model,
    train_dataloaders=train_dataloader,
    val_dataloaders=val_dataloader,
    ckpt_path=cfg.get("ckpt_path"),
)
trainer.validate(model=model, dataloaders=val_dataloader, ckpt_path=cfg.get("ckpt_path"))
print("Enter the following in terminal to view the tensorboard logs:")
print("tensorboard --logdir=%s" % get_latest_directory(cfg.paths.output_dir) + "/lightning_logs/")

# Try Generation
model.cfg.generate_id = 0
model.cfg.max_tokens_budget = 24

from meds_torch.input_encoder import INPUT_ENCODER_MASK_KEY, INPUT_ENCODER_TOKENS_KEY

batch = dataset.collate([dataset[i] for i in range(8)])
input_batch = model.input_encoder.forward(batch)
prompts, mask = input_batch[INPUT_ENCODER_TOKENS_KEY], input_batch[INPUT_ENCODER_MASK_KEY]

prompt_lengths = mask.sum(dim=-1)
prompt_lengths
output_batch = model.forward(input_batch)

in_seq = output_batch["code"]
out_seq = output_batch["GENERATE//0"].filter(pl.col("subject_id") == 0)["code/vocab_index"].to_list()
print("###########################################################################################")
print(f"raw_out_seq: {out_seq}")
print(f"in_seq: {list(filter(lambda x: x != 0, in_seq[0].tolist()))}")
print(f"out_seq: {list(filter(lambda x: x not in [0, 4, 5], out_seq))}")


# # TODO: log the histogram generated by the model at each input element
# # Try generation analysis:
print("###########################################################################################")

model.cfg.generate_id = 0
# select the first batch and remove last token which is [NTP]
mask = output_batch["mask"].to(torch.bool)[:1]
code = output_batch["code"][:1][mask].unsqueeze(0)

histogram = output_batch["histogram"][:1][mask, :].unsqueeze(0)
mask = mask[mask].unsqueeze(0)
print(f"in_seq: {code.tolist()}")
print(f"in_code: {code.shape}")
print(f"in_histogram: {histogram.shape}")
assert mask.all().item(), "mask is not all true"
batch = dict(
    code=code,
    histogram=histogram,
    mask=mask,
    prediction_time=[datetime(2025, 1, 1)],
    end_time=[datetime(2025, 1, 1)],
    subject_id=[0],
)
(
    next_token_logits,
    next_token_histogram,
    next_token_histogram_latent_posterior,
    counts,
    last_embeddings,
) = model.get_sample(batch)


print(f"next_token_logits: {next_token_logits.tolist()}")
print(f"next_token_histogram: {next_token_histogram.tolist()}")
sample_histogram = model.histogram_normalizer.reverse_transform(
    model.autoencoder.decode(next_token_histogram_latent_posterior.sample())
)
print(f"sample_histogram: {sample_histogram}")
mean_histogram = model.histogram_normalizer.reverse_transform(
    model.autoencoder.decode(next_token_histogram_latent_posterior.sample())
)
print(f"mean_histogram: {mean_histogram}")

# print(f"next_token_histogram_logits: {next_token_histogram_logits.tolist()}")
print(f"next_token_histogram_counts: {counts.tolist()}")
# print(f"last_embeddings: {last_embeddings.tolist()}")


# model.diffusion.sample(last_embeddings, temperature=1.0)
