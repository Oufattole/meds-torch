import datetime
from pathlib import Path

import polars as pl
import pytest
import torch
from clinical_zeroshot_labeler.labeler import SequenceLabeler, WindowStatus
from hydra.utils import instantiate
from omegaconf import DictConfig

from meds_torch.input_encoder import INPUT_ENCODER_MASK_KEY, INPUT_ENCODER_TOKENS_KEY
from meds_torch.models import (
    BACKBONE_EMBEDDINGS_KEY,
    BACKBONE_TOKENS_KEY,
    GENERATE_PREFIX,
    MODEL_PRED_PROBA_KEY,
    MODEL_PRED_STATUS_KEY,
)
from meds_torch.models.eic_forecasting import EicForecastingModule


class DummyModel:
    """Dummy model that generates two fixed sequences."""

    cfg = DictConfig(dict(token_emb=None, max_tokens_budget=5))

    def __call__(self, batch):
        B, S = batch["code"].shape
        return {BACKBONE_TOKENS_KEY: torch.ones(B, S, 32), BACKBONE_EMBEDDINGS_KEY: None}

    def generate(self, prompts, **kwargs):
        # Always generate two fixed sequences
        generated = torch.tensor(
            [
                [1, 2, 5, 2, 7],  # Sequence 1: Discharge -> High -> Short -> High -> Long
                [1, 3, 6, 4, 5],  # Sequence 2: Discharge -> Low -> Medium -> Normal -> Short
            ]
        )
        out_lengths = torch.tensor([5, 5])
        labels = dict()
        if kwargs.get("trajectory_labeler") is not None:
            labels = dict(
                labels=torch.tensor([1.0, 0.0]),  # Sequence 1 positive, Sequence 2 negative
                status=torch.ones(2) * WindowStatus.SATISFIED.value,
            )
        return generated, out_lengths, labels


class DummyCodeHead:
    def __init__(self, vocab_size):
        self.vocab_size = vocab_size  # Match metadata size

    def __call__(self, x):
        B, S, _ = x.shape
        logits = torch.zeros(B, S, self.vocab_size)
        logits[..., 1:] = 1.0  # All tokens except PAD equally likely
        return logits


class DummyEncoder:
    def __call__(self, batch):
        batch[INPUT_ENCODER_TOKENS_KEY] = batch["code"]
        batch[INPUT_ENCODER_MASK_KEY] = torch.ones_like(batch["mask"]).bool()
        return batch


class DummyOptimizer:
    def __init__(self, params):
        self.params = list(params)

    def step(self):
        pass

    def zero_grad(self):
        pass


class DummyScheduler:
    def step(self):
        pass

    def get_last_lr(self):
        return [0.001]


@pytest.fixture
def metadata_df():
    return pl.DataFrame(
        {
            "code": [
                "PAD",
                "HOSPITAL_DISCHARGE//MEDICAL",
                "LAB//_Q_1",
                "LAB//_Q_2",
                "LAB//_Q_3",
                "TIME//DELTA//TOKEN//_Q_1",
                "TIME//DELTA//TOKEN//_Q_2",
                "TIME//DELTA//TOKEN//_Q_3",
            ],
            "code/vocab_index": [0, 1, 2, 3, 4, 5, 6, 7],
            "values/min": [None, None, 2.0, 0.0, 1.0, 0, 1, 2],
            "values/max": [None, None, 3.0, 1.0, 2.0, 1, 2, 3],
            "values/sum": [None, None, 0.5, 1.5, 2.5, 0.5, 1.5, 2.5],
            "values/n_occurrences": [None, None, 1, 1, 1, 1, 1, 1],
            "values/quantiles": [
                {"values/quantile/0.5": None},
                {"values/quantile/0.5": None},
                {"values/quantile/0.5": 1},
                {"values/quantile/0.5": 1},
                {"values/quantile/0.5": 1},
                {"values/quantile/0.5": 1},
                {"values/quantile/0.5": 1},
                {"values/quantile/0.5": 1},
            ],
        }
    )


@pytest.fixture
def task_config():
    return """
    predicates:
        hospital_discharge:
            code: {regex: "HOSPITAL_DISCHARGE//.*"}
        lab:
            code: {regex: "LAB//.*"}
        abnormal_lab:
            code: {regex: "LAB//HIGH"}

    trigger: hospital_discharge

    windows:
        input:
            start: NULL
            end: trigger
            start_inclusive: True
            end_inclusive: True
            index_timestamp: end
        target:
            start: input.end
            end: start + 4d
            start_inclusive: False
            end_inclusive: True
            has:
                lab: (1, None)
            label: abnormal_lab
    """


@pytest.fixture
def batch():
    return {
        "code": torch.tensor(
            [
                [1, 2, 5, 2, 7],  # Sequence 1: Discharge -> High -> Short -> High -> Long
                [1, 3, 6, 4, 5],  # Sequence 2: Discharge -> Low -> Medium -> Normal -> Short
            ]
        ),
        "mask": torch.ones(2, 5, dtype=torch.bool),
        "subject_id": torch.tensor([1, 2]),
        "prediction_time": [datetime.datetime(2020, 1, 1), datetime.datetime(2020, 1, 1)],
        "end_time": [datetime.datetime(2020, 1, 1), datetime.datetime(2020, 1, 1)],
    }


@pytest.fixture
def model_config(tmp_path, metadata_df):
    # Save metadata to temp file
    meta_path = Path(tmp_path) / "metadata.parquet"
    metadata_df.write_parquet(meta_path)

    vocab_size = len(metadata_df) + 1  # Add 1 for pad token
    cfg = {
        "code_metadata_fp": str(meta_path),
        "backbone": {"_target_": "tests.test_zero_shot.DummyModel"},
        "vocab_size": vocab_size,
        "generate_id": None,
        "store_generated_trajectory": True,
        "max_seq_len": 10,
        "temperature": 1.0,
        "eos_tokens": None,
        "optimizer": {"_target_": "tests.test_zero_shot.DummyOptimizer", "_partial_": True},
        "scheduler": {"_target_": "tests.test_zero_shot.DummyScheduler", "_partial_": True},
        "input_encoder": {"_target_": "tests.test_zero_shot.DummyEncoder"},
        "code_head": {"_target_": "tests.test_zero_shot.DummyCodeHead", "vocab_size": vocab_size},
        "compile": False,
        "top_k_acc": [1],
        "next_token_auc": False,
    }
    return instantiate(cfg)


def test_training(model_config, batch):
    """Test autoregressive training workflow."""
    model = EicForecastingModule(model_config)
    loss = model.training_step(batch)
    assert loss.isfinite().all()


def test_generation(model_config, batch):
    """Test sequence generation workflow."""
    model_config.generate_id = 1
    model = EicForecastingModule(model_config)
    output = model.forward(batch)

    # Check generated trajectories
    assert GENERATE_PREFIX + "1" in output
    df = output[GENERATE_PREFIX + "1"]
    assert "time" in df.columns
    assert "code" in df.columns
    assert "numeric_value" in df.columns

    # Verify sequences match expected patterns
    seq1 = df.filter(pl.col("subject_id").eq(1))
    seq2 = df.filter(pl.col("subject_id").eq(2))
    assert seq1["code"].to_list()[-1] == 7  # Long time token
    assert seq2["code"].to_list()[-1] == 5  # Short time token


def test_zero_shot_labeling(model_config, batch, metadata_df, task_config):
    """Test zero-shot labeling workflow."""
    model_config.generate_id = 1
    model_config.trajectory_labeler = SequenceLabeler.from_yaml_str(task_config, metadata_df, batch_size=2)

    model = EicForecastingModule(model_config)
    output = model.forward(batch)

    # Check labeling outputs
    assert MODEL_PRED_PROBA_KEY in output
    assert MODEL_PRED_STATUS_KEY in output

    # Verify predictions
    probs = output[MODEL_PRED_PROBA_KEY]
    assert probs[0] > 0.5  # Sequence 1 should be positive
    assert probs[1] < 0.5  # Sequence 2 should be negative

    # Verify status
    status = output[MODEL_PRED_STATUS_KEY]
    assert (status == WindowStatus.SATISFIED.value).all()
