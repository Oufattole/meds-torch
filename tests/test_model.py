import os
import shutil
from datetime import datetime
from pathlib import Path

import polars as pl
import pytest
import pytorch_lightning as lightning
import torch
from hydra import compose, initialize

from meds_torch import embedder
from meds_torch.model.architectures.lstm import LstmModel
from meds_torch.model.architectures.transformer_decoder import TransformerDecoderModel
from meds_torch.model.architectures.transformer_encoder import (
    AttentionAveragedTransformerEncoderModel,
    TransformerEncoderModel,
)
from meds_torch.model.supervised_model import SupervisedModule
from meds_torch.pytorch_dataset import PytorchDataset

IN_GITHUB_ACTIONS = os.getenv("GITHUB_ACTIONS") == "true"


@pytest.fixture
def prep_embedding(tmp_path):
    MEDS_cohort_dir = tmp_path / "processed" / "final_cohort"
    shutil.copytree(Path("./tests/test_data"), MEDS_cohort_dir.parent)

    kwargs = {
        "raw_MEDS_cohort_dir": str(MEDS_cohort_dir.parent.resolve()),
        "MEDS_cohort_dir": str(MEDS_cohort_dir.resolve()),
        "max_seq_len": 512,
        "model.embedder.token_dim": 4,
        "collate_type": "triplet",
    }

    with initialize(version_base=None, config_path="../src/meds_torch/configs"):  # path to config.yaml
        overrides = [f"{k}={v}" for k, v in kwargs.items()]
        cfg = compose(config_name="pytorch_dataset", overrides=overrides)  # config.yaml

    # triplet collating
    pyd = PytorchDataset(cfg, split="train")
    batch = pyd.collate([pyd[i] for i in range(2)])
    assert batch.keys() == {
        "mask",
        "static_mask",
        "code",
        "numerical_value",
        "time_delta_days",
        "numerical_value_mask",
    }
    for key in batch.keys():
        assert batch[key].shape == torch.Size([2, 17])

    # check the continuous value embedder works
    data = torch.ones((2, 3), dtype=torch.float32)
    cve_embedding = embedder.CVE(cfg).forward(data[None, :].transpose(2, 0)).permute(1, 2, 0)
    # Output should have shape B x D x T
    assert cve_embedding.shape == torch.Size([2, 4, 3])
    embed_module = embedder.TripletEmbedder(cfg)
    embedding, mask = embed_module.embed(batch)

    # Check that the embedding is shape batch size x embedding dimension size x sequence length
    assert embedding.shape == torch.Size([2, 4, 17])
    # Check that the mask is shape batch size x sequence length
    assert mask.shape == torch.Size([2, 17])
    return embedding, mask, cfg


def test_transformer_encoder(prep_embedding):
    embedding, mask, cfg = prep_embedding
    model = TransformerEncoderModel(cfg)
    output = model(embedding, mask)
    assert output.shape == torch.Size([2, 4])

    model = AttentionAveragedTransformerEncoderModel(cfg)
    output = model.forward(embedding, mask)
    assert output.shape == torch.Size([2, 4])


def test_lstm(prep_embedding):
    embedding, mask, cfg = prep_embedding
    model = LstmModel(cfg)
    output = model(embedding, mask)
    assert output.shape == torch.Size([2, 4])


def test_transformer_decoder(prep_embedding):
    embedding, mask, cfg = prep_embedding
    model = TransformerDecoderModel(cfg)
    output = model(embedding, mask)
    assert output.shape == torch.Size([2, 4])


# @pytest.mark.skipif(IN_GITHUB_ACTIONS, reason="Test does not work in Github Actions due to Mamba setup.")
# def test_mamba(prep_embedding):
#     embedding, mask, cfg = prep_embedding
#     from meds_torch.model.architectures.mamba import MambaModel

#     model = MambaModel(cfg).cuda()
#     output = model(embedding.cuda(), mask.cuda())
#     assert output.shape == torch.Size([2, 4])


def test_train(tmp_path):
    MEDS_cohort_dir = tmp_path / "processed" / "final_cohort"
    shutil.copytree(Path("./tests/test_data"), MEDS_cohort_dir.parent)

    task_name = "bababooey"
    kwargs = {
        "raw_MEDS_cohort_dir": str(MEDS_cohort_dir.parent.resolve()),
        "MEDS_cohort_dir": str(MEDS_cohort_dir.resolve()),
        "max_seq_len": 1024,
        "model.embedder.token_dim": 4,
        "collate_type": "triplet",
        "task_name": task_name,
        "model.train.batch_size": 2,
        "model.embedder.max_seq_len": 8,
    }

    task_df = pl.DataFrame(
        {
            "patient_id": [239684, 1195293, 68729, 814703],
            "start_time": [
                datetime(1980, 12, 28),
                datetime(1978, 6, 20),
                datetime(1978, 3, 9),
                datetime(1976, 3, 28),
            ],
            "end_time": [
                datetime(2010, 5, 11, 18, 25, 35),
                datetime(2010, 6, 20, 20, 12, 31),
                datetime(2010, 5, 26, 2, 30, 56),
                datetime(2010, 2, 5, 5, 55, 39),
            ],
            task_name: [0, 1, 0, 1],
        }
    )
    tasks_dir = MEDS_cohort_dir / "tasks"
    tasks_dir.mkdir(parents=True, exist_ok=True)
    task_df.write_parquet(tasks_dir / f"{task_name}.parquet")

    with initialize(version_base=None, config_path="../src/meds_torch/configs"):  # path to config.yaml
        overrides = [f"{k}={v}" for k, v in kwargs.items()]
        cfg = compose(config_name="pytorch_dataset", overrides=overrides)  # config.yaml

    # triplet collating
    pyd = PytorchDataset(cfg, split="train")

    embedder_module = embedder.TripletEmbedder(cfg)
    # model = LstmModel(cfg)
    for model in [
        LstmModel(cfg),
        TransformerEncoderModel(cfg),
        AttentionAveragedTransformerEncoderModel(cfg),
        TransformerDecoderModel(cfg),
    ]:
        pt_model = SupervisedModule(cfg, embedder_module, model)
        assert isinstance(pt_model, lightning.LightningModule)
        lightning.Trainer(accelerator="cpu", fast_dev_run=True).fit(
            model=pt_model,
            train_dataloaders=torch.utils.data.DataLoader(
                pyd, batch_size=cfg.model.train.batch_size, collate_fn=pyd.collate, num_workers=0
            ),
        )


def test_text_code(tmp_path):
    MEDS_cohort_dir = tmp_path / "processed" / "final_cohort"
    shutil.copytree(Path("./tests/test_data"), MEDS_cohort_dir.parent)

    task_name = "bababooey"
    kwargs = {
        "raw_MEDS_cohort_dir": str(MEDS_cohort_dir.parent.resolve()),
        "MEDS_cohort_dir": str(MEDS_cohort_dir.resolve()),
        "max_seq_len": 512,
        "model.embedder.token_dim": 768,
        "collate_type": "text_code",
        "task_name": task_name,
        "model.train.batch_size": 2,
        "model.embedder.max_seq_len": 8,
        "code_embedder.tokenizer": "bert-base-uncased",
        "code_embedder.pretrained_model": "bert-base-uncased",
    }

    task_df = pl.DataFrame(
        {
            "patient_id": [239684, 1195293, 68729, 814703],
            "start_time": [
                datetime(1980, 12, 28),
                datetime(1978, 6, 20),
                datetime(1978, 3, 9),
                datetime(1976, 3, 28),
            ],
            "end_time": [
                datetime(2010, 5, 11, 18, 25, 35),
                datetime(2010, 6, 20, 20, 12, 31),
                datetime(2010, 5, 26, 2, 30, 56),
                datetime(2010, 2, 5, 5, 55, 39),
            ],
            task_name: [0, 1, 0, 1],
        }
    )
    tasks_dir = MEDS_cohort_dir / "tasks"
    tasks_dir.mkdir(parents=True, exist_ok=True)
    task_df.write_parquet(tasks_dir / f"{task_name}.parquet")

    with initialize(version_base=None, config_path="../src/meds_torch/configs"):  # path to config.yaml
        overrides = [f"++{k}={v}" for k, v in kwargs.items()]
        cfg = compose(config_name="pytorch_dataset", overrides=overrides)  # config.yaml

    # triplet collating
    pyd = PytorchDataset(cfg, split="train")
    embedder_module = embedder.TextCodeEmbedder(cfg)
    # model = LstmModel(cfg)
    for model in [
        LstmModel(cfg),
        TransformerEncoderModel(cfg),
        AttentionAveragedTransformerEncoderModel(cfg),
        TransformerDecoderModel(cfg),
    ]:
        pt_model = SupervisedModule(cfg, embedder_module, model)
        assert isinstance(pt_model, lightning.LightningModule)
        lightning.Trainer(accelerator="cpu", fast_dev_run=True).fit(
            model=pt_model,
            train_dataloaders=torch.utils.data.DataLoader(
                pyd, batch_size=cfg.model.train.batch_size, collate_fn=pyd.collate, num_workers=0
            ),
        )


def test_text_event(tmp_path):
    MEDS_cohort_dir = tmp_path / "processed" / "final_cohort"
    shutil.copytree(Path("./tests/test_data"), MEDS_cohort_dir.parent)

    task_name = "bababooey"
    kwargs = {
        "raw_MEDS_cohort_dir": str(MEDS_cohort_dir.parent.resolve()),
        "MEDS_cohort_dir": str(MEDS_cohort_dir.resolve()),
        "max_seq_len": 512,
        "model.embedder.token_dim": 768,
        "collate_type": "text_event",
        "task_name": task_name,
        "model.train.batch_size": 2,
        "model.embedder.max_seq_len": 8,
        "code_embedder.tokenizer": "bert-base-uncased",
        "code_embedder.pretrained_model": "bert-base-uncased",
    }

    task_df = pl.DataFrame(
        {
            "patient_id": [239684, 1195293, 68729, 814703],
            "start_time": [
                datetime(1980, 12, 28),
                datetime(1978, 6, 20),
                datetime(1978, 3, 9),
                datetime(1976, 3, 28),
            ],
            "end_time": [
                datetime(2010, 5, 11, 18, 25, 35),
                datetime(2010, 6, 20, 20, 12, 31),
                datetime(2010, 5, 26, 2, 30, 56),
                datetime(2010, 2, 5, 5, 55, 39),
            ],
            task_name: [0, 1, 0, 1],
        }
    )
    tasks_dir = MEDS_cohort_dir / "tasks"
    tasks_dir.mkdir(parents=True, exist_ok=True)
    task_df.write_parquet(tasks_dir / f"{task_name}.parquet")

    with initialize(version_base=None, config_path="../src/meds_torch/configs"):  # path to config.yaml
        overrides = [f"++{k}={v}" for k, v in kwargs.items()]
        cfg = compose(config_name="pytorch_dataset", overrides=overrides)  # config.yaml

    # triplet collating
    pyd = PytorchDataset(cfg, split="train")
    embedder_module = embedder.TextEventEmbedder(cfg)
    # model = LstmModel(cfg)
    for model in [
        LstmModel(cfg),
        TransformerEncoderModel(cfg),
        AttentionAveragedTransformerEncoderModel(cfg),
        TransformerDecoderModel(cfg),
    ]:
        pt_model = SupervisedModule(cfg, embedder_module, model)
        assert isinstance(pt_model, lightning.LightningModule)
        lightning.Trainer(accelerator="cpu", fast_dev_run=True).fit(
            model=pt_model,
            train_dataloaders=torch.utils.data.DataLoader(
                pyd, batch_size=cfg.model.train.batch_size, collate_fn=pyd.collate, num_workers=0
            ),
        )
