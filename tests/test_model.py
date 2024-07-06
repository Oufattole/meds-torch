import os
import shutil
from pathlib import Path

import pytest
import torch
from hydra import compose, initialize

from meds_torch import embedder
from meds_torch.model.architectures.lstm import LstmModel
from meds_torch.model.architectures.transformer_decoder import TransformerDecoderModel
from meds_torch.model.architectures.transformer_encoder import (
    AttentionAveragedTransformerEncoderModel,
    TransformerEncoderModel,
)
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


@pytest.mark.skipif(IN_GITHUB_ACTIONS, reason="Test does not work in Github Actions due to Mamba setup.")
def test_mamba(prep_embedding):
    embedding, mask, cfg = prep_embedding
    from meds_torch.model.architectures.mamba import MambaModel

    model = MambaModel(cfg).cuda()
    output = model(embedding.cuda(), mask.cuda())
    assert output.shape == torch.Size([2, 4])
