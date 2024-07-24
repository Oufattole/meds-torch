# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the Llama 2 Community License Agreement.

import dataclasses

import lightning as L
import numpy as np
import torch
import torch.nn.functional as F
from omegaconf import DictConfig
from torch import nn

from meds_torch.model.architectures.transformer_decoder import TransformerDecoderModel
from meds_torch.model.supervised_model import OutputBase

# from meds_torch.model.architectures.mamba import MambaModel


@dataclasses.dataclass
class ForecastLogits:
    cont_val_logits: torch.Tensor
    cat_val_logits: torch.Tensor
    var_logits: torch.Tensor
    time_logits: torch.Tensor


@dataclasses.dataclass
class ForecastOutput(OutputBase):
    forecast_logits: ForecastLogits | None = None
    loss: float | None = None


def _load_gpt_model(cfg: DictConfig) -> TransformerDecoderModel:
    """Load the autregressive sequence model from the config.

    TODO: add support for loading Mamba, if that dependency is installed.
    """
    model = TransformerDecoderModel(cfg)
    return model


def top_k_logits(logits, k):
    if k == 0:
        return logits
    values, _ = torch.topk(logits, k)
    min_values = values[:, -1]
    return torch.where(logits < min_values, torch.ones_like(logits, dtype=logits.dtype) * -1e10, logits)


class TripletGPTModel(L.LightningModule):
    """Triplet based GPT Forecasting Model."""

    def __init__(self, cfg: DictConfig):
        super().__init__()
        self.cfg = cfg
        self.model = _load_gpt_model(cfg)
        self.cont_val_head = nn.Linear(
            cfg.model_config.embed_size,
            cfg.dataset_config.n_variable - cfg.dataset_config.n_cat_variable,
            bias=False,
        )
        self.cat_val_head = nn.Linear(
            cfg.model_config.embed_size,
            cfg.dataset_config.n_cat_value + cfg.model_config.token_offset,
            bias=False,
        )
        self.var_head = nn.Linear(
            cfg.model_config.embed_size,
            cfg.dataset_config.n_variable + cfg.model_config.token_offset,
            bias=False,
        )
        self.time_head = nn.Linear(cfg.model_config.embed_size, 1, bias=False)

        self.temperature = 1
        self.top_k = 0

    def get_loss(
        self,
        cont_val_logits,
        cat_val_logits,
        var_logits,
        time_logits,
        batch,
        split="train",
        log=False,
    ):
        # Masks
        is_cat_mask = batch["is_cat"].unsqueeze(-1).to(dtype=torch.float32)
        not_cat_mask = 1 - is_cat_mask

        # Variable Loss
        var_target = batch["variable"]
        variable_loss = F.cross_entropy(
            var_logits.view(-1, var_logits.size(-1)),
            var_target.view(-1).to(dtype=torch.long),
            reduction="sum",
        )

        # Categorical Value Loss
        cat_val_target = batch["value"] * is_cat_mask.squeeze()
        categorical_value_loss = F.cross_entropy(
            cat_val_logits.view(-1, cat_val_logits.size(-1)),
            cat_val_target.view(-1).to(dtype=torch.long),
            reduction="none",
        )
        categorical_value_loss = (
            categorical_value_loss.view(cat_val_logits.size(0), cat_val_logits.size(1))
            * is_cat_mask.squeeze(-1)
        ).sum()

        # Continuous Value Loss
        cont_val_target = batch["value"]
        continuous_value_loss = F.mse_loss(cont_val_logits, cont_val_target.unsqueeze(-1), reduction="none")
        continuous_value_loss = (continuous_value_loss * not_cat_mask).sum()

        # Time Loss
        time_target = batch["date"]
        time_loss = F.mse_loss(time_logits, time_target.unsqueeze(-1), reduction="sum")

        # Summing all losses
        total_loss = variable_loss + categorical_value_loss + continuous_value_loss + time_loss

        if log:
            assert split in [
                "val",
                "train",
                "test",
            ], f"Invalid split {split} for logging"
            self.log(split + "_variable_loss", variable_loss)
            self.log(split + "_categorical_value_loss", categorical_value_loss)
            self.log(split + "_continuous_value_loss", continuous_value_loss)
            self.log(split + "_time_loss", time_loss)
            self.log(split + "_loss", total_loss)

        return total_loss

    def get_forecast_logits(self, model_output: OutputBase):
        all_token_embeddings = model_output.rep
        cont_val_logits = self.cont_val_head(all_token_embeddings)
        cat_val_logits = self.cat_val_head(all_token_embeddings)
        var_logits = self.var_head(all_token_embeddings)
        time_logits = self.time_head(all_token_embeddings)
        return ForecastLogits(cont_val_logits, cat_val_logits, var_logits, time_logits)

    def forward(self, batch, get_rep=False, split="train"):
        model_output: OutputBase = self.model(batch["early_fusion"], bos_eos_tokens=True)

        if get_rep:
            return model_output
        forecast = self.get_forecast_logits(model_output)

        loss = self.get_loss(
            forecast.cont_val_logits,
            forecast.cat_val_logits,
            forecast.var_logits,
            forecast.time_logits,
            batch["early_fusion"],
            split=split,
            log=True,
        )
        output = ForecastOutput(rep=None, forecast_logits=forecast, loss=loss)
        return output

    def training_step(self, batch):
        output = self(batch, split="train")
        assert not torch.isnan(output.loss), "Loss is NaN"
        return output.loss

    def validation_step(self, batch):
        output = self(batch, split="val")
        assert not torch.isnan(output.loss), "Loss is NaN"
        return output.loss

    def test_step(self, batch, batch_idx):
        output = self(batch, split="test")
        assert not torch.isnan(output.loss), "Loss is NaN"
        return output.loss

    def configure_optimizers(self):
        if self.cfg.model_config.half_dtype:
            eps = 1e-4
        else:
            eps = 1e-8
        optimizer = torch.optim.Adam(self.parameters(), lr=self.cfg.model_config.lr, eps=eps)
        return optimizer

    def generate(self, batch, length=3, sample=False):
        context = {
            "date": batch["early_fusion"]["date"].clone(),
            "variable": batch["early_fusion"]["variable"].clone(),
            "value": batch["early_fusion"]["value"].clone(),
            "is_cat": batch["early_fusion"]["is_cat"].clone(),
            "length": torch.as_tensor(batch["early_fusion"]["length"]),
        }
        generated_dates = []
        generated_variables = []
        generated_values = []

        with torch.no_grad():
            for i in range(length):
                # Get the position of the last valid data for each item in the batch
                valid_indices = context["length"] - 1 + i  # Adjusting index as we generate more items

                # Model output for the last valid elements
                model_output: OutputBase = self.model(context, bos_eos_tokens=True)
                forecast_logits = self.get_forecast_logits(model_output)

                # Generate new variables
                var_logits = forecast_logits.var_logits
                var_probs = F.softmax(var_logits, dim=-1)
                var = (
                    torch.multinomial(var_probs, num_samples=1)
                    if sample
                    else torch.argmax(var_probs, dim=-1, keepdim=True)
                )

                # Generate new values (categorical or continuous)
                last_is_cat = context["is_cat"][:, i]
                if last_is_cat.item() == 1:
                    cat_val_logits = forecast_logits.cat_val_logits
                    cat_val_probs = F.softmax(cat_val_logits, dim=-1)
                    value = (
                        torch.multinomial(cat_val_probs, num_samples=1)
                        if sample
                        else torch.argmax(cat_val_probs, dim=-1, keepdim=True)
                    )
                else:
                    value = forecast_logits.cont_val_logits

                # Generate new times
                time = forecast_logits.time_logits

                # Update context for the next generation step
                index_update = valid_indices + 1
                context["date"][torch.arange(context["date"].size(0)), index_update] = time
                context["variable"][torch.arange(context["variable"].size(0)), index_update] = var
                context["value"][torch.arange(context["value"].size(0)), index_update] = value
                context["length"] += 1  # Increment the valid length for each item in the batch

                # Store generated outputs
                generated_dates.append(time.squeeze(-1).cpu().numpy())
                generated_variables.append(var.squeeze(-1).cpu().numpy())
                generated_values.append(value.squeeze(-1).cpu().numpy())

        # Aggregate collected data
        results = {
            "dates": np.concatenate(generated_dates),
            "variables": np.concatenate(generated_variables),
            "values": np.concatenate(generated_values),
        }
        return results

    @classmethod
    def initialize_pretrain(cls, cfg):
        return cls(cfg)

    @classmethod
    def initialize_finetune(cls, cfg, ckpt_path: str):
        assert not cfg.pretrain
        return cls.load_from_checkpoint(ckpt_path, cfg=cfg)
