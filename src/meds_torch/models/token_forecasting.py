# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the Llama 2 Community License Agreement.


import numpy as np
import torch
import torch.nn.functional as F
from omegaconf import DictConfig
from torch import nn

from meds_torch.input_encoder.triplet_encoder import TripletEncoder
from meds_torch.input_encoder.triplet_prompt_encoder import TripletPromptEncoder
from meds_torch.models import BACKBONE_TOKENS_KEY, MODEL_LOSS_KEY
from meds_torch.models.base_model import BaseModule
from meds_torch.models.utils import OutputBase

# from meds_torch.model.architectures.mamba import MambaModel


NUMERICAL_VALUE_LOGITS = "MODEL//NUMERICAL_VALUE_LOGITS"
CODE_LOGITS = "MODEL//CODE_LOGITS"
TIME_LOGITS = "MODEL//TIME_LOGITS"

NUMERICAL_VALUE_LOSS = "MODEL//NUMERICAL_VALUE_LOSS"
CODE_LOSS = "MODEL//CODE_LOSS"
TIME_LOSS = "MODEL//TIME_LOSS"


def top_k_logits(logits, k):
    if k == 0:
        return logits
    values, _ = torch.topk(logits, k)
    min_values = values[:, -1]
    return torch.where(logits < min_values, torch.ones_like(logits, dtype=logits.dtype) * -1e10, logits)


def select_values_from_logits(logits, target_indices):
    """Selects values from a 3D logits tensor based on indices specified for the last dimension.

    :param logits: A tensor of shape [batch_size, seq_length, num_classes]
    :param target_indices: A tensor of indices with shape [batch_size, seq_length] where each index is valid
        within the range of the last dimension of logits
    :return: A tensor of selected values with shape [batch_size, seq_length]
    """
    batch_size, seq_length, _ = logits.shape

    # Create batch and sequence indices
    batch_indices = torch.arange(batch_size).unsqueeze(1).expand(-1, seq_length).reshape(-1)
    seq_indices = torch.arange(seq_length).repeat(batch_size)

    # Flatten target_indices to match the expanded batch and sequence indices
    flat_target_indices = target_indices.reshape(-1)

    # Use advanced indexing to select the appropriate elements from logits
    selected_values = logits[batch_indices, seq_indices, flat_target_indices].reshape(batch_size, seq_length)

    return selected_values


class TokenForecastingModule(BaseModule):
    """Triplet based GPT Forecasting Model."""

    def __init__(self, cfg: DictConfig):
        super().__init__(cfg)
        self.setup_heads()

    def setup_heads(self):
        if isinstance(self.input_encoder, TripletEncoder):
            self.numerical_value_head = nn.Linear(
                self.cfg.token_dim,
                self.cfg.vocab_size,
                bias=False,
            )
            self.code_head = nn.Linear(
                self.cfg.token_dim,
                self.cfg.vocab_size,
                bias=False,
            )
            self.time_head = nn.Linear(self.cfg.token_dim, 1, bias=False)
        elif isinstance(self.input_encoder, TripletPromptEncoder):
            self.numerical_value_head = nn.Linear(
                self.cfg.token_dim,
                1,
                bias=False,
            )
            # TODO add vocab size + 2 offset to config
            self.code_head = nn.Linear(
                self.cfg.token_dim,
                self.cfg.vocab_size + 2,
                bias=False,
            )
        else:
            raise NotImplementedError(f"Unsupported input encoder type: {type(self.input_encoder)}")

    def process_numerical_values(self, numerical_value_logits, code_target):
        if isinstance(self.input_encoder, TripletEncoder):
            return select_values_from_logits(numerical_value_logits, code_target)
        elif isinstance(self.input_encoder, TripletPromptEncoder):
            return numerical_value_logits.squeeze()
        else:
            raise NotImplementedError(f"Unsupported input encoder type: {type(self.input_encoder)}")

    def get_time_loss(self, time_logits, time_delta_days_target, dynamic_mask):
        if isinstance(self.input_encoder, TripletEncoder):
            # Time Loss
            time_loss = F.mse_loss(time_logits, time_delta_days_target.unsqueeze(-1), reduction="none")
            time_loss = (time_loss.squeeze() * dynamic_mask).sum() / dynamic_mask.sum()
            # Summing all losses
            return time_loss
        return 0

    def get_loss(
        self,
        batch,
    ):
        code_logits = batch[CODE_LOGITS]
        numerical_value_logits = batch[NUMERICAL_VALUE_LOGITS]
        time_logits = batch[TIME_LOGITS]
        # Code Mask
        dynamic_mask = ~batch["static_mask"]
        code_target = batch["code"]
        # Load data
        numerical_value_target = batch["numerical_value"]
        time_delta_days_target = batch["time_delta_days"]
        numerical_value_mask = batch["numerical_value_mask"]
        # Code Loss
        code_loss = F.cross_entropy(
            code_logits.view(-1, code_logits.size(-1)),
            code_target.view(-1).to(dtype=torch.long),
            reduction="mean",
        )
        # Numerical Value Loss
        numerical_value_preds = self.process_numerical_values(numerical_value_logits, code_target)
        numerical_value_loss = F.mse_loss(numerical_value_preds, numerical_value_target, reduction="none")
        numerical_value_loss = (
            numerical_value_loss * numerical_value_mask
        ).sum() / numerical_value_mask.sum()
        # Time Loss
        time_loss = self.get_time_loss(time_logits, time_delta_days_target, dynamic_mask)

        total_loss = code_loss + numerical_value_loss + time_loss

        batch[MODEL_LOSS_KEY] = total_loss
        batch[NUMERICAL_VALUE_LOSS] = numerical_value_loss
        batch[CODE_LOSS] = code_loss
        batch[TIME_LOSS] = time_loss
        return batch

    def get_forecast_logits(self, model_output):
        all_token_embeddings = model_output[BACKBONE_TOKENS_KEY]
        numerical_value_logits = self.numerical_value_head(all_token_embeddings)
        code_logits = self.code_head(all_token_embeddings)
        if isinstance(self.input_encoder, TripletEncoder):
            time_logits = self.time_head(all_token_embeddings)
        else:
            time_logits = None
        return {
            NUMERICAL_VALUE_LOGITS: numerical_value_logits,
            CODE_LOGITS: code_logits,
            TIME_LOGITS: time_logits,
        }

    def forward(self, batch):
        batch = self.input_encoder(batch)
        model_output = self.model(batch)

        forecast = self.get_forecast_logits(model_output)
        batch[NUMERICAL_VALUE_LOGITS] = forecast[NUMERICAL_VALUE_LOGITS]
        batch[CODE_LOGITS] = forecast[CODE_LOGITS]
        batch[TIME_LOGITS] = forecast[TIME_LOGITS]

        batch = self.get_loss(batch)
        return batch

    def _log(self, batch, split):
        self.log(split + "_code_loss", batch[CODE_LOSS])
        self.log(split + "_numerical_value_loss", batch[NUMERICAL_VALUE_LOSS])
        self.log(split + "_time_loss", batch[TIME_LOSS])
        self.log(split + "_loss", batch[MODEL_LOSS_KEY])

    def training_step(self, batch):
        batch = self(batch)
        assert not torch.isnan(batch[MODEL_LOSS_KEY]), "Loss is NaN"
        self._log(batch, "train")
        return batch[MODEL_LOSS_KEY]

    def validation_step(self, batch):
        batch = self(batch)
        assert not torch.isnan(batch[MODEL_LOSS_KEY]), "Loss is NaN"
        self._log(batch, "val")
        return batch[MODEL_LOSS_KEY]

    def test_step(self, batch):
        batch = self(batch)
        assert not torch.isnan(batch[MODEL_LOSS_KEY]), "Loss is NaN"
        self._log(batch, "test")
        return batch[MODEL_LOSS_KEY]

    def generate(self, batch, length=3, sample=False):
        context = {
            "date": batch["date"].clone(),
            "variable": batch["variable"].clone(),
            "value": batch["value"].clone(),
            "is_cat": batch["is_cat"].clone(),
            "length": torch.as_tensor(batch["length"]),
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
