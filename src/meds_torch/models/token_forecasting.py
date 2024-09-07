import torch
import torch.nn.functional as F
from omegaconf import DictConfig
from torch import nn

from meds_torch.input_encoder.eic_encoder import EicEncoder
from meds_torch.input_encoder.triplet_encoder import TripletEncoder
from meds_torch.input_encoder.triplet_prompt_encoder import TripletPromptEncoder
from meds_torch.models import BACKBONE_TOKENS_KEY, MODEL_LOSS_KEY
from meds_torch.models.base_model import BaseModule
from meds_torch.models.components import AUTOREGRESSIVE_MODELS

NUMERIC_VALUE_LOGITS = "MODEL//NUMERIC_VALUE_LOGITS"
CODE_LOGITS = "MODEL//CODE_LOGITS"
TIME_LOGITS = "MODEL//TIME_LOGITS"

NUMERIC_VALUE_LOSS = "MODEL//NUMERIC_VALUE_LOSS"
CODE_LOSS = "MODEL//CODE_LOSS"
TIME_LOSS = "MODEL//TIME_LOSS"


def top_k_logits(logits, k):
    if k == 0:
        return logits
    values, _ = torch.topk(logits, k)
    min_values = values[:, -1]
    return torch.where(logits < min_values, torch.ones_like(logits, dtype=logits.dtype) * -1e10, logits)


def align_right(t, lens, pad_id=0):
    batch, seq_len, _, device = *t.shape, t.device

    assert lens.ndim == 1 and lens.shape[0] == batch
    assert lens.amax() <= seq_len

    pad_lens = seq_len - lens
    max_pad_len = pad_lens.amax()

    batch_arange = torch.arange(batch, device=device, dtype=torch.long)[..., None]
    prompt_len_arange = torch.arange(seq_len, device=device, dtype=torch.long)

    t = F.pad(t, (0, 0, max_pad_len, 0, 0, 0), value=0)
    offset = max_pad_len - pad_lens

    # TODO: you may need to mask the padding out, x_transformers might take care of this double check
    aligned = t[batch_arange, prompt_len_arange + offset[..., None], :]
    return aligned


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
        if not isinstance(self.model, AUTOREGRESSIVE_MODELS):
            raise ValueError(
                f"Unsupported model type: {type(self.model)}, choose one from {AUTOREGRESSIVE_MODELS}"
            )
        self.setup_heads()

    def setup_heads(self):
        if isinstance(self.input_encoder, TripletEncoder):
            self.numeric_value_head = nn.Linear(
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
        elif isinstance(self.input_encoder, EicEncoder):
            self.code_head = nn.Linear(
                self.cfg.token_dim,
                self.cfg.vocab_size,
                bias=False,
            )
        else:
            raise NotImplementedError(f"Unsupported input encoder type: {type(self.input_encoder)}")

    def process_numeric_values(self, numeric_value_logits, code_target):
        if isinstance(self.input_encoder, TripletEncoder):
            return select_values_from_logits(numeric_value_logits, code_target)
        elif isinstance(self.input_encoder, TripletPromptEncoder):
            return numeric_value_logits.squeeze(dim=-1)
        else:
            raise NotImplementedError(f"Unsupported input encoder type: {type(self.input_encoder)}")

    def get_time_loss(self, time_logits, time_delta_days_target, dynamic_mask):
        if isinstance(self.input_encoder, TripletEncoder):
            # Time Loss
            time_loss = F.mse_loss(time_logits, time_delta_days_target.unsqueeze(-1), reduction="none")
            time_loss = (time_loss.squeeze(dim=-1) * dynamic_mask).sum() / dynamic_mask.sum()
            # Summing all losses
            return time_loss
        return 0

    def get_loss(
        self,
        batch,
    ):
        code_logits = batch[CODE_LOGITS]
        numeric_value_logits = batch[NUMERIC_VALUE_LOGITS]
        time_logits = batch[TIME_LOGITS]
        # Code Mask
        dynamic_mask = ~batch["static_mask"]
        code_target = batch["code"]
        # Load data
        numeric_value_target = batch["numeric_value"]
        time_delta_days_target = batch["time_delta_days"]
        numeric_value_mask = batch["numeric_value_mask"]
        # Code Loss
        code_loss = F.cross_entropy(
            code_logits.view(-1, code_logits.size(-1)),
            code_target.view(-1).to(dtype=torch.long),
            reduction="mean",
        )
        if isinstance(self.input_encoder, (TripletEncoder, TripletPromptEncoder)):
            # Numerical Value Loss
            numeric_value_preds = self.process_numeric_values(numeric_value_logits, code_target)
            numeric_value_loss = F.mse_loss(numeric_value_preds, numeric_value_target, reduction="none")
            numeric_value_loss = (numeric_value_loss * numeric_value_mask).sum() / numeric_value_mask.sum()
            # Time Loss
            time_loss = self.get_time_loss(time_logits, time_delta_days_target, dynamic_mask)

            total_loss = code_loss + numeric_value_loss + time_loss
        else:
            total_loss = code_loss
            numeric_value_loss = 0
            time_loss = 0

        batch[MODEL_LOSS_KEY] = total_loss
        batch[NUMERIC_VALUE_LOSS] = numeric_value_loss
        batch[CODE_LOSS] = code_loss
        batch[TIME_LOSS] = time_loss
        return batch

    def get_forecast_logits(self, model_output):
        if isinstance(model_output, torch.Tensor):
            all_token_embeddings = model_output
        else:
            all_token_embeddings = model_output[BACKBONE_TOKENS_KEY]
        if isinstance(self.input_encoder, (TripletEncoder, TripletPromptEncoder)):
            numeric_value_logits = self.numeric_value_head(all_token_embeddings)
            code_logits = self.code_head(all_token_embeddings)
        else:
            code_logits = all_token_embeddings
            if not code_logits.shape[-1] == self.cfg.vocab_size:
                code_logits = self.code_head(all_token_embeddings)
            numeric_value_logits = None

        if isinstance(self.input_encoder, TripletEncoder):
            time_logits = self.time_head(all_token_embeddings)
        else:
            time_logits = None
        return {
            NUMERIC_VALUE_LOGITS: numeric_value_logits,
            CODE_LOGITS: code_logits,
            TIME_LOGITS: time_logits,
        }

    def forward(self, batch):
        batch = self.input_encoder(batch)
        model_output = self.model(batch)

        forecast = self.get_forecast_logits(model_output)
        batch[NUMERIC_VALUE_LOGITS] = forecast[NUMERIC_VALUE_LOGITS]
        batch[CODE_LOGITS] = forecast[CODE_LOGITS]
        batch[TIME_LOGITS] = forecast[TIME_LOGITS]

        batch = self.get_loss(batch)
        return batch

    def _log(self, batch, split):
        self.log(split + "/code_loss", batch[CODE_LOSS])
        self.log(split + "/numeric_value_loss", batch[NUMERIC_VALUE_LOSS])
        self.log(split + "/time_loss", batch[TIME_LOSS])
        self.log(split + "/loss", batch[MODEL_LOSS_KEY])

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
