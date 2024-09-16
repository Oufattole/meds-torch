import torch
import torch.nn.functional as F
from omegaconf import DictConfig

from meds_torch.input_encoder.eic_encoder import EicEncoder
from meds_torch.models import BACKBONE_TOKENS_KEY, MODEL_LOSS_KEY
from meds_torch.models.base_model import BaseModule
from meds_torch.models.components import AUTOREGRESSIVE_MODELS

CODE_LOGITS = "MODEL//CODE_LOGITS"
CODE_LOSS = "MODEL//CODE_LOSS"
VAL_PREFIX = "VAL_METRIC//"


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


class EicForecastingModule(BaseModule):
    """EIC token based GPT Forecasting Model."""

    def __init__(self, cfg: DictConfig):
        super().__init__(cfg)
        if not isinstance(self.model, AUTOREGRESSIVE_MODELS):
            raise ValueError(
                f"Unsupported model type: {type(self.model)}, choose one from {AUTOREGRESSIVE_MODELS}"
            )
        self.setup_heads()

    def setup_heads(self):
        if not isinstance(self.input_encoder, EicEncoder):
            raise NotImplementedError(f"Unsupported input encoder type: {type(self.input_encoder)}")
        self.code_head = self.cfg.code_head

    def get_loss(
        self,
        batch,
    ):
        code_logits = batch[CODE_LOGITS]
        assert not torch.isnan(code_logits).any(), "code_logits is NaN"
        # Code Mask
        code_target = batch["code"]
        # Code Loss
        code_loss = F.cross_entropy(
            code_logits.view(-1, code_logits.size(-1)),
            code_target.view(-1).to(dtype=torch.long),
            reduction="mean",
        )

        assert not torch.isnan(code_loss).any(), "code_loss is NaN"

        total_loss = code_loss

        batch[MODEL_LOSS_KEY] = total_loss
        batch[VAL_PREFIX + CODE_LOSS] = code_loss
        return batch

    def get_forecast_logits(self, model_output):
        if isinstance(model_output, torch.Tensor):
            all_token_embeddings = model_output
        else:
            all_token_embeddings = model_output[BACKBONE_TOKENS_KEY]
        code_logits = self.code_head(all_token_embeddings)
        return {
            CODE_LOGITS: code_logits,
        }

    def forward(self, batch):
        batch = self.input_encoder(batch)
        model_output = self.model(batch)

        forecast = self.get_forecast_logits(model_output)
        batch[CODE_LOGITS] = forecast[CODE_LOGITS]

        batch = self.get_loss(batch)
        return batch

    def _log(self, batch, split):
        for key in batch:
            if key.startswith(VAL_PREFIX):
                self.log(split + "/" + key, batch[key], on_step=False, on_epoch=True)
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
