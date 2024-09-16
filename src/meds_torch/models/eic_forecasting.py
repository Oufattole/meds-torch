import torch
import torch.nn.functional as F
from meds_torch.input_encoder import INPUT_ENCODER_MASK_KEY, INPUT_ENCODER_TOKENS_KEY
from meds_torch.input_encoder.eic_encoder import EicEncoder
from meds_torch.models import BACKBONE_TOKENS_KEY, MODEL_LOSS_KEY
from meds_torch.models.base_model import BaseModule
from meds_torch.models.components import AUTOREGRESSIVE_MODELS
from omegaconf import DictConfig
from x_transformers import AutoregressiveWrapper
from x_transformers.autoregressive_wrapper import eval_decorator

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

        # TODO convert this to a yaml config
        # self.code_head = nn.Linear(
        #     self.cfg.token_dim,
        #     self.cfg.vocab_size,
        #     bias=False,
        # )

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
        batch = self.generate_evaluation(batch, seq_len=self.cfg.max_seq_len)
        self._log(batch, "val")
        return batch[MODEL_LOSS_KEY]

    def test_step(self, batch):
        batch = self(batch)
        assert not torch.isnan(batch[MODEL_LOSS_KEY]), "Loss is NaN"
        self._log(batch, "test")
        return batch[MODEL_LOSS_KEY]

    @torch.no_grad()
    @eval_decorator
    def generate_evaluation(
        self,
        batch,
        seq_len,
        **kwargs,
    ):
        """Generate evaluation metrics for the model.

        TODO(Oufattole): add prediction evaluation TODO(Oufattole): add ??? evaluation
        """
        model = AutoregressiveWrapper(self.model.model)
        batch = self.input_encoder(batch)
        if self.cfg.backbone.cfg.token_emb:
            raise NotImplementedError(
                "Token embeddings not supported, use x-transformers library for token embeddings"
            )
        else:
            prompts, mask = batch[INPUT_ENCODER_TOKENS_KEY], batch[INPUT_ENCODER_MASK_KEY]

        # Calculate actual lengths of prompts using the mask
        prompt_lengths = mask.sum(dim=1)

        # Calculate half lengths for each prompt
        half_lengths = prompt_lengths // 2

        # Create input prompts
        max_len = prompts.shape[1]
        input_prompts = prompts.clone()

        # Create a mask for input prompts
        input_mask = torch.arange(max_len, device=input_prompts.device).unsqueeze(0) < half_lengths.unsqueeze(
            1
        )
        input_prompts = input_prompts * input_mask

        # Create target mask
        target_mask = torch.arange(max_len, device=input_prompts.device).unsqueeze(
            0
        ) >= half_lengths.unsqueeze(1)
        target_mask = target_mask & mask  # Ensure we only consider valid tokens

        # Generate output using the first half of each prompt
        out = model.generate(input_prompts, seq_len, prompt_lens=half_lengths, **kwargs)
        if seq_len > max_len:
            padding = torch.ones(
                (input_mask.shape[0], seq_len - max_len), dtype=input_mask.dtype, device=input_mask.device
            )
            out_mask = ~torch.cat([input_mask, padding], dim=1)
        else:
            out_mask = ~input_mask

        batch[VAL_PREFIX + "MARGINAL"] = self.evaluate_marginals(
            self.cfg.vocab_size, prompts, target_mask, out, out_mask
        )
        batch[VAL_PREFIX + "COVARIANCE"] = self.evaluate_covariance(
            self.cfg.vocab_size, prompts, target_mask, out, out_mask
        )
        logits, _ = self.model.model(prompts, mask=mask, return_logits_and_embeddings=True)
        batch[VAL_PREFIX + "PERPLEXITY"] = self.evaluate_perplexity(logits, prompts, target_mask)

        return batch

    @classmethod
    def evaluate_marginals(
        cls,
        vocab_size: int,
        data: torch.Tensor,
        target_mask: torch.Tensor,
        out: torch.Tensor,
        out_mask: torch.Tensor,
    ):
        """Assess the distance between each patient's marginal distribution and the forecasted distribution.

        To assess the fitting of the marginal distribution, we compute the average of Wasserstein distance
        between each marginal distribution $S(X_k)$ of the real time series and generated time series over all
        patients $k$ and codes $i$ as the marginal distribution metric and $S(X_k)^{(j)}$.

        Args:
            vocab_size (int): Number of unique codes in the vocabulary.
            data (torch.Tensor): Includes all patient data and has shape (batch_size, seq_len), where each
                element is a code.
            target_mask (torch.Tensor): Mask for 'data' where true values indicate tokens we are evaluating.
            out (torch.Tensor): generated output from the model
            out_mask (torch.Tensor): Mask for 'out' where true values indicate tokens that were generated
                and we wish to evaluate.
        Returns:
            torch.Tensor: Mean Wasserstein distance between real and generated marginal distributions.

        Example:
            >>> import torch
            >>> data = torch.tensor([[1, 2, 3], [4, 5, 6]])
            >>> target_mask = torch.tensor([[1, 1, 1], [1, 1, 0]], dtype=torch.bool)
            >>> out = torch.tensor([[1, 2, 4], [4, 5, 7]])
            >>> out_mask = torch.tensor([[1, 1, 1], [1, 1, 0]], dtype=torch.bool)
            >>> vocab_size = 8
            >>> result = EicTokenForecastingModule.evaluate_marginals(
            >>>     vocab_size, data, target_mask, out, out_mask)
            >>> isinstance(result, torch.Tensor) and result.numel() == 1
            True
        """
        # Create one-hot encoded tensors
        data_onehot = F.one_hot(data, num_classes=vocab_size).float()
        out_onehot = F.one_hot(out, num_classes=vocab_size).float()

        # Apply masks
        data_onehot = data_onehot * target_mask.unsqueeze(-1)
        out_onehot = out_onehot * out_mask.unsqueeze(-1)

        # Compute marginal distributions
        data_marginals = data_onehot.sum(dim=1) / target_mask.sum(dim=1, keepdim=True)
        out_marginals = out_onehot.sum(dim=1) / out_mask.sum(dim=1, keepdim=True)

        # Compute Wasserstein distance
        wasserstein_dist = torch.norm(
            torch.cumsum(data_marginals, dim=1) - torch.cumsum(out_marginals, dim=1), p=1, dim=1
        )

        return wasserstein_dist.mean()

    @classmethod
    def evaluate_covariance(
        cls,
        vocab_size: int,
        data: torch.Tensor,
        target_mask: torch.Tensor,
        out: torch.Tensor,
        out_mask: torch.Tensor,
    ):
        """Assess the distance between each patient's covariance distribution and forecasted distribution.

        This function computes the difference between the covariance matrices of the real data and the model's
        generated data for each patient. The result is averaged over all patients and evaluated only on the
        tokens specified by the masks.

        Args:
            vocab_size (int): Number of unique codes in the vocabulary.
            data (torch.Tensor): Real patient data with shape (batch_size, seq_len), where each element
                represents a code. target_mask (torch.Tensor): A binary mask for 'data' with shape
                (batch_size, seq_len), where True values indicate the tokens that should be included in
                the evaluation.
            out (torch.Tensor): Generated output from the model with shape (batch_size, seq_len),
                corresponding to the same codes as in 'data'.
            out_mask (torch.Tensor): A binary mask for 'out' with shape (batch_size, seq_len), where
                True values indicate the tokens in the generated data that should be included in the
                evaluation.

        Returns:
            torch.Tensor: Mean Frobenius norm of the difference between real and generated covariance
                matrices.

        Example:
            >>> data = torch.tensor([[1, 2, 3], [4, 5, 6]])
            >>> target_mask = torch.tensor([[1, 1, 1], [1, 1, 0]], dtype=torch.bool)
            >>> out = torch.tensor([[1, 2, 4], [4, 5, 7]])
            >>> out_mask = torch.tensor([[1, 1, 1], [1, 1, 0]], dtype=torch.bool)
            >>> vocab_size = 8
            >>> result = EicTokenForecastingModule.evaluate_covariance(
            >>>     vocab_size, data, target_mask, out, out_mask)
            >>> isinstance(result, torch.Tensor) and result.numel() == 1
            True
        """
        # Create one-hot encoded tensors
        data_onehot = F.one_hot(data, num_classes=vocab_size).float()
        out_onehot = F.one_hot(out, num_classes=vocab_size).float()

        # Apply masks
        data_onehot = data_onehot * target_mask.unsqueeze(-1)
        out_onehot = out_onehot * out_mask.unsqueeze(-1)

        # Compute covariance matrices
        data_mean = data_onehot.mean(dim=1, keepdim=True)
        out_mean = out_onehot.mean(dim=1, keepdim=True)

        data_cov = torch.bmm((data_onehot - data_mean).transpose(1, 2), (data_onehot - data_mean))
        out_cov = torch.bmm((out_onehot - out_mean).transpose(1, 2), (out_onehot - out_mean))

        # Normalize covariance matrices
        data_cov = data_cov / (target_mask.sum(dim=1, keepdim=True).unsqueeze(-1) - 1).clamp(min=1)
        out_cov = out_cov / (out_mask.sum(dim=1, keepdim=True).unsqueeze(-1) - 1).clamp(min=1)

        # Compute Frobenius norm of the difference
        cov_diff = torch.norm(data_cov - out_cov, p="fro", dim=(1, 2))

        return cov_diff.mean()

    @classmethod
    def evaluate_perplexity(cls, logits: torch.Tensor, data: torch.Tensor, target_mask: torch.Tensor):
        """Evaluate the perplexity of the model's forecasted distribution.

        Args:
            logits (torch.Tensor): Forecasted logits from the model with shape
                (batch_size, seq_len, num_classes).
            data (torch.Tensor): Includes all patient data and has shape (batch_size, seq_len),
                where each element is a code.
            target_mask (torch.Tensor): Mask for 'data' where true values indicate tokens
                we are evaluating.
        Returns:
            torch.Tensor: Perplexity of the model's forecasted distribution.

        Example:
            >>> batch_size, seq_len, num_classes = 2, 3, 5
            >>> logits = torch.randn(batch_size, seq_len, num_classes)
            >>> data = torch.randint(0, num_classes, (batch_size, seq_len))
            >>> target_mask = torch.tensor([[1, 1, 1], [1, 1, 0]], dtype=torch.float)
            >>> perplexity = EicTokenForecastingModule.evaluate_perplexity(logits, data, target_mask)
            >>> isinstance(perplexity, torch.Tensor) and perplexity.numel() == 1
            True
            >>> perplexity > 0
            tensor(True)
        """

        # Compute cross entropy loss
        loss = F.cross_entropy(logits.transpose(1, 2), data, reduction="none")

        # Apply mask and compute perplexity
        masked_loss = loss * target_mask
        perplexity = torch.exp(masked_loss.sum() / target_mask.sum())

        return perplexity