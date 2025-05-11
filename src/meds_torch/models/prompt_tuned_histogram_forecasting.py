import re
from collections.abc import Iterator

import torch

from meds_torch.models import (
    BACKBONE_EMBEDDINGS_KEY,
    MODEL_BATCH_LOSS_KEY,
    MODEL_LOSS_KEY,
)
from meds_torch.models.histogram_forecasting import HistogramForecastingModule
from meds_torch.utils import RankedLogger

log = RankedLogger(__name__, rank_zero_only=True)


class PromptTunedHistogram(HistogramForecastingModule):
    def __init__(self, cfg):
        super().__init__(cfg)
        # add your prompt-mlp
        if not hasattr(self.input_encoder, "prompt_mlp"):
            raise ValueError("You must implement a prompt_mlp in your input encoder")

        # freeze everything in the base class
        self.freeze_base_parameters()

    def freeze_base_parameters(self):
        if not self.cfg.finetune_all_params:
            base = [p for p in self.parameters() if p not in set(self.input_encoder.prompt_mlp.parameters())]
            for p in base:
                p.requires_grad = False

    @staticmethod
    def _is_norm_bias_param(name: str) -> bool:
        # matches "bias" or "layernorm.weight" / "LayerNorm.bias" etc.
        return bool(re.search(r"(bias|layer(_?)norm(\d*)\.weight)", name, re.IGNORECASE))

    def _norm_bias_params(self) -> Iterator[torch.nn.Parameter]:
        for n, p in self.named_parameters():
            if p.requires_grad and self._is_norm_bias_param(n):
                yield p

    def _decay_params(self) -> Iterator[torch.nn.Parameter]:
        for n, p in self.named_parameters():
            if p.requires_grad and not self._is_norm_bias_param(n):
                yield p

    def configure_optimizers(self):
        # !!!
        # TODO, quick check it's not training other parameters
        # prompt_params = list(self.input_encoder.prompt_mlp.parameters())
        # other_params  = [p for p in self.parameters() if p.requires_grad and p not in prompt_params]

        trainable_names = [n for n, p in self.named_parameters() if p.requires_grad]
        log.info("Unfrozen (trainable) parameters:\n" + "\n".join(trainable_names))

        param_groups = [
            {"params": list(self._decay_params()), "weight_decay": self.cfg.weight_decay},
            {"params": list(self._norm_bias_params()), "weight_decay": 0.0},
        ]

        optimizer = self.optimizer(param_groups)
        if self.scheduler is not None:
            scheduler = self.scheduler.instantiate(optimizer)
            return {
                "optimizer": optimizer,
                "lr_scheduler": {"scheduler": scheduler, **self.scheduler.extra_kwargs},
            }
        return optimizer

    def forward(self, batch, keep_code_logits=False):
        if self.cfg.do_inference:
            with torch.inference_mode():
                return super().forward(batch, keep_code_logits=keep_code_logits)

        with torch.no_grad():
            batch = super().forward(batch, keep_code_logits=keep_code_logits)
            histogram_embeddings = batch[BACKBONE_EMBEDDINGS_KEY]
            if self.cfg.encode_for_prompt_tuning:
                histogram_embeddings = self.autoencoder.encode(histogram_embeddings).mean
        batch = self.input_encoder(batch, histogram_embedding=histogram_embeddings)
        batch = super().forward(batch, keep_code_logits=keep_code_logits, skip_input_encoder=True)

        if self.input_encoder.cfg.do_vector_quantize_prompts:
            vq = self.input_encoder.prompt_quantizer
            vq_loss = vq.codebook_loss + vq.commit_loss
            batch[MODEL_LOSS_KEY] += vq_loss
            batch[MODEL_BATCH_LOSS_KEY] += vq_loss

        return batch
