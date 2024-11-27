import torch

from meds_torch.input_encoder import INPUT_ENCODER_MASK_KEY, INPUT_ENCODER_TOKENS_KEY
from meds_torch.models import (
    BACKBONE_EMBEDDINGS_KEY,
    MODEL_BATCH_LOSS_KEY,
    MODEL_EMBEDDINGS_KEY,
    MODEL_LOGITS_KEY,
    MODEL_PRED_PROBA_KEY,
)
from meds_torch.models.supervised_model import SupervisedModule


class MultimodalSupervisedModule(SupervisedModule):
    def forward(self, batch) -> dict:
        batched_embeddings = self.input_encoder(batch)

        ehr_batch = {
            INPUT_ENCODER_MASK_KEY: batched_embeddings[INPUT_ENCODER_MASK_KEY],
            INPUT_ENCODER_TOKENS_KEY: batched_embeddings[INPUT_ENCODER_TOKENS_KEY]["EHR_embedding"],
        }
        # Shape: (B, seq_len, token_dim)
        ehr_batch = self.model(ehr_batch)
        ehr_embeddings = ehr_batch[BACKBONE_EMBEDDINGS_KEY]  # Shape: (B, token_dim)

        ecg_embeddings = batched_embeddings[INPUT_ENCODER_TOKENS_KEY][
            "ECG_embedding"
        ]  # Shape: (B, token_dim)

        # Intermediate fusion
        #
        # TODO(zberger): Currently we only support taking the mean of the
        # two modalities, but should make this a configurable option and
        # provide other mechanisms of intermediate fusion. Depending on results,
        # might consider normalizing embeddings first.
        #
        # Stack the two embeddings along a new dimension (modality dimension)
        # Shape: (2, B, token_dim)
        stacked_embeddings = torch.stack([ehr_embeddings, ecg_embeddings])
        # Compute the mean across the modality dimension (dim=0)
        # Shape: (B, token_dim)
        fused_embeddings = stacked_embeddings.mean(dim=0)

        logits = self.projection(fused_embeddings)
        if self.cfg.get_representations:
            loss = None
        else:
            loss = self.criterion(logits.squeeze(dim=-1), batch["boolean_value"].float())
        batch[MODEL_EMBEDDINGS_KEY] = fused_embeddings
        batch[MODEL_LOGITS_KEY] = logits
        batch[MODEL_PRED_PROBA_KEY] = torch.sigmoid(logits)
        batch[MODEL_BATCH_LOSS_KEY] = loss

        return batch
