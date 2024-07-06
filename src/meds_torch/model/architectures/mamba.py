import torch
from omegaconf import DictConfig

try:
    from mamba_ssm.models.config_mamba import MambaConfig
    from mamba_ssm.models.mixer_seq_simple import MambaLMHeadModel
except ImportError:
    raise ImportError("Please install mamba-ssm to use this model.")


class MambaModel(torch.nn.Module):
    """Wrapper of MambaLMHeadModel for use in MEDS with triplet token embeddings."""

    def __init__(self, cfg: DictConfig):
        super().__init__()
        self.cfg = cfg
        model = MambaLMHeadModel(
            config=MambaConfig(
                d_model=cfg.model.embedder.token_dim,
                vocab_size=2,
                n_layer=cfg.model.params.n_layers,
            )
        )
        model.backbone.embedding = torch.nn.Identity()
        model.lm_head = torch.nn.Identity()
        self.model = model
        self.get_last_token = cfg.model.get_last_token

    def forward(self, batch, mask):
        output = self.model(batch.transpose(1, 2))
        if self.get_last_token:
            logits = output.logits
            # extract the final non-padding token
            mask_max = (~mask).max(dim=1)
            lengths = mask_max.indices
            lengths[~mask_max.values] = mask.shape[1]
            lengths -= 1
            # expand to match the shape of next_token_embedding
            indices = lengths.view(-1, 1, 1).expand(-1, -1, logits.shape[-1])
            rep = torch.gather(logits, dim=1, index=indices).squeeze(1)  # gather the last token
            return rep
        else:
            return output

    # output = self.model(batch, mask=mask)
    #     if self.get_last_token:
    #         lengths = mask.argmax(dim=1)  # get the length of each sequence
    #         # extract the final non-padding token
    #         indices = (
    #             torch.tensor(lengths, device=output.device) - 1
    #         )  # indices of the last token
    #         indices = indices.view(-1, 1, 1).expand(
    #             -1, -1, output.shape[-1]
    #         )  # expand to match the shape of all_token_embeddings
    #         last_token = torch.gather(output, dim=1, index=indices).squeeze(
    #             1
    #         )  # gather the last token
    #         return last_token
    #     else:
    #         return output
