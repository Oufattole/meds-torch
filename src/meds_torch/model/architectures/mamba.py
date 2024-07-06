import torch
from mamba_ssm.models.config_mamba import MambaConfig
from mamba_ssm.models.mixer_seq_simple import MambaLMHeadModel


class MambaModel(torch.nn.Module):
    """Container module with an encoder, a recurrent or transformer module, and a decoder.

    Copied from: https://github.com/pytorch/examples/blob/main/word_language_model/model.py
    """

    def __init__(self, cfg: TrainConfig):
        super().__init__(cfg)
        self.cfg = cfg
        model = MambaLMHeadModel(
            config=MambaConfig(
                d_model=cfg.model_config.embed_size,
                vocab_size=2,
            )
        )
        model.backbone.embedding = torch.nn.Identity()
        model.lm_head = torch.nn.Identity()
        self.model = model

    def forward(self, batch, bos_eos_tokens=True):
        src, lengths = self.embed(batch, bos_eos_tokens=bos_eos_tokens)
        output = self.model(src.transpose(1, 2))
        logits = output.logits
        # extract the final non-padding token
        indices = torch.tensor(lengths, device=logits.device) - 1  # indices of the last token
        indices = indices.view(-1, 1, 1).expand(
            -1, -1, logits.shape[-1]
        )  # expand to match the shape of next_token_embedding
        rep = torch.gather(logits, dim=1, index=indices).squeeze(1)  # gather the last token
        return rep
