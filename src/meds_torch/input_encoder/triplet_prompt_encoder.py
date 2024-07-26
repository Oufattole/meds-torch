import torch
from omegaconf import DictConfig
from torch import nn

from meds_torch.input_encoder.triplet_encoder import CVE
from meds_torch.utils.module_class import Module


class TripletPromptEncoder(nn.Module, Module):
    def __init__(self, cfg: DictConfig):
        super().__init__()
        self.cfg = cfg
        self.date_embedder = CVE(cfg)
        self.code_embedder = torch.nn.Embedding(cfg.vocab_size, embedding_dim=cfg.token_dim)
        self.numerical_value_embedder = CVE(cfg)
        # Special Tokens
        self.ts_token = torch.nn.Parameter(torch.randn(cfg.token_dim))
        self.code_prefix = torch.nn.Parameter(torch.randn(cfg.token_dim))
        self.val_prefix = torch.nn.Parameter(torch.randn(cfg.token_dim))

    def embed_func(self, embedder, x):
        return embedder(x.unsqueeze(0).transpose(2, 0)).permute(1, 2, 0)

    def get_embedding(self, batch):
        # Extract batch data
        static_mask = batch["static_mask"]
        code = batch["code"]
        numerical_value = batch["numerical_value"]
        time_delta_days = batch["time_delta_days"]
        numerical_value_mask = batch["numerical_value_mask"]

        # Create masks for conditions
        time_valid_mask = time_delta_days != 0
        value_valid_mask = numerical_value_mask

        # Embed data
        time_emb = self.embed_func(self.date_embedder, time_delta_days)
        code_emb = self.code_embedder(code)
        val_emb = self.embed_func(self.numerical_value_embedder, numerical_value)

        # Apply masks
        ts_tokens = torch.where(
            static_mask.unsqueeze(dim=-1) & time_valid_mask.unsqueeze(dim=-1),
            time_emb,
            self.ts_token.expand_as(time_emb),
        )
        val_tokens = torch.where(
            value_valid_mask.unsqueeze(dim=-1), val_emb, self.val_prefix.expand_as(val_emb)
        )

        # Initialize an empty list to collect embeddings
        embeddings = []

        # Append tokens conditionally
        for t, c, v, t_valid, v_valid in zip(
            ts_tokens, code_emb, val_tokens, time_valid_mask, value_valid_mask
        ):
            temp_emb = []
            if t_valid:
                temp_emb.append(t)
            temp_emb.append(self.code_prefix)
            temp_emb.append(c)
            if v_valid:
                temp_emb.append(self.val_prefix)
                temp_emb.append(v)
            embeddings.append(torch.cat(temp_emb))

        # Stack embeddings to form batch output
        embedding = torch.stack(embeddings)

        assert embedding.isfinite().all(), "Embedding is not finite"
        return embedding

    def print_readable_sequence(self, embedding, code_vocab):
        result = []
        for i, token in enumerate(embedding):
            if i % 3 == 0:  # Time token
                result.append(f"[TS = {token.item():.2f} days]")
            elif i % 3 == 1:  # Code token
                code_str = code_vocab[token.item()] if token.item() in code_vocab else "Unknown"
                result.append(f"[CODE: {code_str}]")
            else:  # Value token
                result.append(f"[VAL = {token.item()}]")
        return " ".join(result)

    def forward(self, batch):
        embedding = self.get_embedding(batch)
        batch["input_encoder_mask_key"] = batch["mask"]
        batch["input_encoder_tokens_key"] = embedding
        return batch
