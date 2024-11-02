import dataclasses
import enum
import torch
from omegaconf import DictConfig
from torch import nn
from transformers import AutoTokenizer, AutoModel
from meds_torch.input_encoder import INPUT_ENCODER_MASK_KEY, INPUT_ENCODER_TOKENS_KEY
from meds_torch.utils.module_class import Module

@dataclasses.dataclass
class ModelOutput:
    rep: torch.Tensor
    hidden_states: torch.Tensor = None

class Triplet(enum.Enum):
    DATE = "date"
    VALUE = "value"
    VARIABLE = "variable"

def sequence_mask(lengths, maxlen, dtype=torch.bool):
    row_vector = torch.arange(0, maxlen, 1, device=lengths.device)
    matrix = torch.unsqueeze(lengths, dim=-1)
    mask = row_vector < matrix
    mask.type(dtype)
    return mask

class CVE(nn.Module):
    """Continuous Value Encoder (CVE) module."""
    def __init__(self, cfg):
        super().__init__()
        self.layer = nn.Linear(1, cfg.token_dim)

    def forward(self, x):
        return self.layer(x)

class TextEncoder(nn.Module):
    """Text Encoder module using BioClinicalBERT for embedding text data."""
    def __init__(self, cfg):
        super().__init__()
        # Load BioClinicalBERT model and tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained("emilyalsentzer/Bio_ClinicalBERT")
        self.text_model = AutoModel.from_pretrained("emilyalsentzer/Bio_ClinicalBERT")
        self.token_dim = cfg.token_dim  # Expected embedding dimension

    def forward(self, text_data):
        # Tokenize the input text data
        inputs = self.tokenizer(text_data, padding=True, truncation=True, return_tensors="pt")
        
        # Pass the tokens through the model to get embeddings
        outputs = self.text_model(**inputs)
        
        # Use the [CLS] token embedding as a summary of the sequence
        cls_embedding = outputs.last_hidden_state[:, 0, :]
        
        # Project to the desired token dimension if needed
        if cls_embedding.size(-1) != self.token_dim:
            cls_embedding = nn.Linear(cls_embedding.size(-1), self.token_dim)(cls_embedding)

        return cls_embedding

class MultiModalTripletEncoder(nn.Module, Module):
    """Triplet Encoder with an additional modality for textual data."""
    def __init__(self, cfg: DictConfig):
        super().__init__()
        self.cfg = cfg
        # Define Triplet Embedders
        self.date_embedder = CVE(cfg)
        self.code_embedder = torch.nn.Embedding(cfg.vocab_size, embedding_dim=cfg.token_dim)
        self.numeric_value_embedder = CVE(cfg)
        # Define Text Embedder using BioClinicalBERT
        self.text_encoder = TextEncoder(cfg)

    def embed_func(self, embedder, x):
        out = embedder.forward(x[None, :].transpose(2, 0)).permute(1, 2, 0)
        return out

    def get_triplet_embedding(self, batch):
        # Process triplet data as in the original TripletEncoder
        static_mask = batch["static_mask"]
        code = batch["code"]
        numeric_value = batch["numeric_value"]
        time_delta_days = batch["time_delta_days"]
        numeric_value_mask = batch["numeric_value_mask"]

        # Embed times and mask static value times
        time_emb = self.embed_func(self.date_embedder, time_delta_days) * ~static_mask.unsqueeze(dim=1)
        # Embed codes
        code_emb = self.code_embedder.forward(code).permute(0, 2, 1)
        # Embed numerical values and mask nan values
        val_emb = self.embed_func(self.numeric_value_embedder, numeric_value) * numeric_value_mask.unsqueeze(dim=1)

        # Sum the (time, code, value) triplets to create the triplet embedding
        triplet_embedding = time_emb + code_emb + val_emb
        assert triplet_embedding.isfinite().all(), "Triplet embedding is not finite"
        
        # Check sequence length
        if triplet_embedding.shape[-1] > self.cfg.max_seq_len:
            raise ValueError(
                f"Triplet embedding length {triplet_embedding.shape[-1]} "
                f"is greater than max_seq_len {self.cfg.max_seq_len}"
            )
        
        return triplet_embedding.transpose(1, 2)

    def get_text_embedding(self, batch):
        # Process text data with BioClinicalBERT
        text_data = batch["text_data"]
        text_embedding = self.text_encoder(text_data)
        return text_embedding

    def forward(self, batch):
        # Get triplet embedding
        triplet_embedding = self.get_triplet_embedding(batch)
        # Get text embedding
        text_embedding = self.get_text_embedding(batch)

        # Add modality-specific embeddings to batch with unique keys
        batch[INPUT_ENCODER_TOKENS_KEY + "_triplet"] = triplet_embedding
        batch[INPUT_ENCODER_TOKENS_KEY + "_text"] = text_embedding
        batch[INPUT_ENCODER_MASK_KEY] = batch["mask"]

        return batch