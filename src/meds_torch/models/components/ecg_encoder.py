import torch
from torch import nn


class ECGEncoder(nn.Module):
    """
    ECG Encoder for processing 12-lead ECG signals.

    This module takes as input a tensor of size (E, 12, 5000), representing some number E of
    12-lead ECG signals sampled at 500 Hz over 10 seconds. It encodes these signals into a
    fixed-size representation using a patch-based embedding mechanism and transformer layers.
    The steps performed are as follows:

    1. Split each ECG lead into 50 patches along the time dimension. Each patch has a size of
       100 samples. This reshapes the input from (E, 12, 5000) to (E, 12, 50, 100), where
       50 represents the number of patches and 100 is the patch size.

    2. Project each patch into a token_dim-dimensional space using a linear layer, resulting
       in a tensor of shape (E, 12, 50, token_dim).

    3. Form positional encodings:
       - Temporal positional encodings of shape (1, 50, token_dim) are generated using a sinusoidal
         positional encodings to capture temporal order within each lead.
       - Learned lead positional embeddings of shape (12, token_dim) are used to distinguish between
         the different ECG leads.

    4. Combine the temporal and lead positional encodings with the patch embeddings to form the
       positional-aware representation of the ECG patches.

    5. Flatten the patches across the leads and time dimensions, resulting in a tensor of shape
       (E, 600, token_dim), where 600 = 12 (leads) * 50 (patches per lead).

    6. Append a learnable class token to the sequence, extending the tensor to (E, 601, token_dim).
       The class token is used to aggregate global information from the entire input.

    7. Pass the resulting sequence through a stack of 4 transformer encoder layers, which model
       relationships across patches and leads.

    8. Extract the final representation of the class token, producing a tensor of shape (E, token_dim).

    Args:
        cfg (DictConfig): Configuration object with the following attributes:
            - token_dim (int): Dimensionality of the token embeddings.
            - num_leads (int): Number of EKG leads (default 12).
            - num_patches (int): Number of patches to divide EKG into.
            - patch_size (int): Size of each EKG patch.

    Inputs:
        x (torch.Tensor): Input tensor of shape (E, 12, 5000), where:
            - E is the number of ECGs in the sample.
            - 12 represents the 12 ECG leads.
            - 5000 corresponds to the number of samples per lead.

    Outputs:
        torch.Tensor: A tensor of shape (E, token_dim), representing the final encoded
        ECG signals as fixed-size embeddings.

    Example:
        >>> import torch
        >>> from omegaconf import OmegaConf
        >>> # Configuration for the encoder
        >>> cfg = OmegaConf.create({"token_dim": 128, "num_leads": 12, "num_patches": 50, "patch_size": 100})
        >>> # Initialize the ECGEncoder
        >>> encoder = ECGEncoder(cfg)
        >>> # Create a dummy input tensor of shape (batch_size=2, leads=12, time=5000)
        >>> x = torch.randn(2, 12, 5000)
        >>> # Perform a forward pass
        >>> output = encoder(x)
        >>> # Check the output shape
        >>> output.shape
        torch.Size([2, 128])
    """

    def __init__(self, cfg):
        super().__init__()
        self.token_dim = cfg.token_dim
        self.num_patches = cfg.num_patches
        self.patch_size = cfg.patch_size
        self.num_leads = cfg.num_leads
        self.num_tokens = self.num_patches * self.num_leads

        # Linear projection from patch_size to token_dim.
        self.patch_proj = nn.Linear(self.patch_size, self.token_dim)

        # Learned positional embeddings for lead dimension.
        self.lead_pos_emb = nn.Embedding(self.num_leads, self.token_dim)

        # Sinusoidal positional encodings for time dimension.
        self.time_pos_enc = self._generate_sinusoidal_positional_encoding(self.num_patches, self.token_dim)
        self.register_buffer("time_pos_enc_buffer", self.time_pos_enc)

        # Class token (learned parameter).
        self.cls_token = nn.Parameter(torch.zeros(1, 1, self.token_dim))

        # Transformer Encoder with 4 layers.
        encoder_layer = nn.TransformerEncoderLayer(d_model=self.token_dim, nhead=2)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=4)

    def _generate_sinusoidal_positional_encoding(self, num_positions, dim):
        """
        Generates a sinusoidal positional encoding matrix.

        Args:
            num_positions (int): Number of positions (e.g., 50 patches).
            dim (int): Dimension of the embeddings (token_dim).

        Returns:
            pe (torch.Tensor): Positional encoding of shape (1, num_positions, dim).
        """
        position = torch.arange(num_positions).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, dim, 2) * (-torch.log(torch.tensor(10000.0)) / dim))
        pe = torch.zeros(num_positions, dim)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # Shape: (1, num_positions, dim)
        return pe

    def forward(self, x):
        """
        Forward pass of the ECGEncoder.

        Args:
            x (torch.Tensor): Input tensor of shape (E, 12, 5000).

        Returns:
            cls_token_final (torch.Tensor): Output representation of shape (E, token_dim).
        """
        # x shape: (E, 12, 5000)
        E = x.size(0)

        # Step 1: Patch each ECG lead into 50 patches
        # Reshape to (E, 12, 50, 100)
        x = x.view(E, self.num_leads, self.num_patches, self.patch_size)

        # Step 2: Project each patch to token_dim
        # Maintain structure of ECG/leads; merge E and leads for projection to
        # make sure linear layer operates on each patch independently given a ECG/lead.
        x = x.view(E * self.num_leads, self.num_patches, self.patch_size)  # (E*12, 50, 100)
        x = self.patch_proj(x)  # (E*12, 50, token_dim)
        x = x.view(E, self.num_leads, self.num_patches, self.token_dim)  # (E, 12, 50, token_dim)

        # Step 3: Add time positional encodings
        time_pe = self.time_pos_enc_buffer.to(x.device)  # (1, 50, token_dim)
        x = x + time_pe.unsqueeze(1)  # Broadcast to (E, 12, 50, token_dim)

        # Step 4: Add lead positional embeddings
        lead_indices = torch.arange(self.num_leads, device=x.device)  # (12,)
        lead_pe = self.lead_pos_emb(lead_indices)  # (12, token_dim)
        x = x + lead_pe.unsqueeze(0).unsqueeze(2)  # Broadcast to (E, 12, 50, token_dim)

        # Step 5: Sum up the positional embeddings (already completed by now)

        # Step 6: Flatten and add class token
        x = x.view(E, self.num_tokens, self.token_dim)  # (E, 600, token_dim) since 12*50=600
        cls_tokens = self.cls_token.expand(E, -1, -1)  # (E, 1, token_dim)
        x = torch.cat((cls_tokens, x), dim=1)  # (E, 601, token_dim)

        # Step 7: Pass through transformer encoder
        x = x.permute(1, 0, 2)  # Transformer expects (seq_len, batch_size, token_dim)
        x = self.transformer_encoder(x)  # Output shape: (seq_len, E, token_dim)

        # Step 8: Select the class token (first token)
        cls_token_final = x[0, :, :]  # Shape: (E, token_dim)
        return cls_token_final


if __name__ == "__main__":
    import doctest

    doctest.testmod(verbose=True)
