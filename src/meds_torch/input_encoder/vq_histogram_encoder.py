import torch
from torch import nn
from torchvision.ops import MLP

from meds_torch.data.components.histogram_pytorch_dataset import TokenInsertionStrategy
from meds_torch.input_encoder import INPUT_ENCODER_MASK_KEY, INPUT_ENCODER_TOKENS_KEY
from meds_torch.models import (
    BACKBONE_EMBEDDINGS_KEY,
    BACKBONE_TOKENS_KEY,
    MODEL_BATCH_LOSS_KEY,
    MODEL_LOSS_KEY,
)
from meds_torch.utils import RankedLogger
from meds_torch.utils.module_class import Module

log = RankedLogger(__name__, rank_zero_only=True)
import numpy as np
import torch
import torch.nn as nn
from einops import rearrange

from meds_torch.utils.module_class import Module


class VectorQuantizer2(nn.Module):
    """
    Vector Quantization layer adapted from VQ-VAE.

    This is the **image‑oriented** implementation shipped with many VQ‑GAN repos.
    Your goal is to reuse *most* of the logic for histogram‑shaped inputs, so the
    extra comments below flag every image‑specific assumption (shape handling,
    convolution‑compatibility, etc.) you must rewrite when wiring it into your
    private VAE built on 1‑D histogram tensors of shape ``(B, C)``.

    ---------------------------------------------------------------------------
    Quick recap of the contract:
        • ``n_e``   – number of embeddings in the discrete code‑book
        • ``e_dim`` – dimensionality of each embedding vector
        • ``beta``  – commitment loss coefficient (see VQ‑VAE paper)

    The class returns
        z_q          – quantized latent (same shape as input after reshape)
        loss         – codebook + commitment losses
        aux tuple    – (perplexity, one‑hot encodings, indices)

    ⚠️  **Histogram migration guide**  ⚠️
        ▸  Eliminate height/width handling – histograms do *not* have spatial
           axes. The most important edits are marked with “# HISTOGRAM‑EDIT”.
        ▸  Replace convolutional encoder/decoder blocks *outside* of this class
           with MLPs or whatever you already use for 1‑D data. Nothing inside
           VectorQuantizer2 relies on convolutions; only the shape juggling does.
    """

    # NOTE: due to a historical bug the beta term was applied to the wrong part
    #       of the loss. For backwards compatibility ``legacy=True`` keeps that
    #       behaviour. Set ``legacy=False`` to use the correct formulation.
    def __init__(
        self, n_e, e_dim, beta, remap=None, unknown_index="random", sane_index_shape=False, legacy=True
    ):
        super().__init__()
        self.n_e = n_e
        self.e_dim = e_dim
        self.beta = beta
        self.legacy = legacy

        # --------------------------------------------------------------------
        # Embedding table holding the code‑book vectors (shape ``n_e × e_dim``)
        # --------------------------------------------------------------------
        self.embedding = nn.Embedding(self.n_e, self.e_dim)
        # Uniform initialisation as in the VQ‑VAE paper.
        self.embedding.weight.data.uniform_(-1.0 / self.n_e, 1.0 / self.n_e)

        # Optional index remapping support – irrelevant for histogram vs image,
        # but preserved intact. Skip unless you also need a reduced code‑book.
        self.remap = remap
        if self.remap is not None:
            # ``used`` is a Boolean mask of active code indices loaded from npy.
            self.register_buffer("used", torch.tensor(np.load(self.remap)))
            self.re_embed = self.used.shape[0]
            self.unknown_index = unknown_index  # "random" | "extra" | int
            if self.unknown_index == "extra":
                self.unknown_index = self.re_embed  # extra slot appended
                self.re_embed = self.re_embed + 1
            print(
                f"Remapping {self.n_e} indices to {self.re_embed} indices. "
                f"Using {self.unknown_index} for unknown indices."
            )
        else:
            self.re_embed = n_e

        # When ``sane_index_shape=True`` we keep the index tensor shaped like the
        # feature map (B, H, W) instead of flattening. Histogram case: you will
        # likely set this *False* and squeeze the spatial dims entirely.
        self.sane_index_shape = sane_index_shape

    # ------------------------------------------------------------------------
    # Helper: remap original (full) indices -> compact set of *used* indices.
    # ------------------------------------------------------------------------
    def remap_to_used(self, inds):
        ishape = inds.shape
        assert len(ishape) > 1  # expects batch dimension + something else
        inds = inds.reshape(ishape[0], -1)
        used = self.used.to(inds)
        match = (inds[:, :, None] == used[None, None, ...]).long()
        new = match.argmax(-1)
        unknown = match.sum(2) < 1
        if self.unknown_index == "random":
            new[unknown] = torch.randint(0, self.re_embed, size=new[unknown].shape, device=new.device)
        else:
            new[unknown] = self.unknown_index
        return new.reshape(ishape)

    # ------------------------------------------------------------------------
    # Helper: inverse of the above
    # ------------------------------------------------------------------------
    def unmap_to_all(self, inds):
        ishape = inds.shape
        assert len(ishape) > 1
        inds = inds.reshape(ishape[0], -1)
        used = self.used.to(inds)
        if self.re_embed > self.used.shape[0]:  # extra token case
            inds[inds >= self.used.shape[0]] = 0  # map to 0
        back = torch.gather(used[None, :][inds.shape[0] * [0], :], 1, inds)
        return back.reshape(ishape)

    # ---------------------------------------------------------------------
    # Main forward pass – this is where most edits for histogram inputs live
    # ---------------------------------------------------------------------
    def forward(self, z, temp=None, rescale_logits=False, return_logits=False):
        # ``z`` should come from your encoder and have latent dim = e_dim.
        # For images: shape (B, C, H, W).  For histograms you likely have
        # (B, C). You can keep the channel dim = e_dim and *drop* H, W.

        assert temp is None or temp == 1.0, "Only kept for Gumbel compat."
        assert not rescale_logits, "Only kept for Gumbel compat."
        assert not return_logits, "Only kept for Gumbel compat."

        # ------------------------------------------------------------------
        # Reshape so that channel is last and the spatial dims are grouped.
        # This is **image‑specific** – replace with an identity or simple view
        # for 1‑D histogram tensors.
        # HISTOGRAM‑EDIT: if ``z`` is (B, C) just do:
        #     z_flattened = z.view(-1, self.e_dim)
        # and drop every rearrange / view linked to H,W below.
        z = rearrange(z, "b c h w -> b h w c").contiguous()
        z_flattened = z.view(-1, self.e_dim)

        # ------------------------------------------------------------------
        # Euclidean distance between each latent vector and every code vector
        # without explicit broadcasting.  Unchanged for histogram data.
        d = (
            torch.sum(z_flattened**2, dim=1, keepdim=True)
            + torch.sum(self.embedding.weight**2, dim=1)
            - 2 * torch.einsum("bd,dn -> bn", z_flattened, rearrange(self.embedding.weight, "n d -> d n"))
        )

        # Index of closest codebook vector for every latent position.
        min_encoding_indices = torch.argmin(d, dim=1)
        z_q = self.embedding(min_encoding_indices).view(z.shape)

        # Optional stats (unused downstream in many repos)
        perplexity = None
        min_encodings = None

        # --------------------------------------------------------------
        # VQ‑VAE loss: embedding loss + commitment loss. Beta scales one of
        # them (see note on legacy bug).
        # --------------------------------------------------------------
        if not self.legacy:
            loss = self.beta * torch.mean((z_q.detach() - z) ** 2) + torch.mean((z_q - z.detach()) ** 2)
        else:  # legacy = buggy ordering kept for compatibility
            loss = torch.mean((z_q.detach() - z) ** 2) + self.beta * torch.mean((z_q - z.detach()) ** 2)

        # Straight‑through estimator: copy gradients from z to z_q.
        z_q = z + (z_q - z).detach()

        # ------------------------------------------------------------------
        # Reshape back to (B, C, H, W).  HISTOGRAM‑EDIT: drop this and keep the
        # squeezed (B, C) tensor, or unsqueeze back to (B, C, 1, 1) if the rest
        # of your pipeline still expects 4‑D tensors.
        z_q = rearrange(z_q, "b h w c -> b c h w").contiguous()

        # ------------------------------------------------------------------
        # Optional index remapping handling – unaffected by histogram change.
        # ------------------------------------------------------------------
        if self.remap is not None:
            min_encoding_indices = min_encoding_indices.reshape(z.shape[0], -1)
            min_encoding_indices = self.remap_to_used(min_encoding_indices)
            min_encoding_indices = min_encoding_indices.reshape(-1, 1)

        # Keep output index shape consistent with ``sane_index_shape`` flag.
        if self.sane_index_shape:
            min_encoding_indices = min_encoding_indices.reshape(z_q.shape[0], z_q.shape[2], z_q.shape[3])

        return z_q, loss, (perplexity, min_encodings, min_encoding_indices)

    # ---------------------------------------------------------------------
    # Utility to retrieve codebook embeddings given indices (used by decoder)
    # ---------------------------------------------------------------------
    def get_codebook_entry(self, indices, shape):
        # ``shape`` is (B, H, W, C) in image case.
        # HISTOGRAM‑EDIT: pass shape=(B, 1, 1, C) or None and adjust below.

        if self.remap is not None:
            indices = indices.reshape(shape[0], -1)  # add batch axis
            indices = self.unmap_to_all(indices)
            indices = indices.reshape(-1)  # flatten again

        # Fetch embeddings from the table.
        z_q = self.embedding(indices)

        if shape is not None:
            z_q = z_q.view(shape)
            # Image code expects (B, C, H, W) – convert back.
            # HISTOGRAM‑EDIT: if you set shape without H,W just skip permute.
            z_q = z_q.permute(0, 3, 1, 2).contiguous()

        return z_q


class HistogramNormalizer(torch.nn.Module):
    def __init__(
        self,
        vocab_size,
        histogram_head_loss,
        max_count=4,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.histogram_head_loss = histogram_head_loss
        self.max_count = max_count
        # Create powers of 2 as a buffer to avoid recomputing

    def get_normalized_size(self):
        if self.histogram_head_loss.startswith("multinomial"):
            return self.vocab_size * self.max_count
        elif self.histogram_head_loss.startswith("cont_softmax") or self.histogram_head_loss.startswith(
            "softmax"
        ):
            return self.vocab_size

        else:
            raise ValueError(f"Unknown histogram_head_loss: {self.histogram_head_loss}")

    def multinomial_to_count(self, multinomial):
        """
        Recovers the count from a multinomial (one-hot) vector.
        """
        # multinomial is expected to be (…, max_count+1). We take the index of the maximum value.
        return multinomial.argmax(dim=-1, keepdim=True)

    def multinomial_transform(self, x):
        """
        Converts a batch of histograms (shape: [batch, vocab_size]) into
        a normalized representation using a multinomial expansion.
        """

        return x

    def multinomial_reverse_transform(self, x):
        """
        Converts the normalized representation back into a histogram.
        Returns:
            counts: a tensor of shape [batch, vocab_size] with the recovered counts.
            total: a tensor of shape [batch] with the sum of counts per histogram.
        """
        # if len(x.shape) == 2:
        #     return x
        # Recover each count by taking the argmax over the multinomial dimension.
        x = x.reshape(x.shape[0], self.vocab_size, -1)
        counts = self.multinomial_to_count(x).squeeze(-1)
        return counts

    def transform(self, x):
        if self.histogram_head_loss.startswith("multinomial"):
            return self.multinomial_transform(x)
        elif self.histogram_head_loss.startswith("cont_softmax") or self.histogram_head_loss.startswith(
            "softmax"
        ):
            count = x.sum(dim=-1) + 2  # Add H and NTP token
            data = x / count.unsqueeze(-1)
        else:
            raise ValueError(f"Unknown histogram_head_loss: {self.histogram_head_loss}")
        return data

    def reverse_transform(self, x):
        if self.histogram_head_loss.startswith("multinomial"):
            return self.multinomial_reverse_transform(x)
        elif self.histogram_head_loss.startswith("cont_softmax") or self.histogram_head_loss.startswith(
            "softmax"
        ):
            x = torch.softmax(x, dim=-1)
            x = (x * self.max_count).round()
        else:
            raise ValueError(f"Unknown histogram_head_loss: {self.histogram_head_loss}")

        return x


class VAEEncoder(nn.Module):
    def __init__(
        self,
        input_dim=3,
        latent_dim=16,
        hidden_dims=[64, 32],
        dropout=0.0,
    ):
        super().__init__()
        # Create MLP encoder using torchvision
        self.encoder = MLP(
            in_channels=input_dim,
            hidden_channels=hidden_dims,
            norm_layer=nn.LayerNorm,
            activation_layer=nn.SiLU,
            dropout=dropout,
            inplace=None,  # Explicit None for clarity
        )
        self.fc_mu = nn.Linear(hidden_dims[-1], latent_dim)
        self.fc_var = nn.Linear(hidden_dims[-1], latent_dim)

    def forward(self, x):
        result = self.encoder(x)
        mu = self.fc_mu(result)
        log_var = self.fc_var(result)
        return mu, log_var


class VAEDecoder(nn.Module):
    def __init__(
        self,
        output_dim=3,
        latent_dim=16,
        hidden_dims=[32, 64],
        dropout=0.0,
    ):
        super().__init__()
        self.decoder = MLP(
            in_channels=latent_dim,
            hidden_channels=hidden_dims + [output_dim],
            norm_layer=nn.LayerNorm,
            activation_layer=nn.SiLU,
            dropout=dropout,
            inplace=None,  # Explicit None for clarity
        )

    def forward(self, z):
        return self.decoder(z)


class AutoencoderVQ(nn.Module):
    """
    Histogram‑specific VQ‑VAE:
        • Encoder  -> continuous latents z_e
        • VectorQuantizer2 -> discrete indices + quantised latents z_q + vq_loss
        • Decoder  -> reconstruction
    """

    def __init__(
        self,
        embed_dim,
        input_dim,
        output_dim,
        num_categories,
        vq_num_embeddings=512,
        vq_beta=0.25,
        encoder_hidden_dims=[64, 32],
        decoder_hidden_dims=[32, 64],
        histogram_head_loss="l2",
        commitment_scale=1.0,  # β_q in loss
    ):
        super().__init__()
        self.embed_dim = embed_dim

        # --- encoder & decoder (exactly the same code you have) ----------------
        self.encoder = VAEEncoder(input_dim=input_dim, latent_dim=embed_dim, hidden_dims=encoder_hidden_dims)
        self.decoder = VAEDecoder(
            output_dim=output_dim, latent_dim=embed_dim, hidden_dims=decoder_hidden_dims
        )

        # --- NEW: vector‑quantiser -------------------------------------------
        self.quantiser = VectorQuantizer2(
            n_e=vq_num_embeddings,  # size of code‑book
            e_dim=embed_dim,  # latent dimensionality
            beta=vq_beta,  # internal beta (bug‑compatible)
            sane_index_shape=False,  # we flatten histogram anyway
        )
        self.commitment_scale = commitment_scale
        self.histogram_head_loss = histogram_head_loss
        self.num_categories = num_categories

    # ---------------------------------------------------------------------
    # encode: *no* sampling, just produce continuous latents
    # ---------------------------------------------------------------------
    def encode(self, x):
        z_e, _ = self.encoder(x)  # we only need μ
        return z_e  # shape: (B, embed_dim)

    def decode(self, z_q):
        return self.decoder(z_q)

    def reshape_before_quantize(self, x):
        return x.reshape(-1, 1, 1, self.embed_dim)

    def reshape_after_quantize(self, x):
        return x.reshape(-1, self.embed_dim)

    def quantize(self, x):
        x = self.reshape_before_quantize(x)
        x, vq_loss, metadata = self.quantiser(x)
        x = self.reshape_after_quantize(x)
        return x, vq_loss, metadata

    # ---------------------------------------------------------------------
    # forward / training_step
    # ---------------------------------------------------------------------
    def forward(self, inputs, targets=None):
        if targets is None:
            targets = inputs

        # 1) continuous latents
        z_e = self.encode(inputs)  # (B, D)

        # 2) vector quantisation
        #    VectorQuantizer2 expects shape (B, D) or (B, C, H, W).
        z_q, vq_loss, (_, _, vq_codes) = self.quantize(z_e)  # z_q: same shape as z_e

        # 3) reconstruction
        dec = self.decode(z_q)

        # 4) reconstruction loss  (reuse all the branches you already wrote)
        rec_loss = self._reconstruction_loss(dec, targets)

        # 5) total loss
        loss = rec_loss + self.commitment_scale * vq_loss

        return {
            "vq_loss": loss,
            "vq_rec_loss": rec_loss,
            "vq_commit_loss": vq_loss * self.commitment_scale,
            "vq_reconstruction": dec,
            "vq_codes": vq_codes,
        }

    # copy your histogram loss block here — unchanged
    def _reconstruction_loss(self, dec, outputs):
        if self.histogram_head_loss == "softmax_l2":
            pred_histogram = dec
            target_histogram = outputs
            pred_histogram = torch.nn.functional.softmax(pred_histogram, dim=-1)
            rec_loss = torch.nn.functional.mse_loss(pred_histogram, target_histogram, reduction="mean")
        elif self.histogram_head_loss == "softmax_l1":
            pred_histogram = dec
            target_histogram = outputs
            pred_histogram = torch.nn.functional.softmax(pred_histogram, dim=-1)
            rec_loss += torch.nn.functional.l1_loss(pred_histogram, target_histogram, reduction="mean")
        elif self.histogram_head_loss == "cont_softmax_multinomial":
            pred_histogram = dec
            target_histogram = outputs
            log_probs = torch.nn.functional.log_softmax(pred_histogram, dim=-1)
            rec_loss = -(target_histogram * log_probs).sum(dim=-1).mean()
        elif self.histogram_head_loss == "cont_softmax_multinomial_balanced":
            pred_histogram = dec
            target_histogram = outputs
            rec_loss = self.balanced_ce_loss(pred_histogram, target_histogram)

        elif self.histogram_head_loss == "multinomial":
            dec = dec.reshape(*outputs.shape, -1)
            rec_loss = torch.nn.functional.cross_entropy(
                dec.transpose(1, 2), outputs.long(), reduction="mean"
            )
        else:
            raise ValueError(f"Invalid model.histogram_head_loss of: {self.histogram_head_loss}")

        return rec_loss


import torch
import torch.nn as nn
from torch import nn


class VQEicEmbedder(Module, nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg

        # Get NTP token index from metadata
        if self.cfg.token_insertion_strategy != TokenInsertionStrategy.TOKEN_COUNT_NO_DECREMENT:
            raise ValueError("VQEicEmbedder only supports TOKEN_COUNT_NO_DECREMENT token insertion strategy")
        self.histogram_normalizer: HistogramNormalizer = self.cfg.histogram_normalizer
        self.autoencoder_vq: AutoencoderVQ = self.cfg.autoencoder_vq(
            output_dim=self.histogram_normalizer.get_normalized_size()
        )
        assert isinstance(
            self.autoencoder_vq, AutoencoderVQ
        ), "autoencoder_vq must be an instance of AutoencoderVQ"
        assert isinstance(
            self.histogram_normalizer, HistogramNormalizer
        ), "histogram_normalizer must be an instance of HistogramNormalizer"

    def forward(self, batch):
        if self.cfg.pretrain:
            histograms = batch["histogram"].reshape(-1, batch["histogram"].shape[-1])
            full_histograms = histograms[histograms.sum(dim=1) == self.cfg.max_count]
            normalized_full_histograms = self.histogram_normalizer.transform(full_histograms)
            loss_dict = self.autoencoder_vq(normalized_full_histograms, normalized_full_histograms)
            batch[MODEL_BATCH_LOSS_KEY] = loss_dict["vq_loss"]
            batch[MODEL_LOSS_KEY] = loss_dict["vq_loss"]
            batch[BACKBONE_TOKENS_KEY] = self.histogram_normalizer.reverse_transform(
                loss_dict["vq_reconstruction"]
            ).float()
            batch[BACKBONE_EMBEDDINGS_KEY] = loss_dict["vq_codes"]
            batch["vq_true_hist"] = full_histograms
            batch.update(
                {
                    "MODEL//VQ//" + k: v
                    for k, v in loss_dict.items()
                    if k not in ["vq_reconstruction", "vq_codes"]
                }
            )
        else:
            with torch.no_grad():
                self.autoencoder_vq.eval()
                histograms = batch["histogram"].reshape(-1, batch["histogram"].shape[-1])
                normalized_histograms = self.histogram_normalizer.transform(histograms)
                histogram_embeds = self.autoencoder_vq.encode(normalized_histograms)
                _, _, (_, _, histogram_code) = self.autoencoder_vq.quantize(histogram_embeds)
                histogram_code += self.cfg.cluster_token_offset
                ntp_mask = batch["code"] == self.cfg.ntp_token
                batch["code"] = torch.where(
                    ntp_mask, histogram_code.reshape(*batch["code"].shape), batch["code"]
                )
            batch[INPUT_ENCODER_MASK_KEY] = batch["mask"]
            batch[INPUT_ENCODER_TOKENS_KEY] = batch["code"]
        return batch

    # def process_sample(self, codes, histograms):
    #     """
    #     Process a single sample (or independent tensors) with the same logic.

    #     If histogram_embedding is passed, prompt tuning is applied
    #     """
    #     histogram_codes = self.autoencoder_vq.encode(histograms)
    #     ntp_mask = (codes == self.ntp_token).unsqueeze(-1)
    #     codes[ntp_mask] = histogram_codes[ntp_mask]
    #     return codes
