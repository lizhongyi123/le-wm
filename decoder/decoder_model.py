# decoder_model.py
import torch
from torch import nn
import torch.nn.functional as F
from einops import rearrange

from module import FeedForward


class CrossAttention(nn.Module):
    """Cross-attention: query tokens attend to CLS latent."""

    def __init__(self, dim, heads=8, dim_head=64, dropout=0.0):
        super().__init__()
        inner_dim = heads * dim_head
        self.heads = heads
        self.dropout = dropout

        self.norm_q = nn.LayerNorm(dim)
        self.norm_kv = nn.LayerNorm(dim)

        self.to_q = nn.Linear(dim, inner_dim, bias=False)
        self.to_k = nn.Linear(dim, inner_dim, bias=False)
        self.to_v = nn.Linear(dim, inner_dim, bias=False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout),
        )

    def forward(self, q, kv):
        """
        q:  (B, P, D)  learnable patch query tokens
        kv: (B, 1, D)  projected CLS token
        """
        q = self.norm_q(q)
        kv = self.norm_kv(kv)

        q = self.to_q(q)
        k = self.to_k(kv)
        v = self.to_v(kv)

        q = rearrange(q, "b p (h d) -> b h p d", h=self.heads)
        k = rearrange(k, "b n (h d) -> b h n d", h=self.heads)
        v = rearrange(v, "b n (h d) -> b h n d", h=self.heads)

        drop = self.dropout if self.training else 0.0

        out = F.scaled_dot_product_attention(
            q, k, v, dropout_p=drop, is_causal=False
        )

        out = rearrange(out, "b h p d -> b p (h d)")
        return self.to_out(out)


class CrossAttentionBlock(nn.Module):
    """Decoder block: cross-attention + residual MLP."""

    def __init__(self, dim, heads=8, dim_head=64, mlp_dim=None, dropout=0.0):
        super().__init__()
        mlp_dim = mlp_dim or 4 * dim

        self.cross_attn = CrossAttention(
            dim=dim,
            heads=heads,
            dim_head=dim_head,
            dropout=dropout,
        )

        self.mlp = FeedForward(
            dim=dim,
            hidden_dim=mlp_dim,
            dropout=dropout,
        )

    def forward(self, q, kv):
        q = q + self.cross_attn(q, kv)
        q = q + self.mlp(q)
        return q


class VisualizationDecoder(nn.Module):
    """
    Decode CLS token embedding into an RGB image.

    Input:
        cls_emb: (B, cls_dim)

    Output:
        img: (B, 3, image_size, image_size)
    """

    def __init__(
        self,
        cls_dim=192,
        hidden_dim=256,
        image_size=224,
        patch_size=16,
        channels=3,
        depth=4,
        heads=8,
        dim_head=64,
        mlp_dim=None,
        dropout=0.0,
    ):
        super().__init__()

        assert image_size % patch_size == 0

        self.image_size = image_size
        self.patch_size = patch_size
        self.channels = channels

        self.num_patches_per_side = image_size // patch_size
        self.num_patches = self.num_patches_per_side ** 2

        self.cls_proj = nn.Linear(cls_dim, hidden_dim)

        self.query_tokens = nn.Parameter(
            torch.randn(1, self.num_patches, hidden_dim) * 0.02
        )

        self.layers = nn.ModuleList(
            [
                CrossAttentionBlock(
                    dim=hidden_dim,
                    heads=heads,
                    dim_head=dim_head,
                    mlp_dim=mlp_dim or 4 * hidden_dim,
                    dropout=dropout,
                )
                for _ in range(depth)
            ]
        )

        self.to_patch_pixels = nn.Linear(
            hidden_dim,
            patch_size * patch_size * channels,
        )

    def forward(self, cls_emb):
        """
        cls_emb: (B, 192)
        return:
            img: (B, 3, 224, 224)
        """
        B = cls_emb.size(0)

        kv = self.cls_proj(cls_emb).unsqueeze(1)  # (B, 1, hidden_dim)
        q = self.query_tokens.expand(B, -1, -1)  # (B, P, hidden_dim)

        for layer in self.layers:
            q = layer(q, kv)

        patches = self.to_patch_pixels(q)
        # (B, P, patch_size*patch_size*channels)

        h = w = self.num_patches_per_side
        p = self.patch_size
        c = self.channels

        patches = patches.view(B, h, w, p, p, c)

        img = patches.permute(0, 5, 1, 3, 2, 4).contiguous()
        img = img.view(B, c, h * p, w * p)

        return img


class DecoderTrainingModel(nn.Module):
    """
    Frozen encoder + trainable visualization decoder.
    """

    def __init__(self, encoder, decoder, freeze_encoder=True):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder

        if freeze_encoder:
            self.encoder.eval()
            self.encoder.requires_grad_(False)

    def extract_cls(self, pixels):
        """
        pixels: (B, 3, 224, 224)
        return:
            cls_emb: (B, 192)
        """
        return extract_cls_from_encoder_output(self.encoder(pixels))

    def forward(self, pixels):
        with torch.no_grad():
            cls_emb = self.extract_cls(pixels)

        recon = self.decoder(cls_emb)
        return recon

    def compute_loss(self, pixels):
        recon = self.forward(pixels)

        loss_l1 = F.l1_loss(recon, pixels)
        loss_l2 = F.mse_loss(recon, pixels)
        loss = loss_l1 + 0.1 * loss_l2

        return {
            "loss": loss,
            "loss_l1": loss_l1.detach(),
            "loss_l2": loss_l2.detach(),
            "recon": recon,
        }


def extract_cls_from_encoder_output(out):
    if hasattr(out, "last_hidden_state"):
        return out.last_hidden_state[:, 0]

    if isinstance(out, dict) and "last_hidden_state" in out:
        return out["last_hidden_state"][:, 0]

    if torch.is_tensor(out):
        if out.ndim == 3:
            return out[:, 0]
        if out.ndim == 2:
            return out
        raise ValueError(f"Unsupported encoder tensor output shape: {out.shape}")

    raise ValueError(f"Unsupported encoder output type: {type(out)}")
