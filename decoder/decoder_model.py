# decoder_model.py
import torch
from torch import nn
import torch.nn.functional as F
from einops import rearrange

from module import FeedForward


class CrossAttention(nn.Module):
    """Cross-attention: query tokens attend to one latent token."""

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
        q:  (B, P, D)
        kv: (B, 1, D)
        """
        q = self.norm_q(q)
        kv = self.norm_kv(kv)

        q = self.to_q(q)
        k = self.to_k(kv)
        v = self.to_v(kv)

        q = rearrange(q, "b p (h d) -> b h p d", h=self.heads)
        k = rearrange(k, "b n (h d) -> b h n d", h=self.heads)
        v = rearrange(v, "b n (h d) -> b h n d", h=self.heads)

        dropout_p = self.dropout if self.training else 0.0

        out = F.scaled_dot_product_attention(
            q, k, v, dropout_p=dropout_p, is_causal=False
        )

        out = rearrange(out, "b h p d -> b p (h d)")
        return self.to_out(out)


class CrossAttentionBlock(nn.Module):
    """Cross-attention + MLP."""

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
    def __init__(
        self,
        latent_dim=192,
        hidden_dim=256,
        image_size=224,
        patch_size=16,
        channels=3,
        depth=4,
        heads=8,
        dim_head=64,
        mlp_dim=None,
        dropout=0.0,
        num_latent_tokens=16,   # 新增
    ):
        super().__init__()

        assert image_size % patch_size == 0

        self.image_size = image_size
        self.patch_size = patch_size
        self.channels = channels
        self.num_latent_tokens = num_latent_tokens

        self.num_patches_per_side = image_size // patch_size
        self.num_patches = self.num_patches_per_side ** 2

        # 改这里：一个 latent 映射成多个 memory tokens
        self.latent_proj = nn.Linear(
            latent_dim,
            num_latent_tokens * hidden_dim,
        )

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

    def forward(self, latent):
        """
        输入:
            latent: (B, T, D)

        输出:
            img: (B, T, C, H, W)
        """
        B, T, D = latent.shape

        # 先把时间维展开，方便 decoder 对每一帧 latent 独立重建
        latent = latent.reshape(B * T, D)

        # (B*T, D) -> (B*T, num_latent_tokens * hidden_dim)
        kv = self.latent_proj(latent)

        # (B*T, num_latent_tokens * hidden_dim)
        # -> (B*T, num_latent_tokens, hidden_dim)
        kv = kv.view(B * T, self.num_latent_tokens, -1)

        # query tokens:
        # (1, num_patches, hidden_dim)
        # -> (B*T, num_patches, hidden_dim)
        q = self.query_tokens.expand(B * T, -1, -1)

        for layer in self.layers:
            q = layer(q, kv)

        patches = self.to_patch_pixels(q)

        h = w = self.num_patches_per_side
        p = self.patch_size
        c = self.channels

        # (B*T, num_patches, patch_pixels)
        # -> (B*T, h, w, p, p, c)
        patches = patches.view(B * T, h, w, p, p, c)

        # -> (B*T, c, h*p, w*p)
        img = patches.permute(0, 5, 1, 3, 2, 4).contiguous()
        img = img.view(B * T, c, h * p, w * p)

        # 恢复时间维:
        # (B*T, C, H, W) -> (B, T, C, H, W)
        img = img.view(B, T, c, h * p, w * p)

        return img


class LeWMEncodeWrapper(nn.Module):
    """
    Wrap full LeWM world_model as an encoder.

    It uses:
        world_model.encode(info)["emb"]

    info can contain:
        pixels
        action

    This keeps the input format consistent with LeWM training.
    """

    def __init__(self, world_model):
        super().__init__()
        self.world_model = world_model

        self.world_model.eval()
        for p in self.world_model.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def forward(self, batch):
        """
        batch:
            dict containing at least batch["pixels"]

        return:
            latent:
                (B, T, D) if pixels is (B, T, C, H, W)
                (B, D) if pixels is (B, C, H, W)
        """
        if not isinstance(batch, dict):
            raise TypeError(f"Expected batch to be dict, got {type(batch)}")


        info = self.world_model.encode(batch)

        return info


class DecoderTrainingModel(nn.Module):
    """
    Frozen LeWM encoder + trainable visualization decoder.
    """

    def __init__(self, lewm_encoder, decoder):
        super().__init__()
        self.lewm_encoder = lewm_encoder
        self.decoder = decoder

        self.lewm_encoder.eval()
        self.lewm_encoder.requires_grad_(False)

    def forward(self, batch):
        """
        batch:
            batch["pixels"]: (B, T, C, H, W) or (B, C, H, W)
            batch["action"]: optional
        """
        if not isinstance(batch, dict):
            raise TypeError(f"Expected batch to be dict, got {type(batch)}")


        with torch.no_grad():
            info = self.lewm_encoder(batch)

        target = info["pixels"] # B T 3 224 224
        latent  = info["emb"] #B T 192

        recon = self.decoder(latent)

        return recon, target

    def compute_loss(self, batch):
        recon, target = self.forward(batch)

        loss_l1 = F.l1_loss(recon, target)
        loss_l2 = F.mse_loss(recon, target)
        loss = loss_l1 + 0.1 * loss_l2

        return {
            "loss": loss,
            "loss_l1": loss_l1.detach(),
            "loss_l2": loss_l2.detach(),
            "recon": recon.detach(),
            "target": target.detach(),
        }