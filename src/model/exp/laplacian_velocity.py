"""Fully convolutional velocity fields for Laplacian-band OT-CFM.

D3 (32² topology): AdaGN ResBlocks + Vision Mamba ``VSSBlock2d`` + ``SelfAttention2d``.
D2 (64² mid residual): AdaGN ResBlocks only.
D1 (128² high-frequency residual): very shallow conv + AdaGN.

All nets accept arbitrary H×W (patch training). Conditioning matches the main
CFM split AdaGN path: timestep + scalar (DI / direction) gated with GRU ``g``.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.model.adm import (
    AdaGNResBlock,
    SelfAttention2d,
    sinusoidal_timestep_embedding,
)
from src.model.vss import VSSBlock2d


class SplitTimeCond(nn.Module):
    """``h = e_c + alpha * e_g`` with ``alpha ≡ 0`` at init (same as ADMUNetCFM split)."""

    def __init__(self, model_channels, cond_dim, latent_dim, physics_dim, use_split=True):
        super().__init__()
        self.model_channels = int(model_channels)
        self.cond_dim = int(cond_dim)
        self.use_split = bool(use_split)
        self.time_mlp = nn.Sequential(
            nn.Linear(self.model_channels, cond_dim),
            nn.SiLU(),
            nn.Linear(cond_dim, cond_dim),
        )
        if self.use_split:
            self.scalar_mlp = nn.Sequential(
                nn.Linear(latent_dim, cond_dim),
                nn.SiLU(),
                nn.Linear(cond_dim, cond_dim),
            )
            self.joint_mlp = nn.Sequential(
                nn.SiLU(),
                nn.Linear(cond_dim, cond_dim),
            )
            self.e_c_ln = nn.LayerNorm(cond_dim)
            self.physics_linear = nn.Linear(physics_dim, cond_dim)
            self.e_g_ln = nn.LayerNorm(cond_dim)
            self.alpha = nn.Parameter(torch.zeros(1))
        else:
            self.cond_mlp = nn.Sequential(
                nn.Linear(latent_dim, cond_dim),
                nn.SiLU(),
                nn.Linear(cond_dim, cond_dim),
            )
            self.joint_mlp = nn.Sequential(
                nn.SiLU(),
                nn.Linear(cond_dim, cond_dim),
            )

    def forward(self, t, c, gru_feat=None):
        t_emb = self.time_mlp(sinusoidal_timestep_embedding(t, self.model_channels))
        if self.use_split:
            if gru_feat is None:
                raise ValueError("gru_feat is required for split AdaGN conditioning")
            e_c = self.e_c_ln(self.joint_mlp(t_emb + self.scalar_mlp(c)))
            e_g = self.e_g_ln(self.physics_linear(gru_feat))
            return e_c + self.alpha * e_g
        return self.joint_mlp(t_emb + self.cond_mlp(c))


class DummyVelocityNet(nn.Module):
    """Placeholder ``Model(C_in, C_out)`` — 3×3 conv stack, ignores cond."""

    def __init__(self, in_channels, out_channels, hidden=32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, hidden, 3, padding=1),
            nn.SiLU(),
            nn.Conv2d(hidden, hidden, 3, padding=1),
            nn.SiLU(),
            nn.Conv2d(hidden, out_channels, 3, padding=1),
        )

    def forward(self, z_t, t=None, c=None, gru_feat=None, **kwargs):
        return self.net(z_t)


class _BandVelocityBase(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        hidden,
        cond_dim,
        latent_dim,
        physics_dim,
        use_split=True,
        dropout=0.0,
    ):
        super().__init__()
        self.in_channels = int(in_channels)
        self.out_channels = int(out_channels)
        self.hidden = int(hidden)
        self.cond = SplitTimeCond(hidden, cond_dim, latent_dim, physics_dim, use_split=use_split)
        self.stem = nn.Conv2d(in_channels, hidden, kernel_size=3, padding=1)
        self.out_norm = nn.GroupNorm(min(8, hidden), hidden)
        self.out_conv = nn.Conv2d(hidden, out_channels, kernel_size=3, padding=1)
        self.dropout = float(dropout)

    def _head(self, h):
        return self.out_conv(F.silu(self.out_norm(h)))


class D1VelocityNet(_BandVelocityBase):
    """128² high-frequency residual: two AdaGN ResBlocks, no attention."""

    def __init__(self, in_channels, out_channels, hidden=32, **kw):
        super().__init__(in_channels, out_channels, hidden, **kw)
        self.b1 = AdaGNResBlock(hidden, hidden, self.cond.cond_dim, dropout=self.dropout)
        self.b2 = AdaGNResBlock(hidden, hidden, self.cond.cond_dim, dropout=self.dropout)

    def forward(self, z_t, t, c, gru_feat=None, **kwargs):
        cond = self.cond(t, c, gru_feat=gru_feat)
        h = self.stem(z_t)
        h = self.b1(h, cond)
        h = self.b2(h, cond)
        return self._head(h)


class D2VelocityNet(_BandVelocityBase):
    """64² mid residual: AdaGN ResBlocks only (no VSS / attention)."""

    def __init__(self, in_channels, out_channels, hidden=64, num_blocks=4, **kw):
        super().__init__(in_channels, out_channels, hidden, **kw)
        self.blocks = nn.ModuleList(
            [
                AdaGNResBlock(hidden, hidden, self.cond.cond_dim, dropout=self.dropout)
                for _ in range(int(num_blocks))
            ]
        )

    def forward(self, z_t, t, c, gru_feat=None, **kwargs):
        cond = self.cond(t, c, gru_feat=gru_feat)
        h = self.stem(z_t)
        for blk in self.blocks:
            h = blk(h, cond)
        return self._head(h)


class D3VelocityNet(_BandVelocityBase):
    """32² low-frequency topology: AdaGN + VSS + self-attention."""

    def __init__(self, in_channels, out_channels, hidden=96, attn_heads=4, vss_d_state=16, **kw):
        super().__init__(in_channels, out_channels, hidden, **kw)
        if hidden % int(attn_heads) != 0:
            attn_heads = 1
        self.pre = nn.ModuleList(
            [
                AdaGNResBlock(hidden, hidden, self.cond.cond_dim, dropout=self.dropout)
                for _ in range(2)
            ]
        )
        self.vss = VSSBlock2d(hidden, d_state=int(vss_d_state), expand=2)
        self.attn = SelfAttention2d(hidden, num_heads=int(attn_heads))
        self.post = nn.ModuleList(
            [
                AdaGNResBlock(hidden, hidden, self.cond.cond_dim, dropout=self.dropout)
                for _ in range(2)
            ]
        )

    def forward(self, z_t, t, c, gru_feat=None, **kwargs):
        cond = self.cond(t, c, gru_feat=gru_feat)
        h = self.stem(z_t)
        for blk in self.pre:
            h = blk(h, cond)
        h = self.vss(h)
        h = self.attn(h)
        for blk in self.post:
            h = blk(h, cond)
        return self._head(h)


def build_band_velocity_net(
    band,
    in_channels,
    out_channels,
    dummy=False,
    cond_dim=128,
    latent_dim=2,
    physics_dim=1024,
    use_split=True,
    dropout=0.0,
):
    band = str(band).lower()
    kw = dict(
        cond_dim=cond_dim,
        latent_dim=latent_dim,
        physics_dim=physics_dim,
        use_split=use_split,
        dropout=dropout,
    )
    if dummy:
        return DummyVelocityNet(in_channels, out_channels)
    if band in ("d1", "x1", "1"):
        return D1VelocityNet(in_channels, out_channels, hidden=32, **kw)
    if band in ("d2", "x2", "2"):
        return D2VelocityNet(in_channels, out_channels, hidden=64, num_blocks=4, **kw)
    if band in ("d3", "x3", "3"):
        return D3VelocityNet(in_channels, out_channels, hidden=96, attn_heads=4, **kw)
    raise ValueError(f"Unknown laplacian band {band!r}")
