"""Vision Mamba-style VSS blocks with BCHW <-> BHWC adapters for U-Net integration.

Provenance
----------
This is **not** a vendored copy of the official VMamba repository
(https://github.com/MzeroMiko/VMamba) and does **not** depend on
``mamba-ssm`` or CUDA ``selective_scan`` kernels.

It is a lightweight, pure-PyTorch reimplementation inspired by the VSS
(Vision State Space) block design from VMamba / Vision Mamba literature:
  - SS2D-style 4-direction cross-scan + merge (row / col / reversed)
  - Pre-norm residual + SS2D branch + MLP branch (VSSBlock)
  - ``VSSBlock2d`` permutes U-Net ``(B, C, H, W)`` <-> ``(B, H, W, C)``

The selective scan here is simplified (cumsum + gating) for portability
and to support arbitrary H/W (16x16, 8x8, 4x4) without fixed-resolution
assumptions. Swap in official VMamba SS2D if you need full Mamba fidelity.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


class SS2D(nn.Module):
    """Pure-PyTorch SS2D-style 2D selective mixing (works for any H/W >= 1)."""

    def __init__(self, dim, d_state=16, expand=2):
        super().__init__()
        self.dim = int(dim)
        self.d_state = int(d_state)
        self.d_inner = int(dim * expand)

        self.in_proj = nn.Linear(self.dim, self.d_inner * 2, bias=False)
        self.dw_conv = nn.Conv2d(
            self.d_inner,
            self.d_inner,
            kernel_size=3,
            padding=1,
            groups=self.d_inner,
        )
        self.delta_proj = nn.Linear(self.d_inner, self.d_inner)
        self.out_proj = nn.Linear(self.d_inner, self.dim)

        nn.init.zeros_(self.delta_proj.weight)
        nn.init.zeros_(self.delta_proj.bias)

    @staticmethod
    def _cross_scan(x):
        """Build 4 directional sequences from BHWC feature map."""
        x_row = rearrange(x, "b h w c -> b (h w) c")
        x_col = rearrange(x, "b h w c -> b (w h) c")
        x_row_rev = torch.flip(x_row, dims=[1])
        x_col_rev = torch.flip(x_col, dims=[1])
        return torch.stack([x_row, x_col, x_row_rev, x_col_rev], dim=1)

    @staticmethod
    def _cross_merge(scans, height, width):
        """Merge 4 directional sequences back to BHWC by averaging."""
        row = rearrange(scans[:, 0], "b (h w) c -> b h w c", h=height, w=width)
        col = rearrange(scans[:, 1], "b (w h) c -> b h w c", h=height, w=width)
        row_rev = torch.flip(rearrange(scans[:, 2], "b (h w) c -> b h w c", h=height, w=width), dims=[1, 2])
        col_rev = torch.flip(rearrange(scans[:, 3], "b (w h) c -> b h w c", h=height, w=width), dims=[1, 2])
        return 0.25 * (row + col + row_rev + col_rev)

    def _selective_scan(self, seq):
        """Lightweight selective scan along sequence length L."""
        delta = F.softplus(self.delta_proj(seq))
        gate = torch.sigmoid(seq)
        state = torch.cumsum(seq * delta, dim=2)
        return state * gate

    def forward(self, x):
        """x: [B, H, W, C]"""
        b, height, width, channels = x.shape
        if channels != self.dim:
            raise ValueError(f"SS2D expected dim={self.dim}, got {channels}")

        xz = self.in_proj(x)
        x_inner, z = xz.chunk(2, dim=-1)

        x_conv = self.dw_conv(x_inner.permute(0, 3, 1, 2).contiguous())
        x_conv = x_conv.permute(0, 2, 3, 1).contiguous()

        scans = self._cross_scan(x_conv)
        scans = self._selective_scan(scans)
        y = self._cross_merge(scans, height, width)
        y = y * F.silu(z)
        return self.out_proj(y)


class VSSBlock(nn.Module):
    """VSS Block operating on `(B, H, W, C)` tensors."""

    def __init__(self, dim, d_state=16, expand=2, mlp_ratio=4.0, dropout=0.0):
        super().__init__()
        self.dim = int(dim)
        self.norm1 = nn.LayerNorm(self.dim)
        self.ss2d = SS2D(dim=self.dim, d_state=d_state, expand=expand)
        self.norm2 = nn.LayerNorm(self.dim)
        hidden = max(self.dim, int(self.dim * mlp_ratio))
        self.mlp = nn.Sequential(
            nn.Linear(self.dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, self.dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        x = x + self.ss2d(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class VSSBlock2d(nn.Module):
    """Adapter wrapper: U-Net `(B, C, H, W)` <-> VSS `(B, H, W, C)`."""

    def __init__(self, dim, d_state=16, expand=2, mlp_ratio=4.0, dropout=0.0):
        super().__init__()
        self.dim = int(dim)
        self.block = VSSBlock(
            dim=self.dim,
            d_state=d_state,
            expand=expand,
            mlp_ratio=mlp_ratio,
            dropout=dropout,
        )

    def forward(self, x):
        if x.ndim != 4:
            raise ValueError(f"VSSBlock2d expects BCHW, got shape {tuple(x.shape)}")
        b, c, height, width = x.shape
        if c != self.dim:
            raise ValueError(f"VSSBlock2d expected dim={self.dim}, got {c}")

        x = x.permute(0, 2, 3, 1).contiguous()
        x = self.block(x)
        return x.permute(0, 3, 1, 2).contiguous()
