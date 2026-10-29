"""C2 stage 1: conditional flow matching of a 32x32 truncated spall SDF (docs/redesign/phase1_c2.md).

SDF convention (matches scripts/export/export_anylabel_layout_sdf.py, in 32-grid cells):
positive inside spall, negative outside, clipped to +-TRUNC cells and divided by TRUNC -> [-1, 1].
A frame without spall is all -1 (also the "previous layout" of the first frame).
Rollout: new = max(previous projected layout, generated) (monotone projection).
"""
from __future__ import annotations

import math

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

S = 32
TRUNC = 2.0


def mask_to_sdf(mask: np.ndarray, trunc: float = TRUNC) -> np.ndarray:
    m = (mask > 0).astype(np.uint8)
    if m.max() == 0:
        return -np.ones(m.shape, np.float32)
    if m.min() == 1:
        return np.ones(m.shape, np.float32)
    inside = cv2.distanceTransform(m, cv2.DIST_L2, 5)
    outside = cv2.distanceTransform(1 - m, cv2.DIST_L2, 5)
    sdf = np.where(m > 0, inside - 0.5, -(outside - 0.5))
    return (np.clip(sdf, -trunc, trunc) / trunc).astype(np.float32)


def sdf_to_coverage(sdf):
    """Soft coverage in [0,1] with the 0.5 level set on the SDF zero crossing."""
    return np.clip(0.5 + 0.5 * np.asarray(sdf), 0.0, 1.0).astype(np.float32)


def timestep_embedding(t, dim):
    half = dim // 2
    f = torch.exp(-math.log(10000) * torch.arange(half, device=t.device) / half)
    a = t[:, None].float() * 1000.0 * f[None]
    return torch.cat([a.sin(), a.cos()], -1)


class Block(nn.Module):
    def __init__(self, cin, cout, edim):
        super().__init__()
        self.n1 = nn.GroupNorm(8, cin)
        self.c1 = nn.Conv2d(cin, cout, 3, padding=1)
        self.n2 = nn.GroupNorm(8, cout)
        self.c2 = nn.Conv2d(cout, cout, 3, padding=1)
        self.e = nn.Linear(edim, 2 * cout)
        self.skip = nn.Conv2d(cin, cout, 1) if cin != cout else nn.Identity()
        nn.init.zeros_(self.c2.weight)
        nn.init.zeros_(self.c2.bias)

    def forward(self, x, emb):
        h = self.c1(F.silu(self.n1(x)))
        sc, sh = self.e(emb)[:, :, None, None].chunk(2, 1)
        h = self.n2(h) * (1 + sc) + sh
        h = self.c2(F.silu(h))
        return h + self.skip(x)


class LayoutUNet(nn.Module):
    """Input channels: x_t (1) + prev layout (1, zero-free: -1 = empty) + coord (2). Vector cond -> FiLM."""

    def __init__(self, cond_dim, use_prev=True, ch=64, edim=128):
        super().__init__()
        self.use_prev = use_prev
        self.edim = edim
        cin = 1 + (1 if use_prev else 0) + 2
        self.temb = nn.Sequential(nn.Linear(edim, edim), nn.SiLU(), nn.Linear(edim, edim))
        self.cemb = nn.Sequential(nn.Linear(cond_dim, edim), nn.SiLU(), nn.Linear(edim, edim))
        self.inc = nn.Conv2d(cin, ch, 3, padding=1)
        self.d1 = Block(ch, ch, edim)
        self.d2 = Block(ch, 2 * ch, edim)   # 16x16
        self.d3 = Block(2 * ch, 2 * ch, edim)  # 8x8
        self.mid = Block(2 * ch, 2 * ch, edim)
        self.u3 = Block(4 * ch, 2 * ch, edim)
        self.u2 = Block(4 * ch, ch, edim)
        self.u1 = Block(2 * ch, ch, edim)
        self.out = nn.Conv2d(ch, 1, 3, padding=1)
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)
        yy, xx = torch.meshgrid(torch.linspace(-1, 1, S), torch.linspace(-1, 1, S), indexing="ij")
        self.register_buffer("coord", torch.stack([xx, yy])[None])

    def forward(self, x_t, t, cond, prev=None):
        b = x_t.shape[0]
        emb = self.temb(timestep_embedding(t, self.edim)) + self.cemb(cond)
        xs = [x_t]
        if self.use_prev:
            xs.append(prev)
        xs.append(self.coord.expand(b, -1, -1, -1))
        h0 = self.inc(torch.cat(xs, 1))
        h1 = self.d1(h0, emb)
        h2 = self.d2(F.avg_pool2d(h1, 2), emb)
        h3 = self.d3(F.avg_pool2d(h2, 2), emb)
        m = self.mid(h3, emb)
        u = self.u3(torch.cat([m, h3], 1), emb)
        u = self.u2(torch.cat([F.interpolate(u, scale_factor=2), h2], 1), emb)
        u = self.u1(torch.cat([F.interpolate(u, scale_factor=2), h1], 1), emb)
        return self.out(F.silu(u))


def fm_loss(model, x1, cond, prev):
    """Rectified flow: x_t = (1-t) x0 + t x1, target v = x1 - x0."""
    b = x1.shape[0]
    x0 = torch.randn_like(x1)
    t = torch.rand(b, device=x1.device)
    xt = (1 - t)[:, None, None, None] * x0 + t[:, None, None, None] * x1
    v = model(xt, t, cond, prev)
    return F.mse_loss(v, x1 - x0)


@torch.no_grad()
def sample(model, cond, prev, steps=24, generator=None):
    b = cond.shape[0]
    x = torch.randn(b, 1, S, S, device=cond.device, generator=generator)
    ts = torch.linspace(0, 1, steps + 1, device=cond.device)
    for i in range(steps):
        t = ts[i].expand(b)
        dt = ts[i + 1] - ts[i]
        v = model(x, t, cond, prev)
        x = x + dt * v
    return x.clamp(-1, 1)
