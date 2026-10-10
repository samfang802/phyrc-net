"""g-only spall-area head and low-band velocity correction.

The head sees the physics vector, a [-1, 1] coordinate grid, and the fixed
interference mask. It does not see the image being generated. The target is
accumulated dark extent, so static fixtures cannot be cancelled by a frame
difference; cells that are mostly interference are removed from the target
and from the guidance update. Sampling pushes the 16x16 average of the clean
estimate darker where the predicted probability is high, and writes that
shift back into the velocity as a piecewise-constant field.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class SpallAreaHead(nn.Module):
    """16x16 logits of accumulated dark extent from g and column coordinates."""

    def __init__(self, physics_dim: int = 1024, grid: int = 16):
        super().__init__()
        self.grid = int(grid)
        self.from_g = nn.Sequential(
            nn.Linear(int(physics_dim), 256),
            nn.SiLU(),
            nn.Linear(256, self.grid * self.grid),
        )
        self.coord_conv = nn.Sequential(
            nn.Conv2d(2, 16, kernel_size=3, padding=1),
            nn.SiLU(),
            nn.Conv2d(16, 1, kernel_size=1),
        )
        self.mask_conv = nn.Conv2d(1, 1, kernel_size=1)

    def forward(
        self,
        g: torch.Tensor,
        coords: torch.Tensor,
        interference: torch.Tensor | None = None,
    ) -> torch.Tensor:
        spatial = self.from_g(g).view(g.shape[0], 1, self.grid, self.grid)
        if interference is None:
            interference = spatial.new_zeros(spatial.shape)
        else:
            interference = interference.to(device=spatial.device, dtype=spatial.dtype)
            if interference.ndim == 3:
                interference = interference.unsqueeze(1)
            if interference.shape[-2:] != (self.grid, self.grid):
                interference = F.adaptive_avg_pool2d(interference, (self.grid, self.grid))
        return spatial + self.coord_conv(coords) + self.mask_conv(interference)


def full_frame_coords(batch: int, grid: int, device, dtype) -> torch.Tensor:
    axis = torch.linspace(-1.0, 1.0, int(grid), device=device, dtype=dtype)
    gy, gx = torch.meshgrid(axis, axis, indexing="ij")
    base = torch.stack((gx, gy), dim=0).unsqueeze(0)
    return base.expand(int(batch), -1, -1, -1).contiguous()


def pool_interference(mask: torch.Tensor, grid: int) -> torch.Tensor:
    """Average a full-resolution interference map (1 = fixture) onto the spall grid."""
    if mask.ndim == 3:
        mask = mask.unsqueeze(1)
    return F.adaptive_avg_pool2d(mask.float(), (int(grid), int(grid)))


def suppress_interference(target: torch.Tensor, interference: torch.Tensor, level: float = 0.5) -> torch.Tensor:
    """Drop spall-target cells that are mostly a fixed fixture."""
    blocked = pool_interference(interference, target.shape[-1]) >= float(level)
    return target * (~blocked).to(dtype=target.dtype)


def lowfreq_velocity_delta(
    head: nn.Module,
    z: torch.Tensor,
    v: torch.Tensor,
    t_scalar: float,
    gru: torch.Tensor,
    scale: float,
    grid: int = 16,
    interference: torch.Tensor | None = None,
) -> torch.Tensor:
    """Mean-|Δx| = ``scale`` shift of the clean estimate, folded into v.

    The update is constant on each pooled cell, so it only moves the low band.
    The weight (m - mean(m)) has zero mean, so the image mean brightness stays put.
    """
    remain = max(1.0 - float(t_scalar), 1e-3)
    x1 = (z.detach() + remain * v.detach()).to(dtype=torch.float32)
    luma = x1 if x1.shape[1] == 1 else x1.mean(dim=1, keepdim=True)
    ll = F.adaptive_avg_pool2d(luma, (int(grid), int(grid)))
    ll = ll.detach().requires_grad_(True)
    coords = full_frame_coords(gru.shape[0], grid, ll.device, ll.dtype)
    with torch.enable_grad():
        logits = head(gru.detach().to(dtype=ll.dtype), coords, interference)
        prob = torch.sigmoid(logits)
        if interference is None:
            keep = torch.ones_like(prob)
        else:
            blocked = pool_interference(interference, int(grid)) >= 0.5
            keep = (~blocked).to(dtype=prob.dtype)
        denom = keep.sum(dim=(-2, -1), keepdim=True).clamp(min=1.0)
        centered = prob - (prob * keep).sum(dim=(-2, -1), keepdim=True) / denom
        weight = (centered * keep).detach()
        energy = (weight * ll).mean()
        grad = torch.autograd.grad(energy, ll)[0]
    direction = -grad.detach()
    denom = direction.abs().mean().clamp(min=1e-8)
    delta_ll = float(scale) * direction / denom
    delta_x = F.interpolate(delta_ll, size=x1.shape[-2:], mode="nearest")
    if x1.shape[1] != delta_x.shape[1]:
        delta_x = delta_x.expand(-1, x1.shape[1], -1, -1)
    return (delta_x / remain).to(dtype=v.dtype), prob.detach(), energy.detach()
