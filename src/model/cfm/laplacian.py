"""Image Laplacian pyramid (bilinear) + aligned patch crops for Laplacian-CFM.

Pyramid (matches the GT visualization script)::

    x^(3) = down(down(x))                         # 32×32 low-frequency topology
    x^(2) = down(x) - up(x^(3))                   # 64×64 mid residual
    x^(1) = x - up(down(x))                       # 128×128 high-frequency residual

``down`` / ``up`` are bilinear ×2.  Reconstruction is exact up to interpolation::

    x ≈ x^(1) + up( x^(2) + up(x^(3)) )
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def pyramid_down(x):
    return F.interpolate(x, scale_factor=0.5, mode="bilinear", align_corners=False)


def pyramid_up(x, size):
    return F.interpolate(x, size=size, mode="bilinear", align_corners=False)


def decompose_laplacian(x):
    """Split ``x`` [B,C,H,W] into ``(x1, x2, x3)`` at H, H/2, H/4."""
    if x.shape[-1] < 4 or x.shape[-2] < 4:
        raise ValueError(f"Need spatial size >= 4 for two downsamples, got {tuple(x.shape[-2:])}")
    if x.shape[-1] % 4 != 0 or x.shape[-2] % 4 != 0:
        raise ValueError(f"Laplacian pyramid expects H,W divisible by 4, got {tuple(x.shape[-2:])}")
    x_d1 = pyramid_down(x)
    x3 = pyramid_down(x_d1)
    x2 = x_d1 - pyramid_up(x3, size=x_d1.shape[-2:])
    x1 = x - pyramid_up(x_d1, size=x.shape[-2:])
    return x1, x2, x3


def reconstruct_laplacian(x1, x2, x3):
    """Invert :func:`decompose_laplacian` (interpolation-consistent)."""
    mid = x2 + pyramid_up(x3, size=x2.shape[-2:])
    return x1 + pyramid_up(mid, size=x1.shape[-2:])


def select_band(x1, x2, x3, band):
    band = str(band).lower()
    if band in ("d1", "x1", "1"):
        return x1
    if band in ("d2", "x2", "2"):
        return x2
    if band in ("d3", "x3", "3"):
        return x3
    raise ValueError(f"band must be d1|d2|d3, got {band!r}")


def min_snr_gamma_weight(t, gamma=5.0):
    """Velocity Min-SNR that **keeps** weight as t→1 (high-frequency stage).

    ``snr = (t/(1-t))^2``, ``w = min(snr, γ) / (min(snr, γ) + 1)``.
    Unlike ``min(snr,γ)/(snr+1)`` this saturates at ``γ/(γ+1)`` instead of 0.
    """
    t_c = t.clamp(min=1e-5, max=1.0 - 1e-5)
    snr = (t_c / (1.0 - t_c)) ** 2
    snr_c = torch.clamp(snr, max=float(gamma))
    return snr_c / (snr_c + 1.0)


def sample_patch_size(full_size, probs, generator=None):
    """``probs`` = (p_full, p_half, p_quarter); quarter is skipped if its mass is 0."""
    p_full, p_half, p_q = [float(p) for p in probs]
    total = p_full + p_half + p_q
    if total <= 0:
        return int(full_size)
    u = float(torch.rand((), generator=generator).item()) * total
    if u < p_full:
        return int(full_size)
    if u < p_full + p_half or p_q <= 0:
        return max(1, int(full_size // 2))
    return max(1, int(full_size // 4))


def crop_xt_vt(xt, vt, patch_size, valid_mask=None):
    """Crop ``x_t`` and ``v_t`` with the same per-sample (top, left).

    Returns ``xt, vt, coords, valid_mask`` where ``coords`` is [B,2,P,P] in [-1,1]
    over the *full* band extent (so a 32-crop of a 128 band still knows location).
    """
    if xt.shape != vt.shape:
        raise ValueError(f"xt {tuple(xt.shape)} must match vt {tuple(vt.shape)}")
    b, _, height, width = xt.shape
    if height != width:
        raise ValueError(f"Expected square band maps, got {height}x{width}")
    patch_size = max(1, min(int(patch_size), height))
    ys = torch.linspace(-1.0, 1.0, steps=height, device=xt.device, dtype=xt.dtype)
    xs = torch.linspace(-1.0, 1.0, steps=width, device=xt.device, dtype=xt.dtype)
    grid_y, grid_x = torch.meshgrid(ys, xs, indexing="ij")
    if patch_size == height:
        coords = torch.stack([grid_x, grid_y], dim=0).unsqueeze(0).expand(b, -1, -1, -1)
        return xt, vt, coords.contiguous(), valid_mask

    max_off = height - patch_size
    tops = torch.randint(0, max_off + 1, (b,), device=xt.device)
    lefts = torch.randint(0, max_off + 1, (b,), device=xt.device)
    xt_c, vt_c, coord_c, mask_c = [], [], [], []
    for i in range(b):
        top, left = int(tops[i]), int(lefts[i])
        slh = slice(top, top + patch_size)
        slw = slice(left, left + patch_size)
        xt_c.append(xt[i : i + 1, :, slh, slw])
        vt_c.append(vt[i : i + 1, :, slh, slw])
        coord_c.append(
            torch.stack(
                [grid_x[slh, slw], grid_y[slh, slw]],
                dim=0,
            )
        )
        if valid_mask is not None:
            if valid_mask.ndim == 3:
                mask_c.append(valid_mask[i : i + 1, slh, slw])
            else:
                mask_c.append(valid_mask[i : i + 1, :, slh, slw])
    mask_out = torch.cat(mask_c, dim=0) if valid_mask is not None else None
    return torch.cat(xt_c, dim=0), torch.cat(vt_c, dim=0), torch.stack(coord_c, dim=0), mask_out


def apply_band_patch(xt, vt, band, training=True, valid_mask=None):
    """Dynamic patch schedule for D1/D2/D3. No-op for D3 and at eval."""
    band = str(band).lower()
    full = int(xt.shape[-1])
    probs = BAND_PATCH_PROBS[band if band in BAND_PATCH_PROBS else "d3"]
    if (not training) or band in ("d3", "x3", "3"):
        patch = full
    else:
        patch = sample_patch_size(full, probs)
    return crop_xt_vt(xt, vt, patch, valid_mask=valid_mask)


def ot_cfm_path(x1, t=None):
    """Linear OT-CFM: ``x_t = (1-t) x_0 + t x_1``, ``v_t = x_1 - x_0``."""
    x0 = torch.randn_like(x1)
    b = x1.shape[0]
    if t is None:
        t = torch.rand(b, device=x1.device, dtype=x1.dtype).clamp(min=1e-5, max=1.0 - 1e-5)
    else:
        t = t.reshape(b).to(device=x1.device, dtype=x1.dtype)
    t_ = t.view(b, 1, 1, 1)
    xt = (1.0 - t_) * x0 + t_ * x1
    vt = x1 - x0
    return xt, vt, t


def weighted_velocity_mse(pred, vt, t, gamma=5.0, valid_mask=None):
    """Per-sample MSE × Min-SNR-γ, then batch mean."""
    err = F.mse_loss(pred, vt, reduction="none")
    if valid_mask is None:
        per = err.mean(dim=(1, 2, 3))
    else:
        mask = valid_mask.to(device=err.device, dtype=err.dtype)
        if mask.ndim == 3:
            mask = mask.unsqueeze(1)
        if mask.shape[-2:] != err.shape[-2:]:
            mask = F.interpolate(mask, size=err.shape[-2:], mode="nearest")
        denom = mask.sum(dim=(1, 2, 3)).clamp_min(1e-8) * err.shape[1]
        per = (err * mask).sum(dim=(1, 2, 3)) / denom
    w = min_snr_gamma_weight(t, gamma=gamma).to(dtype=per.dtype)
    return (per * w).mean(), per.detach(), w.detach()


BAND_PATCH_PROBS = {
    "d1": (0.50, 0.30, 0.20),  # 128 / 64 / 32
    "d2": (0.70, 0.30, 0.00),  # 64 / 32  (no quarter)
    "d3": (1.00, 0.00, 0.00),  # 32 only
}
