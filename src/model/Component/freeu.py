"""FreeU (Si et al.): training-free decoder reweight of backbone vs skip features.

Backbone ``x_l`` carries denoising / structure; skip ``h_l`` injects high frequency
but can overpower the backbone at inference.  FreeU (1) spatially scales the first
half of backbone channels by a map derived from channel-mean magnitude, and
(2) attenuates skip low frequencies in the Fourier domain, then concatenates.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def backbone_structure_scale(x, b):
    """Structure-related scaling on the first half of backbone channels.

    ``alpha = (b-1) * (x_bar - min) / (max - min) + 1``, broadcast over H×W.
    Second-half channels are left unchanged.
    """
    b = float(b)
    if abs(b - 1.0) < 1e-8:
        return x
    x_bar = x.mean(dim=1, keepdim=True)
    x_min = x_bar.amin(dim=(-2, -1), keepdim=True)
    x_max = x_bar.amax(dim=(-2, -1), keepdim=True)
    denom = (x_max - x_min).clamp_min(1e-6)
    alpha = (b - 1.0) * (x_bar - x_min) / denom + 1.0
    half = x.shape[1] // 2
    if half < 1:
        return x * alpha
    x_out = x.clone()
    x_out[:, :half] = x[:, :half] * alpha
    return x_out


def skip_spectral_modulation(h, s, r_thresh=0.25):
    """Attenuate skip low frequencies with a circular Fourier mask.

    After ``fftshift``, bins with radius ``r = hypot(dy/H, dx/W) < r_thresh``
    are multiplied by ``s``; the rest stay at 1.  Then IFFT back to spatial.
    """
    s = float(s)
    r_thresh = float(r_thresh)
    if abs(s - 1.0) < 1e-8:
        return h
    orig_dtype = h.dtype
    h_f = h.float()
    spec = torch.fft.fftn(h_f, dim=(-2, -1))
    spec = torch.fft.fftshift(spec, dim=(-2, -1))
    _, _, height, width = spec.shape
    device = spec.device
    yy = torch.linspace(-0.5, 0.5, height, device=device, dtype=torch.float32)
    xx = torch.linspace(-0.5, 0.5, width, device=device, dtype=torch.float32)
    gy, gx = torch.meshgrid(yy, xx, indexing="ij")
    radius = torch.sqrt(gy * gy + gx * gx)
    beta = torch.ones((1, 1, height, width), device=device, dtype=torch.float32)
    beta = torch.where(radius < r_thresh, torch.full_like(beta, s), beta)
    spec = spec * beta
    spec = torch.fft.ifftshift(spec, dim=(-2, -1))
    rec = torch.fft.ifftn(spec, dim=(-2, -1)).real
    return rec.to(dtype=orig_dtype)


def freeu_fuse(backbone, skip, b, s, r_thresh=0.25):
    """Apply FreeU then concat ``[x', h']`` along channels (U-Net skip merge)."""
    x_p = backbone_structure_scale(backbone, b)
    h_p = skip_spectral_modulation(skip, s, r_thresh=r_thresh)
    if x_p.shape[-2:] != h_p.shape[-2:]:
        h_p = F.interpolate(h_p, size=x_p.shape[-2:], mode="nearest")
    return torch.cat([x_p, h_p], dim=1)


def freeu_params_for_stage(stage_idx, b1, b2, s1, s2):
    """Map decoder stage index (0 = deepest) to (b_l, s_l). Stages ≥2 → identity."""
    if int(stage_idx) == 0:
        return float(b1), float(s1)
    if int(stage_idx) == 1:
        return float(b2), float(s2)
    return 1.0, 1.0
