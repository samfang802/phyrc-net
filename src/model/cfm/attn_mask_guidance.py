"""Masked self-attention guidance (SAG-style, no unconditional branch).

Bottleneck self-attn ``A`` → spatial saliency → punch the top-k region on ``x_t``
→ second conditional forward → ``v ← v_clean + γ (v_clean − v_mask)``.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def saliency_from_self_attn(
    attn: torch.Tensor,
    hw,
    reduce: str = "mean",
    axis: str = "key",
) -> torch.Tensor:
    """``attn`` [B, heads, HW, HW] → saliency [B, 1, Hb, Wb].

    ``axis=key``: mean/max over queries (how much each location is attended *to*).
    ``axis=query``: mean/max over keys (how peaked each query's attention is).
    """
    if attn.ndim != 4:
        raise ValueError(f"Expected attn [B, heads, HW, HW], got {tuple(attn.shape)}")
    hb, wb = int(hw[0]), int(hw[1])
    a = attn.mean(dim=1)
    if axis == "key":
        s = a.max(dim=-2).values if reduce == "max" else a.mean(dim=-2)
    elif axis == "query":
        s = a.max(dim=-1).values if reduce == "max" else a.mean(dim=-1)
    else:
        raise ValueError(f"axis must be key|query, got {axis!r}")
    if reduce not in ("mean", "max"):
        raise ValueError(f"reduce must be mean|max, got {reduce!r}")
    n = hb * wb
    if int(s.shape[-1]) != n:
        raise ValueError(f"attn tokens {s.shape[-1]} != Hb*Wb={n} (hw={hw})")
    return s.view(attn.shape[0], 1, hb, wb)


def topk_mask_from_saliency(
    saliency: torch.Tensor,
    spatial_hw,
    mask_frac: float = 0.3,
) -> torch.Tensor:
    """Bilinear-upsample saliency and mark the top ``mask_frac`` pixels. [B,1,H,W] bool."""
    if not (0.0 < float(mask_frac) < 1.0):
        raise ValueError(f"mask_frac must be in (0,1), got {mask_frac}")
    h, w = int(spatial_hw[0]), int(spatial_hw[1])
    s = F.interpolate(saliency.float(), size=(h, w), mode="bilinear", align_corners=False)
    flat = s.flatten(1)
    n = flat.shape[1]
    k = max(1, int(round(float(mask_frac) * n)))
    thresh = torch.kthvalue(flat, n - k + 1, dim=1).values.unsqueeze(1)
    return (flat >= thresh).view(s.shape[0], 1, h, w)


def mask_xt(
    x_t: torch.Tensor,
    binary_mask: torch.Tensor,
    fill: str = "zero",
    local_k: int = 5,
) -> torch.Tensor:
    """Destroy masked spatial locations on ``x_t`` (all channels share the mask)."""
    m = binary_mask.to(device=x_t.device)
    if m.dtype != torch.bool:
        m = m > 0.5
    if m.shape[-2:] != x_t.shape[-2:]:
        raise ValueError(f"mask HW {tuple(m.shape[-2:])} != x_t {tuple(x_t.shape[-2:])}")
    if m.shape[0] != x_t.shape[0]:
        raise ValueError(f"mask batch {m.shape[0]} != x_t {x_t.shape[0]}")
    fill = str(fill).lower()
    if fill == "zero":
        return x_t.masked_fill(m, 0.0)
    if fill == "local_mean":
        k = max(1, int(local_k))
        if k % 2 == 0:
            k += 1
        avg = F.avg_pool2d(x_t, kernel_size=k, stride=1, padding=k // 2)
        return torch.where(m, avg, x_t)
    raise ValueError(f"fill must be zero|local_mean, got {fill!r}")
