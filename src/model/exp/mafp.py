"""MAFP: Mixed Augmentation with Fixed Probability (pixel space, before VAE).

Two independent Bernoulli stages (training only):
  p1: Diff-Augment translation (mostly horizontal; tiny vertical).
  p2: StyleGAN2-ADA ops at fixed p, adaptive controller OFF.
      Enabled: xflip, isotropic scale, brightness, contrast.
      Disabled: rotate, rotate90, aniso, yflip, lumaflip (and hue/sat — grayscale).

Augmentation parameters are returned as a per-sample vector for AdaGN
conditional-input (identity at inference → zeros). Spatial ops use
``grid_sample`` so they stay autograd-compatible.
"""

import torch
import torch.nn.functional as F

# [apply_t, tx, ty, xflip, log_scale, brightness, contrast, unused]
MAFP_PARAM_DIM = 8


def _affine_grid_sample(images, theta, padding_mode="border", mode="bilinear"):
    """theta: [B, 2, 3] mapping output coords → input coords."""
    grid = F.affine_grid(theta, images.size(), align_corners=False)
    return F.grid_sample(
        images,
        grid,
        mode=mode,
        padding_mode=padding_mode,
        align_corners=False,
    )


def _translate(images, tx, ty, mode="bilinear"):
    """tx, ty in normalized image units (fraction of W/H). Positive tx → right."""
    b = images.shape[0]
    theta = torch.zeros(b, 2, 3, device=images.device, dtype=images.dtype)
    theta[:, 0, 0] = 1.0
    theta[:, 1, 1] = 1.0
    theta[:, 0, 2] = -tx.to(dtype=images.dtype)
    theta[:, 1, 2] = -ty.to(dtype=images.dtype)
    return _affine_grid_sample(images, theta, mode=mode)


def _scale_isotropic(images, scale, mode="bilinear"):
    """scale>1 zooms in. Identity when scale==1."""
    b = images.shape[0]
    s = scale.to(dtype=images.dtype).clamp(min=0.5, max=2.0)
    theta = torch.zeros(b, 2, 3, device=images.device, dtype=images.dtype)
    theta[:, 0, 0] = s
    theta[:, 1, 1] = s
    return _affine_grid_sample(images, theta, mode=mode)


def _maybe_spatial(tensor, fn):
    if tensor is None:
        return None
    return fn(tensor)


def apply_mafp(
    images,
    *,
    condition_mask=None,
    valid_mask=None,
    p_diff=0.1,
    p_ada=0.1,
    tx_max=0.125,
    ty_max=0.02,
    scale_std=0.08,
    brightness_std=0.1,
    contrast_std=0.1,
    training=True,
):
    """Apply MAFP to a pixel batch in model range (typically [-1, 1]).

    Returns:
        images, condition_mask, valid_mask, params [B, MAFP_PARAM_DIM]
    """
    if images.ndim != 4:
        raise ValueError(f"MAFP expects NCHW, got {tuple(images.shape)}")
    b = images.shape[0]
    device = images.device
    dtype = images.dtype
    params = torch.zeros(b, MAFP_PARAM_DIM, device=device, dtype=dtype)
    if not training or b == 0:
        return images, condition_mask, valid_mask, params

    # --- p1: Diff-Augment translation (horizontal-dominant) ---
    apply_t = torch.rand(b, device=device) < float(p_diff)
    tx = (torch.empty(b, device=device, dtype=dtype).uniform_(-float(tx_max), float(tx_max))) * apply_t.to(dtype)
    ty = (torch.empty(b, device=device, dtype=dtype).uniform_(-float(ty_max), float(ty_max))) * apply_t.to(dtype)
    if apply_t.any():
        images = _translate(images, tx, ty)
        condition_mask = _maybe_spatial(
            condition_mask, lambda m: _translate(m, tx, ty, mode="nearest")
        )
        valid_mask = _maybe_spatial(
            valid_mask, lambda m: _translate(m, tx, ty, mode="nearest")
        )
    params[:, 0] = apply_t.to(dtype)
    params[:, 1] = tx
    params[:, 2] = ty

    # --- p2: fixed-p ADA (no adaptive p controller) ---
    # xflip
    do_flip = torch.rand(b, device=device) < float(p_ada)
    if do_flip.any():
        flipped = torch.flip(images, dims=(-1,))
        images = torch.where(do_flip.view(b, 1, 1, 1), flipped, images)
        if condition_mask is not None:
            condition_mask = torch.where(
                do_flip.view(b, 1, 1, 1), torch.flip(condition_mask, dims=(-1,)), condition_mask
            )
        if valid_mask is not None:
            valid_mask = torch.where(
                do_flip.view(b, 1, 1, 1), torch.flip(valid_mask, dims=(-1,)), valid_mask
            )
    params[:, 3] = do_flip.to(dtype)

    # isotropic scale (log-normal around 1)
    do_scale = torch.rand(b, device=device) < float(p_ada)
    log_s = torch.randn(b, device=device, dtype=dtype) * float(scale_std)
    log_s = torch.where(do_scale, log_s, torch.zeros_like(log_s))
    scale = torch.exp(log_s)
    if do_scale.any():
        images = _scale_isotropic(images, scale)
        condition_mask = _maybe_spatial(
            condition_mask, lambda m: _scale_isotropic(m, scale, mode="nearest")
        )
        valid_mask = _maybe_spatial(
            valid_mask, lambda m: _scale_isotropic(m, scale, mode="nearest")
        )
    params[:, 4] = log_s

    # brightness / contrast (pixel photometric; grayscale-safe)
    do_b = torch.rand(b, device=device) < float(p_ada)
    delta_b = torch.randn(b, device=device, dtype=dtype) * float(brightness_std)
    delta_b = torch.where(do_b, delta_b, torch.zeros_like(delta_b))
    images = images + delta_b.view(b, 1, 1, 1)
    params[:, 5] = delta_b

    do_c = torch.rand(b, device=device) < float(p_ada)
    delta_c = torch.randn(b, device=device, dtype=dtype) * float(contrast_std)
    delta_c = torch.where(do_c, delta_c, torch.zeros_like(delta_c))
    mean = images.mean(dim=(1, 2, 3), keepdim=True)
    images = (images - mean) * (1.0 + delta_c.view(b, 1, 1, 1)) + mean
    params[:, 6] = delta_c
    # params[:, 7] reserved (hue/sat / lumaflip disabled)

    images = images.clamp(-1.0, 1.0)
    if condition_mask is not None:
        condition_mask = condition_mask.clamp(0.0, 1.0)
    if valid_mask is not None:
        valid_mask = valid_mask.clamp(0.0, 1.0)
    return images, condition_mask, valid_mask, params
