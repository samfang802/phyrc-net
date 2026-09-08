"""FreCaS (Frequency-aware Cascaded Sampling) utilities for training-free 64→128 generation.

Implements SNR-aligned stage transition, frequency-aware model-output scaling, and
Mid-block self-attention capture / fusion for OT-CFM (velocity prediction) models.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def build_alphas_cumprod_from_flow(num_steps: int, device=None, dtype=torch.float32):
    """Build DDPM-style ``alpha_bar`` from linear OT-CFM flow times.

    For flow time ``t in (0, 1]``, training Min-SNR uses
    ``SNR(t) = (t / (1 - t))^2``.  We define ``alpha_bar = SNR / (1 + SNR)`` so
    ``SNR = alpha_bar / (1 - alpha_bar)`` as in VP diffusion.
    """
    if num_steps < 1:
        raise ValueError(f"num_steps must be >= 1, got {num_steps}")
    t = torch.linspace(0.0, 1.0, num_steps + 1, device=device, dtype=dtype)[1:]
    t_safe = t.clamp(min=1e-5, max=1.0 - 1e-5)
    snr = (t_safe / (1.0 - t_safe)) ** 2
    return snr / (snr + 1.0)


def calculate_shifted_timestep(L, s1=64, s2=128, gamma=2.0, alphas_cumprod=None):
    """Compute Stage-2 start index ``F`` from Stage-1 stop index ``L`` via SNR alignment.

    FreCaS relation (resolution change from ``s1`` to ``s2``)::

        SNR(z_L) = SNR(z_F) * (s1 / s2) ** gamma

    Solving for the Stage-2 SNR and matching against ``alphas_cumprod`` yields ``F``.
    """
    if alphas_cumprod is None:
        raise ValueError("alphas_cumprod is required")
    if isinstance(alphas_cumprod, torch.Tensor):
        alpha = alphas_cumprod.detach().float().cpu()
    else:
        alpha = torch.tensor(alphas_cumprod, dtype=torch.float32)

    n = int(alpha.numel())
    L = int(L)
    if L < 0 or L >= n:
        raise ValueError(f"L must be in [0, {n - 1}], got {L}")

    alpha_L = alpha[L].clamp(min=1e-8, max=1.0 - 1e-8)
    snr_L = alpha_L / (1.0 - alpha_L)
    scale = float(s2) / float(s1)
    if scale <= 0.0:
        raise ValueError(f"s1 and s2 must be positive, got s1={s1}, s2={s2}")
    snr_F = snr_L * (scale ** float(gamma))

    snr_all = alpha / (1.0 - alpha).clamp(min=1e-8)
    F = int(torch.argmin((snr_all - snr_F).abs()).item())
    return F


def flow_time_from_schedule_index(index, num_steps):
    """Map discrete schedule index in ``[0, num_steps)`` to continuous flow time ``t``."""
    num_steps = max(1, int(num_steps))
    index = int(index)
    if index < 0:
        index = 0
    if index >= num_steps:
        index = num_steps - 1
    return float(index + 1) / float(num_steps)


def separate_frequency_components(x, pool_kernel=3):
    """Split spatial tensor into low / high frequency parts via avg-pool low-pass."""
    k = int(pool_kernel)
    if k < 3:
        k = 3
    if k % 2 == 0:
        k += 1
    pad = k // 2
    x_pad = F.pad(x, (pad, pad, pad, pad), mode="reflect")
    x_low = F.avg_pool2d(x_pad, kernel_size=k, stride=1)
    x_high = x - x_low
    return x_low, x_high


def apply_frequency_aware_scaling(pred, w_h=2.0, pool_kernel=3):
    """Frequency-aware scaling on conditional model output (``eps`` or CFM velocity).

    ``pred_out = pred_low + w_h * pred_high``
    """
    w_h = float(w_h)
    pred_low, pred_high = separate_frequency_components(pred, pool_kernel=pool_kernel)
    return pred_low + w_h * pred_high


def interpolate_self_attn_map(attn, src_hw, dst_hw):
    """Bilinear-resize a self-attention map ``[B, heads, sq, sk]`` to ``dst_hw`` grid."""
    if attn.ndim != 4:
        raise ValueError(f"Expected attn [B, heads, sq, sk], got {tuple(attn.shape)}")
    sh, sw = int(src_hw[0]), int(src_hw[1])
    dh, dw = int(dst_hw[0]), int(dst_hw[1])
    sq, sk = int(attn.shape[-2]), int(attn.shape[-1])
    if (sh, sw) == (dh, dw) and sq == sh * sw and sk == sh * sw:
        return attn

    b, heads = attn.shape[0], attn.shape[1]
    x = attn.reshape(b * heads, 1, sq, sk)
    x = F.interpolate(
        x,
        size=(dh * dw, dh * dw),
        mode="bilinear",
        align_corners=False,
    )
    return x.reshape(b, heads, dh * dw, dh * dw)


class MidBlockAttentionManager:
    """Capture / fuse Mid-block ``SelfAttention2d`` maps via hooks or direct module flags.

    Usage (Stage 1 — capture at stop step)::

        mgr = MidBlockAttentionManager(stage1_model.velocity_net.mid_attn)
        mgr.enable_capture()
        ... final forward at stop step ...
        stage1_attn = mgr.pop_captured()

    Usage (Stage 2 — fuse with Stage-1 layout)::

        mgr = MidBlockAttentionManager(stage2_model.velocity_net.mid_attn)
        mgr.set_reference(stage1_attn, src_hw, w_c=0.6)
        ... sampling loop ...
        mgr.clear_reference()
    """

    def __init__(self, mid_attn_module: nn.Module):
        self.mid_attn = mid_attn_module
        self._hook_handle = None
        self._captured = None
        self._captured_hw = None

    def enable_capture(self):
        """Turn on ``store_attn`` and register a forward hook on Mid self-attention."""
        if not hasattr(self.mid_attn, "store_attn"):
            raise TypeError(
                "mid_attn must be SelfAttention2d with store_attn support; "
                f"got {type(self.mid_attn)}"
            )
        self.mid_attn.store_attn = True
        if self._hook_handle is None:
            self._hook_handle = self.mid_attn.register_forward_hook(self._capture_hook)

    def disable_capture(self):
        if hasattr(self.mid_attn, "store_attn"):
            self.mid_attn.store_attn = False

    def _capture_hook(self, module, inputs, output):
        del inputs, output
        if getattr(module, "last_attn", None) is not None:
            self._captured = module.last_attn
            self._captured_hw = getattr(module, "last_hw", None)

    def pop_captured(self):
        """Return ``(attn, hw)`` and clear the internal buffer."""
        attn, hw = self._captured, self._captured_hw
        self._captured = None
        self._captured_hw = None
        return attn, hw

    def set_reference(self, ref_attn, ref_hw, w_c=0.6, dst_hw=None):
        """Install Stage-1 attention for linear fusion during Stage-2 forwards."""
        if ref_attn is None:
            raise ValueError("ref_attn must not be None")
        attn = ref_attn
        if dst_hw is not None and ref_hw is not None:
            attn = interpolate_self_attn_map(attn, ref_hw, dst_hw)
        if hasattr(self.mid_attn, "set_reference_attn"):
            self.mid_attn.set_reference_attn(attn, float(w_c))
        else:
            raise TypeError(
                "mid_attn must implement set_reference_attn; "
                f"got {type(self.mid_attn)}"
            )

    def clear_reference(self):
        if hasattr(self.mid_attn, "clear_reference_attn"):
            self.mid_attn.clear_reference_attn()
        self.disable_capture()
        if self._hook_handle is not None:
            self._hook_handle.remove()
            self._hook_handle = None
