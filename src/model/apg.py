"""Adaptive Projected Guidance (APG).

Sadat et al., “Eliminating Oversaturation and Artifacts of High Guidance
Scales in Diffusion Models” (APG): reverse momentum → L2 rescaling →
orthogonal projection of ``diff = pred_cond - pred_uncond`` onto ``pred_cond``.

CFG identity (η=1, no momentum / rescale):
    pred_cond + (s - 1) * (pred_cond - pred_uncond)
    == pred_uncond + s * (pred_cond - pred_uncond)
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch


class MomentumBuffer:
    """Running update ``avg ← momentum * avg + diff`` (use negative momentum)."""

    def __init__(self, momentum: float = -0.5):
        self.momentum = float(momentum)
        self.running_average: Optional[torch.Tensor] = None

    def reset(self) -> None:
        self.running_average = None

    def update(self, diff: torch.Tensor) -> torch.Tensor:
        if self.running_average is None:
            self.running_average = diff
        else:
            if self.running_average.shape != diff.shape:
                raise ValueError(
                    f"momentum buffer shape {tuple(self.running_average.shape)} "
                    f"!= diff {tuple(diff.shape)}"
                )
            self.running_average = self.momentum * self.running_average + diff
        return self.running_average


def _reduce_dims(x: torch.Tensor) -> Tuple[int, ...]:
    return tuple(range(1, x.ndim))


def _project_onto(diff: torch.Tensor, reference: torch.Tensor):
    """Split ``diff`` into parallel / orthogonal parts w.r.t. ``reference``.

    ``parallel = (diff · r / ||r||²) r``,  ``orthogonal = diff - parallel``.
    """
    dims = _reduce_dims(reference)
    ref_sq = reference.square().sum(dim=dims, keepdim=True).clamp_min(1e-8)
    coeff = (diff * reference).sum(dim=dims, keepdim=True) / ref_sq
    parallel = coeff * reference
    return parallel, diff - parallel


def apg_guidance(
    pred_cond: torch.Tensor,
    pred_uncond: torch.Tensor,
    guidance_scale: float,
    eta: float = 0.0,
    norm_threshold: float = 0.0,
    momentum_buffer: Optional[MomentumBuffer] = None,
) -> torch.Tensor:
    """CFG with APG on ``diff = pred_cond - pred_uncond``.

    Args:
        pred_cond: Conditional prediction (same layout as ``pred_uncond``).
        pred_uncond: Unconditional (or weaker-condition) prediction.
        guidance_scale: CFG scale ``s``.
        eta: Weight on the component of ``diff`` parallel to ``pred_cond``.
            ``0`` drops that axis (de-saturation). ``1`` keeps vanilla CFG.
        norm_threshold: If ``||diff||_2 > threshold``, scale ``diff`` to that
            length. ``<= 0`` disables rescaling.
        momentum_buffer: Optional reverse-momentum state (mutated in-place).
            Typical ``MomentumBuffer(momentum=-0.5)``.

    Returns:
        Guided prediction, same shape as ``pred_cond``.
    """
    if pred_cond.shape != pred_uncond.shape:
        raise ValueError(
            f"pred_cond {tuple(pred_cond.shape)} != pred_uncond {tuple(pred_uncond.shape)}"
        )

    diff = pred_cond - pred_uncond

    if momentum_buffer is not None:
        diff = momentum_buffer.update(diff)

    if norm_threshold > 0.0:
        dims = _reduce_dims(diff)
        diff_norm = torch.linalg.vector_norm(diff, ord=2, dim=dims, keepdim=True).clamp_min(1e-8)
        scale = (float(norm_threshold) / diff_norm).clamp(max=1.0)
        diff = diff * scale

    if eta != 1.0:
        parallel, orthogonal = _project_onto(diff, pred_cond)
        diff = orthogonal + float(eta) * parallel

    scale = float(guidance_scale)
    return pred_cond + (scale - 1.0) * diff
