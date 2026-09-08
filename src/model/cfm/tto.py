"""Test-Time Optimization (TTO) of the 1024-d physics condition ``gru_feat``.

Freeze the OT-CFM U-Net, run a short differentiable Euler draft, and update
``c_opt`` so the draft image has stronger spatial edges (less MSE blur), while
staying close to the original ``gru_features`` via L2 regularization.

Final high-quality sampling should call ``sample_with_ode`` again with the
returned ``c_opt`` and the full ODE budget (no gradients).
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn


def total_variation_loss(x: torch.Tensor) -> torch.Tensor:
    """Anisotropic Total Variation (mean absolute spatial gradients).

    Higher values ⇒ more edge energy. Used as a *sharpness reward* in TTO
    (objective minimizes ``-TV``, i.e. maximizes this quantity).

    Args:
        x: ``(B, C, H, W)`` image / latent in any scale (e.g. ``[-1, 1]``).
    """
    dh = (x[:, :, 1:, :] - x[:, :, :-1, :]).abs().mean()
    dw = (x[:, :, :, 1:] - x[:, :, :, :-1]).abs().mean()
    return dh + dw


def optimize_gru_features_tto(
    model: nn.Module,
    c: torch.Tensor,
    gru_features: torch.Tensor,
    z0: Optional[torch.Tensor] = None,
    *,
    n_iters: int = 10,
    draft_steps: int = 5,
    lr: float = 1e-2,
    lambda_l2: float = 1.0,
    tv_weight: float = 1.0,
) -> torch.Tensor:
    """Optimize ``gru_features`` at inference to sharpen the generated draft.

    Args:
        model: CFMADM (or mixin) exposing ``sample_with_ode`` / ``_decode_from_target_space``.
        c: frozen scalar / AdaGN condition ``(B, cond_dim)``.
        gru_features: initial physics embedding ``(B, 1024)`` (or matching physics_dim).
        z0: fixed noise seed ``(B, C, H, W)``. If ``None``, drawn once and reused.
        n_iters: TTO Adam steps.
        draft_steps: cheap differentiable Euler steps inside the loop.
        lr: Adam learning rate on ``c_opt``.
        lambda_l2: weight on ``||c_opt - gru_features||^2``.
        tv_weight: weight on the (negated) TV sharpness term.

    Returns:
        Optimized ``c_opt`` detached, ready for full-step ``sample_with_ode``.
    """
    if gru_features is None:
        raise ValueError("gru_features is required for TTO (condition_mode=split)")
    if not isinstance(gru_features, torch.Tensor):
        raise TypeError(f"gru_features must be a Tensor, got {type(gru_features)}")

    device = c.device
    gru0 = gru_features.detach().to(device=device, dtype=torch.float32)
    c = c.detach().to(device=device)

    # Freeze U-Net; only c_opt receives Adam updates.
    velocity_net = getattr(model, "velocity_net", None)
    was_training = bool(velocity_net.training) if velocity_net is not None else False
    param_req = []
    if velocity_net is not None:
        velocity_net.eval()
        for p in velocity_net.parameters():
            param_req.append(p.requires_grad)
            p.requires_grad_(False)

    if z0 is None:
        z0 = torch.randn(
            (
                c.shape[0],
                int(model.model_in_channels),
                int(model.model_spatial_size),
                int(model.model_spatial_size),
            ),
            device=device,
            dtype=torch.float32,
        )
    else:
        z0 = z0.detach().to(device=device, dtype=torch.float32)

    c_opt = gru0.clone().detach().requires_grad_(True)
    opt = torch.optim.Adam([c_opt], lr=float(lr))

    prev_solver = getattr(model, "cfm_solver", "heun")
    model.cfm_solver = "euler"  # differentiable single-stage steps

    try:
        for _ in range(int(n_iters)):
            opt.zero_grad(set_to_none=True)
            # Differentiable draft: gradients flow through Euler steps → c_opt.
            x_gen = model.sample_with_ode(
                c,
                z0=z0,
                num_steps=int(draft_steps),
                gru_feat=c_opt,
                guidance_weight=0.0,
                guidance_mode="none",
            )
            # Pixel CFM: identity decode; VAE path is non-diff (no_grad) — TTO
            # then acts on the latent, which is still valid for edge energy.
            x_vis = model._decode_from_target_space(x_gen)

            tv = total_variation_loss(x_vis)
            l2 = ((c_opt - gru0) ** 2).mean()
            # Maximize TV (sharpen) while staying near the original physics vector.
            loss = (-float(tv_weight) * tv) + (float(lambda_l2) * l2)
            loss.backward()
            opt.step()
    finally:
        model.cfm_solver = prev_solver
        if velocity_net is not None:
            for p, req in zip(velocity_net.parameters(), param_req):
                p.requires_grad_(req)
            if was_training:
                velocity_net.train()

    return c_opt.detach()
