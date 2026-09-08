"""Phase 1 (Oracle Extraction) + Phase 2 (Residual Modality Adapter).

Phase 0 trains the Dual-Track OT-CFM U-Net end-to-end (see ``ADMUNetCFM``).
After Phase 0, freeze the U-Net and:

  Phase 1 — Offline TTO per sample:
      c_opt ← argmin_c MSE(v_θ(x_t, t, c_scalar, c), u_target)
      save converged c_opt as c_oracle

  Phase 2 — Train a lightweight residual MLP:
      Adapter(c_physics) = c_physics + MLP(c_physics)
      Loss = MSE(pred, c_oracle) + λ_cos * (1 - cos_sim)
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Phase 2 — Residual MLP Adapter
# ---------------------------------------------------------------------------

class ResidualPhysicsAdapter(nn.Module):
    """Map GRU ``c_physics`` → ``c_oracle`` with a residual bottleneck MLP.

    Shapes:
        x: [B, D]  (D=1024)
        y: [B, D]  = x + MLP(x)
    """

    def __init__(self, dim: int = 1024, hidden_dim: int = 512, dropout: float = 0.0):
        super().__init__()
        self.dim = int(dim)
        self.mlp = nn.Sequential(
            nn.LayerNorm(self.dim),
            nn.Linear(self.dim, hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, self.dim),
        )
        # Residual starts as identity: Δ ≡ 0.
        nn.init.zeros_(self.mlp[-1].weight)
        nn.init.zeros_(self.mlp[-1].bias)

    def forward(self, c_physics: torch.Tensor) -> torch.Tensor:
        if c_physics.ndim != 2 or c_physics.shape[-1] != self.dim:
            raise ValueError(f"Expected c_physics [B, {self.dim}], got {tuple(c_physics.shape)}")
        return c_physics + self.mlp(c_physics)


def adapter_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    cosine_weight: float = 0.1,
) -> torch.Tensor:
    """MSE + weighted cosine distance."""
    mse = F.mse_loss(pred, target)
    cos = F.cosine_similarity(pred, target, dim=-1).mean()
    return mse + float(cosine_weight) * (1.0 - cos)


# ---------------------------------------------------------------------------
# Phase 1 — Oracle extraction via CFM velocity matching
# ---------------------------------------------------------------------------

@torch.enable_grad()
def extract_oracle_condition(
    model: nn.Module,
    x_target: torch.Tensor,
    c_scalar: torch.Tensor,
    c_physics: torch.Tensor,
    *,
    damage_mask: Optional[torch.Tensor] = None,
    n_iters: int = 50,
    lr: float = 5e-2,
    steps_per_iter: int = 1,
) -> torch.Tensor:
    """Optimize ``c_opt`` so frozen U-Net velocity matches the OT target field.

    Args:
        model: trained CFMADM (velocity_net frozen).
        x_target: clean damaged image / latent [B, C, H, W] in model space.
        c_scalar: AdaGN scalar condition [B, cond_dim] (DI/dir; usually frozen).
        c_physics: initial GRU physics vector [B, 1024].
        damage_mask: optional mask channel [B, 1, H, W].
        n_iters: Adam steps on ``c_opt``.
        lr: Adam learning rate.
        steps_per_iter: random OT samples drawn per Adam step.

    Returns:
        ``c_oracle`` [B, 1024] detached.
    """
    velocity_net = getattr(model, "velocity_net", None)
    if velocity_net is None:
        raise ValueError("model must expose velocity_net")

    # Freeze U-Net completely.
    was_training = velocity_net.training
    velocity_net.eval()
    req_flags = []
    for p in velocity_net.parameters():
        req_flags.append(p.requires_grad)
        p.requires_grad_(False)

    device = x_target.device
    x1 = x_target.detach().to(device=device, dtype=torch.float32)
    c = c_scalar.detach().to(device=device, dtype=torch.float32)
    c0 = c_physics.detach().to(device=device, dtype=torch.float32)

    c_opt = c0.clone().requires_grad_(True)
    opt = torch.optim.Adam([c_opt], lr=float(lr))

    try:
        for _ in range(int(n_iters)):
            opt.zero_grad(set_to_none=True)
            loss_acc = 0.0
            for _s in range(int(steps_per_iter)):
                # OT path: x_t = (1-t) ε + t x1 ,  u = x1 - ε
                eps = torch.randn_like(x1)
                t = torch.rand(x1.shape[0], device=device).clamp(1e-5, 1.0 - 1e-5)
                t_ = t.view(-1, *([1] * (x1.ndim - 1)))
                x_t = (1.0 - t_) * eps + t_ * x1
                u_target = x1 - eps

                coords = None
                if getattr(model, "use_patch_diffusion", False):
                    # Full-frame coords (oracle uses whole image; no random crop).
                    grid_x, grid_y = model._build_full_coordinate_maps(
                        x_t.shape[-2], x_t.shape[-1], device=device, dtype=x_t.dtype
                    )
                    coords = (
                        torch.stack([grid_x, grid_y], dim=0)
                        .unsqueeze(0)
                        .expand(x_t.shape[0], -1, -1, -1)
                        .contiguous()
                    )

                v_pred = model._predict_velocity(
                    x_t,
                    t,
                    c,
                    coords=coords,
                    damage_mask=damage_mask,
                    gru_feat=c_opt,
                    self_cond=None,
                )
                loss_acc = loss_acc + F.mse_loss(v_pred, u_target)

            (loss_acc / float(steps_per_iter)).backward()
            opt.step()
    finally:
        for p, req in zip(velocity_net.parameters(), req_flags):
            p.requires_grad_(req)
        if was_training:
            velocity_net.train()

    return c_opt.detach()


def train_adapter_epoch(
    adapter: ResidualPhysicsAdapter,
    loader,
    optimizer: torch.optim.Optimizer,
    *,
    cosine_weight: float = 0.1,
    device: Optional[torch.device] = None,
) -> float:
    """One epoch of Phase-2 adapter training.

    ``loader`` yields ``(c_physics, c_oracle)`` batches, each ``[B, 1024]``.
    """
    adapter.train()
    total = 0.0
    n = 0
    for c_physics, c_oracle in loader:
        if device is not None:
            c_physics = c_physics.to(device)
            c_oracle = c_oracle.to(device)
        optimizer.zero_grad(set_to_none=True)
        pred = adapter(c_physics)
        loss = adapter_loss(pred, c_oracle, cosine_weight=cosine_weight)
        loss.backward()
        optimizer.step()
        total += float(loss.detach()) * c_physics.shape[0]
        n += c_physics.shape[0]
    return total / max(n, 1)
