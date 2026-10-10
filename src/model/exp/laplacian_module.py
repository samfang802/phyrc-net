"""Independent OT-CFM trainer for one Laplacian band (D1, D2, or D3)."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
import lightning.pytorch as pl

from src.model.cfm.condition import build_generation_condition, build_physics_condition
from src.model.exp.laplacian import (
    apply_band_patch,
    decompose_laplacian,
    ot_cfm_path,
    reconstruct_laplacian,
    select_band,
    weighted_velocity_mse,
)
from src.model.exp.laplacian_velocity import build_band_velocity_net


class LaplacianBandCFM(pl.LightningModule):
    """Train **one** of {D1, D2, D3}. Run three jobs in parallel; they do not share features."""

    def __init__(self, regressor=None, **kwargs):
        super().__init__()
        self.save_hyperparameters(ignore=["kwargs", "result_dir", "regressor"])
        self.regressor = regressor
        self.result_dir = kwargs.get("result_dir", ".")
        self.band = str(kwargs.get("laplacian_band", "d1")).lower()
        if self.band not in ("d1", "d2", "d3"):
            raise ValueError(f"laplacian_band must be d1|d2|d3, got {self.band!r}")
        self.image_size = int(kwargs.get("image_size", 128))
        self.lr = float(kwargs.get("lr", 1e-4))
        self.model_in_channels = 3
        self.use_coords = self.band in ("d1", "d2")
        in_ch = self.model_in_channels + (2 if self.use_coords else 0)
        out_ch = self.model_in_channels

        self.condition_mode = str(kwargs.get("condition_mode", "split")).lower()
        self.latent_conditioning = bool(kwargs.get("latent_conditioning", False))
        self.direction_conditioning = bool(kwargs.get("direction_conditioning", False))
        self.include_di_conditioning = bool(kwargs.get("include_di_conditioning", False))
        self.zero_di_condition = bool(kwargs.get("zero_di_condition", False))
        self.condition_dim = int(kwargs.get("condition_dim", 2))
        self.physics_dim = int(kwargs.get("physics_dim", 1024))
        self.min_snr_gamma = float(kwargs.get("min_snr_gamma", 5.0))
        self.use_velocity_min_snr_weighting = bool(
            kwargs.get("use_velocity_min_snr_weighting", True)
        )
        self.use_object_loss_mask = bool(kwargs.get("use_object_loss_mask", False))
        self.use_damage_mask_condition = False
        self.optimizer_name = str(kwargs.get("optimizer", "adamw")).lower()
        self.lr_schedule = kwargs.get("lr_schedule", "warmup_cosine")
        self.warmup_steps = int(kwargs.get("warmup_steps", 800))
        self.total_steps = int(kwargs.get("max_steps", 15000))
        self.min_lr_ratio = float(kwargs.get("min_lr_ratio", 0.1))
        self.grad_clip_max_norm = float(kwargs.get("grad_clip_max_norm", 1.0))
        self.cfm_ode_steps = int(kwargs.get("cfm_ode_steps", 30))
        self.cfm_solver = str(kwargs.get("cfm_solver", "heun")).lower()
        self.use_laplacian_patch = bool(kwargs.get("use_laplacian_patch", False))

        dummy = bool(kwargs.get("laplacian_dummy", False))
        self.velocity_net = build_band_velocity_net(
            self.band,
            in_ch,
            out_ch,
            dummy=dummy,
            cond_dim=int(kwargs.get("cfm_cond_dim", 128)),
            latent_dim=self.condition_dim,
            physics_dim=self.physics_dim,
            use_split=self.condition_mode == "split",
            dropout=float(kwargs.get("cfm_dropout", 0.0)),
        )
        print(
            f"[LaplacianCFM] band={self.band} dummy={dummy} in_ch={in_ch} "
            f"patch={self.use_laplacian_patch} self_cond=off "
            f"params={sum(p.numel() for p in self.velocity_net.parameters()) / 1e6:.3f}M",
            flush=True,
        )

    def _unpack_generation_batch(self, batch):
        if isinstance(batch, dict):
            real_data = batch.get("image_target", batch["image"])
            hidden_state = batch.get("hidden_state_target", batch["hidden_state"])
            direction = batch.get("direction_target", batch["direction"])
            DI = batch.get("DI_target", batch["DI"])
            info = batch["info"]
            valid_mask = batch.get("loss_mask_target", batch.get("loss_mask"))
        else:
            real_data, hidden_state, direction, _exp, _ts, DI, info = batch[:7]
            valid_mask = batch[7] if len(batch) >= 8 else None
        if not self.use_object_loss_mask:
            valid_mask = None
        return real_data, hidden_state, direction, DI, info, valid_mask

    def _conditions(self, info, hidden_state, direction, DI):
        c = build_generation_condition(
            info,
            hidden_state,
            direction,
            DI,
            self.latent_conditioning,
            self.direction_conditioning,
            self.include_di_conditioning,
            self.condition_mode,
            zero_di_condition=self.zero_di_condition,
        )
        gru_feat = build_physics_condition(hidden_state, self.condition_mode)
        return c, gru_feat

    def _band_target(self, real_data):
        x1, x2, x3 = decompose_laplacian(real_data)
        return select_band(x1, x2, x3, self.band)

    def _align_mask(self, mask, spatial):
        if mask is None:
            return None
        if mask.ndim == 3:
            mask = mask.unsqueeze(1)
        if tuple(mask.shape[-2:]) != tuple(spatial):
            mask = F.interpolate(mask, size=spatial, mode="nearest")
        return mask

    def _net_in(self, xt, coords):
        if self.use_coords:
            return torch.cat([xt, coords], dim=1)
        return xt

    def flow_matching_loss(self, x_band, c, gru_feat, valid_mask=None, apply_patch=True):
        xt, vt, t = ot_cfm_path(x_band)
        mask = self._align_mask(valid_mask, x_band.shape[-2:])
        if apply_patch:
            xt, vt, coords, mask = apply_band_patch(
                xt, vt, self.band, training=bool(self.training), valid_mask=mask
            )
        else:
            _, _, coords, _ = apply_band_patch(xt, vt, "d3", training=False)
        pred = self.velocity_net(self._net_in(xt, coords), t, c, gru_feat=gru_feat)
        if self.use_velocity_min_snr_weighting:
            loss, _, _ = weighted_velocity_mse(
                pred, vt, t, gamma=self.min_snr_gamma, valid_mask=mask
            )
        else:
            loss = F.mse_loss(pred, vt)
        return loss

    def training_step(self, batch, batch_idx):
        real, hidden, direction, DI, info, valid_mask = self._unpack_generation_batch(batch)
        c, gru_feat = self._conditions(info, hidden, direction, DI)
        x_band = self._band_target(real)
        loss = self.flow_matching_loss(
            x_band, c, gru_feat, valid_mask=valid_mask, apply_patch=self.use_laplacian_patch
        )
        if getattr(self, "_trainer", None) is not None:
            self.log("train/loss", loss, prog_bar=True, on_step=True, on_epoch=False)
            trainer = self._trainer
            if trainer.optimizers:
                self.log(
                    "train/lr",
                    float(trainer.optimizers[0].param_groups[0]["lr"]),
                    on_step=True,
                )
        return loss

    def validation_step(self, batch, batch_idx):
        real, hidden, direction, DI, info, valid_mask = self._unpack_generation_batch(batch)
        c, gru_feat = self._conditions(info, hidden, direction, DI)
        x_band = self._band_target(real)
        loss = self.flow_matching_loss(x_band, c, gru_feat, valid_mask=valid_mask, apply_patch=False)
        self.log("val/loss", loss, prog_bar=True, on_step=False, on_epoch=True, sync_dist=True)
        return loss

    @torch.no_grad()
    def sample_band(self, c, gru_feat, steps=None, generator=None):
        """ODE 0→1 for this band (Heun by default; Euler if ``cfm_solver=euler``)."""
        steps = int(steps or self.cfm_ode_steps)
        size = {"d1": self.image_size, "d2": self.image_size // 2, "d3": self.image_size // 4}[self.band]
        b = c.shape[0]
        x = torch.randn(
            b, self.model_in_channels, size, size, device=c.device, dtype=c.dtype, generator=generator
        )
        _, _, coords, _ = apply_band_patch(x, x, "d3", training=False)
        times = [float(i) / float(steps) for i in range(steps + 1)]
        use_heun = self.cfm_solver != "euler"

        def _v(state, t_scalar):
            t = torch.full((b,), t_scalar, device=c.device, dtype=c.dtype)
            return self.velocity_net(self._net_in(state, coords), t, c, gru_feat=gru_feat)

        for i in range(steps):
            t0, t1 = times[i], times[i + 1]
            dt = t1 - t0
            v0 = _v(x, t0)
            if use_heun:
                v1 = _v(x + dt * v0, t1)
                x = x + 0.5 * dt * (v0 + v1)
            else:
                x = x + dt * v0
        return x

    def configure_optimizers(self):
        if self.optimizer_name == "adam":
            opt = torch.optim.Adam(self.velocity_net.parameters(), lr=self.lr)
        else:
            opt = torch.optim.AdamW(self.velocity_net.parameters(), lr=self.lr)
        if self.lr_schedule != "warmup_cosine":
            return opt

        def lr_lambda(step_idx):
            if self.warmup_steps > 0 and step_idx < self.warmup_steps:
                return float(step_idx + 1) / float(self.warmup_steps)
            remain = max(self.total_steps - self.warmup_steps, 1)
            progress = min(max((step_idx - self.warmup_steps) / remain, 0.0), 1.0)
            cosine = 0.5 * (1.0 + np.cos(np.pi * progress))
            return self.min_lr_ratio + (1.0 - self.min_lr_ratio) * cosine

        return {
            "optimizer": opt,
            "lr_scheduler": {
                "scheduler": torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda=lr_lambda),
                "interval": "step",
                "frequency": 1,
            },
        }

    def on_before_optimizer_step(self, optimizer):
        if self.grad_clip_max_norm > 0:
            torch.nn.utils.clip_grad_norm_(self.velocity_net.parameters(), self.grad_clip_max_norm)


def compose_laplacian_sample(x1, x2, x3):
    """Merge independently sampled bands back to a 128 image."""
    return reconstruct_laplacian(x1, x2, x3)
