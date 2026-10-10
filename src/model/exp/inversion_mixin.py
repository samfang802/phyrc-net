"""OT-CFM inversion, partial-noise, TTO, and inpainting helpers."""

import torch
import torch.nn.functional as F

from src.model.cfm.ode_mixin import CFMODEMixin


class CFMInversionMixin:
    def optimize_gru_features_tto(
        self,
        c,
        gru_features,
        z0=None,
        n_iters=10,
        draft_steps=5,
        lr=1e-2,
        lambda_l2=1.0,
        tv_weight=1.0,
    ):
        """Test-time optimize ``gru_features`` (see ``src.model.exp.tto``)."""
        from src.model.exp.tto import optimize_gru_features_tto as _tto

        return _tto(
            self,
            c,
            gru_features,
            z0=z0,
            n_iters=n_iters,
            draft_steps=draft_steps,
            lr=lr,
            lambda_l2=lambda_l2,
            tv_weight=tv_weight,
        )


    def sample_with_tto(
        self,
        c,
        gru_feat,
        z0=None,
        num_steps=None,
        n_iters=10,
        draft_steps=5,
        lr=1e-2,
        lambda_l2=1.0,
        tv_weight=1.0,
        **sample_kwargs,
    ):
        """TTO on ``gru_feat``, then full-step ODE sample (no grad on final pass).

        Returns ``(z, gru_feat_opt)``.
        """
        if z0 is None:
            z0 = torch.randn(
                (
                    c.shape[0],
                    self.model_in_channels,
                    self.model_spatial_size,
                    self.model_spatial_size,
                ),
                device=c.device,
            )
        gru_opt = self.optimize_gru_features_tto(
            c,
            gru_feat,
            z0=z0,
            n_iters=n_iters,
            draft_steps=draft_steps,
            lr=lr,
            lambda_l2=lambda_l2,
            tv_weight=tv_weight,
        )
        with torch.no_grad():
            z = self.sample_with_ode(
                c,
                z0=z0,
                num_steps=num_steps,
                gru_feat=gru_opt,
                **sample_kwargs,
            )
        return z, gru_opt


    @staticmethod
    def ot_partial_noise(z_clean, t_start, eps=None):
        """Push a clean latent onto the OT path at ``t_start``.

            z_t = (1 - t_start) * eps + t_start * z_clean

        Returns ``(z_t, eps)``.
        """
        t = float(t_start)
        if t < 0.0 or t >= 1.0:
            raise ValueError(f"t_start must be in [0, 1), got {t}")
        if eps is None:
            eps = torch.randn_like(z_clean)
        else:
            eps = eps.to(device=z_clean.device, dtype=z_clean.dtype)
            if eps.shape != z_clean.shape:
                raise ValueError(
                    f"eps shape {tuple(eps.shape)} must match z_clean {tuple(z_clean.shape)}"
                )
        z_t = (1.0 - t) * eps + t * z_clean
        return z_t, eps


    def invert_with_ode(self, z1, c, num_steps=None, gru_feat=None):
        """Backward Heun ODE inversion: map a clean latent ``z1`` → noise ``z0``.

        Solves the CFM ODE **backward** in time (t: 1 → 0) under condition ``c``
        (typically the undamaged / DI=0 condition used to create ``z1``)::

            dz/dt = v_θ(z, t, c)
            z_{t+Δt} = Heun(z_t, v_θ)   with Δt < 0

        Heun predictor-corrector with negative ``dt`` (same formulas as forward)::

            v_t     = v_θ(z, t)
            z_pred  = z + dt * v_t
            v_next  = v_θ(z_pred, t+dt)
            z       = z + 0.5 * dt * (v_t + v_next)

        No CFG / frequency filters. Pair with ``sample_with_ode(..., z0=z0_inv)``
        under a *damaged* condition for structure-preserving forward generation.

        Args:
            z1: clean / undamaged state in target space (B,C,H,W)
            c: undamaged generation condition
            num_steps: ODE steps (default: self.cfm_ode_steps)
            gru_feat: physics embedding when condition_mode=split

        Returns:
            z0_inverted with the same shape as ``z1``
        """
        if self.use_cascade_sr:
            raise ValueError("invert_with_ode is not supported with use_cascade_sr")
        if num_steps is None:
            num_steps = self.cfm_ode_steps
        if self.use_split_conditioning and gru_feat is None:
            raise ValueError(
                "gru_feat is required for invert_with_ode when condition_mode=split"
            )

        z = z1.to(device=c.device, dtype=torch.float32).clone()
        if z.shape[0] != c.shape[0]:
            raise ValueError(
                f"z1 batch {z.shape[0]} must match condition batch {c.shape[0]}"
            )

        coords = None
        if self.use_patch_diffusion:
            grid_x, grid_y = self._build_full_coordinate_maps(
                z.shape[-2],
                z.shape[-1],
                device=z.device,
                dtype=z.dtype,
            )
            coords = (
                torch.stack([grid_x, grid_y], dim=0)
                .unsqueeze(0)
                .expand(c.shape[0], -1, -1, -1)
                .contiguous()
            )

        condition_mask = None
        if self.use_damage_mask_condition:
            condition_mask = torch.ones(
                (c.shape[0], 1, z.shape[-2], z.shape[-1]),
                device=z.device,
                dtype=z.dtype,
            )

        # At t≈1 the state is already (near) clean → seed self-cond with z1.
        self_cond = z.clone() if self.use_self_conditioning else None

        # Negative step: integrate from t=1 down to t=0.
        dt = -1.0 / float(num_steps)
        for i in range(num_steps):
            t_scalar = 1.0 - float(i) / float(num_steps)
            t_next_scalar = 1.0 - float(i + 1) / float(num_steps)
            t = torch.full((c.shape[0],), t_scalar, device=c.device)

            v_t = self._predict_velocity(
                z,
                t,
                c,
                coords=coords,
                damage_mask=condition_mask,
                gru_feat=gru_feat,
                self_cond=self_cond,
            )
            z1_hat = (
                self._estimate_clean_from_velocity(z, t, v_t).detach()
                if self.use_self_conditioning
                else None
            )

            if self.cfm_solver == "heun":
                # Predictor with negative dt (moves toward lower t / more noise).
                z_pred = z + dt * v_t
                t_next = torch.full((c.shape[0],), t_next_scalar, device=c.device)
                # Corrector: average velocities at t and t+dt (same self_cond).
                v_next = self._predict_velocity(
                    z_pred,
                    t_next,
                    c,
                    coords=coords,
                    damage_mask=condition_mask,
                    gru_feat=gru_feat,
                    self_cond=self_cond,
                )
                z = z + 0.5 * dt * (v_t + v_next)
            else:
                z = z + dt * v_t

            if self.use_self_conditioning:
                self_cond = z1_hat

        return z
    @staticmethod
    def _inpaint_replace(z_pred, z0, z1_gt, t_scalar, mask):
        """
        Replace known pixels with the OT-path GT state at time t.

        Mask semantics: 1 = KEEP (GT), 0 = INPAINT (model prediction).
        ``mask`` shape: (B, 1, H, W) or (B, C, H, W).
        """
        z_gt = CFMODEMixin._ot_path_state(z0, z1_gt, t_scalar)
        if mask.shape[1] == 1 and z_pred.shape[1] != 1:
            mask = mask.expand(-1, z_pred.shape[1], -1, -1)
        return z_gt * mask + z_pred * (1.0 - mask)


    def sample_with_ode_inpaint(
        self,
        c,
        z1_gt,
        inpaint_mask,
        z0=None,
        num_steps=None,
        gru_feat=None,
    ):
        """
        Heun/Euler CFM ODE with OT-path replacement inpainting.

        Spatial roles are intentionally split:
          - U-Net damage-mask channel: **all-ones** (tell the net the whole
            128×128 is a clean pier — same as normal ``sample_with_ode`` inference).
          - ODE replacement: real SAM / DamagePattern mask with hole (0) dilated
            by 5px via max_pool (1=KEEP GT OT path, 0=INPAINT keep model prediction).

        At every Heun prediction point and correction point, known pixels (mask=1)
        are replaced by the GT OT state at that time. AdaGN conditions ``c`` /
        ``gru_feat`` are still fed as in normal sampling.

        Args:
            c: scalar / joint generation condition (DI, direction, …)
            z1_gt: clean GT in target space (same shape as latent/image state)
            inpaint_mask: (B, 1, H, W) real SAM mask, 1=KEEP, 0=INPAINT
            z0: optional shared noise; if None, sampled once and reused for OT path
            num_steps: ODE steps (default: self.cfm_ode_steps)
            gru_feat: physics embedding for gated AdaGN (required when condition_mode=split)
        """
        if self.use_cascade_sr:
            raise ValueError("sample_with_ode_inpaint is not supported with use_cascade_sr")
        if num_steps is None:
            num_steps = self.cfm_ode_steps
        if z1_gt.shape[0] != c.shape[0]:
            raise ValueError(
                f"z1_gt batch {z1_gt.shape[0]} must match condition batch {c.shape[0]}"
            )
        if inpaint_mask.ndim != 4 or inpaint_mask.shape[1] != 1:
            raise ValueError(
                f"inpaint_mask must be (B,1,H,W), got {tuple(inpaint_mask.shape)}"
            )
        if inpaint_mask.shape[0] != c.shape[0]:
            raise ValueError(
                f"inpaint_mask batch {inpaint_mask.shape[0]} must match condition {c.shape[0]}"
            )

        if z0 is None:
            z0 = torch.randn_like(z1_gt)
        else:
            if z0.shape != z1_gt.shape:
                raise ValueError(
                    f"z0 shape {tuple(z0.shape)} must match z1_gt {tuple(z1_gt.shape)}"
                )
            z0 = z0.to(device=z1_gt.device, dtype=z1_gt.dtype)

        z = z0.clone()
        # Dilate hole (0) by 5px before ODE — expand interference region for replacement.
        real_sam_mask = inpaint_mask.to(device=z.device, dtype=z.dtype)
        if real_sam_mask.shape[-2:] != z.shape[-2:]:
            raise ValueError(
                f"inpaint_mask spatial {tuple(real_sam_mask.shape[-2:])} must match state {tuple(z.shape[-2:])}"
            )
        dilated_sam_mask = 1.0 - F.max_pool2d(
            1.0 - real_sam_mask, kernel_size=11, stride=1, padding=5
        )
        mask = dilated_sam_mask

        # U-Net spatial condition: all-ones (clean pier). Dilated SAM mask is replace-only.
        condition_mask = None
        if self.use_damage_mask_condition:
            condition_mask = torch.ones_like(mask)

        # AdaGN / split conditions (c, gru_feat) — same as normal CFM sampling.
        if self.use_split_conditioning and gru_feat is None:
            raise ValueError(
                "gru_feat (physics condition) is required for inpaint when condition_mode=split"
            )

        coords = None
        if self.use_patch_diffusion:
            grid_x, grid_y = self._build_full_coordinate_maps(
                z.shape[-2],
                z.shape[-1],
                device=z.device,
                dtype=z.dtype,
            )
            coords = (
                torch.stack([grid_x, grid_y], dim=0)
                .unsqueeze(0)
                .expand(c.shape[0], -1, -1, -1)
                .contiguous()
            )

        # Match normal inference: self_cond starts at zeros, then carries z1_hat.
        self_cond = torch.zeros_like(z) if self.use_self_conditioning else None

        # t=0: known region already equals OT path (z0); still apply for consistency.
        z = self._inpaint_replace(z, z0, z1_gt, 0.0, mask)

        dt = 1.0 / float(num_steps)
        for i in range(num_steps):
            t_scalar = float(i) / float(num_steps)
            t_next_scalar = float(i + 1) / float(num_steps)
            t = torch.full((c.shape[0],), t_scalar, device=c.device)
            t_next = torch.full((c.shape[0],), t_next_scalar, device=c.device)

            v_t = self._predict_velocity(
                z,
                t,
                c,
                coords=coords,
                damage_mask=condition_mask,
                gru_feat=gru_feat,
                self_cond=self_cond,
            )
            z1_hat = (
                self._estimate_clean_from_velocity(z, t, v_t).detach()
                if self.use_self_conditioning
                else None
            )

            if self.cfm_solver == "heun":
                # Heun prediction (Euler intermediate) — replace at t_next with SAM mask.
                z_pred = z + dt * v_t
                z_pred = self._inpaint_replace(z_pred, z0, z1_gt, t_next_scalar, mask)
                # Same self_cond for both Heun evaluations.
                v_next = self._predict_velocity(
                    z_pred,
                    t_next,
                    c,
                    coords=coords,
                    damage_mask=condition_mask,
                    gru_feat=gru_feat,
                    self_cond=self_cond,
                )
                # Heun correction — replace again at t_next with SAM mask.
                z = z + 0.5 * dt * (v_t + v_next)
                z = self._inpaint_replace(z, z0, z1_gt, t_next_scalar, mask)
            else:
                z = z + dt * v_t
                z = self._inpaint_replace(z, z0, z1_gt, t_next_scalar, mask)

            if self.use_self_conditioning:
                self_cond = z1_hat

        return z
