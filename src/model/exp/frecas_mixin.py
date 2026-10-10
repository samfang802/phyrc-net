"""FreCaS sampling helpers."""

import torch
import torch.nn.functional as F


class CFMFrecasMixin:
    def _frecas_ode_segment(
        self,
        model,
        z,
        c,
        gru_feat,
        t_start,
        t_end,
        num_steps,
        damage_mask=None,
        apply_hf_scaling=False,
        w_h=2.0,
        hf_pool_kernel=3,
        capture_mid_attn=False,
        mid_attn_mgr=None,
    ):
        """Integrate OT-CFM ODE on ``model`` from ``t_start`` to ``t_end`` (inclusive end node)."""
        if num_steps is None:
            num_steps = model.cfm_ode_steps
        t_start = float(t_start)
        t_end = float(t_end)
        if t_end <= t_start:
            raise ValueError(f"t_end ({t_end}) must be > t_start ({t_start})")

        coords = None
        if model.use_patch_diffusion:
            grid_x, grid_y = model._build_full_coordinate_maps(
                z.shape[-1],
                z.shape[-2],
                device=z.device,
                dtype=z.dtype,
            )
            coords = (
                torch.stack([grid_x, grid_y], dim=0)
                .unsqueeze(0)
                .expand(c.shape[0], -1, -1, -1)
                .contiguous()
            )

        condition_mask = damage_mask
        if condition_mask is None and model.use_damage_mask_condition:
            condition_mask = torch.ones(
                (c.shape[0], 1, z.shape[-2], z.shape[-1]),
                device=z.device,
                dtype=z.dtype,
            )
        elif condition_mask is not None:
            condition_mask = condition_mask.to(device=z.device, dtype=z.dtype)
            if condition_mask.ndim == 3:
                condition_mask = condition_mask.unsqueeze(1)
            if condition_mask.shape[-2:] != z.shape[-2:]:
                condition_mask = F.interpolate(
                    condition_mask, size=z.shape[-2:], mode="nearest"
                )

        self_cond = torch.zeros_like(z) if model.use_self_conditioning else None
        span = t_end - t_start
        n_steps = max(1, int(round(float(num_steps) * span)))
        times = [t_start + span * float(i) / float(n_steps) for i in range(n_steps + 1)]

        from src.model.exp.frecas_sampling import apply_frequency_aware_scaling

        if capture_mid_attn and mid_attn_mgr is not None:
            mid_attn_mgr.enable_capture()

        for i in range(n_steps):
            t_scalar = float(times[i])
            t_next_scalar = float(times[i + 1])
            dt = t_next_scalar - t_scalar
            t = torch.full((c.shape[0],), t_scalar, device=c.device)

            v = model._predict_velocity(
                z,
                t,
                c,
                coords=coords,
                damage_mask=condition_mask,
                gru_feat=gru_feat,
                self_cond=self_cond,
                lr_cond=None,
            )
            if apply_hf_scaling:
                v = apply_frequency_aware_scaling(
                    v, w_h=w_h, pool_kernel=hf_pool_kernel
                )

            z1_hat = (
                model._estimate_clean_from_velocity(z, t, v).detach()
                if model.use_self_conditioning
                else None
            )

            if model.cfm_solver == "heun":
                z_pred = z + dt * v
                t_next = torch.full((c.shape[0],), t_next_scalar, device=c.device)
                v_next = model._predict_velocity(
                    z_pred,
                    t_next,
                    c,
                    coords=coords,
                    damage_mask=condition_mask,
                    gru_feat=gru_feat,
                    self_cond=self_cond,
                    lr_cond=None,
                )
                if apply_hf_scaling:
                    v_next = apply_frequency_aware_scaling(
                        v_next, w_h=w_h, pool_kernel=hf_pool_kernel
                    )
                z = z + 0.5 * dt * (v + v_next)
            else:
                z = z + dt * v

            if model.use_self_conditioning:
                self_cond = z1_hat

        if capture_mid_attn and mid_attn_mgr is not None:
            mid_attn_mgr.disable_capture()

        return z


    def sample_with_frecas(
        self,
        c,
        z0=None,
        num_steps=None,
        gru_feat=None,
        damage_mask=None,
        stage1_model=None,
        stage2_model=None,
        stage1_stop_index=None,
        s1=None,
        s2=None,
        gamma=None,
        w_h=None,
        w_c=None,
        frecas_pool_kernel=None,
    ):
        """FreCaS two-stage training-free sampling: ``s1`` → upsample → ``s2``.

        Stage 1 runs on ``stage1_model`` until schedule index ``L``.  The latent ``z_L``
        is bilinearly upsampled to ``s2`` and Stage 2 continues from SNR-shifted time ``F``.

        **Single-model mode** (``frecas_single_model=True``): the same pretrained weights
        run Stage 1 at ``s1`` and Stage 2 at ``s2`` — the core FreCaS setup for breaking
        resolution limits without a separate HR checkpoint.

        Applies frequency-aware velocity scaling (``w_h``) and Mid-block attention fusion
        (``w_c``) in Stage 2.  Conditional-only — no CFG.
        """
        from src.model.exp.frecas_sampling import (
            MidBlockAttentionManager,
            build_alphas_cumprod_from_flow,
            calculate_shifted_timestep,
            flow_time_from_schedule_index,
        )

        if num_steps is None:
            num_steps = self.cfm_ode_steps
        num_steps = max(1, int(num_steps))

        if s1 is None:
            s1 = int(getattr(self, "frecas_stage1_size", 64))
        if s2 is None:
            s2 = int(getattr(self, "frecas_stage2_size", 128))
        if gamma is None:
            gamma = float(getattr(self, "frecas_gamma", 2.0))
        if w_h is None:
            w_h = float(getattr(self, "frecas_w_h", 2.0))
        if w_c is None:
            w_c = float(getattr(self, "frecas_w_c", 0.6))
        if frecas_pool_kernel is None:
            frecas_pool_kernel = int(getattr(self, "frecas_pool_kernel", 3))

        if stage1_stop_index is None:
            cfg_stop = int(getattr(self, "frecas_stage1_stop_index", -1))
            stage1_stop_index = num_steps // 2 if cfg_stop < 0 else cfg_stop
        L = int(stage1_stop_index)
        if L < 0 or L >= num_steps:
            raise ValueError(f"stage1_stop_index L must be in [0, {num_steps - 1}], got {L}")

        s1 = int(s1)
        s2 = int(s2)
        gamma = float(gamma)
        w_h = float(w_h)
        w_c = float(w_c)

        if stage2_model is None:
            stage2_model = self
        single_model = bool(getattr(self, "frecas_single_model", False))
        if stage1_model is None:
            if single_model:
                stage1_model = self
                stage2_model = self
            elif hasattr(self, "_get_cascade_stage1"):
                stage1_model = self._get_cascade_stage1()
            if stage1_model is None and int(self.model_spatial_size) == s1:
                stage1_model = self
            if stage1_model is None:
                raise ValueError(
                    "stage1_model is required when no cascade Stage-1 is attached "
                    f"and self.model_spatial_size={self.model_spatial_size} != s1={s1}"
                )
        if single_model or stage1_model is stage2_model:
            stage2_model = stage1_model
            single_model = True

        if int(stage1_model.model_spatial_size) != s1:
            raise ValueError(
                f"stage1_model spatial size {stage1_model.model_spatial_size} != s1={s1}"
            )
        if not single_model and int(stage2_model.model_spatial_size) != s2:
            raise ValueError(
                f"stage2_model spatial size {stage2_model.model_spatial_size} != s2={s2}"
            )

        stage1_model.eval()
        stage2_model.eval()

        if z0 is None:
            z0 = torch.randn(
                (c.shape[0], stage1_model.model_in_channels, s1, s1),
                device=c.device,
            )
        else:
            z0 = z0.to(device=c.device)
            if tuple(z0.shape[-2:]) != (s1, s1):
                z0 = F.interpolate(z0, size=(s1, s1), mode="bilinear", align_corners=False)

        dm_s1 = damage_mask
        if dm_s1 is not None:
            dm_s1 = dm_s1.to(device=c.device, dtype=torch.float32)
            if dm_s1.ndim == 3:
                dm_s1 = dm_s1.unsqueeze(1)
            if tuple(dm_s1.shape[-2:]) != (s1, s1):
                dm_s1 = F.interpolate(dm_s1, size=(s1, s1), mode="nearest")

        t_stop = flow_time_from_schedule_index(L, num_steps)
        stage1_mid_mgr = MidBlockAttentionManager(stage1_model.velocity_net.mid_attn)
        z_L = self._frecas_ode_segment(
            stage1_model,
            z0.clone(),
            c,
            gru_feat,
            t_start=0.0,
            t_end=t_stop,
            num_steps=num_steps,
            damage_mask=dm_s1,
            capture_mid_attn=True,
            mid_attn_mgr=stage1_mid_mgr,
        )
        stage1_attn, stage1_hw = stage1_mid_mgr.pop_captured()
        stage1_mid_mgr.clear_reference()

        z_F = F.interpolate(z_L, size=(s2, s2), mode="bilinear", align_corners=False)

        alphas_cumprod = build_alphas_cumprod_from_flow(num_steps, device=c.device)
        F_idx = calculate_shifted_timestep(
            L, s1=s1, s2=s2, gamma=gamma, alphas_cumprod=alphas_cumprod
        )
        t_start_s2 = flow_time_from_schedule_index(F_idx, num_steps)

        dm_s2 = damage_mask
        if dm_s2 is not None:
            dm_s2 = dm_s2.to(device=c.device, dtype=torch.float32)
            if dm_s2.ndim == 3:
                dm_s2 = dm_s2.unsqueeze(1)
            if tuple(dm_s2.shape[-2:]) != (s2, s2):
                dm_s2 = F.interpolate(dm_s2, size=(s2, s2), mode="nearest")

        stage2_mid_mgr = MidBlockAttentionManager(stage2_model.velocity_net.mid_attn)
        if stage1_attn is not None:
            n_down = len(stage2_model.velocity_net.downsamples)
            mid_side_s2 = max(1, s2 // (2 ** n_down))
            stage2_mid_mgr.set_reference(
                stage1_attn,
                stage1_hw,
                w_c=w_c,
                dst_hw=(mid_side_s2, mid_side_s2),
            )

        try:
            z_out = self._frecas_ode_segment(
                stage2_model,
                z_F,
                c,
                gru_feat,
                t_start=t_start_s2,
                t_end=1.0,
                num_steps=num_steps,
                damage_mask=dm_s2,
                apply_hf_scaling=True,
                w_h=w_h,
                hf_pool_kernel=frecas_pool_kernel,
            )
        finally:
            stage2_mid_mgr.clear_reference()

        return z_out
