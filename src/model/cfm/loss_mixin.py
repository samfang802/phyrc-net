"""OT-CFM training loss helpers."""

import torch
import torch.nn.functional as F


class CFMLossMixin:
    def _compute_velocity_min_snr_weight(self, t):
        """Min-SNR-gamma weighting for velocity prediction loss."""
        t_clamped = t.clamp(min=1e-5, max=1.0 - 1e-5)
        snr = (t_clamped / (1.0 - t_clamped)) ** 2
        return torch.clamp(snr, max=self.min_snr_gamma) / (snr + 1.0)


    def _cfm_mse(self, pred, target, valid_mask=None):
        unreduced_loss = F.mse_loss(pred, target, reduction="none")
        if valid_mask is None:
            return unreduced_loss.mean(dim=(1, 2, 3))
        valid_mask = valid_mask.to(device=unreduced_loss.device, dtype=unreduced_loss.dtype)
        if valid_mask.ndim == 3:
            valid_mask = valid_mask.unsqueeze(1)
        if valid_mask.shape[-2:] != unreduced_loss.shape[-2:]:
            valid_mask = F.interpolate(valid_mask, size=unreduced_loss.shape[-2:], mode="nearest")
        masked_loss = unreduced_loss * valid_mask
        valid_count = valid_mask.sum(dim=(1, 2, 3)).clamp_min(1e-8)
        return masked_loss.sum(dim=(1, 2, 3)) / (valid_count * unreduced_loss.shape[1])


    def _gamma_fm_weights(self, z1):
        """Per-sample γ from patch variance of z1 (complexity proxy).

        γ̂_b = Var_{C,H,W}(z1_b). Batch min-max → [eps, 1], then divide by mean
        so the *average* loss scale stays ~1 (does not silently change lr).

        Returns:
            weight: [B] float tensor, detached. Broadcasts onto per-sample MSE [B].
        """
        eps = float(getattr(self, "gamma_fm_eps", 0.1))
        # [B, C, H, W] → [B]；與像素 MSE 無關，不回傳梯度。
        gamma = z1.detach().flatten(1).var(dim=1, unbiased=False)
        gmin = gamma.min()
        gmax = gamma.max()
        # 整批幾乎同質（例如全是 DI≈0）時 min-max 無意義，退回均勻權重。
        if float((gmax - gmin).item()) < 1e-8:
            weight = torch.ones_like(gamma)
        else:
            # [B] in [0, 1]
            norm = (gamma - gmin) / (gmax - gmin)
            weight = eps + (1.0 - eps) * norm
        # 保持 E[weight]=1，避免與 min-SNR 疊加後整體 loss 被放大/縮小。
        weight = weight / weight.mean().clamp_min(1e-8)
        self._last_gamma_fm_weight = float(weight.mean().detach())
        self._last_gamma_fm_std = float(weight.std(unbiased=False).detach())
        return weight


    def _estimate_clean_from_velocity(self, z_t, t, v_t):
        """CFM identity: z_1 = z_t + (1 - t) * v  when v predicts (z_1 - z_0)."""
        t_ = t.view(-1, *([1] * (z_t.ndim - 1)))
        return z_t + (1.0 - t_) * v_t


    @staticmethod
    def _gray_luma(x):
        """[B,C,H,W] → [B,1,H,W] mean over channels (grayscale-expanded RGB OK)."""
        return x.mean(dim=1, keepdim=True)


    @staticmethod
    def _sobel_grad_mag(x):
        """Sobel edge magnitude from GT / prediction (no external crack mask).

        Args:
            x: [B, C, H, W] in model space (e.g. [-1, 1]).
        Returns:
            [B, 1, H, W] edge magnitude.
        """
        gray = CFMLossMixin._gray_luma(x)
        device, dtype = gray.device, gray.dtype
        # Sobel kernels
        kx = torch.tensor(
            [[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]],
            device=device,
            dtype=dtype,
        ).view(1, 1, 3, 3)
        ky = torch.tensor(
            [[-1.0, -2.0, -1.0], [0.0, 0.0, 0.0], [1.0, 2.0, 1.0]],
            device=device,
            dtype=dtype,
        ).view(1, 1, 3, 3)
        gx = F.conv2d(gray, kx, padding=1)
        gy = F.conv2d(gray, ky, padding=1)
        return torch.sqrt(gx * gx + gy * gy + 1e-8)


    @staticmethod
    def _highpass_residual(x, kernel_size=3):
        """x - box-blur(x); captures mid/high spatial frequencies from the image itself."""
        k = int(kernel_size)
        if k < 3:
            k = 3
        if k % 2 == 0:
            k += 1
        pad = k // 2
        # Reflect pad then avg-pool with stride 1 to avoid zero-padding dark rings.
        x_pad = F.pad(x, (pad, pad, pad, pad), mode="reflect")
        blur = F.avg_pool2d(x_pad, kernel_size=k, stride=1)
        return x - blur


    def _late_t_hf_aux_loss(self, z_t, t, v_pred, z1, valid_mask=None):
        """Late-t image aux on OT clean estimate (does not change u_target).

            x̂₁ = z_t + (1 - t) · v_θ
            L = edge_match(x̂₁, z1) + γ · highpass_match(x̂₁, z1)
            only for samples with t ≥ hf_aux_t_min

        Uses gradients of the GT image itself — not spall/crack masks.
        """
        if not bool(getattr(self, "use_hf_aux_loss", False)):
            self._last_hf_aux_loss = 0.0
            return z_t.new_zeros(())
        # Pixel-space only; latent CFM would need a differentiable decode.
        if getattr(self, "vae", None) is not None:
            self._last_hf_aux_loss = 0.0
            return z_t.new_zeros(())

        t_min = float(getattr(self, "hf_aux_t_min", 0.7))
        edge_w = float(getattr(self, "hf_aux_edge_weight", 1.0))
        hp_w = float(getattr(self, "hf_aux_hp_weight", 0.5))
        hp_k = int(getattr(self, "hf_aux_hp_kernel", 3))

        gate = (t.detach() >= t_min).to(dtype=z_t.dtype)
        if float(gate.sum().item()) < 1.0:
            self._last_hf_aux_loss = 0.0
            # Keep graph-friendly zero tied to v_pred.
            return (v_pred * 0.0).mean()

        x1_hat = self._estimate_clean_from_velocity(z_t, t, v_pred)

        # Per-sample spatial losses → [B]
        def _masked_mean(diff):
            # diff: [B,C,H,W]
            if valid_mask is None:
                return diff.mean(dim=(1, 2, 3))
            vm = valid_mask.to(device=diff.device, dtype=diff.dtype)
            if vm.ndim == 3:
                vm = vm.unsqueeze(1)
            if vm.shape[-2:] != diff.shape[-2:]:
                vm = F.interpolate(vm, size=diff.shape[-2:], mode="nearest")
            denom = (vm.sum(dim=(1, 2, 3)) * diff.shape[1]).clamp_min(1e-8)
            return (diff * vm).sum(dim=(1, 2, 3)) / denom

        loss = x1_hat.new_zeros(x1_hat.shape[0])
        if edge_w > 0.0:
            e_hat = self._sobel_grad_mag(x1_hat)
            e_gt = self._sobel_grad_mag(z1)
            loss = loss + edge_w * _masked_mean((e_hat - e_gt).abs())
        if hp_w > 0.0:
            hp_hat = self._highpass_residual(x1_hat, kernel_size=hp_k)
            hp_gt = self._highpass_residual(z1, kernel_size=hp_k)
            loss = loss + hp_w * _masked_mean((hp_hat - hp_gt).abs())

        gated = (loss * gate).sum() / gate.sum().clamp_min(1.0)
        self._last_hf_aux_loss = float(gated.detach())
        return gated


    def _ar_self_cond_drop_prob_at_step(self, step=None):
        """Linear curriculum for AR self-cond dropout.

        drop_prob: probability of zeroing self_cond (exposure-bias mitigation).
          step 0      -> ar_drop_prob_start (default 0.50)
          step end    -> ar_drop_prob_end   (default 0.15)
          step > end  -> ar_drop_prob_end
        """
        if step is None:
            step = int(self.global_step) if getattr(self, "_trainer", None) is not None else 0
        step = int(step)
        p0 = float(getattr(self, "ar_drop_prob_start", 0.5))
        p1 = float(getattr(self, "ar_drop_prob_end", 0.15))
        end = max(1, int(getattr(self, "ar_drop_prob_end_step", 80000)))
        if step <= 0:
            return p0
        if step >= end:
            return p1
        return p0 + (p1 - p0) * (float(step) / float(end))


    def _refiner_draw_full(self):
        """One full ``refiner_roll_nfe`` rollout with prob ``refiner_roll_prob`` (always in eval)."""
        return (not self.training) or (
            float(torch.rand(1).item()) < float(self.refiner_roll_prob)
        )


    @torch.no_grad()
    def _refiner_base_sample(self, info, hs, dr, di, mask, full):
        """Frozen single-frame model sample for the frame described by (hs, dr, di)."""
        self._attach_refiner_base()
        base = self._refiner_base_ref[0]
        c_b, g_b = base._unpack_conditions(info, hs, dr, di)
        solver = base.cfm_solver
        try:
            base.cfm_solver = "heun" if full else "euler"
            steps = max(1, int(self.refiner_roll_nfe) // 2) if full else 1
            return base.sample_with_ode(c_b, gru_feat=g_b, num_steps=steps, damage_mask=mask)
        finally:
            base.cfm_solver = solver


    @torch.no_grad()
    def _refiner_state_step(self, state, c, g, mask, full, gru_delta=None):
        """State track: this network from noise, ``state`` locked, returns ``clamp(state + residual)``."""
        solver = self.cfm_solver
        try:
            self.cfm_solver = "heun" if full else "euler"
            steps = max(1, int(self.refiner_roll_nfe) // 2) if full else 1
            return self.sample_with_ode(
                c, gru_feat=g, num_steps=steps, damage_mask=mask, lr_cond=state, gru_delta=gru_delta
            )
        finally:
            self.cfm_solver = solver


    @torch.no_grad()
    def _refiner_base_prev_state(self, ar_pack, info, mask_combined):
        """S_{t-1} as the frozen single-frame model would produce it, not the real photo.

        Returns None if the batch carries no previous-frame conditions.
        """
        hs = ar_pack.get("hidden_state_prev")
        dr = ar_pack.get("direction_prev")
        di = ar_pack.get("DI_prev")
        if hs is None or dr is None or di is None:
            return None
        full = self._refiner_draw_full()
        self._last_refiner_full_rollout = float(full)
        return self._refiner_base_sample(info, hs, dr, di, mask_combined, full)


    def refiner_residual_flow_matching_loss(
        self,
        z_target,
        ar_pack,
        info,
        c_target,
        gru_feat_target=None,
        valid_mask=None,
        mask_combined=None,
    ):
        """Residual refiner with a locked state channel and a stop-gradient state track.

        State track (no grad): build ``S_{t-1}``. Single hop uses the frozen single-frame
        model for frame t-1. Chain (prob ``refiner_chain_prob``) draws frame t-2 from that
        model, then lets this network produce ``S_{t-1}`` from it, so the state carries the
        refiner's own error. One Euler step usually, a ``refiner_roll_nfe`` Heun rollout with
        prob ``refiner_roll_prob``.

        Loss track: noise ``x0``, target residual ``r = x_t - S_{t-1}``,
        ``z_τ = (1-τ) x0 + τ r``, ``v = r - x0``, with ``S_{t-1}`` as an input channel.
        """
        b = z_target.shape[0]
        device = z_target.device
        dtype = z_target.dtype

        def _flag(key):
            v = ar_pack.get(key)
            if v is None:
                return torch.zeros(b, dtype=torch.bool, device=device)
            return v.reshape(b).to(device=device) >= 0.5

        has_prev = _flag("has_prev")
        has_prev2 = _flag("has_prev2") & has_prev
        hs1, dr1, di1 = (
            ar_pack["hidden_state_prev"],
            ar_pack["direction_prev"],
            ar_pack["DI_prev"],
        )
        use_chain = (not self.training) or (
            float(torch.rand(1).item()) < float(self.refiner_chain_prob)
        )
        chain_ok = use_chain and ar_pack.get("hidden_state_prev2") is not None
        self._last_refiner_chain = float(chain_ok)

        with torch.no_grad():
            if chain_ok:
                def _pick(a2, a1):
                    return torch.where(has_prev2.view(b, *([1] * (a1.ndim - 1))), a2, a1)

                hs0 = _pick(ar_pack["hidden_state_prev2"], hs1)
                dr0 = _pick(ar_pack["direction_prev2"], dr1)
                di0 = _pick(ar_pack["DI_prev2"], di1)
                s0 = self._refiner_base_sample(
                    info, hs0, dr0, di0, mask_combined, self._refiner_draw_full()
                )
                c1, g1 = self._unpack_conditions(info, hs1, dr1, di1)
                dg_hop = None
                if self.use_delta_g:
                    dg_hop = g1 - self.process_physics_condition(hs0)
                s1 = self._refiner_state_step(
                    s0, c1, g1, mask_combined, self._refiner_draw_full(), gru_delta=dg_hop
                )
                s_in = torch.where(has_prev2.view(b, 1, 1, 1), s1, s0)
            else:
                s_in = self._refiner_base_sample(
                    info, hs1, dr1, di1, mask_combined, self._refiner_draw_full()
                )
            s_in = s_in.to(dtype=dtype)

        noise_max = float(self.refiner_state_noise_max) if self.training else 0.0
        state_std = torch.rand(b, device=device, dtype=dtype) * noise_max
        s_cond = s_in + state_std.view(b, 1, 1, 1) * torch.randn_like(s_in)

        resid = z_target - s_in
        z0 = torch.randn_like(resid)
        t = torch.rand(b, device=device, dtype=dtype).clamp(min=1e-5, max=1.0 - 1e-5)
        coords = None
        if self.use_patch_diffusion:
            resid, valid_mask, coords, mask_combined, z0, s_cond = self._apply_patch_crop(
                resid,
                valid_mask=valid_mask,
                condition_mask=mask_combined,
                z0=z0,
                z_aux=s_cond,
                flow_t=t,
            )
        t_ = t.view(b, 1, 1, 1)
        zt = (1.0 - t_) * z0 + t_ * resid
        target_u = resid - z0

        dg_target = None
        if self.use_delta_g:
            dg_target = gru_feat_target - self.process_physics_condition(hs1)
        v_pred = self._predict_velocity(
            zt,
            t,
            c_target,
            coords=coords,
            damage_mask=mask_combined,
            gru_feat=gru_feat_target,
            lr_cond=s_cond,
            gru_delta=dg_target,
        )
        per_sample_loss = self._cfm_mse(v_pred, target_u, valid_mask=valid_mask)
        if self.use_velocity_min_snr_weighting:
            per_sample_loss = per_sample_loss * self._compute_velocity_min_snr_weight(t).to(
                per_sample_loss.dtype
            )
        weight = has_prev.to(per_sample_loss.dtype)
        self._last_refiner_resid_std = float(resid.detach().std())
        return (per_sample_loss * weight).sum() / weight.sum().clamp_min(1.0)


    def paired_bridge_flow_matching_loss(
        self,
        z_target,
        z_prev,
        c_target,
        gru_feat_target=None,
        valid_mask=None,
        mask_combined=None,
        has_prev=None,
    ):
        """Straight bridge between one real pair. First frames stay noise-to-image.

        With a previous frame, training draws ``s ~ U(0, noise_max)`` and

            x0 = (1 - s) * x_prev + s * ε
            z_t = (1 - t) * x0 + t * x_next
            v = x_next - x0

        ``s = 0`` at validation. Samples with no previous frame use ``x0 ~ N(0, I)``.
        """
        if z_prev.shape != z_target.shape:
            raise ValueError(
                f"z_prev shape {tuple(z_prev.shape)} must match z_target {tuple(z_target.shape)}"
            )
        b = z_target.shape[0]
        device = z_target.device
        dtype = z_target.dtype
        if has_prev is None:
            use_bridge = torch.ones(b, device=device, dtype=torch.bool)
        else:
            use_bridge = has_prev.reshape(b).to(device=device) >= 0.5
        self._last_bridge_frac = float(use_bridge.float().mean().item())

        flow_t = torch.rand(b, device=device, dtype=dtype).clamp(min=1e-5, max=1.0 - 1e-5)
        coords = None
        if self.use_patch_diffusion:
            z_target, valid_mask, coords, mask_combined, z_prev = self._apply_patch_crop(
                z_target,
                valid_mask=valid_mask,
                condition_mask=mask_combined,
                z_aux=z_prev,
                flow_t=flow_t,
            )

        noise_max = float(self.paired_bridge_noise_max) if self.training else 0.0
        s = torch.rand(b, device=device, dtype=dtype) * noise_max
        s = s * use_bridge.to(dtype)
        s_ = s.view(b, *([1] * (z_target.ndim - 1)))
        eps = torch.randn_like(z_target)
        x0_bridge = (1.0 - s_) * z_prev.detach() + s_ * eps
        x0_noise = torch.randn_like(z_target)
        use_bridge_ = use_bridge.view(b, *([1] * (z_target.ndim - 1)))
        x0 = torch.where(use_bridge_, x0_bridge, x0_noise)
        t_ = flow_t.view(b, *([1] * (z_target.ndim - 1)))
        z_t = (1.0 - t_) * x0 + t_ * z_target
        target_u = z_target - x0

        v_pred = self._predict_velocity(
            z_t,
            flow_t,
            c_target,
            coords=coords,
            damage_mask=mask_combined,
            gru_feat=gru_feat_target,
        )
        per_sample_loss = self._cfm_mse(v_pred, target_u, valid_mask=valid_mask)
        if self.use_velocity_min_snr_weighting:
            min_snr_weight = self._compute_velocity_min_snr_weight(flow_t).to(per_sample_loss.dtype)
            per_sample_loss = per_sample_loss * min_snr_weight
        return per_sample_loss.mean()


    def ar_temporal_flow_matching_loss(
        self,
        z_target,
        z_prev,
        c_target,
        gru_feat_target=None,
        valid_mask=None,
        mask_combined=None,
        has_prev=None,
        z0=None,
    ):
        """Single-stage AR OT-CFM: teacher-force noisy GT ``z_prev`` as self_cond.

        Path: ``z0 ~ N(0,I) → z_target``. Spatial self_cond is ``z_prev + ε`` with
        per-sample Gaussian std in ``[ar_noise_std_min, ar_noise_std_max]``, then
        curriculum-dropped to zeros (and always dropped when ``has_prev`` is False).
        Only TARGET AdaGN / GRU conditions are used (no prev-physics draft pass).
        """
        if z0 is None:
            z0 = torch.randn_like(z_target)
        if z_prev.shape != z_target.shape:
            raise ValueError(
                f"z_prev shape {tuple(z_prev.shape)} must match z_target {tuple(z_target.shape)}"
            )

        b = z_target.shape[0]
        flow_t = torch.rand(b, device=z_target.device).clamp(min=1e-5, max=1.0 - 1e-5)

        coords = None
        if self.use_patch_diffusion:
            z_target, valid_mask, coords, mask_combined, z0, z_prev = self._apply_patch_crop(
                z_target,
                valid_mask=valid_mask,
                condition_mask=mask_combined,
                z0=z0,
                z_aux=z_prev,
                flow_t=flow_t,
            )

        # OT path noise → z_target
        z_t = (1.0 - flow_t[:, None, None, None]) * z0 + flow_t[:, None, None, None] * z_target
        target_u = z_target - z0

        # --- self_cond = noisy GT previous latent ---
        self_cond = z_prev.detach().clone()
        std_min = float(getattr(self, "ar_noise_std_min", 0.05))
        std_max = float(getattr(self, "ar_noise_std_max", 0.10))
        if std_max < std_min:
            raise ValueError(f"ar_noise_std_max ({std_max}) must be >= ar_noise_std_min ({std_min})")
        # Per-sample std in [std_min, std_max], broadcast over C,H,W
        noise_std = torch.empty(b, device=z_target.device, dtype=z_target.dtype).uniform_(std_min, std_max)
        noise_std = noise_std.view(b, *([1] * (self_cond.ndim - 1)))
        self_cond = self_cond + torch.randn_like(self_cond) * noise_std

        # Curriculum drop + first-frame force-drop
        drop_prob = self._ar_self_cond_drop_prob_at_step()
        self._last_self_cond_prob = float(1.0 - drop_prob)
        self._last_ar_drop_prob = float(drop_prob)
        drop_mask = torch.rand(b, device=z_target.device) < drop_prob
        if has_prev is not None:
            hp = has_prev.reshape(b).to(device=z_target.device)
            drop_mask = drop_mask | (hp < 0.5)
        if drop_mask.any():
            self_cond = self_cond.clone()
            self_cond[drop_mask] = 0.0

        # Single forward under TARGET physics only
        v_pred = self._predict_velocity(
            z_t,
            flow_t,
            c_target,
            coords=coords,
            damage_mask=mask_combined,
            gru_feat=gru_feat_target,
            self_cond=self_cond,
        )

        per_sample_loss = self._cfm_mse(
            v_pred, target_u, valid_mask=valid_mask
        )
        if self.use_velocity_min_snr_weighting:
            min_snr_weight = self._compute_velocity_min_snr_weight(flow_t).to(per_sample_loss.dtype)
            per_sample_loss = per_sample_loss * min_snr_weight
        return per_sample_loss.mean()


    def ar_chain_flow_matching_loss(
        self,
        z_target,
        z_prev,
        c_target,
        gru_feat_target=None,
        valid_mask=None,
        mask_combined=None,
        has_prev=None,
        z0=None,
    ):
        """Train OT-CFM to match inference DI/frame chain (img2img).

        For samples with a previous frame (aligned with ``sample_with_ode`` chain)::

            z_start = (1 - α)·ε + α·z_prev          # α = chain_t_start
            τ ~ U(α, 1)
            s = (τ - α) / (1 - α)
            z_τ = (1 - s)·z_start + s·z_target
            u = (z_target - z_start) / (1 - α)       # dz/dτ under absolute time τ

        First frames (no prev) fall back to the standard noise → z_target path.
        TARGET physics only; self_cond is zeros if the channel exists.
        """
        if z_prev.shape != z_target.shape:
            raise ValueError(
                f"z_prev shape {tuple(z_prev.shape)} must match z_target {tuple(z_target.shape)}"
            )
        if z0 is None:
            z0 = torch.randn_like(z_target)

        b = z_target.shape[0]
        device = z_target.device
        dtype = z_target.dtype
        alpha = float(getattr(self, "chain_t_start", 0.5))
        if not (0.0 < alpha < 1.0):
            raise ValueError(f"chain_t_start must be in (0,1), got {alpha}")
        self._last_chain_t_start = alpha

        # Per-sample: True → chain path; False → standard noise→target.
        if has_prev is not None:
            use_chain = has_prev.reshape(b).to(device=device) >= 0.5
        else:
            use_chain = torch.ones(b, device=device, dtype=torch.bool)

        s_rand = torch.rand(b, device=device, dtype=dtype).clamp(min=1e-5, max=1.0 - 1e-5)
        tau_chain = alpha + (1.0 - alpha) * s_rand
        tau_noise = torch.rand(b, device=device, dtype=dtype).clamp(min=1e-5, max=1.0 - 1e-5)
        tau = torch.where(use_chain, tau_chain, tau_noise)

        coords = None
        if self.use_patch_diffusion:
            z_target, valid_mask, coords, mask_combined, z0, z_prev = self._apply_patch_crop(
                z_target,
                valid_mask=valid_mask,
                condition_mask=mask_combined,
                z0=z0,
                z_aux=z_prev,
                flow_t=tau,
            )

        eps = z0  # reuse OT noise
        z_start = (1.0 - alpha) * eps + alpha * z_prev.detach()

        # Progress along chain segment (only meaningful when use_chain).
        s = ((tau - alpha) / (1.0 - alpha)).clamp(min=0.0, max=1.0)
        s_ = s.view(b, *([1] * (z_target.ndim - 1)))
        tau_ = tau.view(b, *([1] * (z_target.ndim - 1)))

        z_t_chain = (1.0 - s_) * z_start + s_ * z_target
        u_chain = (z_target - z_start) / (1.0 - alpha)

        z_t_noise = (1.0 - tau_) * eps + tau_ * z_target
        u_noise = z_target - eps

        use_chain_ = use_chain.view(b, *([1] * (z_target.ndim - 1)))
        z_t = torch.where(use_chain_, z_t_chain, z_t_noise)
        target_u = torch.where(use_chain_, u_chain, u_noise)

        self_cond = None
        if self.use_self_conditioning:
            self_cond = torch.zeros_like(z_t)

        v_pred = self._predict_velocity(
            z_t,
            tau,
            c_target,
            coords=coords,
            damage_mask=mask_combined,
            gru_feat=gru_feat_target,
            self_cond=self_cond,
        )

        per_sample_loss = self._cfm_mse(
            v_pred, target_u, valid_mask=valid_mask
        )
        if self.use_velocity_min_snr_weighting:
            # Weight by absolute flow time τ (same schedule as standard CFM).
            min_snr_weight = self._compute_velocity_min_snr_weight(tau).to(per_sample_loss.dtype)
            per_sample_loss = per_sample_loss * min_snr_weight
        return per_sample_loss.mean()


    def flow_matching_loss(
        self,
        z1,
        c,
        valid_mask=None,
        condition_mask=None,
        gru_feat=None,
        z0=None,
    ):
        """OT-CFM path loss (single-scale velocity MSE; no deep supervision)."""
        if z0 is None:
            z0 = torch.randn_like(z1)

        gru_feat = self._drop_gru_feat_cfg(gru_feat)

        lr_cond = None
        if self.use_cascade_sr:
            z1, lr_cond = self._prepare_cascade_target(z1)

        b = z1.shape[0]
        t = torch.rand(b, device=z1.device).clamp(min=1e-5, max=1.0 - 1e-5)

        coords = None
        if self.use_patch_diffusion:
            if lr_cond is not None:
                z1, valid_mask, coords, condition_mask, z0, lr_cond = self._apply_patch_crop(
                    z1,
                    valid_mask=valid_mask,
                    condition_mask=condition_mask,
                    z0=z0,
                    z_aux=lr_cond,
                    flow_t=t,
                )
            else:
                z1, valid_mask, coords, condition_mask, z0 = self._apply_patch_crop(
                    z1,
                    valid_mask=valid_mask,
                    condition_mask=condition_mask,
                    z0=z0,
                    flow_t=t,
                )

        zt = (1.0 - t[:, None, None, None]) * z0 + t[:, None, None, None] * z1
        target_u = z1 - z0

        self_cond = None
        if self.use_self_conditioning:
            # Batch-level Bernoulli with step-dependent schedule.
            p_self = self._self_cond_prob_at_step()
            self._last_self_cond_prob = float(p_self)
            use_self_cond_batch = bool(torch.rand((), device=z1.device) < p_self)
            if use_self_cond_batch:
                # Pass 1: estimate z1 under no_grad (no activation storage / no graph).
                with torch.no_grad():
                    v_est = self._predict_velocity(
                        zt,
                        t,
                        c,
                        coords=coords,
                        damage_mask=condition_mask,
                        gru_feat=gru_feat,
                        self_cond=torch.zeros_like(zt),
                        lr_cond=lr_cond,
                    )
                    z1_pred = self._estimate_clean_from_velocity(zt, t, v_est)
                # Detach so Pass 2 never backprops into the estimate.
                self_cond = z1_pred.detach()
            else:
                self_cond = torch.zeros_like(zt)
        else:
            self._last_self_cond_prob = 0.0

        noise_pred = self._predict_velocity(
            zt,
            t,
            c,
            coords=coords,
            damage_mask=condition_mask,
            gru_feat=gru_feat,
            self_cond=self_cond,
            lr_cond=lr_cond,
        )

        # per_sample_loss: [B] = mean_{C,H,W} ||v_pred - u_target||^2
        hf_boost = float(getattr(self, "wavelet_hf_boost", 0.0))
        if hf_boost > 0.0:
            from src.model.exp.wavelet_unet import wavelet_velocity_loss

            per_sample_loss = wavelet_velocity_loss(
                noise_pred, target_u, valid_mask=valid_mask, hf_boost=hf_boost
            )
        else:
            per_sample_loss = self._cfm_mse(
                noise_pred, target_u, valid_mask=valid_mask
            )

        if bool(getattr(self, "use_gamma_fm_loss", False)):
            # γ: [B]，對齊 [B] MSE。不要乘在 [B,C,H,W] 上再 mean，否則等價但難讀。
            # cascade 時 z1 已是 residual，Var(z1) ≈ 該 crop 要補的高頻量。
            gamma = self._gamma_fm_weights(z1).to(dtype=per_sample_loss.dtype)
            per_sample_loss = per_sample_loss * gamma
        else:
            self._last_gamma_fm_weight = 1.0
            self._last_gamma_fm_std = 0.0

        if self.use_velocity_min_snr_weighting:
            min_snr_weight = self._compute_velocity_min_snr_weight(t).to(per_sample_loss.dtype)
            per_sample_loss = per_sample_loss * min_snr_weight

        loss_fm = per_sample_loss.mean()
        self._last_fm_loss = float(loss_fm.detach())

        # Late-t HF / edge aux on x̂₁ — OT target u unchanged.
        lambda_hf = float(getattr(self, "hf_aux_lambda", 0.0))
        if bool(getattr(self, "use_hf_aux_loss", False)) and lambda_hf > 0.0:
            loss_hf = self._late_t_hf_aux_loss(
                zt, t, noise_pred, z1, valid_mask=valid_mask
            )
            return loss_fm + lambda_hf * loss_hf

        self._last_hf_aux_loss = 0.0
        return loss_fm

    def _prepare_spall_cond(self, z1, z_prev, g_curr, g_prev, coords, has_prev):
        """Mask loss plus detached ``[pool, m]`` channels. Velocity loss weight is separate."""
        import torch.nn.functional as F
        from src.model.cfm.spall_mask import (
            coarse_coords,
            coarse_pool,
            extent_mask,
            feather_mask,
            growth_from_extent,
            neighbor_tv,
            upsample_mask,
        )

        b = z1.shape[0]
        grid = int(self.spall_mask_grid)
        hp = has_prev.reshape(b).to(device=z1.device) >= 0.5
        hp_f = hp.view(b, 1, 1, 1).to(dtype=z1.dtype)
        if bool(getattr(self, "spall_cond_prob", False)):
            loss_m, cond, weight, lock = self._spall_prob_cond(
                z_prev, g_curr, g_prev, coords, has_prev, hp, hp_f
            )
            return loss_m, cond, weight, lock, None

        level = float(self.spall_dark_level)
        # Accumulated extent. Same fixed luminance on both frames, no max-norm.
        m_star = extent_mask(
            z1, grid, level, self.spall_bg_quantile, self.spall_bg_target, self.spall_border_gap,
            seed_margin=float(getattr(self, "spall_seed_margin", 0.2)),
        )
        m_prev = extent_mask(
            z_prev, grid, level, self.spall_bg_quantile, self.spall_bg_target, self.spall_border_gap,
            seed_margin=float(getattr(self, "spall_seed_margin", 0.2)),
        ) * hp_f
        pool = coarse_pool(z_prev, grid) * hp_f
        if self.training and float(self.spall_pool_noise_std) > 0.0:
            pool = (pool + torch.randn_like(pool) * float(self.spall_pool_noise_std)) * hp_f
        g_prev = g_prev * hp.view(b, 1).to(dtype=g_prev.dtype)
        logits = self.spall_mask_head(
            pool, g_prev, g_curr, coarse_coords(coords, grid, pool)
        )
        m_prob = torch.sigmoid(logits)
        # Cells that crossed the dark level this frame. The stable hook is
        # weight 1; these new cells are upweighted so Y cannot ignore them.
        new = ((m_star > 0.5) & (m_prev < 0.5)).to(dtype=logits.dtype)
        cell_w = 1.0 + (float(self.spall_new_cell_weight) - 1.0) * new
        loss_m = F.binary_cross_entropy_with_logits(logits, m_star, weight=cell_w)
        if float(self.spall_tv_weight) > 0.0:
            loss_m = loss_m + float(self.spall_tv_weight) * neighbor_tv(m_prob)
        spatial = z1.shape[-2:]
        if bool(getattr(self, "spall_soft_paint", False)):
            return self._spall_soft_paint_cond(
                z1, z_prev, m_prob, m_prev, hp, hp_f, grid, spatial, loss_m
            )
        # G sees the dilated increment, not the accumulated blob.
        m_grow = growth_from_extent(
            m_prob, m_prev, self.spall_mask_threshold, dilate_k=self.spall_dilate_k
        )
        m_soft = feather_mask(m_grow, spatial, self.spall_feather_px)
        cond = torch.cat(
            [upsample_mask(pool.detach(), spatial), m_soft.detach()],
            dim=1,
        )
        # Old dark cells are out of the velocity loss. New spall, cracks, and
        # the rest of the frame stay in. First frame has no old region.
        lock = upsample_mask(m_prev, spatial)
        weight = 1.0 - lock
        return loss_m, cond, weight, lock, None

    def spall_growth_flow_matching_loss(
        self,
        z1,
        z_prev,
        c,
        gru_feat,
        g_prev,
        valid_mask=None,
        condition_mask=None,
        has_prev=None,
        z0=None,
    ):
        """Velocity loss outside old spall, plus Dice/TV on the accumulated extent."""
        if z0 is None:
            z0 = torch.randn_like(z1)
        if has_prev is None:
            has_prev = torch.ones(z1.shape[0], device=z1.device, dtype=z1.dtype)
        g_for_mask = gru_feat
        gru_feat = self._drop_gru_feat_cfg(gru_feat)

        b = z1.shape[0]
        t = torch.rand(b, device=z1.device).clamp(min=1e-5, max=1.0 - 1e-5)
        coords = None
        if self.use_patch_diffusion:
            z1, valid_mask, coords, condition_mask, z0, z_prev = self._apply_patch_crop(
                z1,
                valid_mask=valid_mask,
                condition_mask=condition_mask,
                z0=z0,
                z_aux=z_prev,
                flow_t=t,
            )

        loss_m, spall_cond, weight, lock, endpoint = self._prepare_spall_cond(
            z1, z_prev, g_for_mask, g_prev, coords, has_prev
        )
        if getattr(self, "spall_mask_only", False):
            self._last_fm_loss = 0.0
            self._last_spall_mask_loss = float(loss_m.detach())
            return loss_m
        if valid_mask is None:
            valid_mask = weight
        else:
            valid_mask = valid_mask * weight

        t_view = t[:, None, None, None]
        if endpoint is None:
            zt = (1.0 - t_view) * z0 + t_view * z1
            from src.model.cfm.spall_mask import repaint_known
            zt = repaint_known(zt, z0, z_prev, t, lock)
            target_u = z1 - z0
        else:
            # Same mixture the ODE writes back: paint toward this frame, lock toward x_prev.
            zt = (1.0 - t_view) * z0 + t_view * endpoint
            target_u = endpoint - z0
        noise_pred = self._predict_velocity(
            zt,
            t,
            c,
            coords=coords,
            damage_mask=condition_mask,
            gru_feat=gru_feat,
            spall_cond=spall_cond,
        )
        per_sample_loss = self._cfm_mse(noise_pred, target_u, valid_mask=valid_mask)
        if self.use_velocity_min_snr_weighting:
            min_snr_weight = self._compute_velocity_min_snr_weight(t).to(per_sample_loss.dtype)
            per_sample_loss = per_sample_loss * min_snr_weight
        loss_fm = per_sample_loss.mean()
        self._last_fm_loss = float(loss_fm.detach())
        self._last_spall_mask_loss = float(loss_m.detach())
        return loss_fm + float(self.spall_loss_weight) * loss_m

    def _spall_soft_paint_cond(self, z1, z_prev, m_prob, m_prev, hp, hp_f, grid, spatial, loss_m):
        """MaskFlow path. ``α`` is the paint weight; locked pixels follow ``x_prev``.

        ``M``'s BCE already used the real previous frame. Only ``G`` sees a
        previous frame pushed a short way toward noise.
        """
        from src.model.cfm.spall_mask import coarse_pool, context_ot, soft_paint_alpha, upsample_mask

        alpha = soft_paint_alpha(
            z_prev,
            m_prob,
            m_prev,
            spatial,
            grid=grid,
            threshold=self.spall_mask_threshold,
            close_k=int(self.spall_close_k),
            bg_quantile=self.spall_bg_quantile,
            bg_target=self.spall_bg_target,
            border_gap=self.spall_border_gap,
            healthy_margin=float(self.spall_healthy_margin),
            sigma=float(self.spall_feather_px),
            lock_healthy=bool(getattr(self, "spall_lock_healthy", True)),
        )
        alpha = alpha * hp_f
        # No previous frame: generate the whole image and show G an empty condition,
        # matching inference when ``spall_prev`` is absent.
        free = torch.where(hp.view(hp.shape[0], 1, 1, 1), alpha, torch.ones_like(alpha))
        ctx = z_prev
        if self.training and float(self.spall_ctx_noise_max) > 0.0:
            ctx = context_ot(z_prev, self.spall_ctx_noise_max)
        ctx = torch.where(hp.view(hp.shape[0], 1, 1, 1), ctx, z_prev)
        cond = torch.cat(
            [
                upsample_mask(coarse_pool(ctx, grid).detach() * hp_f, spatial),
                alpha.detach(),
            ],
            dim=1,
        )
        endpoint = (free * z1 + (1.0 - free) * ctx).detach()
        return loss_m, cond, free.detach(), (1.0 - free).detach(), endpoint

    def _spall_prob_cond(self, z_prev, g_curr, g_prev, coords, has_prev, hp, hp_f):
        """Detached sigmoid of frozen M. No m* and no hard mask."""
        import torch.nn.functional as F
        from src.model.cfm.spall_mask import coarse_coords, coarse_pool, upsample_mask

        b = z_prev.shape[0]
        grid = int(self.spall_mask_grid)
        pool = coarse_pool(z_prev, grid) * hp_f
        if self.training and float(self.spall_pool_noise_std) > 0.0:
            pool = (pool + torch.randn_like(pool) * float(self.spall_pool_noise_std)) * hp_f
        g_prev = g_prev * hp.view(b, 1).to(dtype=g_prev.dtype)
        with torch.no_grad():
            logits = self.spall_mask_head(
                pool, g_prev, g_curr, coarse_coords(coords, grid, pool)
            )
            prob = torch.sigmoid(logits)
        spatial = z_prev.shape[-2:]
        prob_up = F.interpolate(prob, size=spatial, mode="bilinear", align_corners=False)
        cond = torch.cat([upsample_mask(pool.detach(), spatial), prob_up.detach()], dim=1)
        weight = torch.ones(b, 1, spatial[0], spatial[1], device=z_prev.device, dtype=z_prev.dtype)
        loss_m = prob.new_zeros(())
        return loss_m, cond, weight, None
