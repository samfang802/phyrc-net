"""OT-CFM training loss and ODE sampling (including inpainting)."""

import torch
import torch.nn.functional as F


class CFMSamplingMixin:
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
        gray = CFMSamplingMixin._gray_luma(x)
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

        coords = None
        if self.use_patch_diffusion:
            z_target, valid_mask, coords, mask_combined, z0, z_prev = self._apply_patch_crop(
                z_target,
                valid_mask=valid_mask,
                condition_mask=mask_combined,
                z0=z0,
                z_aux=z_prev,
            )

        b = z_target.shape[0]
        # OT path noise → z_target
        flow_t = torch.rand(b, device=z_target.device).clamp(min=1e-5, max=1.0 - 1e-5)
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

        per_sample_loss = self._cfm_mse(v_pred, target_u, valid_mask=valid_mask)
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

        coords = None
        if self.use_patch_diffusion:
            z_target, valid_mask, coords, mask_combined, z0, z_prev = self._apply_patch_crop(
                z_target,
                valid_mask=valid_mask,
                condition_mask=mask_combined,
                z0=z0,
                z_aux=z_prev,
            )

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

        eps = z0  # reuse OT noise
        z_start = (1.0 - alpha) * eps + alpha * z_prev.detach()

        # Absolute time τ on [α,1] for chain; on (0,1) for fallback.
        s_rand = torch.rand(b, device=device, dtype=dtype).clamp(min=1e-5, max=1.0 - 1e-5)
        tau_chain = alpha + (1.0 - alpha) * s_rand
        tau_noise = torch.rand(b, device=device, dtype=dtype).clamp(min=1e-5, max=1.0 - 1e-5)
        tau = torch.where(use_chain, tau_chain, tau_noise)

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

        per_sample_loss = self._cfm_mse(v_pred, target_u, valid_mask=valid_mask)
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

        lr_cond = None
        if self.use_cascade_sr:
            z1, lr_cond = self._prepare_cascade_target(z1)

        coords = None
        if self.use_patch_diffusion:
            if lr_cond is not None:
                z1, valid_mask, coords, condition_mask, z0, lr_cond = self._apply_patch_crop(
                    z1,
                    valid_mask=valid_mask,
                    condition_mask=condition_mask,
                    z0=z0,
                    z_aux=lr_cond,
                )
            else:
                z1, valid_mask, coords, condition_mask, z0 = self._apply_patch_crop(
                    z1, valid_mask=valid_mask, condition_mask=condition_mask, z0=z0
                )

        b = z1.shape[0]
        t = torch.rand(b, device=z1.device).clamp(min=1e-5, max=1.0 - 1e-5)
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
            from src.model.Component.wavelet_unet import wavelet_velocity_loss

            per_sample_loss = wavelet_velocity_loss(
                noise_pred, target_u, valid_mask=valid_mask, hf_boost=hf_boost
            )
        else:
            per_sample_loss = self._cfm_mse(noise_pred, target_u, valid_mask=valid_mask)

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


    @staticmethod
    def _ot_path_state(z0, z1_gt, t_scalar):
        """Linear OT interpolation z_t = (1-t) z0 + t z1_gt."""
        t = float(t_scalar)
        return (1.0 - t) * z0 + t * z1_gt


    @staticmethod
    def _blend_latent_toward_ot_ref(z, z0, z1_ref, t_scalar, guidance_weight):
        """Training-free reference guidance via latent blending (no FFT).

        Pulls the query state toward the deterministic OT reference path
            z_t_ref = (1 - t) * z0 + t * z1_ref
        before the velocity network is evaluated::

            z_query = (1 - α) * z + α * z_t_ref
        """
        alpha = float(guidance_weight)
        if alpha <= 0.0 or z1_ref is None:
            return z
        if alpha > 1.0:
            alpha = 1.0
        z_t_ref = CFMSamplingMixin._ot_path_state(z0, z1_ref, t_scalar)
        return (1.0 - alpha) * z + alpha * z_t_ref


    @staticmethod
    def _blend_velocity_toward_ot_ref(v_theta, z0, z1_ref, guidance_weight):
        """Training-free reference guidance via velocity blending (no FFT).

        For linear OT, the reference velocity is constant along the path:
            v_ref = z1_ref - z0
        Blend with the network velocity::

            v = (1 - α) * v_θ + α * v_ref
        """
        alpha = float(guidance_weight)
        if alpha <= 0.0 or z1_ref is None:
            return v_theta
        if alpha > 1.0:
            alpha = 1.0
        v_ref = z1_ref - z0
        return (1.0 - alpha) * v_theta + alpha * v_ref


    def sample_with_ode(
        self,
        c,
        z0=None,
        num_steps=None,
        gru_feat=None,
        z1_ref=None,
        guidance_weight=None,
        guidance_mode=None,
        t_start=0.0,
        z_init=None,
        lr_cond=None,
        damage_mask=None,
    ):
        """Heun/Euler OT-CFM ODE sampling with optional training-free reference guidance.

        Reference path (shared by both modes):
            z_t_ref = (1 - t) * z0 + t * z1_ref

        Modes (no FFT / frequency filters):
          - ``latent``:   evaluate v_θ at z_query = (1-α)z + α z_t_ref
          - ``velocity``: evaluate v_θ at z, then v = (1-α)v_θ + α(z1_ref - z0)

        Partial / img2img integration:
          - ``t_start=0`` (default): start from noise ``z0``, integrate t: 0 → 1.
          - ``t_start∈(0,1)``: start from ``z_init`` (state at ``t_start``) and
            integrate only t: t_start → 1. Number of Heun steps scales with
            ``(1 - t_start)`` so the last node lands exactly on t=1.

        ``guidance_weight`` / α defaults to ``self.ref_guidance_weight`` (usually 0).
        """
        if num_steps is None:
            num_steps = self.cfm_ode_steps
        if guidance_weight is None:
            guidance_weight = float(getattr(self, "ref_guidance_weight", 0.0))
        else:
            guidance_weight = float(guidance_weight)
        if guidance_mode is None:
            guidance_mode = str(getattr(self, "ref_guidance_mode", "latent")).lower()
        else:
            guidance_mode = str(guidance_mode).lower()
        # Blend modes only; "inversion"/"chain" handled outside (shared z0 / seq).
        if guidance_mode not in ("latent", "velocity", "inversion", "chain", "none"):
            raise ValueError(
                f"guidance_mode must be latent|velocity|inversion|chain|none, got {guidance_mode!r}"
            )
        if guidance_mode in ("inversion", "chain", "none"):
            guidance_weight = 0.0

        t_start = float(t_start)
        if t_start < 0.0 or t_start >= 1.0:
            raise ValueError(f"t_start must be in [0, 1), got {t_start}")

        if t_start <= 0.0:
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
            else:
                z0 = z0.to(device=c.device)
            z = z0.clone()
        else:
            if z_init is None:
                raise ValueError("z_init is required when t_start > 0")
            z = z_init.to(device=c.device, dtype=torch.float32).clone()
            if z.shape[0] != c.shape[0]:
                raise ValueError(
                    f"z_init batch {z.shape[0]} must match condition batch {c.shape[0]}"
                )
            # Keep a placeholder z0 for optional latent/velocity ref blend math.
            if z0 is None:
                z0 = torch.randn_like(z)
            else:
                z0 = z0.to(device=z.device, dtype=z.dtype)

        use_ref = z1_ref is not None and guidance_weight > 0.0
        if use_ref:
            z1_ref = z1_ref.to(device=z.device, dtype=z.dtype)
            if z1_ref.shape[0] == 1 and z.shape[0] > 1:
                z1_ref = z1_ref.expand(z.shape[0], -1, -1, -1).contiguous()
            if z1_ref.shape != z.shape:
                raise ValueError(
                    f"z1_ref shape {tuple(z1_ref.shape)} must match state {tuple(z.shape)} "
                    f"(or broadcast from batch=1)"
                )

        if self.use_cascade_sr:
            lr_cond = self._ensure_cascade_lr_cond(
                lr_cond,
                spatial_hw=z.shape[-2:],
                batch=c.shape[0],
                device=z.device,
                dtype=z.dtype,
            )
        elif lr_cond is not None:
            raise ValueError("lr_cond provided but use_cascade_sr=False")

        coords = None
        if self.use_patch_diffusion:
            grid_x, grid_y = self._build_full_coordinate_maps(
                self.model_spatial_size,
                self.model_spatial_size,
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
        if condition_mask is None and self.use_damage_mask_condition:
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

        # Inference self-conditioning: carry clean estimate across ODE steps.
        # When starting mid-trajectory, seed with zeros (no prior clean estimate).
        self_cond = torch.zeros_like(z) if self.use_self_conditioning else None

        def _guided_velocity(z_state, t_tensor, t_scalar):
            """Predict v_θ then apply latent- or velocity-space reference guidance."""
            if use_ref and guidance_mode == "latent":
                z_in = self._blend_latent_toward_ot_ref(
                    z_state, z0, z1_ref, t_scalar, guidance_weight
                )
            else:
                z_in = z_state
            v = self._predict_velocity(
                z_in,
                t_tensor,
                c,
                coords=coords,
                damage_mask=condition_mask,
                gru_feat=gru_feat,
                self_cond=self_cond,
                lr_cond=lr_cond,
            )
            if use_ref and guidance_mode == "velocity":
                v = self._blend_velocity_toward_ot_ref(
                    v, z0, z1_ref, guidance_weight
                )
            return v, z_in

        # Integrate only over [t_start, 1]; scale step count with remaining span.
        span = 1.0 - t_start
        n_steps = max(1, int(round(float(num_steps) * span)))
        # Exact endpoint t=1 via linspace nodes (avoids drift from repeated dt adds).
        times = [t_start + span * float(i) / float(n_steps) for i in range(n_steps + 1)]

        for i in range(n_steps):
            t_scalar = float(times[i])
            t_next_scalar = float(times[i + 1])
            dt = t_next_scalar - t_scalar
            t = torch.full((c.shape[0],), t_scalar, device=c.device)

            # --- Heun stage 1 ---
            v_t, z_for_clean = _guided_velocity(z, t, t_scalar)
            z1_hat = (
                self._estimate_clean_from_velocity(z_for_clean, t, v_t).detach()
                if self.use_self_conditioning
                else None
            )

            if self.cfm_solver == "heun":
                z_pred = z + dt * v_t
                t_next = torch.full((c.shape[0],), t_next_scalar, device=c.device)
                # --- Heun stage 2 (same self_cond for both evaluations) ---
                v_next, _ = _guided_velocity(z_pred, t_next, t_next_scalar)
                z = z + 0.5 * dt * (v_t + v_next)
            else:
                z = z + dt * v_t

            if self.use_self_conditioning:
                self_cond = z1_hat
        if self.use_cascade_sr:
            z = self._compose_cascade_sample(z, lr_cond)
        return z


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

        from src.model.cfm.frecas_sampling import apply_frequency_aware_scaling

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
        from src.model.cfm.frecas_sampling import (
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
        """Test-time optimize ``gru_features`` (see ``src.model.cfm.tto``)."""
        from src.model.cfm.tto import optimize_gru_features_tto as _tto

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
        z_gt = CFMSamplingMixin._ot_path_state(z0, z1_gt, t_scalar)
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
