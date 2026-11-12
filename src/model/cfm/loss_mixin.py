"""OT-CFM training loss helpers."""

import torch
import torch.nn.functional as F

from src.model.cfm.refiner_memory import inject_memory_errors, memory_error


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
        if mask is not None and mask.shape[1] > 1 and not getattr(base, "layout_cond", False):
            mask = mask[:, :1]  # layout channel only for a layout-conditioned refiner
        solver = base.cfm_solver
        try:
            base.cfm_solver = "heun" if full else "euler"
            steps = max(1, int(self.refiner_roll_nfe) // 2) if full else 1
            return base.sample_with_ode(c_b, gru_feat=g_b, num_steps=steps, damage_mask=mask)
        finally:
            base.cfm_solver = solver


    @torch.no_grad()
    def _refiner_state_step(self, state, c, g, mask, full, gru_delta=None):
        """State track: this network from noise, ``state`` locked, returns ``clamp(state + residual)``.
        With ``refiner_memory`` the hop reads an empty memory (the dropped value)."""
        n_mem = int(getattr(self, "refiner_memory_hint_channels", 0))
        if n_mem:
            state = torch.cat([state, torch.zeros_like(state[:, :1]).expand(-1, n_mem, -1, -1)], dim=1)
        state = self._refiner_conf_hint(state, float(self.refiner_state_conf_value))
        solver = self.cfm_solver
        try:
            self.cfm_solver = "heun" if full else "euler"
            steps = max(1, int(self.refiner_roll_nfe) // 2) if full else 1
            return self.sample_with_ode(
                c, gru_feat=g, num_steps=steps, damage_mask=mask, lr_cond=state, gru_delta=gru_delta
            )
        finally:
            self.cfm_solver = solver


    def _refiner_memory_time_gate(self, hint, t):
        """Zero the memory channels of ``hint`` ([S | memory | confidence]) where flow time ``t`` (float or [B])
        is >= ``refiner_memory_tmax``."""
        n_mem = int(getattr(self, "refiner_memory_hint_channels", 0))
        tmax = float(getattr(self, "refiner_memory_tmax", 1.0))
        # Sampling-time memory schedule (rfmv2): scale_early for tau < scale_t, scale_late after; also in the
        # self-resampling hops, which stand in for sampling.
        lo, hi = float(getattr(self, "refiner_memory_scale_early", 1.0)), float(getattr(self, "refiner_memory_scale_late", 1.0))
        sched = (lo != 1.0 or hi != 1.0) and (not self.training or getattr(self, "_in_hop", False))
        if n_mem <= 0 or hint is None or (tmax >= 1.0 and not sched):
            return hint
        b = hint.shape[0]
        tt = torch.as_tensor(t, device=hint.device).reshape(-1).to(hint.dtype).expand(b)
        keep = (tt < tmax).to(hint.dtype)
        if sched:
            t_sw = float(getattr(self, "refiner_memory_scale_t", 0.5))
            keep = keep * torch.where(tt < t_sw, torch.full_like(tt, lo), torch.full_like(tt, hi))
        c0 = int(self.model_in_channels)
        mem = hint[:, c0:c0 + n_mem] * keep.view(b, 1, 1, 1)
        return torch.cat([hint[:, :c0], mem, hint[:, c0 + n_mem:]], dim=1)

    def _refiner_conf_hint(self, hint, conf, hops=None):
        """Append the S confidence channel (``refiner_state_conf``): ``conf`` is a float or a [B] tensor.
        With ``refiner_state_conf_hops`` a second channel: ``hops`` (float or [B]) self-generated hops behind S,
        min(h, L) / L. Not given: 1 hop where conf < 1 (one resampling hop), 0 for a real frame."""
        if not getattr(self, "refiner_state_conf", False):
            return hint
        b = hint.shape[0]
        if not torch.is_tensor(conf):
            conf = torch.full((b,), float(conf), device=hint.device, dtype=hint.dtype)
        conf = conf.to(device=hint.device, dtype=hint.dtype).reshape(-1).expand(b)
        chans = [conf]
        if getattr(self, "refiner_state_conf_hops", False):
            L = float(max(int(getattr(self, "refiner_window", 0)), 1))
            if hops is None:
                hops = (conf < 1.0).to(hint.dtype) * min(1.0, 1.0 / L)
            elif not torch.is_tensor(hops):
                hops = torch.full((b,), min(float(hops), L) / L, device=hint.device, dtype=hint.dtype)
            else:
                hops = hops.to(device=hint.device, dtype=hint.dtype).reshape(-1).expand(b).clamp(max=L) / L
            chans.append(hops)
        extra = torch.stack(chans, dim=1).view(b, len(chans), 1, 1)
        return torch.cat([hint, extra.expand(-1, -1, hint.shape[-2], hint.shape[-1])], dim=1)

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
        mask_hop = ar_pack.get("mask_hop")
        return self._refiner_base_sample(info, hs, dr, di, mask_combined if mask_hop is None else mask_hop, full)


    def _refiner_state_track(self, ar_pack, info, mask_combined, b, device, dtype):
        """State track (no grad). Returns (has_prev [B] bool, S_in [B,C,H,W]).

        Single hop: ``S_in`` = base sample of frame t-1. Chain (prob ``refiner_chain_prob``, always
        in eval): base sample of frame t-2, then one hop of this network to frame t-1.
        The base hop reads the offline full-ODE cache (``state_prev`` / ``state_prev2`` in the
        batch) when the dataset provides it, else samples the frozen base on the fly.
        """

        def _flag(key):
            v = ar_pack.get(key)
            if v is None:
                return torch.zeros(b, dtype=torch.bool, device=device)
            return v.reshape(b).to(device=device) >= 0.5

        has_prev = _flag("has_prev")
        # Frame t-1 is drawn with frame t-1's own layout (layout_cond: dataset "mask_hop"), as at inference.
        mask_hop = ar_pack.get("mask_hop")
        mask_hop = mask_combined if mask_hop is None else mask_hop.to(device=device)
        if self.refiner_state_source == "gt" and ar_pack.get("image_prev") is not None:
            # Teacher forcing ablation: the real previous photo, no state track.
            self._last_refiner_chain = 0.0
            s_in = self._encode_to_target_space(ar_pack["image_prev"]).to(device=device)
            return has_prev, s_in.to(dtype=dtype)
        has_prev2 = _flag("has_prev2") & has_prev
        if self.refiner_state_source == "resample":
            self._last_refiner_chain = 0.0
            return has_prev, self._refiner_resampled_prev(ar_pack, info, mask_hop, has_prev2, b, device, dtype)
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
        cached1 = ar_pack.get("state_prev")
        cached2 = ar_pack.get("state_prev2")
        if self.training and self.refiner_state_cache and cached1 is None:
            raise RuntimeError("--refiner_state_cache is on but the train batch has no state_prev")

        with torch.no_grad():
            if chain_ok:
                def _pick(a2, a1):
                    return torch.where(has_prev2.view(b, *([1] * (a1.ndim - 1))), a2, a1)

                hs0 = _pick(ar_pack["hidden_state_prev2"], hs1)
                if cached1 is not None and cached2 is not None:
                    s0 = _pick(cached2, cached1).to(device=device)
                else:
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
                    s0, c1, g1, mask_hop, self._refiner_draw_full(), gru_delta=dg_hop
                )
                s_in = torch.where(has_prev2.view(b, 1, 1, 1), s1, s0)
            elif cached1 is not None:
                s_in = cached1.to(device=device)
            else:
                s_in = self._refiner_base_sample(
                    info, hs1, dr1, di1, mask_hop, self._refiner_draw_full()
                )
        return has_prev, s_in.to(dtype=dtype)


    @torch.no_grad()
    def _refiner_resampled_prev(self, ar_pack, info, mask_hop, has_prev2, b, device, dtype):
        """Resampling Forcing (arXiv 2512.15702): the real frame t-1 noised to ``t_s`` and integrated
        back to t = 1 by this network (no grad), reading the real frame t-2 as its own S, so the state
        carries the model's own error at a strength set by ``t_s`` (t_s -> 1: teacher forcing, t_s -> 0:
        a fresh draw of frame t-1). One ``t_s = sigmoid(N(mu, sigma))`` per batch. Rows without a frame
        t-2, and every row before ``refiner_resample_warmup`` steps, keep the real frame t-1; after the
        warm-up the resampled share of the rest ramps linearly from 0 to 1 over ``refiner_resample_ramp``
        steps (per sample). Sets ``_refiner_state_conf_rows`` (t_s if resampled, else 1)."""
        x_prev = self._encode_to_target_space(ar_pack["image_prev"]).to(device=device, dtype=dtype)
        step = int(self.global_step) if getattr(self, "_trainer", None) is not None else 0
        self._refiner_state_conf_rows = torch.ones(b, device=device, dtype=dtype)
        warm = int(self.refiner_resample_warmup)
        ramp = int(self.refiner_resample_ramp)
        p = 1.0 if (ramp <= 0 or not self.training) else min(1.0, max(0.0, (step - warm) / ramp))
        rows = has_prev2 & (torch.rand(b, device=device) < p)
        if step < warm or not bool(rows.any()):
            self._last_refiner_resample_t = 1.0
            self._last_refiner_resample_frac = 0.0
            return x_prev
        u = float(self.refiner_resample_mu) + float(self.refiner_resample_sigma) * float(torch.randn(()))
        t_s = min(float(torch.sigmoid(torch.tensor(u))), 0.95)
        self._last_refiner_resample_t = t_s
        self._last_refiner_resample_frac = float(rows.float().mean())
        x_prev2 = self._encode_to_target_space(ar_pack["image_prev2"]).to(device=device, dtype=dtype)
        c1, g1 = self._unpack_conditions(
            info, ar_pack["hidden_state_prev"], ar_pack["direction_prev"], ar_pack["DI_prev"]
        )
        dg = None
        if self.use_delta_g:
            dg = g1 - self.process_physics_condition(ar_pack["hidden_state_prev2"])
        hint2 = x_prev2
        if getattr(self, "refiner_memory", False):
            # The regenerated frame t-1 reads the real memory before t-1 (M_{t-2}), as the sampler would.
            if getattr(self, "refiner_memory_recycle", False):
                raise ValueError("refiner_memory with resample does not support recycle")
            if self.refiner_memory_encoder:
                # Learned memory: E over the writing frames before t-1 (no grad: the hop is not trained).
                if ar_pack.get("memory_rows_prev2") is None:
                    raise KeyError("refiner_memory_encoder + resample needs memory_rows_prev2 (use_prev2 + memory rows)")
                with torch.no_grad():
                    mem2 = self._refiner_learned_memory_hint(ar_pack, b, device, dtype, "memory_rows_prev2", drop=False)
            else:
                mem2 = ar_pack.get("memory_prev2")
            if mem2 is None:
                raise KeyError("refiner_memory + resample needs memory_prev2 in the batch (use_prev2 + memory cache)")
            hint2 = torch.cat([x_prev2, mem2.to(device=device, dtype=dtype)], dim=1)
        eps = self._refiner_start_noise(None, b, device).to(dtype)
        z_init = (1.0 - t_s) * eps + t_s * x_prev
        solver = self.cfm_solver
        try:
            self.cfm_solver = "euler"
            s1 = self.sample_with_ode(
                c1, z0=eps, z_init=z_init, t_start=t_s, num_steps=int(self.refiner_resample_steps),
                gru_feat=g1, damage_mask=mask_hop, lr_cond=self._refiner_conf_hint(hint2, 1.0), gru_delta=dg,
            )
        finally:
            self.cfm_solver = solver
        self._refiner_state_conf_rows = torch.where(rows, torch.full_like(self._refiner_state_conf_rows, t_s),
                                                    self._refiner_state_conf_rows)
        return torch.where(rows.view(b, 1, 1, 1), s1.to(dtype), x_prev)


    def _refiner_memory_hint(self, ar_pack, b, device, dtype):
        """The real frames' structure memory before the target (dataset ``memory_prev``); zeroed per
        sample with ``refiner_memory_drop_prob`` in training."""
        mem = ar_pack.get("memory_prev")
        if mem is None:
            raise KeyError("refiner_memory needs memory_prev in the batch (dataset.attach_memory_cache)")
        mem = mem.to(device=device, dtype=dtype)
        if self._refiner_memory_recycle_on():
            mem = self._refiner_memory_recycle(mem, ar_pack, b, device)
        return self._refiner_memory_drop(mem, b, device, dtype)

    def _refiner_memory_recycle_on(self):
        """Training with ``refiner_memory_recycle`` from step ``refiner_memory_recycle_start`` on (real memory
        before it, as a teacher-forcing warm-up); both banking and injection."""
        if not (self.training and getattr(self, "refiner_memory_recycle", False)):
            return False
        step = int(self.global_step) if getattr(self, "_trainer", None) is not None else 0
        return step >= int(self.refiner_memory_recycle_start)

    def attach_memory_error_bank(self, n_rows):
        """One empty slot per global frame row for the recycled structure errors (``refiner_memory_recycle``)."""
        size = int(self.model_spatial_size)
        self._memory_err = torch.zeros(int(n_rows), 1, size, size, dtype=torch.float16)
        self._memory_err_ok = torch.zeros(int(n_rows), dtype=torch.bool)

    def _memory_err_bank(self, device):
        if self._memory_err is None:
            raise RuntimeError("refiner_memory_recycle needs attach_memory_error_bank")
        if self._memory_err.device != device:
            self._memory_err = self._memory_err.to(device)
            self._memory_err_ok = self._memory_err_ok.to(device)
        return self._memory_err, self._memory_err_ok

    @torch.no_grad()
    def _refiner_memory_recycle(self, mem, ar_pack, b, device):
        """Corrupt the real memory with banked errors: a ``refiner_memory_recycle_clean`` share stays clean, the
        rest takes 1..n errors from the earlier frames of its own walk or (share ``_transplant``) from any banked
        frame, flipped with the sample (transplants: at random)."""
        bank, ok = self._memory_err_bank(device)
        rows = ar_pack.get("memory_rows")
        flipped = ar_pack.get("flipped")
        if rows is None or flipped is None:
            raise KeyError("refiner_memory_recycle needs memory_rows / flipped in the batch (dataset.attach_memory_rows)")
        n = int(self.refiner_memory_recycle_n)
        filled = torch.nonzero(ok).flatten()
        corrupt = torch.rand(b, device=device) >= float(self.refiner_memory_recycle_clean)
        if filled.numel() == 0 or not bool(corrupt.any()):
            self._last_memory_recycle_frac = 0.0
            return mem
        transplant = torch.rand(b, device=device) < float(self.refiner_memory_recycle_transplant)
        k = torch.randint(1, n + 1, (b,), device=device)
        slot = torch.arange(n, device=device).view(1, n) < k.view(b, 1)
        # Own walk: up to n random earlier rows that are banked.
        rows = rows.to(device)
        own_ok = (rows >= 0) & ok[rows.clamp(min=0)]
        score = torch.where(own_ok, torch.rand(rows.shape, device=device), torch.full(rows.shape, -1.0, device=device))
        top = score.topk(min(n, rows.shape[1]), dim=1)
        own = torch.full((b, n), -1, dtype=torch.long, device=device)
        own[:, : top.indices.shape[1]] = torch.where(top.values >= 0, rows.gather(1, top.indices), torch.full_like(top.indices, -1))
        other = filled[torch.randint(0, filled.numel(), (b, n), device=device)]
        pick = torch.where(transplant.view(b, 1), other, own)
        valid = slot & (pick >= 0) & corrupt.view(b, 1)
        errs = bank[pick.clamp(min=0)].float().squeeze(2)
        flip = torch.where(transplant, torch.rand(b, device=device) < 0.5, flipped.to(device).reshape(b) >= 0.5)
        errs = torch.where(flip.view(b, 1, 1, 1), torch.flip(errs, dims=(-1,)), errs)
        self._last_memory_recycle_frac = float(valid.any(dim=1).float().mean())
        return inject_memory_errors(mem.float(), errs, valid).to(mem.dtype)

    @torch.no_grad()
    def _refiner_memory_bank(self, ar_pack, zt, t, v_pred, x1):
        """Bank this step's structure errors (x̂1 against the real frame, full-frame crops with flow time >=
        ``refiner_memory_recycle_tmin``) at the frames' own rows, unflipped."""
        size = int(self.model_spatial_size)
        row = ar_pack.get("memory_row")
        if row is None:
            raise KeyError("refiner_memory_recycle needs memory_row in the batch (attach_memory_rows with row_of)")
        if x1.shape[-1] != size or x1.shape[-2] != size:
            return
        keep = t >= float(self.refiner_memory_recycle_tmin)
        if not bool(keep.any()):
            return
        bank, ok = self._memory_err_bank(x1.device)
        x_hat = self._estimate_clean_from_velocity(zt, t, v_pred)
        err = memory_error(x_hat[keep], x1[keep], self.refiner_memory_sigma, self.refiner_memory_tau)
        flip = ar_pack["flipped"].to(x1.device).reshape(-1)[keep] >= 0.5
        err = torch.where(flip.view(-1, 1, 1, 1), torch.flip(err, dims=(-1,)), err)
        rows = row.to(x1.device).reshape(-1)[keep]
        bank[rows] = err.to(bank.dtype)
        ok[rows] = True
        self._last_memory_bank_fill = float(ok.float().mean())

    def _refiner_memory_drop(self, mem, b, device, dtype):
        if self.training and float(self.refiner_memory_drop_prob) > 0.0:
            keep = torch.rand(b, device=device) >= float(self.refiner_memory_drop_prob)
            mem = mem * keep.to(dtype).view(b, 1, 1, 1)
        return mem

    @torch.no_grad()
    def _refiner_erase_cracks(self, S, M, f_tgt, static, mem_keep, eps=0.03, sigma=3.0, grid=8):
        """Ring Forcing (arXiv 2608.26794) appear-disappear-reappear for a fixed camera and monotone damage: in a
        ``refiner_erase_prob`` share of the rows whose memory is not dropped, the damage cracks held by both the
        memory ``M`` and the target (``f_tgt``, its structure map), minus frame 0's static structure, are erased
        from ``S`` inside random cells of a ``grid`` x ``grid`` partition (each kept with a per-row rate
        U[0.5, 1]), dilated 2 px, and filled from the unmasked surroundings (normalised Gaussian convolution).
        The target is unchanged, so only the memory can supply the erased cracks."""
        from src.model.cfm.refiner_memory import _gauss_1d

        Bt, C, H, W = S.shape
        dev = S.device
        not_static = F.max_pool2d((static < -eps).float(), 5, stride=1, padding=2) <= 0
        cand = (M.float() < -eps) & (f_tgt.float() < -eps) & not_static
        rows = mem_keep & (torch.rand(Bt, device=dev) < float(self.refiner_erase_prob))
        rate = 0.5 + 0.5 * torch.rand(Bt, 1, 1, 1, device=dev)
        cells = F.interpolate((torch.rand(Bt, 1, grid, grid, device=dev) < rate).float(), size=(H, W), mode="nearest")
        m = cand.float() * cells * rows.float().view(Bt, 1, 1, 1)
        m = F.max_pool2d(m, 5, stride=1, padding=2)
        self._last_erase_frac = float(rows.float().mean())
        self._last_erase_px = float(m.mean())
        if not bool((m > 0).any()):
            return S
        k, r = _gauss_1d(sigma, dev, torch.float32)

        def blur(x):
            n = x.shape[1]
            y = x.reshape(-1, 1, H, W)
            y = F.conv2d(F.pad(y, (r, r, 0, 0), mode="reflect"), k.view(1, 1, 1, -1))
            y = F.conv2d(F.pad(y, (0, 0, r, r), mode="reflect"), k.view(1, 1, -1, 1))
            return y.reshape(Bt, n, H, W)

        s = S.float()
        w = 1.0 - m
        den = blur(w)
        fill = torch.where(den > 1e-3, blur(s * w) / den.clamp(min=1e-3), blur(s))
        return (s * w + fill * m).to(S.dtype)

    @torch.no_grad()
    def _refiner_bank_paste(self, S, M, rows, flip, mem_keep):
        """rfmv2b (src/model/cfm/rollout_bank.py): in a ``refiner_erase_prob`` share of the targets, S gets the
        model's own rollout degradation of its frame (delta against the real frame) inside random feathered cells;
        in a ``refiner_bank_mem_err_prob`` share of the targets whose memory is kept, the memory gets the same
        rollout's structure error (spurious entries added, missed ones removed). Flipped with the window."""
        from src.model.cfm.rollout_bank import feathered_cells

        Bt, _C, H, W = S.shape
        dev = S.device
        delta, err, ok = self._rollout_bank.sample(rows, dev)
        delta = torch.where(flip.view(Bt, 1, 1, 1), torch.flip(delta, dims=(-1,)), delta)
        err = torch.where(flip.view(Bt, 1, 1, 1), torch.flip(err, dims=(-1,)), err)
        paste = ok & (torch.rand(Bt, device=dev) < float(self.refiner_erase_prob))
        m = feathered_cells(Bt, H, W, int(self.refiner_bank_grid), self.refiner_bank_rate[0],
                            self.refiner_bank_rate[1], float(self.refiner_bank_feather), dev)
        m = m * paste.float().view(Bt, 1, 1, 1)
        S = (S.float() + m * delta).to(S.dtype)
        inj = ok & mem_keep & (torch.rand(Bt, device=dev) < float(self.refiner_bank_mem_err_prob))
        if bool(inj.any()):
            M = inject_memory_errors(M.float(), err, inj.view(Bt, 1)).to(M.dtype)
        self._last_bank_paste_frac = float(paste.float().mean())
        self._last_bank_paste_px = float(m.mean())
        self._last_bank_err_frac = float(inj.float().mean())
        return S, M

    def attach_memory_frames(self, frames):
        """Real frames [N, C, H, W] (train then val rows) that the learned memory encoder reads."""
        self._memory_frames = frames

    def _refiner_learned_memory_hint(self, ar_pack, b, device, dtype, rows_key="memory_rows", drop=True):
        """``max over the sample's earlier writing frames of E(frame)`` (0 if none), flipped with the sample,
        at the model resolution. Gradient reaches E through the max. ``rows_key="memory_rows_prev2"``: the
        memory before frame t-1 (Resampling Forcing hop), without the per-sample drop."""
        rows, flipped = ar_pack.get(rows_key), ar_pack.get("flipped")
        if rows is None or flipped is None or self._memory_frames is None:
            raise KeyError("refiner_memory_encoder needs memory_rows / flipped in the batch and attach_memory_frames")
        if self._memory_frames.device != device:
            self._memory_frames = self._memory_frames.to(device)
        rows = rows.to(device)
        valid = rows >= 0
        size = int(self.model_spatial_size)
        side = (size + 3) // 4
        field = torch.zeros(b, self.refiner_memory_channels, side, side, device=device, dtype=torch.float32)
        aux = self.training and drop and float(self.refiner_memory_aux_weight) > 0.0
        sparse_w = float(getattr(self, "refiner_memory_sparse_weight", 0.0)) if (self.training and drop) else 0.0
        if rows_key == "memory_rows":
            self._refiner_memory_aux_term = None
        if bool(valid.any()):
            uniq, inv = torch.unique(rows[valid], return_inverse=True)
            inputs = self._memory_frames[uniq]
            x = inputs.float()
            a = float(getattr(self, "refiner_memory_contrast_aug", 1.0))
            if self.training and a < 1.0:
                # Fogged (contrast-compressed) generated frames write weaker inputs; the target stays unaugmented.
                x = x * (a + (1.0 - a) * torch.rand(x.shape[0], 1, 1, 1, device=device))
            e = self.memory_encoder(x)
            owner = torch.arange(b, device=device).view(b, 1).expand_as(rows)[valid]
            maxed, targets = [], []
            for i in range(b):
                pick = inv[owner == i]
                maxed.append(e[pick].amax(dim=0) if pick.numel() else torch.zeros_like(e[0]))
                if aux:
                    targets.append(self.memory_encoder.aux_target(inputs[pick]) if pick.numel()
                                   else torch.zeros(inputs.shape[1:], device=device))
            field = torch.stack(maxed)
            if drop:
                self._last_memory_field_mean = float(field.detach().mean())
            term = None
            if aux:
                # Unflipped field vs the unflipped bank: the decoder must place the accumulated thin structure.
                loss_aux, parts = self.memory_encoder.aux_loss_value(field, torch.stack(targets))
                self._last_memory_aux_loss = float(loss_aux.detach())
                self._last_memory_aux_parts = parts
                term = float(self.refiner_memory_aux_weight) * loss_aux
            if sparse_w > 0.0:
                sp = e.mean()  # E >= 0: L1 on every written frame's output
                self._last_memory_sparse = float(sp.detach())
                term = sparse_w * sp if term is None else term + sparse_w * sp
            if term is not None:
                self._refiner_memory_aux_term = term
        flip = flipped.to(device).reshape(b) >= 0.5
        field = torch.where(flip.view(b, 1, 1, 1), torch.flip(field, dims=(-1,)), field)
        mem = self.memory_encoder.hint(field, size).to(dtype)
        return self._refiner_memory_drop(mem, b, device, dtype) if drop else mem


    def _refiner_state_condition(self, s_in):
        """What the network reads as S_{t-1}: perturbed (optional) plus Gaussian noise, training only."""
        if not self.training:
            return s_in
        s = self._perturb_refiner_state(s_in) if self.refiner_state_perturb else s_in
        b = s.shape[0]
        state_std = torch.rand(b, device=s.device, dtype=s.dtype) * float(self.refiner_state_noise_max)
        return s + state_std.view(b, 1, 1, 1) * torch.randn_like(s)


    @torch.no_grad()
    def _perturb_refiner_state(self, s):
        """Make S_{t-1} look like a wrongly exposed / under-damaged previous output.

        Per sample: exposure offset (half the batch), contrast gain about the image mean, a smooth
        brightness field on 32 px cells, dark (damage) pixels pulled toward the median, and
        sometimes a blur. Clipped to the image range, so a pushed-up frame also saturates.
        """
        b = s.shape[0]
        dev, dt = s.device, s.dtype
        shape = (b, 1, 1, 1)
        mean = s.mean(dim=(1, 2, 3), keepdim=True)
        gain = 1.0 + (torch.rand(shape, device=dev, dtype=dt) * 2.0 - 1.0) * float(self.refiner_state_gain_jitter)
        out = mean + (s - mean) * gain
        lo, hi = float(self.refiner_state_bias_min), float(self.refiner_state_bias_max)
        bias = lo + (hi - lo) * torch.rand(shape, device=dev, dtype=dt)
        out = out + bias * (torch.rand(shape, device=dev) < 0.5).to(dt)
        h, w = s.shape[-2:]
        field = torch.randn(b, 1, max(h // 32, 2), max(w // 32, 2), device=dev, dtype=dt)
        field = F.interpolate(field, size=(h, w), mode="bicubic", align_corners=False)
        out = out + float(self.refiner_state_lowfreq_std) * field
        median = s.flatten(1).median(dim=1).values.view(shape)
        lam = float(self.refiner_state_damage_shrink)
        keep = lam + (1.0 - lam) * torch.rand(shape, device=dev, dtype=dt)
        dark = (s < median - 0.3).to(dt)
        out = out + dark * (1.0 - keep) * (median - s)
        blur_on = torch.rand(b, device=dev) < float(self.refiner_state_blur_prob)
        if bool(blur_on.any()):
            sigma = torch.where(blur_on, 0.5 + torch.rand(b, device=dev), torch.zeros(b, device=dev))
            out = self._gaussian_blur_nchw(out, sigma)
        return out.clamp(-1.0, 1.0)


    @staticmethod
    def _lowpass_tone(x, block=16):
        """Block means: the global / low-frequency brightness the tone loss compares."""
        k = max(1, min(int(block), int(x.shape[-1]), int(x.shape[-2])))
        return F.avg_pool2d(x.mean(dim=1, keepdim=True), kernel_size=k, stride=k)


    @staticmethod
    def _refiner_neighbor_pack(prev, k_prev, nxt, k_next):
        """Control hint [prev | next | k_prev | k_next]; k is a float or a [B] tensor."""
        b, _, h, w = prev.shape

        def plane(k):
            k = torch.as_tensor(k, device=prev.device, dtype=prev.dtype)
            return k.reshape(-1, 1, 1, 1).expand(b, 1, h, w)

        return torch.cat([prev, nxt, plane(k_prev), plane(k_next)], dim=1)

    def _refiner_neighbor_hint(self, ar_pack, z0, t):
        """Neighbors of the target at their own noise levels, ``(1-k) ε + k x`` (Diffusion Forcing).
        Per sample: both at the target level τ (tied), one random neighbor at τ + U(0, ahead_max) and
        the other at τ (ahead, what the sweep sampler shows), or each k ~ U(0, 1). Each neighbor is then
        dropped (k = 0) with ``refiner_nbr_drop_prob``, and always where it does not exist. ε shares
        the target's low band with ``refiner_shared_lowfreq_noise``, as all frames do at sampling."""
        b = z0.shape[0]
        device, dtype = z0.device, z0.dtype

        def exists(key):
            v = ar_pack.get(key)
            if v is None:
                return torch.zeros(b, dtype=torch.bool, device=device)
            return v.reshape(b).to(device) >= 0.5

        u = torch.rand(b, device=device)
        p_tied = float(self.refiner_nbr_tied_prob)
        tied = u < p_tied
        ahead = (u >= p_tied) & (u < p_tied + float(self.refiner_nbr_ahead_prob))
        lead = (t + float(self.refiner_nbr_ahead_max) * torch.rand(b, device=device, dtype=dtype)).clamp(max=1.0)
        prev_leads = torch.rand(b, device=device) < 0.5
        k_prev = torch.rand(b, device=device, dtype=dtype)
        k_next = torch.rand(b, device=device, dtype=dtype)
        k_prev = torch.where(tied | (ahead & ~prev_leads), t, k_prev)
        k_next = torch.where(tied | (ahead & prev_leads), t, k_next)
        k_prev = torch.where(ahead & prev_leads, lead, k_prev)
        k_next = torch.where(ahead & ~prev_leads, lead, k_next)
        drop = float(self.refiner_nbr_drop_prob) if self.training else 0.0
        keep_prev = exists("has_prev") & (torch.rand(b, device=device) >= drop)
        keep_next = exists("has_next") & (torch.rand(b, device=device) >= drop)
        k_prev = k_prev * keep_prev.to(dtype)
        k_next = k_next * keep_next.to(dtype)

        shared = z0 if self.refiner_shared_lowfreq_noise else None

        def noised(key, k):
            x = self._encode_to_target_space(ar_pack[key].to(device))
            k_ = k.view(b, 1, 1, 1)
            return (1.0 - k_) * self._refiner_start_noise(shared, b, device).to(dtype) + k_ * x

        return self._refiner_neighbor_pack(
            noised("image_prev", k_prev), k_prev, noised("image_next", k_next), k_next
        )

    # ------------------------------------------------------------------------------------------------------
    # rfmv2 (docs/RFMV2_ARCHITECTURE.md): window training with autoregressive self-resampling, the memory
    # written from the resampled frames, and no loss on keeping history the real walk does not have.
    # ------------------------------------------------------------------------------------------------------
    def _refiner_cfm_core(self, x1, s_cond, c, g, valid_mask, mask, dg):
        """The CFM loss of refiner_implicit_flow_matching_loss (target_x1, every row weighted 1) on a ready hint;
        training memory gain instead of the time gate."""
        b, device, dtype = x1.shape[0], x1.device, x1.dtype
        z0 = torch.randn_like(x1)
        t = torch.rand(b, device=device, dtype=dtype).clamp(min=1e-5, max=1.0 - 1e-5)
        coords = None
        if self.use_patch_diffusion:
            x1, valid_mask, coords, mask, z0, s_cond = self._apply_patch_crop(
                x1, valid_mask=valid_mask, condition_mask=mask, z0=z0, z_aux=s_cond, flow_t=t)
        t_ = t.view(b, 1, 1, 1)
        zt = (1.0 - t_) * z0 + t_ * x1
        if self.refiner_memory and self.refiner_memory_gain and self.training:
            # alpha ~ U[0, 1] with probability 0.25 + 0.5 tau, else 1: every tau sees full, weak and no memory.
            use = torch.rand(b, device=device) < (0.25 + 0.5 * t)
            alpha = torch.where(use, torch.rand(b, device=device, dtype=dtype), torch.ones_like(t))
            c0, nm = int(self.model_in_channels), int(self.refiner_memory_hint_channels)
            s_cond = torch.cat([s_cond[:, :c0], s_cond[:, c0:c0 + nm] * alpha.view(b, 1, 1, 1), s_cond[:, c0 + nm:]], 1)
        v_pred = self._predict_velocity(zt, t, c, coords=coords, damage_mask=mask, gru_feat=g, lr_cond=s_cond,
                                        gru_delta=dg)
        per = self._cfm_mse(v_pred, x1 - z0, valid_mask=valid_mask)
        if self.use_velocity_min_snr_weighting:
            per = per * self._compute_velocity_min_snr_weight(t).to(per.dtype)
        loss = per.mean()
        self._last_fm_loss = float(loss.detach())
        tone_w = float(self.refiner_tone_loss_weight)
        if tone_w > 0.0:
            x1_hat = self._estimate_clean_from_velocity(zt, t, v_pred)
            block = int(self.refiner_lowfreq_block)
            diff = (self._lowpass_tone(x1_hat, block) - self._lowpass_tone(x1, block)).pow(2)
            tone = (diff.mean(dim=(1, 2, 3)) * t).mean()
            self._last_refiner_tone_loss = float(tone.detach())
            loss = loss + tone_w * tone
        topo_w = float(getattr(self, "refiner_topo_loss_weight", 0.0))
        if topo_w > 0.0:
            # Crack-map recall / precision on the clean estimate (t > 0.6): soft black-hat of x1_hat (5x5 grey
            # closing, as damage_texture.crack_map) against the target's hard crack map; pixels the false-memory
            # mask down-weights (valid < 0.5) are left out.
            x1_hat = self._estimate_clean_from_velocity(zt, t, v_pred).float()
            thr, temp = 12.0 / 127.5, 0.02

            def blackhat(x):
                g = x.mean(dim=1, keepdim=True)
                close = -F.max_pool2d(-F.max_pool2d(g, 5, stride=1, padding=2), 5, stride=1, padding=2)
                return close - g

            ok = (valid_mask.float() > 0.5).float() if valid_mask is not None else torch.ones_like(x1[:, :1]).float()
            if ok.shape[1] != 1:
                ok = ok[:, :1]
            real = (blackhat(x1.float()) > thr).float() * ok
            p = torch.sigmoid((blackhat(x1_hat) - thr) / temp) * ok
            near = F.max_pool2d(real, 5, stride=1, padding=2)
            rec = 1.0 - (p * real).sum(dim=(1, 2, 3)) / real.sum(dim=(1, 2, 3)).clamp(min=1.0)
            prec = (p * (1.0 - near)).sum(dim=(1, 2, 3)) / p.sum(dim=(1, 2, 3)).clamp(min=1.0)
            sel = (t > 0.6).float() * (real.sum(dim=(1, 2, 3)) > 0).float()
            topo = ((rec + prec) * sel).sum() / sel.sum().clamp(min=1.0)
            self._last_refiner_topo_loss = float(topo.detach())
            loss = loss + topo_w * topo
        return loss

    def refiner_window_loss(self, batch):
        """One batch of rfmv2 windows (dataset.enable_window): [warm real frames | anchor | L targets].

        No grad: the anchor and the L-1 frames after it are resampled in order (each real frame noised to the
        batch t_s and integrated back reading the previous resampled frame and the memory written so far); the
        memory is the rule track continued over the resampled frames, a second track over the real frames gives
        the real memory. Then target j reads S = resampled frame j-1, that memory, confidence (t_s, hops), and the
        CFM loss on target j is weighted refiner_false_mem_weight where the history memory holds structure the
        real memory up to target j (or target j itself) lacks."""
        from src.model.cfm.refiner_memory import memory_feature

        img = batch["win_image"]
        B, F_ = img.shape[:2]
        L = int(self.refiner_window)
        a = F_ - L - 1
        device = img.device
        valid = batch["win_valid"].to(device) > 0.5
        flat = lambda x: x.reshape(B * F_, *x.shape[2:])
        X = self._encode_to_target_space(flat(img).to(device))
        dtype = X.dtype
        X = X.view(B, F_, *X.shape[1:])
        c, g = self._unpack_conditions(flat(batch["win_info"]).to(device), flat(batch["win_hidden"]).to(device),
                                       flat(batch["win_direction"]).to(device), flat(batch["win_DI"]).to(device))
        c = c.view(B, F_, *c.shape[1:])
        g = None if g is None else g.view(B, F_, *g.shape[1:])
        lmask = self._align_spatial_to_target(flat(batch["win_loss_mask"]).to(device))
        lmask = lmask.view(B, F_, *lmask.shape[1:])
        cmask = None
        if batch.get("win_mask") is not None:
            cmask = self._align_spatial_to_target(flat(batch["win_mask"]).to(device).to(dtype))
            cmask = cmask.view(B, F_, *cmask.shape[1:])

        def mask_at(q):  # frame q given frame q-1: [max(object q-1, q) | layout q]
            if cmask is None:
                return None
            m = cmask[:, q]
            obj = torch.maximum(cmask[:, q - 1, :1], m[:, :1]) if q > 0 else m[:, :1]
            return torch.cat([obj, m[:, 1:]], dim=1)

        def dg_at(q):
            if not self.use_delta_g or g is None:
                return None
            ok = valid[:, q - 1].view(B, *([1] * (g.dim() - 2))).to(g.dtype)
            return (g[:, q] - g[:, q - 1]) * ok

        # --- rule memory tracks (MemoryTrack semantics, per-row valid history) ---
        sig, tau, win = float(self.refiner_memory_sigma), float(self.refiner_memory_tau), int(self.refiner_memory_window)
        eps = 0.03
        nan = torch.tensor(float("nan"), device=device)

        def feat(x, ok=None):
            f = memory_feature(x, sig, tau).float()
            return f if ok is None else torch.where(ok.view(B, 1, 1, 1), f, nan)

        def write(recent):
            return torch.nanmedian(torch.stack(recent[-win:]), dim=0).values

        recent0 = [feat(X[:, q], valid[:, q]) for q in range(a)]
        mem0 = batch["win_mem0"].to(device).float()
        v_hist, r_hist = mem0, list(recent0)
        v_real, r_real = mem0.clone(), list(recent0)

        # --- self-resampling decision (teacher-forcing warm-up, then the resampled share ramps up) ---
        step = int(self.global_step) if getattr(self, "_trainer", None) is not None else 0
        warm, ramp = int(self.refiner_resample_warmup), int(self.refiner_resample_ramp)
        p = 1.0 if ramp <= 0 else min(1.0, max(0.0, (step - warm) / ramp))
        p = min(p, float(getattr(self, "refiner_resample_max_p", 1.0)))
        rs = (torch.rand(B, device=device) < p) if step >= warm else torch.zeros(B, dtype=torch.bool, device=device)
        n_grp = min(int(getattr(self, "refiner_resample_groups", 1)), B)
        grp = torch.arange(B, device=device) % n_grp  # window -> t_s group
        u = float(self.refiner_resample_mu) + float(self.refiner_resample_sigma) * torch.randn(n_grp)
        ts_grp = [min(float(v), 0.95) for v in torch.sigmoid(u)]
        ts_row = torch.tensor(ts_grp, device=device)[grp]  # [B] t_s of each window
        t_s = float(ts_row.mean())

        S_list, M_list, conf_list, hop_list, w_list, F_list = [], [], [], [], [], []
        prev = X[:, a - 1] if a > 0 else X[:, a]
        prev_hops = torch.zeros(B, device=device)  # hops behind prev (0 = real frame)
        n_false = n_hist = 0.0
        solver = self.cfm_solver
        with torch.no_grad():
            for j in range(L):
                q = a + j
                prev_ok = valid[:, q - 1] if q > 0 else torch.zeros(B, dtype=torch.bool, device=device)
                hop_rows = rs & (prev_ok if j == 0 else torch.ones_like(rs))
                xq = X[:, q]
                xh = xq
                if bool(hop_rows.any()):
                    conf_in = torch.where(prev_hops > 0, ts_row.to(prev_hops.dtype), torch.ones_like(prev_hops))
                    hint = self._refiner_conf_hint(torch.cat([prev, v_hist.to(dtype)], dim=1), conf_in, hops=prev_hops)
                    epsn = self._refiner_start_noise(None, B, device).to(dtype)
                    out = xq.clone()
                    sub = lambda v, r: None if v is None else v[r]
                    try:
                        self.cfm_solver, self._in_hop = self.refiner_resample_solver, True
                        for gi, tg in enumerate(ts_grp):
                            r = torch.nonzero(hop_rows & (grp == gi)).flatten()
                            if r.numel() == 0:
                                continue
                            out[r] = self.sample_with_ode(
                                c[r, q], z0=epsn[r], z_init=(1.0 - tg) * epsn[r] + tg * xq[r], t_start=tg,
                                num_steps=int(self.refiner_resample_steps),
                                gru_feat=None if g is None else g[r, q], damage_mask=sub(mask_at(q), r),
                                lr_cond=hint[r], gru_delta=sub(dg_at(q), r)).to(dtype)
                    finally:
                        self.cfm_solver, self._in_hop = solver, False
                    xh = torch.where(hop_rows.view(B, 1, 1, 1), out, xq)
                hops = torch.where(hop_rows, prev_hops + 1, torch.zeros_like(prev_hops))
                # memory after frame q: history track from the resampled frame, real track from the real frame
                r_hist.append(feat(xh))
                v_hist = torch.minimum(v_hist, write(r_hist))
                r_real.append(feat(xq))
                v_real = torch.minimum(v_real, write(r_real))
                # real memory including target q+1 (and target q+1's own structure): what may be kept
                f_next = feat(X[:, q + 1])
                v_real_next = torch.minimum(v_real, write(r_real + [f_next]))
                allowed = (v_real_next < -eps) | (f_next < -eps)
                allowed = F.max_pool2d(allowed.float(), 5, stride=1, padding=2) > 0
                false_mem = (v_hist < -eps) & ~allowed
                n_false += float(false_mem.float().sum())
                n_hist += float((v_hist < -eps).float().sum())
                lam = float(self.refiner_false_mem_weight)
                w_list.append(lmask[:, q + 1] * (1.0 - (1.0 - lam) * false_mem.to(dtype)))
                S_list.append(xh)
                M_list.append(v_hist.to(dtype))
                F_list.append(f_next)
                conf_list.append(torch.where(hops > 0, ts_row.to(hops.dtype), torch.ones_like(hops)))
                hop_list.append(hops)
                prev, prev_hops = xh, hops

        Bt = B * L
        stackT = lambda lst: torch.stack(lst, dim=1).reshape(Bt, *lst[0].shape[1:])
        tgt = slice(a + 1, a + L + 1)
        x1 = X[:, tgt].reshape(Bt, *X.shape[2:])
        c_t = c[:, tgt].reshape(Bt, *c.shape[2:])
        g_t = None if g is None else g[:, tgt].reshape(Bt, *g.shape[2:])
        dg_t = None if dg_at(a + 1) is None else torch.stack([dg_at(q) for q in range(a + 1, a + L + 1)], 1).reshape(
            Bt, *g.shape[2:])
        m_t = None if cmask is None else torch.stack([mask_at(q) for q in range(a + 1, a + L + 1)], 1).reshape(
            Bt, -1, *cmask.shape[-2:])
        S, M = stackT(S_list), stackT(M_list)
        mem_keep = torch.ones(Bt, dtype=torch.bool, device=device)
        if float(self.refiner_memory_drop_prob) > 0.0:
            mem_keep = torch.rand(Bt, device=device) >= float(self.refiner_memory_drop_prob)
            M = M * mem_keep.to(dtype).view(Bt, 1, 1, 1)
        if self.refiner_erase_source == "bank":
            bank = self._rollout_bank
            self._last_bank_paste_frac = self._last_bank_paste_px = self._last_bank_err_frac = 0.0
            if bank is not None and bank.ready():
                rows = batch.get("win_rows")
                if rows is None:
                    raise KeyError("refiner_erase_source=bank needs win_rows in the window batch")
                rows = rows[:, a:a + L].reshape(Bt).to(device)  # S of target q+1 is frame q
                flip = (batch["win_flipped"].to(device).float() >= 0.5).repeat_interleave(L)
                S, M = self._refiner_bank_paste(S, M, rows, flip, mem_keep)
        elif float(self.refiner_erase_prob) > 0.0:
            static = batch.get("win_static")
            if static is None:
                raise KeyError("refiner_erase_prob needs win_static in the window batch")
            static = static.to(device).float().repeat_interleave(L, dim=0)
            S = self._refiner_erase_cracks(S, M, stackT(F_list), static, mem_keep)
        S = self._refiner_state_condition(S)
        s_cond = self._refiner_conf_hint(torch.cat([S, M], dim=1), stackT(conf_list), hops=stackT(hop_list))
        loss = self._refiner_cfm_core(x1, s_cond, c_t, g_t, stackT(w_list), m_t, dg_t)
        self._last_refiner_resample_t = t_s
        self._last_refiner_resample_frac = float(rs.float().mean())
        self._last_window_hops = float(torch.stack(hop_list).float().mean())
        self._last_false_mem_frac = n_false / max(n_hist, 1.0)
        return loss

    def refiner_implicit_flow_matching_loss(
        self,
        z_target,
        ar_pack,
        info,
        c_target,
        gru_feat_target=None,
        valid_mask=None,
        mask_combined=None,
    ):
        """Refiners that generate x_t itself (v10.5 implicit, v10.6 target_x1).

        State track as in V10 (``_refiner_state_track``). Loss track is plain CFM to the frame:
        ``z_τ = (1-τ) x0 + τ x_t``, ``v = x_t - x0``, with ``S_{t-1}`` as the control hint
        (v10.5) or as concatenated input channels (v10.6). Optional tone loss on the block means
        of ``x̂1 = z_τ + (1-τ) v``.

        v10.5: samples without a previous frame, and a ``refiner_control_drop_prob`` share of the
        rest, run with the control branch off (the single-frame task); every sample counts.
        v10.6 (``refiner_target_x1``, with or without the control branch): no single-frame task;
        samples without a previous frame have weight 0, as in V10.
        """
        b = z_target.shape[0]
        device = z_target.device
        dtype = z_target.dtype
        hs1 = ar_pack["hidden_state_prev"]
        if self.refiner_neighbors:
            has_prev = torch.ones(b, dtype=torch.bool, device=device)
            s_cond = None
        else:
            has_prev, s_in = self._refiner_state_track(ar_pack, info, mask_combined, b, device, dtype)
            s_cond = self._refiner_state_condition(s_in)
            if self.refiner_memory:
                hint = self._refiner_learned_memory_hint if self.refiner_memory_encoder else self._refiner_memory_hint
                s_cond = torch.cat([s_cond, hint(ar_pack, b, device, dtype)], dim=1)
            s_cond = self._refiner_conf_hint(s_cond, self._refiner_state_conf_rows)

        control_scale = None
        weight = has_prev.to(dtype)
        if self.refiner_implicit and not self.refiner_target_x1:
            keep = has_prev
            if self.training and float(self.refiner_control_drop_prob) > 0.0:
                keep = keep & (torch.rand(b, device=device) >= float(self.refiner_control_drop_prob))
            control_scale = keep.to(dtype)
            weight = torch.ones_like(weight)
            self._last_refiner_control_frac = float(control_scale.mean())

        z0 = torch.randn_like(z_target)
        t = torch.rand(b, device=device, dtype=dtype).clamp(min=1e-5, max=1.0 - 1e-5)
        if self.refiner_neighbors:
            s_cond = self._refiner_neighbor_hint(ar_pack, z0, t)
        coords = None
        x1 = z_target
        if self.use_patch_diffusion:
            x1, valid_mask, coords, mask_combined, z0, s_cond = self._apply_patch_crop(
                x1,
                valid_mask=valid_mask,
                condition_mask=mask_combined,
                z0=z0,
                z_aux=s_cond,
                flow_t=t,
            )
        t_ = t.view(b, 1, 1, 1)
        zt = (1.0 - t_) * z0 + t_ * x1
        target_u = x1 - z0

        dg_target = None
        if self.use_delta_g:
            dg_target = gru_feat_target - self.process_physics_condition(hs1)
        dg_next = None
        if self.refiner_nbr_delta_g:
            dg_next = self.process_physics_condition(ar_pack["hidden_state_next"]) - gru_feat_target
        if self.refiner_memory:
            s_cond = self._refiner_memory_time_gate(s_cond, t)
        v_pred = self._predict_velocity(
            zt,
            t,
            c_target,
            coords=coords,
            damage_mask=mask_combined,
            gru_feat=gru_feat_target,
            lr_cond=s_cond,
            gru_delta=dg_target,
            control_scale=control_scale,
            gru_delta_next=dg_next,
        )
        per_sample_loss = self._cfm_mse(v_pred, target_u, valid_mask=valid_mask)
        if self.use_velocity_min_snr_weighting:
            per_sample_loss = per_sample_loss * self._compute_velocity_min_snr_weight(t).to(
                per_sample_loss.dtype
            )
        if self._refiner_memory_recycle_on():
            self._refiner_memory_bank(ar_pack, zt.detach(), t, v_pred.detach(), x1)
        denom = weight.sum().clamp_min(1.0)
        loss = (per_sample_loss * weight).sum() / denom
        self._last_fm_loss = float(loss.detach())

        tone_w = float(self.refiner_tone_loss_weight)
        if tone_w > 0.0:
            # Weighted by τ: x̂1 is only a blurry mean near the noise end.
            x1_hat = self._estimate_clean_from_velocity(zt, t, v_pred)
            block = int(self.refiner_lowfreq_block)
            diff = (self._lowpass_tone(x1_hat, block) - self._lowpass_tone(x1, block)).pow(2)
            tone = (diff.mean(dim=(1, 2, 3)) * t * weight).sum() / denom
            self._last_refiner_tone_loss = float(tone.detach())
            loss = loss + tone_w * tone
        aux = getattr(self, "_refiner_memory_aux_term", None)  # already weighted (decoder loss + sparsity)
        if aux is not None:
            loss = loss + aux
            self._refiner_memory_aux_term = None
        return loss


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
        has_prev, s_in = self._refiner_state_track(ar_pack, info, mask_combined, b, device, dtype)
        hs1 = ar_pack["hidden_state_prev"]
        s_cond = self._refiner_state_condition(s_in)

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
