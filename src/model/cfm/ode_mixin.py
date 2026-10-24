"""OT-CFM ODE sampling helpers."""

import math

import torch
import torch.nn.functional as F

from src.model.cfm.attn_mask_guidance import (
    mask_xt,
    saliency_from_self_attn,
    topk_mask_from_saliency,
)
from src.model.apg import MomentumBuffer, _project_onto, apg_guidance


class CFMODEMixin:
    def _feat_guidance_enabled(self, gru_feat) -> bool:
        scale = float(getattr(self, "feat_guidance_scale", 0.0))
        fn = getattr(self, "feat_guidance_fn", None)
        return fn is not None and scale != 0.0 and gru_feat is not None

    def _apply_feat_guidance(self, z_in, t_tensor, t_scalar, v, gru_feat):
        """Classifier guidance: v ← v − λ(t) ∇_{x_t} L_feat(x̂₁, g).

        UNet velocity is detached; gradient is through the OT identity
        ``x̂₁ = x_t + (1-t) v``. λ(t) = scale · t^{p} (p=0 → constant).
        """
        if not self._feat_guidance_enabled(gru_feat):
            return v
        fn = self.feat_guidance_fn
        scale = float(self.feat_guidance_scale)
        power = float(getattr(self, "feat_guidance_t_power", 1.0))
        lam = scale * (float(t_scalar) ** power)
        if abs(lam) < 1e-12:
            return v
        with torch.enable_grad():
            z_req = z_in.detach().requires_grad_(True)
            t_ = t_tensor.view(-1, *([1] * (z_req.ndim - 1))).to(dtype=z_req.dtype)
            x1_hat = z_req + (1.0 - t_) * v.detach()
            img = self._decode_from_target_space(x1_hat)
            loss = fn(img, gru_feat)
            grad = torch.autograd.grad(loss, z_req, create_graph=False)[0]
        if not getattr(self, "_feat_g_logged", False):
            self._feat_g_logged = True
            print(
                f"[featg] t={float(t_scalar):.3f} lam={lam:.4g} "
                f"loss={float(loss.detach()):.4g} |grad|mean={float(grad.detach().abs().mean()):.4g} "
                f"|v|mean={float(v.detach().abs().mean()):.4g}",
                flush=True,
            )
        return v.detach() - lam * grad.detach()

    def _apply_lowfreq_spall_guide(self, z_in, t_scalar, v, gru_feat):
        """Push the low band of x̂₁ toward the g-only spall-area probability."""
        scale = float(getattr(self, "lowfreq_guide_scale", 0.0))
        head = getattr(self, "spall_area_head", None)
        if scale == 0.0 or head is None or gru_feat is None:
            return v
        t_min = float(getattr(self, "lowfreq_guide_t_min", 0.2))
        t_max = float(getattr(self, "lowfreq_guide_t_max", 0.7))
        if not (t_min <= float(t_scalar) < t_max):
            return v
        from src.model.cfm.lowfreq_guide import lowfreq_velocity_delta

        delta, prob, energy = lowfreq_velocity_delta(
            head,
            z_in,
            v,
            float(t_scalar),
            gru_feat,
            scale,
            grid=int(getattr(head, "grid", 16)),
            interference=getattr(self, "_lowfreq_interference", None),
        )
        if not getattr(self, "_lowfreq_logged", False):
            self._lowfreq_logged = True
            print(
                f"[lowfreq] t={float(t_scalar):.3f} scale={scale:g} "
                f"p_mean={float(prob.mean()):.3f} E={float(energy):.4g} "
                f"|dv|mean={float(delta.abs().mean()):.4g} "
                f"|v|mean={float(v.detach().abs().mean()):.4g}",
                flush=True,
            )
        return v.detach() + delta

    def _sag_enabled(self, t_scalar) -> bool:
        gamma = float(getattr(self, "sag_scale", 0.0))
        t_min = float(getattr(self, "sag_t_min", 0.2))
        mid = getattr(getattr(self, "velocity_net", None), "mid_attn", None)
        return (
            gamma != 0.0
            and float(t_scalar) >= t_min
            and mid is not None
            and hasattr(mid, "store_attn")
        )

    def _apply_sag_guidance(
        self,
        z_in,
        t_tensor,
        t_scalar,
        v_clean,
        predict_fn,
    ):
        """``v ← v_clean + γ (v_clean − v_mask)`` with the same condition in ``predict_fn``."""
        if not self._sag_enabled(t_scalar):
            return v_clean
        mid = self.velocity_net.mid_attn
        attn = getattr(mid, "last_attn", None)
        hw = getattr(mid, "last_hw", None)
        if attn is None or hw is None:
            return v_clean
        sal = saliency_from_self_attn(
            attn,
            hw,
            reduce=str(getattr(self, "sag_reduce", "mean")),
            axis=str(getattr(self, "sag_axis", "key")),
        )
        binary = topk_mask_from_saliency(
            sal, z_in.shape[-2:], mask_frac=float(getattr(self, "sag_mask_frac", 0.3))
        )
        z_mask = mask_xt(
            z_in,
            binary,
            fill=str(getattr(self, "sag_fill", "zero")),
        )
        v_mask = predict_fn(z_mask)
        gamma = float(self.sag_scale)
        delta = v_clean - v_mask
        if not getattr(self, "_sag_logged", False):
            self._sag_logged = True
            print(
                f"[sag] t={float(t_scalar):.3f} γ={gamma:g} mask_frac={self.sag_mask_frac:g} "
                f"fill={self.sag_fill} |Δv|mean={float(delta.detach().abs().mean()):.4g} "
                f"|v|mean={float(v_clean.detach().abs().mean()):.4g}",
                flush=True,
            )
        return v_clean + gamma * delta

    def _coord_guidance_enabled(self) -> bool:
        if not bool(getattr(self, "use_patch_diffusion", False)):
            return False
        if bool(getattr(self, "zero_patch_coords", False)):
            return False
        apg_on_coords = (
            float(getattr(self, "apg_guidance_scale", 0.0)) != 0.0
            and not str(getattr(self, "apg_g0", "") or "")
        )
        coord_on_coords = (
            float(getattr(self, "coord_guidance_scale", 0.0)) != 0.0
            and not str(getattr(self, "coord_g0", "") or "")
        )
        return coord_on_coords or apg_on_coords

    def _apply_coord_guidance(self, z_in, v_xy, predict_fn, t_scalar=None):
        """Same-c coord pair: vanilla ``v_xy + γ(v_xy−v_0)`` or APG on that pair."""
        if not self._coord_guidance_enabled():
            return v_xy
        t_min = float(getattr(self, "coord_guidance_t_min", 0.0))
        t_max = float(getattr(self, "coord_guidance_t_max", 1.0))
        if t_scalar is not None and not (t_min <= float(t_scalar) < t_max):
            return v_xy
        prev = bool(getattr(self, "zero_patch_coords", False))
        self.zero_patch_coords = True
        try:
            v_0 = predict_fn(z_in)
        finally:
            self.zero_patch_coords = prev
        apg_s = float(getattr(self, "apg_guidance_scale", 0.0))
        if apg_s != 0.0:
            buf = getattr(self, "_apg_momentum_buffer", None)
            guided = apg_guidance(
                v_xy,
                v_0,
                guidance_scale=apg_s,
                eta=float(getattr(self, "apg_eta", 0.0)),
                norm_threshold=float(getattr(self, "apg_norm_threshold", 0.0)),
                momentum_buffer=buf,
            )
            if not getattr(self, "_apg_logged", False):
                self._apg_logged = True
                delta = v_xy - v_0
                print(
                    f"[apg] s={apg_s:g} eta={float(self.apg_eta):g} "
                    f"norm_thr={float(self.apg_norm_threshold):g} "
                    f"mom={float(self.apg_momentum):g} "
                    f"|Δv|mean={float(delta.detach().abs().mean()):.4g}",
                    flush=True,
                )
            return guided
        gamma = float(self.coord_guidance_scale)
        delta = self._highpass_coord_delta(v_xy - v_0)
        if not getattr(self, "_coord_g_logged", False):
            self._coord_g_logged = True
            hp = float(getattr(self, "coord_guidance_hipass_sigma", 0.0))
            print(
                f"[coordg] γ={gamma:g} hipassσ={hp:g} t=[{t_min:g},{t_max:g}) "
                f"|Δv|mean={float(delta.detach().abs().mean()):.4g} "
                f"|v_xy|mean={float(v_xy.detach().abs().mean()):.4g}",
                flush=True,
            )
        return v_xy + gamma * delta

    def _apply_facfg(self, v_cond, predict_unc):
        """FreCaS FA-CFG: separate CFG weights on low and high frequency of v.

        ``v = (1-w_l) v_unc^l + w_l v_c^l + (1-w_h) v_unc^h + w_h v_c^h``.
        ``v_unc`` is the same state with the caller-supplied g=0 feature.
        Low frequency is a 2× downsample and resize, matching the paper's split.
        """
        wl = float(self.facfg_wl)
        wh = float(self.facfg_wh)
        if wl == 1.0 and wh == 1.0:
            return v_cond
        v_unc = predict_unc()
        low_c, high_c = self._facfg_split(v_cond)
        low_u, high_u = self._facfg_split(v_unc)
        guided = (1.0 - wl) * low_u + wl * low_c + (1.0 - wh) * high_u + wh * high_c
        if not getattr(self, "_facfg_logged", False):
            self._facfg_logged = True
            delta = (v_cond - v_unc).detach()
            print(
                f"[facfg] wl={wl:g} wh={wh:g} "
                f"|v_c-v_g0|mean={float(delta.abs().mean()):.4g} "
                f"|v_c|mean={float(v_cond.detach().abs().mean()):.4g}",
                flush=True,
            )
        return guided

    @staticmethod
    def _facfg_split(v):
        h, w = int(v.shape[-2]), int(v.shape[-1])
        low = F.interpolate(
            v,
            size=(max(1, h // 2), max(1, w // 2)),
            mode="bilinear",
            align_corners=False,
        )
        low = F.interpolate(low, size=(h, w), mode="bilinear", align_corners=False)
        return low, v - low

    def _highpass_coord_delta(self, delta):
        """Drop low-frequency (blob/area) from coord CFG residual."""
        sigma = float(getattr(self, "coord_guidance_hipass_sigma", 0.0))
        if sigma <= 0.0 or delta.ndim != 4:
            return delta
        k = int(max(3, 2 * round(sigma * 2.0) + 1))
        if k % 2 == 0:
            k += 1
        pad = k // 2
        low = F.avg_pool2d(delta, kernel_size=k, stride=1, padding=pad)
        if tuple(low.shape[-2:]) != tuple(delta.shape[-2:]):
            low = F.interpolate(low, size=delta.shape[-2:], mode="nearest")
        return delta - low

    def _apply_repel_prev(self, z, t_tensor, v, prev_ref):
        """Push the current estimate away from the previous ODE step's result.

        ``x1``: away from last clean estimate ``z+(1-t)v``.
        ``state``: away from last latent/pixel state ``z``.
        """
        gamma = float(getattr(self, "repel_prev_scale", 0.0))
        if gamma == 0.0 or prev_ref is None:
            return v
        mode = str(getattr(self, "repel_prev_mode", "x1")).lower()
        t_ = t_tensor.view(-1, *([1] * (z.ndim - 1))).to(dtype=z.dtype)
        if mode == "state":
            cur = z
        else:
            cur = z + (1.0 - t_) * v
        away = cur - prev_ref
        dims = tuple(range(1, away.ndim))
        nrm = torch.linalg.vector_norm(away, ord=2, dim=dims, keepdim=True).clamp_min(1e-8)
        v = v + gamma * (away / nrm)
        if not getattr(self, "_repel_logged", False):
            self._repel_logged = True
            print(
                f"[repel] mode={mode} γ={gamma:g} |away|/n={float((away / nrm).abs().mean()):.4g}",
                flush=True,
            )
        return v

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
        z_t_ref = CFMODEMixin._ot_path_state(z0, z1_ref, t_scalar)
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


    def _spall_infer_cond(self, x_prev, g_curr, g_prev, coords, spatial_hw):
        """``[pool, feathered m_grow]`` and the old-dark pixels locked to ``x_prev``."""
        from src.model.cfm.spall_mask import (
            coarse_coords,
            coarse_pool,
            dilate_mask,
            extent_mask,
            feather_mask,
            hard_mask,
            solidify_extent,
            upsample_mask,
        )

        if g_curr is None:
            raise ValueError("gru_feat is required for spall growth inference")
        if g_prev is None:
            g_prev = torch.zeros_like(g_curr)
        grid = int(self.spall_mask_grid)
        pool = coarse_pool(x_prev, grid)
        m_prev = extent_mask(
            x_prev,
            grid,
            self.spall_dark_level,
            self.spall_bg_quantile,
            self.spall_bg_target,
            self.spall_border_gap,
            seed_margin=float(getattr(self, "spall_seed_margin", 0.2)),
        )
        logits = self.spall_mask_head(
            pool, g_prev, g_curr, coarse_coords(coords, grid, pool)
        )
        prob = torch.sigmoid(logits)
        if bool(getattr(self, "spall_guide_m", False)):
            from src.model.cfm.spall_mask import core_feather

            extent = solidify_extent(
                hard_mask(prob, self.spall_mask_threshold),
                close_k=int(getattr(self, "spall_close_k", 3)),
            )
            alpha = core_feather(extent, spatial_hw, float(self.spall_feather_px)).to(dtype=x_prev.dtype)
            lock = (1.0 - alpha).detach()
            self._last_spall_vis = {
                "m_prob": upsample_mask(prob, spatial_hw).detach(),
                "m_grow": alpha.detach(),
                "lock": lock,
            }
            return torch.cat([upsample_mask(pool, spatial_hw), alpha.detach()], dim=1), lock
        if bool(getattr(self, "spall_soft_paint", False)):
            from src.model.cfm.spall_mask import soft_paint_alpha

            alpha = soft_paint_alpha(
                x_prev,
                prob,
                m_prev,
                spatial_hw,
                grid=grid,
                threshold=self.spall_mask_threshold,
                close_k=int(getattr(self, "spall_close_k", 3)),
                bg_quantile=self.spall_bg_quantile,
                bg_target=self.spall_bg_target,
                border_gap=self.spall_border_gap,
                healthy_margin=float(getattr(self, "spall_healthy_margin", 0.10)),
                sigma=float(self.spall_feather_px),
                lock_healthy=bool(getattr(self, "spall_lock_healthy", True)),
            ).to(dtype=x_prev.dtype)
            lock = (1.0 - alpha).detach()
            self._last_spall_vis = {
                "m_prob": upsample_mask(prob, spatial_hw).detach(),
                "m_grow": alpha.detach(),
                "lock": lock,
            }
            return torch.cat([upsample_mask(pool, spatial_hw), alpha.detach()], dim=1), lock
        if bool(getattr(self, "spall_cond_prob", False)):
            prob_up = F.interpolate(prob, size=spatial_hw, mode="bilinear", align_corners=False)
            self._last_spall_vis = {
                "m_prob": prob_up.detach(),
                "m_grow": prob_up.detach(),
                "lock": torch.zeros_like(prob_up),
            }
            return torch.cat([upsample_mask(pool, spatial_hw), prob_up], dim=1), None
        mode = str(getattr(self, "spall_apply_mode", "retrain"))
        spatial_up = upsample_mask(pool, spatial_hw)
        if mode == "legacy":
            # Trained behavior: incremental m, paste the complement after the ODE.
            m_soft = feather_mask(
                hard_mask(prob, self.spall_mask_threshold),
                spatial_hw,
                self.spall_feather_px,
            )
            keep = (m_soft > 1e-3).to(dtype=x_prev.dtype)
            self._last_spall_vis = {
                "m_prob": upsample_mask(prob, spatial_hw).detach(),
                "m_grow": m_soft.detach(),
                "lock": keep.detach(),
            }
            return torch.cat([spatial_up, m_soft], dim=1), keep
        if mode == "tricks":
            # Same incremental head. Dilate m, lock old-dark cells, RePaint those.
            grown = dilate_mask(hard_mask(prob, self.spall_mask_threshold), 3)
            m_soft = feather_mask(grown, spatial_hw, self.spall_feather_px)
            lock = upsample_mask(m_prev, spatial_hw).to(dtype=x_prev.dtype)
            self._last_spall_vis = {
                "m_prob": upsample_mask(prob, spatial_hw).detach(),
                "m_grow": m_soft.detach(),
                "lock": lock.detach(),
            }
            return torch.cat([spatial_up, m_soft], dim=1), lock
        extent = solidify_extent(
            hard_mask(prob, self.spall_mask_threshold),
            close_k=int(getattr(self, "spall_close_k", 3)),
        )
        # Full solid extent, not the rim left after subtracting the previous frame.
        m_grow = extent
        m_soft = feather_mask(m_grow, spatial_hw, self.spall_feather_px)
        lock = upsample_mask(m_prev, spatial_hw).to(dtype=x_prev.dtype)
        lock = lock * (1.0 - upsample_mask(extent, spatial_hw))
        self._last_spall_vis = {
            "m_prob": upsample_mask(prob, spatial_hw).detach(),
            "m_grow": m_soft.detach(),
            "lock": lock.detach(),
        }
        return torch.cat([spatial_up, m_soft], dim=1), lock


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
        spall_prev=None,
        spall_g_prev=None,
        gru_delta=None,
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

        if self.use_cascade_sr or self.use_refiner_residual:
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

        spall_cond = None
        spall_lock = None
        sdedit_strength = None
        if getattr(self, "use_spall_growth_mask", False) and spall_prev is not None:
            spall_prev = spall_prev.to(device=z.device, dtype=z.dtype)
            if tuple(spall_prev.shape[-2:]) != tuple(z.shape[-2:]):
                spall_prev = F.interpolate(
                    spall_prev, size=z.shape[-2:], mode="bilinear", align_corners=False
                )
            if spall_g_prev is not None:
                spall_g_prev = spall_g_prev.to(device=z.device)
            if bool(getattr(self, "spall_sdedit", False)):
                from src.model.cfm.spall_mask import coarse_coords, coarse_pool

                grid = int(self.spall_mask_grid)
                pool = coarse_pool(spall_prev, grid)
                g_curr = gru_feat
                g_for_prev = spall_g_prev if spall_g_prev is not None else torch.zeros_like(g_curr)
                logits = self.spall_mask_head(
                    pool, g_for_prev, g_curr, coarse_coords(coords, grid, pool)
                )
                prob = torch.sigmoid(logits)
                prob_up = F.interpolate(
                    prob, size=z.shape[-2:], mode="bilinear", align_corners=False
                )
                floor = float(getattr(self, "spall_sdedit_floor", 0.0))
                if floor > 0.0:
                    # Drop the faint bottom haze. Only mass above the floor can edit.
                    prob_up = ((prob_up - floor) / max(1e-6, 1.0 - floor)).clamp(0.0, 1.0)
                cap = float(getattr(self, "spall_sdedit_max", 1.0))
                sdedit_strength = (prob_up * cap).clamp(0.0, 1.0).detach()
                self._last_spall_vis = {
                    "m_prob": F.interpolate(
                        prob, size=z.shape[-2:], mode="bilinear", align_corners=False
                    ).detach(),
                    "m_grow": sdedit_strength,
                    "lock": (1.0 - sdedit_strength).detach(),
                }
            else:
                spall_cond, spall_lock = self._spall_infer_cond(
                    spall_prev, gru_feat, spall_g_prev, coords, z.shape[-2:]
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

        feat_on = self._feat_guidance_enabled(gru_feat)
        if float(getattr(self, "apg_guidance_scale", 0.0)) != 0.0:
            self._apg_momentum_buffer = MomentumBuffer(
                momentum=float(getattr(self, "apg_momentum", -0.5))
            )
        else:
            self._apg_momentum_buffer = None

        autoguide_w = float(getattr(self, "autoguide_weight", 0.0))
        autoguide_bad = None
        if autoguide_w != 0.0:
            self._attach_autoguide()
            autoguide_bad = self._autoguide_ref[0]

        prev_ref = None
        repel_mode = str(getattr(self, "repel_prev_mode", "x1")).lower()
        self._repel_logged = False

        def _guided_velocity(z_state, t_tensor, t_scalar, apply_repel=True):
            """Predict v_θ then apply latent- or velocity-space reference guidance."""
            if use_ref and guidance_mode == "latent":
                z_in = self._blend_latent_toward_ot_ref(
                    z_state, z0, z1_ref, t_scalar, guidance_weight
                )
            else:
                z_in = z_state

            def _predict_at(z_eval, capture_attn=False, gru_override=None):
                mid = getattr(self.velocity_net, "mid_attn", None)
                if capture_attn and mid is not None and hasattr(mid, "store_attn"):
                    mid.store_attn = True
                v_pred = self._predict_velocity(
                    z_eval,
                    t_tensor,
                    c,
                    coords=coords,
                    damage_mask=condition_mask,
                    gru_feat=gru_feat if gru_override is None else gru_override,
                    self_cond=self_cond,
                    lr_cond=lr_cond,
                    spall_cond=spall_cond,
                    gru_delta=gru_delta,
                )
                if capture_attn and mid is not None and hasattr(mid, "store_attn"):
                    mid.store_attn = False
                return v_pred

            capture_sag = self._sag_enabled(t_scalar)
            v = _predict_at(z_in, capture_attn=capture_sag)
            if autoguide_w != 0.0:
                c_bad = c
                bad_in = int(autoguide_bad.velocity_net.scalar_mlp[0].in_features)
                if c_bad.shape[1] != bad_in and c_bad.shape[1] == bad_in + 1:
                    c_bad = c_bad[:, 1:]
                with torch.no_grad():
                    v_bad = autoguide_bad._predict_velocity(
                        z_in,
                        t_tensor,
                        c_bad,
                        coords=coords,
                        damage_mask=condition_mask,
                        gru_feat=gru_feat,
                        self_cond=self_cond,
                        lr_cond=lr_cond,
                        spall_cond=spall_cond,
                        ref_feat=getattr(self, "_undamaged_ref_embed", None),
                    )
                diff = v - v_bad
                ag_wl = float(getattr(self, "autoguide_wl", 1.0))
                ag_wh = float(getattr(self, "autoguide_wh", 1.0))
                if ag_wl != 1.0 or ag_wh != 1.0:
                    d_low, d_high = self._facfg_split(diff)
                    diff = ag_wl * d_low + ag_wh * d_high
                eta = float(getattr(self, "autoguide_eta", 1.0))
                if eta != 1.0:
                    parallel, orthogonal = _project_onto(diff, v)
                    diff = orthogonal + eta * parallel
                gamma = float(getattr(self, "autoguide_gamma", 0.0))
                w_min = float(getattr(self, "autoguide_w_min", 0.0))
                w_t = max(w_min, autoguide_w * (1.0 - float(t_scalar)) ** gamma)
                if not getattr(self, "_autoguide_logged", False):
                    self._autoguide_logged = True
                    raw = (v.detach() - v_bad).abs().mean()
                    print(
                        f"[autoguide] w_ref={autoguide_w:g} w(t)={w_t:.4g} gamma={gamma:g} "
                        f"w_min={w_min:g} eta={eta:g} t={float(t_scalar):.3f} "
                        f"|v_s-v_bad|mean={float(raw):.4g} "
                        f"|diff|mean={float(diff.detach().abs().mean()):.4g} "
                        f"|v_s|mean={float(v.detach().abs().mean()):.4g}",
                        flush=True,
                    )
                v = v + w_t * diff
            g0 = getattr(self, "_facfg_gru_unc", None)
            apg_g0_on = bool(str(getattr(self, "apg_g0", "") or "")) and (
                float(getattr(self, "apg_guidance_scale", 0.0)) != 0.0
            )
            if g0 is not None and getattr(self, "facfg_wl", None) is not None and not apg_g0_on:
                v = self._apply_facfg(
                    v, lambda: _predict_at(z_in, capture_attn=False, gru_override=g0)
                )
            apg_g0 = str(getattr(self, "apg_g0", "") or "")
            if apg_g0 and float(getattr(self, "apg_guidance_scale", 0.0)) != 0.0:
                # APG on (v_cond, v_g0): the "unconditional" branch swaps only g.
                g_unc = torch.zeros_like(gru_feat) if apg_g0 == "zero" else getattr(self, "_facfg_gru_unc", None)
                if g_unc is None:
                    raise RuntimeError("apg_g0=undamaged needs _facfg_gru_unc set per sample")
                v_unc = _predict_at(z_in, capture_attn=False, gru_override=g_unc)
                v_cond = v
                facfg_wl = getattr(self, "facfg_wl", None)
                facfg_wh = getattr(self, "facfg_wh", None)
                use_band = facfg_wl is not None and facfg_wh is not None
                # With guidance_scale=2 apg_guidance returns v_cond + diff', where diff' is
                # the momentum / norm-clipped / projected difference.
                v_apg = apg_guidance(
                    v_cond,
                    v_unc,
                    guidance_scale=2.0 if use_band else float(self.apg_guidance_scale),
                    eta=float(getattr(self, "apg_eta", 0.0)),
                    norm_threshold=float(getattr(self, "apg_norm_threshold", 0.0)),
                    momentum_buffer=getattr(self, "_apg_momentum_buffer", None),
                )
                if use_band:
                    # APG + FA-CFG: per-band CFG weight (w - 1) on the processed difference.
                    d_low, d_high = self._facfg_split(v_apg - v_cond)
                    v = v_cond + (float(facfg_wl) - 1.0) * d_low + (float(facfg_wh) - 1.0) * d_high
                else:
                    v = v_apg
                if not getattr(self, "_apg_g0_logged", False):
                    self._apg_g0_logged = True
                    print(
                        f"[apg-g0] g0={apg_g0} s={float(self.apg_guidance_scale):g} "
                        f"bands={'wl=%g wh=%g' % (float(facfg_wl), float(facfg_wh)) if use_band else 'off'} "
                        f"eta={float(self.apg_eta):g} norm_thr={float(self.apg_norm_threshold):g} "
                        f"|v_c-v_g0|mean={float((v_cond - v_unc).detach().abs().mean()):.4g} "
                        f"|v_c|mean={float(v_cond.detach().abs().mean()):.4g}",
                        flush=True,
                    )
            coord_g0 = str(getattr(self, "coord_g0", "") or "")
            if coord_g0 and float(getattr(self, "coord_guidance_scale", 0.0)) != 0.0:
                # High-pass guide on (v_cond, v_g0): v += gamma * highpass(v - v_g0).
                t_lo = float(getattr(self, "coord_guidance_t_min", 0.0))
                t_hi = float(getattr(self, "coord_guidance_t_max", 1.0))
                if t_lo <= float(t_scalar) < t_hi:
                    g_unc = torch.zeros_like(gru_feat) if coord_g0 == "zero" else getattr(self, "_facfg_gru_unc", None)
                    if g_unc is None:
                        raise RuntimeError("coord_g0=undamaged needs _facfg_gru_unc set per sample")
                    v_unc = _predict_at(z_in, capture_attn=False, gru_override=g_unc)
                    delta = self._highpass_coord_delta(v - v_unc)
                    if not getattr(self, "_coord_g0_logged", False):
                        self._coord_g0_logged = True
                        print(
                            f"[coordg-g0] g0={coord_g0} gamma={float(self.coord_guidance_scale):g} "
                            f"hipass_sigma={float(getattr(self, 'coord_guidance_hipass_sigma', 0.0)):g} "
                            f"t=[{t_lo:g},{t_hi:g}) |delta|mean={float(delta.detach().abs().mean()):.4g} "
                            f"|v|mean={float(v.detach().abs().mean()):.4g}",
                            flush=True,
                        )
                    v = v + float(self.coord_guidance_scale) * delta
            if use_ref and guidance_mode == "velocity":
                v = self._blend_velocity_toward_ot_ref(
                    v, z0, z1_ref, guidance_weight
                )
            if self._coord_guidance_enabled():
                v = self._apply_coord_guidance(
                    z_in,
                    v,
                    predict_fn=lambda z_m: _predict_at(z_m, capture_attn=False),
                    t_scalar=t_scalar,
                )
            if capture_sag:
                v = self._apply_sag_guidance(
                    z_in,
                    t_tensor,
                    t_scalar,
                    v,
                    predict_fn=lambda z_m: _predict_at(z_m, capture_attn=False),
                )
            v = self._apply_feat_guidance(z_in, t_tensor, t_scalar, v, gru_feat)
            v = self._apply_lowfreq_spall_guide(z_in, t_scalar, v, gru_feat)
            if apply_repel:
                v = self._apply_repel_prev(z_in, t_tensor, v, prev_ref)
            return v, z_in

        # Integrate only over [t_start, 1]; scale step count with remaining span.
        span = 1.0 - t_start
        n_steps = max(1, int(round(float(num_steps) * span)))
        # Exact endpoint t=1 via linspace nodes (avoids drift from repeated dt adds).
        times = [t_start + span * float(i) / float(n_steps) for i in range(n_steps + 1)]

        from src.model.cfm.spall_mask import repaint_known

        legacy_paste = str(getattr(self, "spall_apply_mode", "retrain")) == "legacy"

        def _project_spall(z_state, t_scalar):
            if sdedit_strength is not None:
                t_now = float(t_scalar)
                anchor = (1.0 - t_now) * z0 + t_now * spall_prev
                # Strength m starts editing at t=1-m. Before that, stay on x_prev's noise path.
                hold = (sdedit_strength <= (1.0 - t_now)).to(dtype=z_state.dtype)
                if t_now >= 1.0 - 1e-8:
                    hold = (sdedit_strength <= 0.0).to(dtype=z_state.dtype)
                    anchor = spall_prev
                return hold * anchor + (1.0 - hold) * z_state
            if legacy_paste or spall_lock is None:
                return z_state
            return repaint_known(z_state, z0, spall_prev, t_scalar, spall_lock)

        for i in range(n_steps):
            t_scalar = float(times[i])
            t_next_scalar = float(times[i + 1])
            dt = t_next_scalar - t_scalar
            t = torch.full((c.shape[0],), t_scalar, device=c.device)
            z = _project_spall(z, t_scalar)

            # --- Heun stage 1 ---
            v_t, z_for_clean = _guided_velocity(z, t, t_scalar)
            t_view = t.view(-1, *([1] * (z.ndim - 1)))
            cur_ref = (
                z
                if repel_mode == "state"
                else (z + (1.0 - t_view) * v_t)
            )
            z1_hat = (
                self._estimate_clean_from_velocity(z_for_clean, t, v_t).detach()
                if self.use_self_conditioning
                else None
            )

            if self.cfm_solver == "heun":
                z_pred = z + dt * v_t
                if feat_on:
                    z_pred = z_pred.detach()
                z_pred = _project_spall(z_pred, t_next_scalar)
                t_next = torch.full((c.shape[0],), t_next_scalar, device=c.device)
                # Stage 2 does not apply a second repulsion (prev_ref only on stage 1).
                v_next, _ = _guided_velocity(z_pred, t_next, t_next_scalar, apply_repel=False)
                z = z + 0.5 * dt * (v_t + v_next)
            else:
                z = z + dt * v_t
            if feat_on:
                z = z.detach()
            z = _project_spall(z, t_next_scalar)
            prev_ref = cur_ref.detach()

            if self.use_self_conditioning:
                self_cond = z1_hat
        if legacy_paste and spall_lock is not None:
            z = spall_lock * z + (1.0 - spall_lock) * spall_prev
        if self.use_cascade_sr or self.use_refiner_residual:
            z = self._compose_cascade_sample(z, lr_cond)
        return z

    def sample_spall_sequence(
        self, conds, gru_feats, num_steps=None, damage_mask=None, return_masks=False
    ):
        """Generate a time sequence. Each step sees Pool of the previous output and both ``g``."""
        if not self.use_spall_growth_mask:
            raise ValueError("sample_spall_sequence requires use_spall_growth_mask=True")
        if len(conds) != len(gru_feats):
            raise ValueError(
                f"conds ({len(conds)}) and gru_feats ({len(gru_feats)}) must have the same length"
            )
        per_frame_mask = isinstance(damage_mask, (list, tuple))
        if per_frame_mask and len(damage_mask) != len(conds):
            raise ValueError(
                f"damage_mask ({len(damage_mask)}) and conds ({len(conds)}) must have the same length"
            )
        frames = []
        mask_frames = []
        x_prev = None
        g_prev = None
        spatial = int(self.model_spatial_size)
        empty = {
            "m_prob": torch.zeros(1, 1, spatial, spatial),
            "m_grow": torch.zeros(1, 1, spatial, spatial),
            "lock": torch.zeros(1, 1, spatial, spatial),
        }
        for i, (c, g) in enumerate(zip(conds, gru_feats)):
            mask_i = damage_mask[i] if per_frame_mask else damage_mask
            self._last_spall_vis = None
            x = self.sample_with_ode(
                c,
                gru_feat=g,
                num_steps=num_steps,
                damage_mask=mask_i,
                spall_prev=x_prev,
                spall_g_prev=g_prev,
            )
            frames.append(x)
            if return_masks:
                vis = getattr(self, "_last_spall_vis", None)
                if vis is None:
                    vis = {k: v.clone() for k, v in empty.items()}
                    vis = {k: v.to(device=x.device, dtype=x.dtype) for k, v in vis.items()}
                mask_frames.append({k: v.detach().cpu() for k, v in vis.items()})
            x_prev = x.detach()
            g_prev = g.detach()
        if return_masks:
            return frames, mask_frames
        return frames

    def sample_paired_bridge_sequence(
        self,
        conds,
        gru_feats,
        num_steps=None,
        damage_mask=None,
    ):
        """Walk a column. Frame 0 is noise-to-image. Later frames start at the previous output.

        The learned field is ``v = x_next - x_prev``, so integrating from the
        previous image is the bridge. An optional ``_paired_bridge_frame0``
        module draws frame 0 instead of this network.
        """
        if len(conds) != len(gru_feats):
            raise ValueError(
                f"conds ({len(conds)}) and gru_feats ({len(gru_feats)}) must have the same length"
            )
        per_frame_mask = isinstance(damage_mask, (list, tuple))
        if per_frame_mask and len(damage_mask) != len(conds):
            raise ValueError(
                f"damage_mask ({len(damage_mask)}) and conds ({len(conds)}) must have the same length"
            )
        frames = []
        x_prev = None
        x_anchor = None
        frame0 = getattr(self, "_paired_bridge_frame0", None)
        corrector = getattr(self, "_paired_bridge_corrector", None)
        corrector_s = float(getattr(self, "bridge_corrector_s", 0.0) or 0.0)
        for i, (c, g) in enumerate(zip(conds, gru_feats)):
            mask_i = damage_mask[i] if per_frame_mask else damage_mask
            if x_prev is None:
                owner = self if frame0 is None else frame0
                x = owner.sample_with_ode(
                    c,
                    gru_feat=g,
                    num_steps=num_steps,
                    damage_mask=mask_i,
                )
                x_anchor = x.detach()
            else:
                x = self.sample_with_ode(
                    c,
                    z0=x_prev,
                    gru_feat=g,
                    num_steps=num_steps,
                    damage_mask=mask_i,
                )
                if corrector is not None and corrector_s > 0.0:
                    x = self._bridge_correct_frame(
                        x,
                        c,
                        g,
                        x_anchor,
                        corrector,
                        corrector_s,
                        damage_mask=mask_i,
                        num_steps=num_steps,
                    )
            frames.append(x)
            x_prev = x.detach()
        return frames

    @staticmethod
    def _block_mean(x, block):
        """Orthogonal projection onto block-constant images (avg-pool, nearest upsample)."""
        low = F.avg_pool2d(x, kernel_size=block, stride=block)
        return F.interpolate(low, size=x.shape[-2:], mode="nearest")

    def _refiner_start_noise(self, shared, batch, device):
        """Start noise of one frame. With ``refiner_shared_lowfreq_noise`` the block-mean band comes
        from ``shared`` (one draw per trajectory) and the rest is fresh; both parts are independent
        projections of white noise, so every frame is still exactly N(0, I)."""
        size = int(self.model_spatial_size)
        eps = torch.randn((batch, self.model_in_channels, size, size), device=device)
        if shared is None:
            return eps
        block = int(self.refiner_lowfreq_block)
        return eps - self._block_mean(eps, block) + self._block_mean(shared, block)

    def sample_refiner_sequence(self, conds, gru_feats, num_steps=None, damage_mask=None):
        """Walk a column. Frame 0 is the frozen single-frame model. Each later frame is drawn
        from noise with ``S_{t-1}`` locked: V10 adds the sampled residual outside the network,
        ``S_t = clamp(S_{t-1} + residual)``; the v10.5 implicit refiner reads ``S_{t-1}`` in its
        control branch and generates ``S_t`` directly. ``refiner_shared_lowfreq_noise`` gives all
        frames of a trajectory the same low-frequency start noise (frame 0 included).
        """
        if len(conds) != len(gru_feats):
            raise ValueError(
                f"conds ({len(conds)}) and gru_feats ({len(gru_feats)}) must have the same length"
            )
        per_frame_mask = isinstance(damage_mask, (list, tuple))
        self._attach_refiner_base()
        base = self._refiner_base_ref[0]
        frames = []
        state = None
        shared = None
        if getattr(self, "refiner_shared_lowfreq_noise", False) and conds:
            shared = self._refiner_start_noise(None, conds[0].shape[0], conds[0].device)
        for i, (c, g) in enumerate(zip(conds, gru_feats)):
            mask_i = damage_mask[i] if per_frame_mask else damage_mask
            dg = None if i == 0 or not getattr(self, "use_delta_g", False) else g - gru_feats[i - 1]
            z0 = self._refiner_start_noise(shared, c.shape[0], c.device)
            if state is None:
                c_base = c
                base_in = int(base.velocity_net.scalar_mlp[0].in_features)
                if c.shape[1] + 1 == base_in:
                    c_base = torch.cat([torch.zeros_like(c[:, :1]), c], dim=1)
                x = base.sample_with_ode(c_base, z0=z0, gru_feat=g, num_steps=num_steps, damage_mask=mask_i)
            else:
                x = self.sample_with_ode(
                    c,
                    z0=z0,
                    gru_feat=g,
                    num_steps=num_steps,
                    damage_mask=mask_i,
                    lr_cond=state,
                    gru_delta=dg,
                )
            if getattr(self, "refiner_gray_state", False):
                x = x.mean(dim=1, keepdim=True).expand_as(x).contiguous()
            frames.append(x)
            state = x.detach()
        return frames

    def _bridge_correct_frame(
        self,
        x,
        c,
        g,
        x_anchor,
        corrector,
        noise_frac,
        damage_mask=None,
        num_steps=None,
    ):
        """Pull a drifted bridge frame back onto the single-frame model.

        ``noise_frac`` is the Gaussian fraction. The frozen corrector integrates
        from that time to t=1. High frequency is then taken from the first frame,
        and a dark global mean is shifted part of the way back toward concrete.
        """
        noise_frac = float(noise_frac)
        if not (0.0 < noise_frac < 1.0):
            raise ValueError(f"bridge corrector noise fraction must be in (0, 1), got {noise_frac}")
        t_start = 1.0 - noise_frac
        noise = torch.randn_like(x)
        z_init = noise_frac * noise + (1.0 - noise_frac) * x
        ode_steps = int(getattr(corrector, "cfm_ode_steps", 30) or 30)
        if num_steps is not None:
            ode_steps = int(num_steps)
        y = corrector.sample_with_ode(
            c,
            gru_feat=g,
            num_steps=max(1, ode_steps),
            t_start=t_start,
            z_init=z_init,
            damage_mask=damage_mask,
        )
        sigma = float(getattr(self, "bridge_anchor_sigma", 1.5) or 1.5)
        y = self._anchor_highpass(y, x_anchor, sigma)
        return self._pull_dark_mean(y)

    @staticmethod
    def _anchor_highpass(x, anchor, sigma):
        from torchvision.transforms.functional import gaussian_blur

        sigma = float(sigma)
        k = max(3, int(2 * math.ceil(3.0 * sigma) + 1))
        if k % 2 == 0:
            k += 1
        low = gaussian_blur(x, [k, k], [sigma, sigma])
        low_anchor = gaussian_blur(anchor, [k, k], [sigma, sigma])
        return low + (anchor - low_anchor)

    @staticmethod
    def _pull_dark_mean(x, floor=0.10, target=0.25, strength=0.5):
        mean = x.mean(dim=(1, 2, 3), keepdim=True)
        gap = (float(target) - mean).clamp(min=0.0)
        apply = (mean < float(floor)).to(dtype=x.dtype)
        return x + float(strength) * gap * apply

    def sample_flowedit_sequence(
        self,
        conds,
        gru_feats,
        num_steps=None,
        t_start=0.5,
        damage_mask=None,
    ):
        """Edit each frame from the previous one by a velocity difference.

        Same noise and time, two conditions. The image moves by
        ``v(g_t) - v(g_{t-1})``. Where ``g`` does not change, the difference
        is near zero and the previous frame stays. The first frame is a
        normal ODE from noise.
        """
        if len(conds) != len(gru_feats):
            raise ValueError(
                f"conds ({len(conds)}) and gru_feats ({len(gru_feats)}) must have the same length"
            )
        t_start = float(t_start)
        if not (0.0 <= t_start < 1.0):
            raise ValueError(f"flowedit t_start must be in [0, 1), got {t_start}")
        edit_steps = int(self.cfm_ode_steps if num_steps is None else num_steps)
        edit_steps = max(1, edit_steps)
        frames = []
        x_prev = None
        g_prev = None
        c_prev = None
        for c, g in zip(conds, gru_feats):
            if x_prev is None:
                x = self.sample_with_ode(c, gru_feat=g, damage_mask=damage_mask)
            else:
                x = self._flowedit_frame(
                    c,
                    g,
                    x_prev,
                    g_prev,
                    c_prev,
                    t_start=t_start,
                    num_steps=edit_steps,
                    damage_mask=damage_mask,
                )
            frames.append(x)
            x_prev = x.detach()
            g_prev = g.detach()
            c_prev = c.detach()
        return frames

    def _flowedit_frame(
        self,
        c,
        g_curr,
        x_prev,
        g_prev,
        c_prev,
        t_start,
        num_steps,
        damage_mask=None,
    ):
        """Euler integration of ``v(g_curr) - v(g_prev)`` on the clean image."""
        x = x_prev.detach().clone()
        device = c.device
        times = [
            float(t_start) + (1.0 - float(t_start)) * float(i) / float(num_steps)
            for i in range(num_steps + 1)
        ]
        coords = None
        if self.use_patch_diffusion:
            grid_x, grid_y = self._build_full_coordinate_maps(
                self.model_spatial_size,
                self.model_spatial_size,
                device=device,
                dtype=x.dtype,
            )
            coords = (
                torch.stack([grid_x, grid_y], dim=0)
                .unsqueeze(0)
                .expand(c.shape[0], -1, -1, -1)
                .contiguous()
            )
        if damage_mask is None and self.use_damage_mask_condition:
            condition_mask = torch.ones(
                (c.shape[0], 1, x.shape[-2], x.shape[-1]),
                device=device,
                dtype=x.dtype,
            )
        else:
            condition_mask = damage_mask

        def _v(z, t_scalar, cond, gru):
            t = torch.full((cond.shape[0],), float(t_scalar), device=device, dtype=z.dtype)
            return self._predict_velocity(
                z,
                t,
                cond,
                coords=coords,
                damage_mask=condition_mask,
                gru_feat=gru,
            )

        for i in range(num_steps):
            t_now = times[i]
            dt = times[i + 1] - t_now
            noise = torch.randn_like(x)
            z_src = (1.0 - t_now) * noise + t_now * x_prev
            z_tar = x + z_src - x_prev
            v_delta = _v(z_tar, t_now, c, g_curr) - _v(z_src, t_now, c_prev, g_prev)
            x = x + dt * v_delta
        # The difference places the damage. A short ODE of v(g_curr) near the
        # image puts the cracks back; the shared grid already cancelled out.
        refine_t = float(getattr(self, "flowedit_refine_t", 0.0) or 0.0)
        if refine_t > 0.0:
            if not (0.0 < refine_t < 1.0):
                raise ValueError(f"flowedit_refine_t must be in (0, 1), got {refine_t}")
            noise = torch.randn_like(x)
            z_init = (1.0 - refine_t) * noise + refine_t * x
            ode_steps = int(getattr(self, "cfm_ode_steps", 30) or 30)
            x = self.sample_with_ode(
                c,
                num_steps=max(1, ode_steps),
                gru_feat=g_curr,
                t_start=refine_t,
                z_init=z_init,
                damage_mask=damage_mask,
            )
        return x

    def sample_with_sde(
        self,
        c,
        z0=None,
        num_steps=None,
        gru_feat=None,
        damage_mask=None,
        lr_cond=None,
        sde_sigma=None,
        ode_tail=0.08,
    ):
        """Euler–Maruyama SDE with the same OT interpolant marginals as the PF-ODE.

        Convention ``t=0`` noise, ``t=1`` data, ``z_t=(1-t)z0+t z1``, ``v≈z1-z0``.
        Score from the interpolant: ``z0_hat=z-t v``, ``∇log p≈-z0_hat/(1-t)^2``.
        Diffusion ``g(t)=σ(1-t)``; last ``ode_tail`` of time is pure ODE so grain dies.
        """
        if self.use_cascade_sr or self.use_refiner_residual:
            raise ValueError("sample_with_sde is not supported with use_cascade_sr / refiner residual")
        if lr_cond is not None:
            raise ValueError("sample_with_sde does not take lr_cond")
        if num_steps is None:
            num_steps = self.cfm_ode_steps
        num_steps = max(1, int(num_steps))
        sigma = float(self.cfm_sde_sigma if sde_sigma is None else sde_sigma)
        if sigma < 0.0:
            raise ValueError(f"sde_sigma must be >= 0, got {sigma}")
        tail = min(0.5, max(0.0, float(ode_tail)))

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

        self_cond = torch.zeros_like(z) if self.use_self_conditioning else None
        times = [float(i) / float(num_steps) for i in range(num_steps + 1)]

        for i in range(num_steps):
            t_scalar = float(times[i])
            t_next_scalar = float(times[i + 1])
            dt = t_next_scalar - t_scalar
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
            if self.use_self_conditioning:
                self_cond = self._estimate_clean_from_velocity(z, t, v_t).detach()

            use_noise = (sigma > 0.0) and (t_scalar < 1.0 - tail)
            if use_noise:
                t_clamped = max(t_scalar, 1e-4)
                z0_hat = z - t_clamped * v_t
                g = sigma * (1.0 - t_scalar)
                drift = v_t - 0.5 * (sigma ** 2) * z0_hat
                eps = torch.randn_like(z)
                z = z + drift * dt + g * (dt ** 0.5) * eps
            elif self.cfm_solver == "heun":
                z_pred = z + dt * v_t
                t_next = torch.full((c.shape[0],), t_next_scalar, device=c.device)
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
        return z
