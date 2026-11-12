"""OT-CFM ODE sampling helpers."""

import contextlib
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


    def _refiner_x1_projection(self, z, v, t_scalar, mem):
        """Correct the clean estimate x1 = z + (1 - t) v by Delta, keeping the implied noise z - t v:
        z += t * Delta, v += Delta. Two optional terms (sum):
          refiner_mono_proj   on established memory entries (mem < -refiner_mono_delta) and t >= refiner_mono_t,
                              the structure deficit D = relu(f(x1) - kappa * mem) (f = memory_feature) is darkened
                              back, high-pass only (D - G_4(D)), times refiner_mono_lambda;
          refiner_tone_env    eps gray: on intact 16 px blocks (no memory entry), the block-mean offset of x1 against
                              self._tone_ref beyond +-eps is removed, spread smoothly to all blocks; t >= refiner_tone_env_t."""
        from src.model.cfm.refiner_memory import _gauss_1d, memory_feature
        x1 = z + (1.0 - t_scalar) * v
        delta = torch.zeros_like(x1[:, :1]).float()

        def blur(x, sigma):
            k, r = _gauss_1d(sigma, x.device, torch.float32)
            y = F.conv2d(F.pad(x, (r, r, 0, 0), mode="reflect"), k.view(1, 1, 1, -1))
            return F.conv2d(F.pad(y, (0, 0, r, r), mode="reflect"), k.view(1, 1, -1, 1))

        if bool(getattr(self, "refiner_mono_proj", 0)) and mem is not None \
                and t_scalar >= float(getattr(self, "refiner_mono_t", 0.4)):
            kappa = float(getattr(self, "refiner_mono_kappa", 0.6))
            est = (mem < -float(getattr(self, "refiner_mono_delta", 0.03))).float()
            f = memory_feature(x1, float(self.refiner_memory_sigma), float(self.refiner_memory_tau)).float()
            D = torch.relu(f - kappa * mem) * est
            delta = delta - float(getattr(self, "refiner_mono_lambda", 1.0)) * (D - blur(D, 4.0))
        eps = float(getattr(self, "refiner_tone_env", 0.0))
        ref = getattr(self, "_tone_ref", None)
        if eps > 0 and ref is not None and t_scalar >= float(getattr(self, "refiner_tone_env_t", 0.5)):
            bs = 16
            g = x1.float().mean(dim=1, keepdim=True)
            r = ref.float().mean(dim=1, keepdim=True).to(g.device)
            diff = F.avg_pool2d(g, bs) - F.avg_pool2d(r, bs)
            if mem is not None:
                # new damage = memory structure the reference frame does not have (the painted grid and the damage
                # already in the reference are in both); a block with any new damage is not intact
                fr = memory_feature(ref.float().to(g.device), float(self.refiner_memory_sigma),
                                    float(self.refiner_memory_tau)).float()
                old_s = F.max_pool2d((fr < -0.03).float(), 5, stride=1, padding=2) > 0
                new_s = ((mem < -0.03) & ~old_s).float()
                intact = (F.max_pool2d(new_s, bs) <= 0).float()
            else:
                intact = torch.ones_like(diff)
            e = torch.sign(diff) * torch.relu(diff.abs() - eps / 127.5) * intact
            k, rr = _gauss_1d(1.5, e.device, torch.float32)
            sm = lambda x: F.conv2d(F.pad(F.conv2d(F.pad(x, (rr, rr, 0, 0), mode="replicate"), k.view(1, 1, 1, -1)),
                                          (0, 0, rr, rr), mode="replicate"), k.view(1, 1, -1, 1))
            field = sm(e) / sm(intact).clamp(min=1e-3)
            field = torch.where(intact.sum(dim=(-2, -1), keepdim=True) > 0, field, torch.zeros_like(field))
            delta = delta - F.interpolate(field, size=g.shape[-2:], mode="bilinear", align_corners=False)
        delta = delta.to(v.dtype).expand_as(v)
        return z + t_scalar * delta, v + delta

    def _refiner_trust_mask(self, lr_cond, mode):
        """[B, 1, H, W] where S may be trusted, from the hint [S | memory | confidence]: "mem" = memory structure,
        "agree" (or "agree_mean") = memory structure that S also draws (memory_feature of S), both dilated by
        refiner_conf_map_dilate px. "mean" uses the "mem" map."""
        from src.model.cfm.refiner_memory import memory_feature
        n_mem = int(getattr(self, "refiner_memory_hint_channels", 0))
        c0 = int(self.model_in_channels)
        dil = int(getattr(self, "refiner_conf_map_dilate", 2))
        g = lr_cond[:, c0:c0 + n_mem].float().amin(dim=1, keepdim=True) < -0.03
        if mode.startswith("agree"):
            fs = memory_feature(lr_cond[:, :c0].float(), float(self.refiner_memory_sigma),
                                float(self.refiner_memory_tau)).float()
            g = g & (F.max_pool2d((fs < -0.03).float(), 5, stride=1, padding=2) > 0)
        g = g.float()
        if dil > 0:
            g = F.max_pool2d(g, 2 * dil + 1, stride=1, padding=dil)
        return g

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

        # Memory-gated spatial confidence (refiner_conf_map, set at sampling): the confidence plane becomes
        # conf + (refiner_conf_map_value - conf) * g, g = memory structure (< -0.03) dilated by refiner_conf_map_dilate
        # px; "mean" feeds the per-sample mean of that map as a constant plane (the non-local control).
        cmap = str(getattr(self, "refiner_conf_map", "") or "")
        cg_mask_mode = str(getattr(self, "refiner_conf_guidance_mask", "") or "")
        cg_mask = None
        if cg_mask_mode and lr_cond is not None:
            cg_mask = self._refiner_trust_mask(lr_cond, cg_mask_mode)
        if cmap and lr_cond is not None:
            n_mem = int(getattr(self, "refiner_memory_hint_channels", 0))
            n_cf = int(getattr(self, "refiner_conf_hint_channels", 0))
            if n_mem <= 0 or n_cf <= 0:
                raise ValueError("refiner_conf_map needs a memory and a confidence channel")
            c0 = int(self.model_in_channels)
            k0 = lr_cond.shape[1] - n_cf
            g = self._refiner_trust_mask(lr_cond, cmap)
            lr_cond = lr_cond.clone()
            base = lr_cond[:, k0:k0 + 1].float()
            cm = base + (float(getattr(self, "refiner_conf_map_value", 1.0)) - base) * g
            if cmap.endswith("mean"):
                cm = cm.mean(dim=(-2, -1), keepdim=True).expand_as(cm)
            lr_cond[:, k0:k0 + 1] = cm.to(lr_cond.dtype)
            if n_cf > 1 and bool(getattr(self, "refiner_conf_map_hops0", False)) and cmap != "mean":
                lr_cond[:, k0 + 1:k0 + 2] = lr_cond[:, k0 + 1:k0 + 2] * (1.0 - g).to(lr_cond.dtype)
        ctrl = getattr(self.velocity_net, "control", None)
        if ctrl is not None:
            dc_ref = getattr(self, "refiner_conf_dc_ref", None)
            n_cf = int(getattr(self, "refiner_conf_hint_channels", 0))
            ctrl._conf_dc = None if dc_ref is None or n_cf <= 0 else (
                float(dc_ref), float(getattr(self, "refiner_conf_dc_keep", 0.0)), n_cf)

        # Memory guidance (refiner_memory, set at sampling): v = v_zero + w (v_mem - v_zero), where v_zero reads
        # the dropped (all-zero) memory the model was trained on; w < 1 trusts the memory less, w > 1 more.
        # Optional two-segment schedule over flow time (t = 0 noise, 1 data): w for t < refiner_memory_guidance_t,
        # refiner_memory_guidance_late for t >= it (unset = constant w).
        mem_w = float(getattr(self, "refiner_memory_guidance", 1.0))
        mem_w_late = getattr(self, "refiner_memory_guidance_late", None)
        mem_w_late = mem_w if mem_w_late is None else float(mem_w_late)
        mem_t_sw = float(getattr(self, "refiner_memory_guidance_t", 0.5))

        def _mem_w_at(t_scalar):
            return mem_w if float(t_scalar) < mem_t_sw else mem_w_late

        lr_zero_mem = None
        if mem_w != 1.0 or mem_w_late != 1.0:
            n_mem = int(getattr(self, "refiner_memory_hint_channels", 0))
            if not getattr(self, "refiner_memory", False) or n_mem <= 0 or lr_cond is None:
                raise ValueError("refiner_memory_guidance needs a refiner_memory model and its hint")
            lr_zero_mem = lr_cond.clone()
            c0 = int(self.model_in_channels)  # hint = [S | memory | conf]
            lr_zero_mem[:, c0:c0 + n_mem] = 0.0

        # S guidance (refiner, set at sampling): v = v_lp + (1 + a) (v - v_lp), v_lp reads S Gaussian-blurred
        # (sigma refiner_state_guidance_sigma px): amplifies what S's fine structure adds (context-aware decoding).
        # Applied for flow time t >= refiner_state_guidance_t. Confidence guidance: v = v + w (v_conf1 - v),
        # v_conf1 reads the confidence channels of a real S (conf 1, hops 0).
        st_a = float(getattr(self, "refiner_state_guidance", 0.0))
        st_t = float(getattr(self, "refiner_state_guidance_t", 0.0))
        cf_w = float(getattr(self, "refiner_conf_guidance", 0.0))
        lr_lp_state = lr_conf1 = None
        if (st_a != 0.0 or cf_w != 0.0) and lr_cond is not None:
            c0 = int(self.model_in_channels)
            if st_a != 0.0:
                from src.model.cfm.refiner_memory import _gauss_1d
                k, r = _gauss_1d(float(getattr(self, "refiner_state_guidance_sigma", 2.0)), lr_cond.device, torch.float32)
                sx = lr_cond[:, :c0].float()
                sx = F.conv2d(F.pad(sx.reshape(-1, 1, *sx.shape[-2:]), (r, r, 0, 0), mode="reflect"), k.view(1, 1, 1, -1))
                sx = F.conv2d(F.pad(sx, (0, 0, r, r), mode="reflect"), k.view(1, 1, -1, 1))
                lr_lp_state = lr_cond.clone()
                lr_lp_state[:, :c0] = sx.reshape(lr_cond.shape[0], c0, *lr_cond.shape[-2:]).to(lr_cond.dtype)
            if cf_w != 0.0:
                n_cf = int(getattr(self, "refiner_conf_hint_channels", 0))
                if n_cf <= 0:
                    raise ValueError("refiner_conf_guidance needs refiner_state_conf")
                k0 = lr_cond.shape[1] - n_cf
                lr_conf1 = lr_cond.clone()
                lr_conf1[:, k0] = 1.0
                if n_cf > 1:
                    lr_conf1[:, k0 + 1] = 0.0

        def _gate(lr):
            return self._refiner_memory_time_gate(lr, float(t_cur[0])) if getattr(self, "refiner_memory", False) else lr

        t_cur = [0.0]

        # x1 projections (refiner, set at sampling): irreversibility (refiner_mono_proj) and an intact-surface tone
        # band (refiner_tone_env), applied to the clean estimate x1 = z + (1 - t) v with the implied noise fixed.
        proj_on = bool(getattr(self, "refiner_mono_proj", 0)) or float(getattr(self, "refiner_tone_env", 0.0)) > 0
        mem_raw = None
        if proj_on and lr_cond is not None:
            n_mem = int(getattr(self, "refiner_memory_hint_channels", 0))
            c0 = int(self.model_in_channels)
            mem_raw = lr_cond[:, c0:c0 + n_mem].float().amin(dim=1, keepdim=True) if n_mem > 0 else None

        prev_ref = None
        repel_mode = str(getattr(self, "repel_prev_mode", "x1")).lower()
        self._repel_logged = False

        def _guided_velocity(z_state, t_tensor, t_scalar, apply_repel=True):
            """Predict v_θ then apply latent- or velocity-space reference guidance."""
            lr_t = self._refiner_memory_time_gate(lr_cond, float(t_scalar)) if getattr(self, "refiner_memory", False) \
                else lr_cond
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
                    lr_cond=lr_t,
                    spall_cond=spall_cond,
                    gru_delta=gru_delta,
                )
                if capture_attn and mid is not None and hasattr(mid, "store_attn"):
                    mid.store_attn = False
                return v_pred

            capture_sag = self._sag_enabled(t_scalar)
            v = _predict_at(z_in, capture_attn=capture_sag)
            w_mem = _mem_w_at(t_scalar)
            if w_mem != 1.0:
                v_zero = self._predict_velocity(
                    z_in, t_tensor, c, coords=coords, damage_mask=condition_mask, gru_feat=gru_feat,
                    self_cond=self_cond, lr_cond=lr_zero_mem, spall_cond=spall_cond, gru_delta=gru_delta,
                )
                v = v_zero + w_mem * (v - v_zero)
            t_cur[0] = float(t_scalar)
            if lr_lp_state is not None and float(t_scalar) >= st_t:
                v_lp = self._predict_velocity(
                    z_in, t_tensor, c, coords=coords, damage_mask=condition_mask, gru_feat=gru_feat,
                    self_cond=self_cond, lr_cond=_gate(lr_lp_state), spall_cond=spall_cond, gru_delta=gru_delta,
                )
                v = v_lp + (1.0 + st_a) * (v - v_lp)
            if lr_conf1 is not None:
                v_c1 = self._predict_velocity(
                    z_in, t_tensor, c, coords=coords, damage_mask=condition_mask, gru_feat=gru_feat,
                    self_cond=self_cond, lr_cond=_gate(lr_conf1), spall_cond=spall_cond, gru_delta=gru_delta,
                )
                v = v + cf_w * (v_c1 - v) * (1.0 if cg_mask is None else cg_mask.to(v.dtype))
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

        churn_s = float(getattr(self, "churn_s", 0.0) or 0.0)
        churn_lo = float(getattr(self, "churn_t_lo", 0.0))
        churn_hi = float(getattr(self, "churn_t_hi", 1.0))
        churn_noise = float(getattr(self, "churn_s_noise", 1.0))

        # Diagnostic (off unless a probe sets _lr_cond_switch = (t_sw, hint)): the hint is replaced by ``hint`` for
        # every step with t >= t_sw. Only for refiner_target_x1 models, where the hint is a condition and not the base
        # the sample is composed on.
        lr_switch = getattr(self, "_lr_cond_switch", None)
        if lr_switch is not None:
            if lr_zero_mem is not None or not getattr(self, "refiner_target_x1", False):
                raise ValueError("_lr_cond_switch needs refiner_target_x1 and no memory guidance")
            lr_late = self._ensure_cascade_lr_cond(lr_switch[1], spatial_hw=z.shape[-2:], batch=c.shape[0],
                                                   device=z.device, dtype=z.dtype)

        for i in range(n_steps):
            t_scalar = float(times[i])
            t_next_scalar = float(times[i + 1])
            dt = t_next_scalar - t_scalar
            t = torch.full((c.shape[0],), t_scalar, device=c.device)
            z = _project_spall(z, t_scalar)
            if lr_switch is not None and t_scalar >= float(lr_switch[0]) - 1e-6:
                lr_cond = lr_late  # read by _guided_velocity at call time

            # --- optional EDM-style churn (default off): inside t in [t_lo, t_hi) step back to
            # t_hat = t(1-gamma) by re-noising (variance-preserving for z_t=(1-t)z0+t z1), then
            # integrate from t_hat. Low-noise steps (t >= t_hi) stay deterministic.
            if churn_s > 0.0 and churn_lo <= t_scalar < churn_hi and t_scalar > 0.0:
                gamma = min(churn_s / float(n_steps), 2.0 ** 0.5 - 1.0)
                t_hat = t_scalar * (1.0 - gamma)
                ratio = t_hat / t_scalar
                std = max((1.0 - t_hat) ** 2 - (ratio * (1.0 - t_scalar)) ** 2, 0.0) ** 0.5
                z = ratio * z + churn_noise * std * torch.randn_like(z)
                t_scalar = t_hat
                dt = t_next_scalar - t_scalar
                t = torch.full((c.shape[0],), t_scalar, device=c.device)

            # --- Heun stage 1 ---
            v_t, z_for_clean = _guided_velocity(z, t, t_scalar)
            t_view = t.view(-1, *([1] * (z.ndim - 1)))
            if proj_on and lr_cond is not None:
                z, v_t = self._refiner_x1_projection(z, v_t, t_scalar, mem_raw)
            # Diagnostic (off unless a probe sets _x1_record_t, a list of flow times): the endpoint estimate
            # x1 = z + (1 - t) v at the first step with t >= t_ref, composed like the returned sample.
            # Stored as self._x1_trace[t_ref] = (t actually used, frame); the caller resets _x1_trace.
            rec_t = getattr(self, "_x1_record_t", None)
            if rec_t:
                for t_ref in rec_t:
                    if t_ref not in self._x1_trace and t_scalar >= float(t_ref) - 1e-6:
                        x1e = (z + (1.0 - t_view) * v_t).detach()
                        if self.use_cascade_sr or self.use_refiner_residual:
                            x1e = self._compose_cascade_sample(x1e, lr_cond)
                        self._x1_trace[t_ref] = (t_scalar, x1e)
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

            # cfm_heun_last_euler: the last step is Euler. Heun's second stage there reads v(z, t = 1), where the
            # network cannot see the start noise any more, so the average keeps about dt / 2 * z0 of it.
            last_euler = i == n_steps - 1 and bool(getattr(self, "cfm_heun_last_euler", False))
            if self.cfm_solver == "heun" and not last_euler:
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
        shape = (int(self.model_in_channels), size, size)
        split = getattr(self, "_refiner_noise_split", None)
        if split:
            # refiner_noise_streams: consecutive row blocks, each from its own generator, so a block's draws do not
            # depend on what else shares the batch.
            if sum(n for _, n in split) != batch:
                raise ValueError(f"noise streams cover {sum(n for _, n in split)} rows, batch is {batch}")
            eps = torch.cat([torch.randn((n, *shape), generator=g, device=device) for g, n in split])
        else:
            eps = torch.randn((batch, *shape), device=device)
        if shared is None:
            return eps
        block = int(self.refiner_lowfreq_block)
        return eps - self._block_mean(eps, block) + self._block_mean(shared, block)

    @contextlib.contextmanager
    def refiner_noise_streams(self, seeds, rows, device):
        """Within the block, ``_refiner_start_noise`` draws row block b (``rows[b]`` rows) from a generator seeded
        ``seeds[b]``: batching several walks / probe positions gives each the noise it gets alone."""
        prev = getattr(self, "_refiner_noise_split", None)
        self._refiner_noise_split = [(torch.Generator(device=device).manual_seed(int(s)), int(r))
                                     for s, r in zip(seeds, rows)]
        try:
            yield
        finally:
            self._refiner_noise_split = prev

    def refiner_memory_track(self):
        """A fresh memory accumulator for one walk (rule or learned encoder)."""
        from src.model.cfm.refiner_memory import LearnedMemoryTrack, MemoryTrack

        if getattr(self, "refiner_memory_encoder", False):
            return LearnedMemoryTrack(self.memory_encoder, self.model_spatial_size)
        novelty = self.refiner_memory_novelty if getattr(self, "refiner_memory_gate", "none") == "novelty" else 0.0
        return MemoryTrack(self.refiner_memory_sigma, self.refiner_memory_tau, self.refiner_memory_window,
                           int(getattr(self, "refiner_memory_ttl", 0)), novelty)

    def sample_refiner_sequence(self, conds, gru_feats, num_steps=None, damage_mask=None, memory=None,
                                memory_gate=None, frame0=None):
        """Walk a column. Frame 0 is the frozen single-frame model. Each later frame is drawn
        from noise with ``S_{t-1}`` locked: V10 adds the sampled residual outside the network,
        ``S_t = clamp(S_{t-1} + residual)``; the v10.5 implicit refiner reads ``S_{t-1}`` in its
        control branch and generates ``S_t`` directly. ``refiner_shared_lowfreq_noise`` gives all
        frames of a trajectory the same low-frequency start noise (frame 0 included).
        ``refiner_state_roll`` (diagnostic, set at sampling): trajectory k reads the state of
        trajectory k - roll, so -1 hands every trajectory its neighbor k+1's ``S_{t-1}``.
        ``memory`` (``refiner_memory`` only): per-frame memory hints given from outside (oracle / zero
        diagnostics) instead of the one accumulated from the generated frames. ``memory_gate``: per-frame
        write flags (``refiner_memory_gate``), all True if None.
        ``refiner_frame0_hop`` (set at sampling): frame 0 gets one refiner hop on the base's draw (S = that
        draw, frame 0's own conditions, zero delta g). ``refiner_state_conf_value``: the S confidence channel.
        ``frame0`` (target space, [B, C, H, W]): frame 0 given from outside (e.g. the real first photo) instead of
        the base draw; it is returned as frame 0, is the first S and the first memory write.
        """
        if len(conds) != len(gru_feats):
            raise ValueError(
                f"conds ({len(conds)}) and gru_feats ({len(gru_feats)}) must have the same length"
            )
        if getattr(self, "refiner_neighbors", False):
            return self.sample_refiner_joint(conds, gru_feats, num_steps=num_steps, damage_mask=damage_mask)
        per_frame_mask = isinstance(damage_mask, (list, tuple))
        self._attach_refiner_base()
        base = self._refiner_base_ref[0]
        frames = []
        self._tone_ref = None
        state = None
        shared = None
        roll = int(getattr(self, "refiner_state_roll", 0) or 0)
        mem_track = self.refiner_memory_track() if getattr(self, "refiner_memory", False) else None
        if getattr(self, "refiner_shared_lowfreq_noise", False) and conds:
            shared = self._refiner_start_noise(None, conds[0].shape[0], conds[0].device)
        for i, (c, g) in enumerate(zip(conds, gru_feats)):
            mask_i = damage_mask[i] if per_frame_mask else damage_mask
            dg = None if i == 0 or not getattr(self, "use_delta_g", False) else g - gru_feats[i - 1]
            z0 = self._refiner_start_noise(shared, c.shape[0], c.device)
            if state is None and frame0 is not None:
                x = frame0.to(device=c.device)
                frames.append(x)
                state = x.detach()
                if mem_track is not None:
                    mem_track.update(state, True if memory_gate is None else bool(memory_gate[i]))
                continue
            if state is None:
                c_base = c
                base_in = int(base.velocity_net.scalar_mlp[0].in_features)
                if c.shape[1] + 1 == base_in:
                    c_base = torch.cat([torch.zeros_like(c[:, :1]), c], dim=1)
                mask_b = mask_i
                if mask_b is not None and mask_b.shape[1] > 1 and not getattr(base, "layout_cond", False):
                    mask_b = mask_b[:, :1]  # layout channel only for a layout-conditioned refiner
                x = base.sample_with_ode(c_base, z0=z0, gru_feat=g, num_steps=num_steps, damage_mask=mask_b)
                if bool(getattr(self, "refiner_frame0_hop", False)):
                    state = x.detach()
                    z0 = self._refiner_start_noise(shared, c.shape[0], c.device)
                    dg = torch.zeros_like(g) if getattr(self, "use_delta_g", False) else None
            if state is not None:
                if mem_track is None:
                    hint = state
                else:
                    mem = mem_track.value if memory is None else memory[i]
                    if mem is None:
                        mem = torch.zeros_like(state[:, :1]).expand(-1, self.refiner_memory_hint_channels, -1, -1)
                    hint = torch.cat([state, mem.to(device=state.device, dtype=state.dtype)], dim=1)
                # Hops behind S since the last real frame: frames 1..i-1 with a given frame 0, else 0..i-1.
                hint = self._refiner_conf_hint(hint, float(getattr(self, "refiner_state_conf_value", 1.0)),
                                               hops=float(i - 1 if frame0 is not None else i))
                x = self.sample_with_ode(
                    c,
                    z0=z0,
                    gru_feat=g,
                    num_steps=num_steps,
                    damage_mask=mask_i,
                    lr_cond=hint if roll == 0 else hint.roll(roll, dims=0),
                    gru_delta=dg,
                )
            if getattr(self, "refiner_gray_state", False):
                x = x.mean(dim=1, keepdim=True).expand_as(x).contiguous()
            frames.append(x)
            state = x.detach()
            if i == 0:
                self._tone_ref = state  # refiner_tone_env: the intact-surface tone band is around frame 0
            if mem_track is not None:
                mem_track.update(state, True if memory_gate is None else bool(memory_gate[i]))
        self._tone_ref = None
        return frames

    @torch.no_grad()
    def sample_refiner_joint(self, conds, gru_feats, num_steps=None, damage_mask=None, mode=None):
        """Bidirectional refiner (``refiner_neighbors``): every frame of the column starts from noise
        and all are integrated together (Heun). Each velocity call reads the neighbors' current
        states with k = their current level (``refiner_joint_mode``):
          parallel  every frame reads the same step's states; Heun's second stage reads the
                    neighbors' first-stage predictions at the next level
          sweep     frames in turn, the direction alternating per step, so the neighbor already
                    visited this step is one level ahead
          causal    forward sweep, next frame absent
          none      both neighbors absent
        An absent neighbor is a fixed noise draw with k = 0, as in training.
        """
        mode = str(mode or getattr(self, "refiner_joint_mode", "sweep"))
        n = len(conds)
        if n == 0:
            return []
        steps = int(num_steps or self.cfm_ode_steps)
        b = conds[0].shape[0]
        device = conds[0].device
        size = int(self.model_spatial_size)
        per_frame_mask = isinstance(damage_mask, (list, tuple))
        use_dg = bool(getattr(self, "use_delta_g", False))

        def frame_mask(i):
            m = damage_mask[i] if per_frame_mask else damage_mask
            if m is None:
                if not self.use_damage_mask_condition:
                    return None
                return torch.ones((b, 1, size, size), device=device)
            m = m.to(device=device, dtype=torch.float32)
            if m.ndim == 3:
                m = m.unsqueeze(1)
            if m.shape[-2:] != (size, size):
                m = F.interpolate(m, size=(size, size), mode="nearest")
            return m

        masks = [frame_mask(i) for i in range(n)]
        dgs = [None if i == 0 or not use_dg else gru_feats[i] - gru_feats[i - 1] for i in range(n)]
        use_dg_next = bool(getattr(self, "refiner_nbr_delta_g", False))
        dgs_next = [None if i == n - 1 or not use_dg_next else gru_feats[i + 1] - gru_feats[i] for i in range(n)]
        coords = None
        if self.use_patch_diffusion:
            grid_x, grid_y = self._build_full_coordinate_maps(size, size, device=device, dtype=torch.float32)
            coords = torch.stack([grid_x, grid_y], dim=0).unsqueeze(0).expand(b, -1, -1, -1).contiguous()
        shared = self._refiner_start_noise(None, b, device) if self.refiner_shared_lowfreq_noise else None
        z = [self._refiner_start_noise(shared, b, device) for _ in range(n)]
        absent = [self._refiner_start_noise(shared, b, device) for _ in range(n)]
        level = [0.0] * n

        def hint(i, states, levels):
            prev, k_prev = absent[i], 0.0
            nxt, k_next = absent[i], 0.0
            if mode != "none" and i > 0:
                prev, k_prev = states[i - 1], levels[i - 1]
            if mode not in ("none", "causal") and i < n - 1:
                nxt, k_next = states[i + 1], levels[i + 1]
            return self._refiner_neighbor_pack(prev, k_prev, nxt, k_next)

        def velocity(i, z_i, t_scalar, h):
            return self._predict_velocity(
                z_i,
                torch.full((b,), float(t_scalar), device=device),
                conds[i],
                coords=coords,
                damage_mask=masks[i],
                gru_feat=gru_feats[i],
                lr_cond=h,
                gru_delta=dgs[i],
                gru_delta_next=dgs_next[i],
            )

        heun = self.cfm_solver == "heun"
        times = [float(s) / float(steps) for s in range(steps + 1)]
        for s in range(steps):
            t0, t1 = times[s], times[s + 1]
            dt = t1 - t0
            if mode == "parallel" or mode == "none":
                v0 = [velocity(i, z[i], t0, hint(i, z, level)) for i in range(n)]
                if heun:
                    z_pred = [z[i] + dt * v0[i] for i in range(n)]
                    lv_pred = [t1] * n
                    v1 = [velocity(i, z_pred[i], t1, hint(i, z_pred, lv_pred)) for i in range(n)]
                    z = [z[i] + 0.5 * dt * (v0[i] + v1[i]) for i in range(n)]
                else:
                    z = [z[i] + dt * v0[i] for i in range(n)]
                level = [t1] * n
                continue
            order = range(n) if (mode == "causal" or s % 2 == 0) else range(n - 1, -1, -1)
            for i in order:
                h = hint(i, z, level)
                v0 = velocity(i, z[i], t0, h)
                if heun:
                    v1 = velocity(i, z[i] + dt * v0, t1, h)
                    z[i] = z[i] + 0.5 * dt * (v0 + v1)
                else:
                    z[i] = z[i] + dt * v0
                level[i] = t1
        return z

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
