"""Fixed-DI inference panels, anchor selection, and related helpers."""

import os

import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import wandb


class CFMInferenceMixin:
    def _condition_includes_di(self):
        mode = str(self.condition_mode).lower()
        if self.zero_di_condition:
            return False
        if mode in ("di_only", "split", "adagn"):
            return True
        if mode == "latent_direction":
            return False
        return bool(self.include_di_conditioning)


    def _should_log_metrics(self):
        trainer = getattr(self, "_trainer", None)
        if trainer is not None and trainer.sanity_checking:
            return False
        step_num = int(self.global_step)
        if step_num <= 0:
            return False
        return step_num % self.metrics_every_n_epochs == 0


    def _should_run_inference(self):
        trainer = getattr(self, "_trainer", None)
        if trainer is not None and trainer.sanity_checking:
            return False
        step_num = int(self.global_step)
        if not self._has_logged_initial_inference and step_num > 0:
            return True
        if step_num <= 0:
            return False
        return step_num % self.inference_every_n_epochs == 0


    def _get_inference_di_values(self):
        # Fixed DI sweep for comparable train/val inference panels.
        return [float(v) for v in self.fixed_di_values[: self.inference_num_images]]


    def _inference_candidate_cap(self):
        # The growth sequence needs every val frame of one column, not a DI sample.
        if (
            getattr(self, "use_spall_growth_mask", False)
            or getattr(self, "use_paired_bridge", False)
            or getattr(self, "use_refiner_residual", False)
        ):
            return 100000
        return max(32, int(self.inference_num_images) * 24)


    @staticmethod
    def _extract_batch_field(field, sample_idx, default=None):
        if isinstance(field, (list, tuple)):
            value = field[sample_idx]
        elif torch.is_tensor(field) and field.ndim > 0:
            value = field[sample_idx]
        else:
            value = default if default is not None else field
        if torch.is_tensor(value):
            value = value.item()
        return value


    def _collect_inference_anchor_candidates(
        self,
        pool_key,
        split,
        real_vis,
        hidden_state,
        direction,
        exp,
        timestep,
        DI,
        info,
        condition_mask=None,
    ):
        pool = self.validation_outputs[pool_key]
        candidate_cap = self._inference_candidate_cap()
        if len(pool) >= candidate_cap:
            return

        remaining = candidate_cap - len(pool)
        take_count = min(remaining, real_vis.shape[0])
        for sample_idx in range(take_count):
            anchor_exp = self._extract_batch_field(exp, sample_idx)
            anchor_timestep = int(self._extract_batch_field(timestep, sample_idx))
            gt_di_value = float(self._extract_batch_field(DI, sample_idx))
            entry = {
                "split": split,
                "real": real_vis[sample_idx].detach().cpu(),
                "hidden_state": hidden_state[sample_idx].detach().cpu(),
                "direction": direction[sample_idx].detach().cpu(),
                "info": info[sample_idx].detach().cpu(),
                "exp": anchor_exp,
                "timestep": anchor_timestep,
                "gt_di": gt_di_value,
                "sample_uid": f"{split}_{anchor_exp}_{anchor_timestep}_{sample_idx}_{len(pool)}",
            }
            if condition_mask is not None:
                entry["condition_mask"] = condition_mask[sample_idx].detach().cpu()
            pool.append(entry)


    @torch.no_grad()
    def _stash_spall_mask_examples(self, z1, ar_pack, g_curr, exp, timestep, di, direction):
        """Full-frame m and m* for the mask-only check. One val pass, then discarded."""
        from src.model.cfm.spall_mask import coarse_coords, coarse_pool, extent_mask

        z_prev = self._encode_to_target_space(ar_pack["image_prev"])
        g_prev = self.process_physics_condition(ar_pack.get("hidden_state_prev"))
        has_prev = ar_pack.get("has_prev")
        if has_prev is None:
            has_prev = torch.ones(z1.shape[0], device=z1.device, dtype=z1.dtype)
        b = z1.shape[0]
        grid = int(self.spall_mask_grid)
        hp = has_prev.reshape(b).to(device=z1.device) >= 0.5
        hp_f = hp.view(b, 1, 1, 1).to(dtype=z1.dtype)
        m_star = extent_mask(
            z1,
            grid,
            float(self.spall_dark_level),
            self.spall_bg_quantile,
            self.spall_bg_target,
            self.spall_border_gap,
            seed_margin=float(getattr(self, "spall_seed_margin", 0.2)),
        )
        pool = coarse_pool(z_prev, grid) * hp_f
        g_prev = g_prev * hp.view(b, 1).to(dtype=g_prev.dtype)
        logits = self.spall_mask_head(
            pool, g_prev, g_curr, coarse_coords(None, grid, pool)
        )
        m_prob = torch.sigmoid(logits)
        stash = getattr(self, "_spall_mask_preview", None)
        if stash is None:
            stash = []
            self._spall_mask_preview = stash
        for i in range(b):
            dir_i = direction[i].detach().float().cpu().reshape(-1)
            stash.append(
                {
                    "image": torch.clamp(z1[i].detach().cpu() * 0.5 + 0.5, 0.0, 1.0),
                    "m_prob": m_prob[i].detach().cpu(),
                    "m_star": m_star[i].detach().cpu(),
                    "timestep": int(self._extract_batch_field(timestep, i)),
                    "di": float(self._extract_batch_field(di, i)),
                    "exp": str(self._extract_batch_field(exp, i)),
                    "direction": tuple(round(float(v), 4) for v in dir_i.tolist()),
                }
            )


    def _save_spall_mask_preview(self, step_num):
        """Photo, target extent, and predicted extent. Bright cells are the dark region."""
        rows = list(getattr(self, "_spall_mask_preview", []) or [])
        if not rows:
            print(f"[CFM] No spall mask preview samples at step {step_num}", flush=True)
            return
        groups = {}
        for row in rows:
            groups.setdefault((row["exp"], row["direction"]), []).append(row)
        key = max(groups, key=lambda k: len({r["timestep"] for r in groups[k]}))
        by_t = {}
        for row in groups[key]:
            by_t.setdefault(int(row["timestep"]), row)
        ordered = [by_t[t] for t in sorted(by_t)]
        if len(ordered) > 6:
            pick = [int(round(i * (len(ordered) - 1) / 5)) for i in range(6)]
            ordered = [ordered[i] for i in pick]
        n = len(ordered)
        fig, axes = plt.subplots(3, n, figsize=(n * 1.45, 4.6))
        if n == 1:
            axes = np.array(axes).reshape(3, 1)
        labels = ("photo", "m*", "m")
        for c, row in enumerate(ordered):
            photo = row["image"]
            if photo.ndim == 3:
                photo = photo[0] if photo.shape[0] != 3 else photo.permute(1, 2, 0)
            axes[0, c].imshow(photo.numpy(), cmap="gray", vmin=0, vmax=1)
            axes[1, c].imshow(row["m_star"][0].numpy(), cmap="gray", vmin=0, vmax=1, interpolation="nearest")
            axes[2, c].imshow(row["m_prob"][0].numpy(), cmap="gray", vmin=0, vmax=1, interpolation="nearest")
            axes[0, c].set_title(f"t={row['timestep']}\nDI {row['di']:.2f}", fontsize=8)
            for r in range(3):
                axes[r, c].set_xticks([])
                axes[r, c].set_yticks([])
                if c == 0:
                    axes[r, c].set_ylabel(labels[r], fontsize=8)
        exp, direction = key
        fig.suptitle(
            f"step {step_num}  {exp} {direction}  bright = darker than {float(self.spall_dark_level):.2f}",
            fontsize=10,
        )
        fig.tight_layout()
        out_dir = os.path.join(self.result_dir, "inference_samples")
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, f"spall_m_step_{int(step_num):05d}.png")
        fig.savefig(out_path, dpi=140, bbox_inches="tight")
        plt.close(fig)
        print(f"[CFM] Spall mask preview: {out_path}", flush=True)


    def _select_fixed_anchor(self, candidates, split="val", anchor_exp=None):
        cache_attr = f"_fixed_{split}_inference_anchor"
        cached = getattr(self, cache_attr, None)
        if cached is not None:
            return cached

        required_exp = str(anchor_exp or "").strip()
        pool = [c for c in candidates if str(c.get("split", split)) == split]
        if required_exp:
            pool = [c for c in pool if str(c["exp"]) == required_exp]
        if not pool:
            print(
                f"[CFM] No {split} inference anchor found for exp={required_exp or 'any'} "
                f"(candidates={len(candidates)})",
                flush=True,
            )
            return None

        if self.inference_anchor_timestep >= 0:
            anchor = min(pool, key=lambda c: abs(int(c["timestep"]) - self.inference_anchor_timestep))
        else:
            anchor = sorted(pool, key=lambda c: (str(c["exp"]), int(c["timestep"]), str(c["sample_uid"])))[0]
        setattr(self, cache_attr, anchor)
        print(
            f"[CFM] Fixed {split} inference anchor: exp={anchor['exp']} "
            f"t={anchor['timestep']} GT_DI={anchor['gt_di']:.2f}",
            flush=True,
        )
        return anchor


    def _select_anchors_for_di_values(self, candidates, di_values, split, anchor_exp=None):
        """Pick one distinct sample per target DI; each anchor carries its own timestep t."""
        cache_attr = f"_fixed_{split}_inference_anchors"
        cached = getattr(self, cache_attr, None)
        if cached is not None and len(cached) == len(di_values):
            return cached

        required_exp = str(anchor_exp or "").strip()
        pool = [c for c in candidates if str(c.get("split", split)) == split]
        if required_exp:
            pool = [c for c in pool if str(c["exp"]) == required_exp]
        if not pool:
            print(
                f"[CFM] No {split}-set inference anchors for exp={required_exp or 'any'} "
                f"(candidates={len(candidates)})",
                flush=True,
            )
            return []

        selected = []
        remaining = list(pool)
        for di_value in di_values:
            search = remaining if remaining else pool
            anchor = min(
                search,
                key=lambda c: (
                    abs(float(c["gt_di"]) - float(di_value)),
                    int(c["timestep"]),
                    str(c["sample_uid"]),
                ),
            )
            selected.append(anchor)
            remaining = [c for c in remaining if c["sample_uid"] != anchor["sample_uid"]]

        setattr(self, cache_attr, selected)
        for di_value, anchor in zip(di_values, selected):
            print(
                f"[CFM] Fixed {split} inference anchor: target_DI={float(di_value):.2f} "
                f"exp={anchor['exp']} t={anchor['timestep']} GT_DI={anchor['gt_di']:.2f}",
                flush=True,
            )
        return selected


    @staticmethod
    def _direction_key(direction):
        if torch.is_tensor(direction):
            vals = direction.detach().float().reshape(-1).tolist()
            return tuple(round(float(v), 4) for v in vals)
        return (str(direction),)


    def _select_spall_sequence_anchors(self, candidates, split="val", anchor_exp=None):
        """One column, one loading direction. Keep ``inference_num_images`` frames, spread in time."""
        cache_attr = f"_fixed_{split}_spall_sequence"
        cached = getattr(self, cache_attr, None)
        if cached:
            return cached

        required_exp = str(anchor_exp or "").strip()
        pool = [c for c in candidates if str(c.get("split", split)) == split]
        if required_exp:
            pool = [c for c in pool if str(c["exp"]) == required_exp]
        if not pool:
            print(
                f"[CFM] No {split} spall sequence for exp={required_exp or 'any'} "
                f"(candidates={len(candidates)})",
                flush=True,
            )
            return []

        groups = {}
        for cand in pool:
            groups.setdefault(self._direction_key(cand["direction"]), []).append(cand)
        direction_key = min(
            groups,
            key=lambda key: (-len(groups[key]), key),
        )
        by_t = {}
        for cand in groups[direction_key]:
            t = int(cand["timestep"])
            prev = by_t.get(t)
            if prev is None or str(cand["sample_uid"]) < str(prev["sample_uid"]):
                by_t[t] = cand
        ordered = [by_t[t] for t in sorted(by_t)]
        n_keep = max(1, int(self.inference_num_images))
        if len(ordered) > n_keep:
            if n_keep == 1:
                keep_idx = [0]
            else:
                keep_idx = []
                for i in range(n_keep):
                    idx = int(round(i * (len(ordered) - 1) / (n_keep - 1)))
                    if idx not in keep_idx:
                        keep_idx.append(idx)
            selected = [ordered[i] for i in keep_idx]
        else:
            selected = ordered
        setattr(self, cache_attr, selected)
        ts = [int(a["timestep"]) for a in selected]
        print(
            f"[CFM] Spall {split} sequence: exp={selected[0]['exp']} "
            f"direction={direction_key} show={len(selected)}/{len(ordered)} "
            f"t={ts}",
            flush=True,
        )
        return selected


    def _select_bridge_sequence_anchors(self, candidates, split="val", anchor_exp=None):
        """Every val frame of one column, in time order. The bridge steps between neighbors."""
        cache_attr = f"_fixed_{split}_bridge_sequence"
        cached = getattr(self, cache_attr, None)
        if cached:
            return cached

        required_exp = str(anchor_exp or "").strip()
        pool = [c for c in candidates if str(c.get("split", split)) == split]
        if required_exp:
            pool = [c for c in pool if str(c["exp"]) == required_exp]
        if not pool:
            print(
                f"[CFM] No {split} bridge sequence for exp={required_exp or 'any'} "
                f"(candidates={len(candidates)})",
                flush=True,
            )
            return []

        groups = {}
        for cand in pool:
            groups.setdefault(self._direction_key(cand["direction"]), []).append(cand)
        direction_key = min(groups, key=lambda key: (-len(groups[key]), key))
        by_t = {}
        for cand in groups[direction_key]:
            t = int(cand["timestep"])
            prev = by_t.get(t)
            if prev is None or str(cand["sample_uid"]) < str(prev["sample_uid"]):
                by_t[t] = cand
        selected = [by_t[t] for t in sorted(by_t)]
        setattr(self, cache_attr, selected)
        ts = [int(a["timestep"]) for a in selected]
        print(
            f"[CFM] Bridge {split} sequence: exp={selected[0]['exp']} "
            f"direction={direction_key} frames={len(selected)} t={ts[0]}..{ts[-1]}",
            flush=True,
        )
        return selected


    def _run_bridge_sequence_inference(self, anchors):
        """Frame 0 from noise. Each later frame integrates from the previous output."""
        conds = []
        gru_feats = []
        masks = []
        for anchor in anchors:
            hidden_state = anchor["hidden_state"].unsqueeze(0).to(self.device)
            direction = anchor["direction"].unsqueeze(0).to(self.device)
            info = anchor["info"].unsqueeze(0).to(self.device)
            di = torch.tensor([[float(anchor["gt_di"])]], device=self.device, dtype=info.dtype)
            with torch.no_grad():
                c, gru_feat = self._unpack_conditions(
                    info, hidden_state, direction, di, exp=anchor.get("exp")
                )
            conds.append(c)
            gru_feats.append(gru_feat)
            damage_mask = None
            if anchor.get("condition_mask") is not None:
                damage_mask = anchor["condition_mask"].unsqueeze(0).to(self.device, dtype=info.dtype)
                if damage_mask.ndim == 3:
                    damage_mask = damage_mask.unsqueeze(1)
                damage_mask = self._align_spatial_to_target(damage_mask)
            masks.append(damage_mask)

        with torch.no_grad():
            if getattr(self, "use_refiner_residual", False):
                frames = self.sample_refiner_sequence(conds, gru_feats, damage_mask=masks)
            else:
                frames = self.sample_paired_bridge_sequence(conds, gru_feats, damage_mask=masks)

        items = []
        for anchor, frame in zip(anchors, frames):
            sampled_image = self._decode_from_target_space(frame)
            sampled_vis = torch.clamp(sampled_image * 0.5 + 0.5, 0.0, 1.0)[0].detach().cpu()
            items.append(
                {
                    "di_value": float(anchor["gt_di"]),
                    "sampled_vis": sampled_vis,
                    "anchor": anchor,
                }
            )
        return items


    def _run_spall_sequence_inference(self, anchors):
        """Walk time with ``sample_spall_sequence``. Later frames see the generated previous frame."""
        conds = []
        gru_feats = []
        masks = []
        for anchor in anchors:
            hidden_state = anchor["hidden_state"].unsqueeze(0).to(self.device)
            direction = anchor["direction"].unsqueeze(0).to(self.device)
            info = anchor["info"].unsqueeze(0).to(self.device)
            di = torch.tensor([[float(anchor["gt_di"])]], device=self.device, dtype=info.dtype)
            with torch.no_grad():
                c, gru_feat = self._unpack_conditions(
                    info, hidden_state, direction, di, exp=anchor.get("exp")
                )
            conds.append(c)
            gru_feats.append(gru_feat)
            damage_mask = None
            if anchor.get("condition_mask") is not None:
                damage_mask = anchor["condition_mask"].unsqueeze(0).to(self.device, dtype=info.dtype)
                if damage_mask.ndim == 3:
                    damage_mask = damage_mask.unsqueeze(1)
                damage_mask = self._align_spatial_to_target(damage_mask)
            masks.append(damage_mask)

        with torch.no_grad():
            frames = self.sample_spall_sequence(conds, gru_feats, damage_mask=masks)

        items = []
        for anchor, frame in zip(anchors, frames):
            sampled_image = self._decode_from_target_space(frame)
            sampled_vis = torch.clamp(sampled_image * 0.5 + 0.5, 0.0, 1.0)[0].detach().cpu()
            items.append(
                {
                    "di_value": float(anchor["gt_di"]),
                    "sampled_vis": sampled_vis,
                    "anchor": anchor,
                }
            )
        return items


    def _get_cascade_stage1(self):
        ref = getattr(self, "_cascade_stage1_ref", None)
        if isinstance(ref, (list, tuple)) and len(ref) > 0:
            return ref[0]
        return getattr(self, "_cascade_stage1", None)


    def _cascade_lr_cond_for_anchor(self, anchor, di_value):
        """LR cond at HR size: Stage-1 sample if attached, else GT bicubic (teacher-force)."""
        stage1 = self._get_cascade_stage1()
        hr = int(self.model_spatial_size)
        if stage1 is not None:
            stage1.to(self.device)
            stage1.eval()
            stage1.facfg_wl = getattr(self, "facfg_wl", None)
            stage1.facfg_wh = getattr(self, "facfg_wh", None)
            stage1._facfg_gru_unc = getattr(self, "_facfg_gru_unc", None)
            stage1._facfg_logged = getattr(self, "_facfg_logged", False)
            item = stage1._run_inference_for_anchor(anchor, di_value)
            self._facfg_logged = bool(getattr(stage1, "_facfg_logged", False))
            vis = item["sampled_vis"].unsqueeze(0).to(device=self.device, dtype=torch.float32)
            z_lr = vis * 2.0 - 1.0
            if tuple(z_lr.shape[-2:]) != (hr, hr):
                z_lr = F.interpolate(z_lr, size=(hr, hr), mode="bicubic", align_corners=False)
            return z_lr
        real = anchor["real"].unsqueeze(0).to(device=self.device, dtype=torch.float32)
        z_hr = real * 2.0 - 1.0
        if tuple(z_hr.shape[-2:]) != (hr, hr):
            z_hr = F.interpolate(z_hr, size=(hr, hr), mode="bicubic", align_corners=False)
        return self._cascade_lr_spatial(z_hr)


    def _run_inference_for_anchor(
        self,
        anchor,
        di_value,
        z0=None,
        z1_ref=None,
        guidance_weight=None,
        return_latent=False,
        t_start=0.0,
        z_init=None,
        lr_cond=None,
    ):
        hidden_state = anchor["hidden_state"].unsqueeze(0).to(self.device)
        direction = anchor["direction"].unsqueeze(0).to(self.device)
        info = anchor["info"].unsqueeze(0).to(self.device)
        di = torch.tensor([[float(di_value)]], device=self.device, dtype=info.dtype)
        exp = anchor.get("exp")
        with torch.no_grad():
            c, gru_feat = self._unpack_conditions(
                info, hidden_state, direction, di, exp=exp
            )
        damage_mask = None
        if anchor.get("condition_mask") is not None:
            damage_mask = anchor["condition_mask"].unsqueeze(0).to(self.device, dtype=info.dtype)
            if damage_mask.ndim == 3:
                damage_mask = damage_mask.unsqueeze(1)
            damage_mask = self._align_spatial_to_target(damage_mask)

        # Optional ResidualPhysicsAdapter (frozen U-Net + adapted c_physics).
        if bool(getattr(self, "use_physics_adapter", False)):
            adapter = getattr(self, "physics_adapter", None)
            if adapter is None:
                raise RuntimeError("use_physics_adapter=True but physics_adapter is missing")
            adapter.eval()
            with torch.no_grad():
                gru_feat = adapter(gru_feat)

        if lr_cond is None and self.use_cascade_sr:
            lr_cond = self._cascade_lr_cond_for_anchor(anchor, di_value)

        use_tto = bool(getattr(self, "use_tto", False)) and gru_feat is not None
        # Mid-trajectory chain starts skip TTO (partial path ≠ full draft objective).
        if use_tto and self.use_cascade_sr:
            raise ValueError("TTO is not supported with use_cascade_sr")
        if use_tto and float(t_start) <= 0.0:
            if z0 is None:
                z0 = torch.randn(
                    (
                        c.shape[0],
                        self.model_in_channels,
                        self.model_spatial_size,
                        self.model_spatial_size,
                    ),
                    device=self.device,
                )
            gru_feat = self.optimize_gru_features_tto(
                c,
                gru_feat,
                z0=z0,
                n_iters=int(getattr(self, "tto_n_iters", 10)),
                draft_steps=int(getattr(self, "tto_draft_steps", 5)),
                lr=float(getattr(self, "tto_lr", 1e-2)),
                lambda_l2=float(getattr(self, "tto_lambda_l2", 1.0)),
                tv_weight=float(getattr(self, "tto_tv_weight", 1.0)),
            )

        with torch.no_grad():
            if bool(getattr(self, "use_frecas", False)):
                if float(t_start) > 0.0:
                    raise ValueError("use_frecas does not support t_start > 0")
                if lr_cond is not None or self.use_cascade_sr:
                    raise ValueError("use_frecas is mutually exclusive with cascade SR / lr_cond")
                if use_tto:
                    raise ValueError("use_frecas is mutually exclusive with TTO")
                sampled = self.sample_with_frecas(
                    c,
                    z0=z0,
                    gru_feat=gru_feat,
                    damage_mask=damage_mask,
                )
            elif bool(getattr(self, "use_sde_sample", False)):
                if float(t_start) > 0.0:
                    raise ValueError("SDE sampling does not support t_start > 0")
                sampled = self.sample_with_sde(
                    c,
                    z0=z0,
                    gru_feat=gru_feat,
                    damage_mask=damage_mask,
                    sde_sigma=float(getattr(self, "cfm_sde_sigma", 1.0)),
                )
            else:
                sampled = self.sample_with_ode(
                    c,
                    z0=z0,
                    gru_feat=gru_feat,
                    z1_ref=z1_ref,
                    guidance_weight=guidance_weight,
                    t_start=float(t_start),
                    z_init=z_init,
                    lr_cond=lr_cond,
                    damage_mask=damage_mask,
                )
            sampled_image = self._decode_from_target_space(sampled)
            sampled_vis = torch.clamp(sampled_image * 0.5 + 0.5, 0.0, 1.0)[0].detach().cpu()
        out = {
            "di_value": float(di_value),
            "sampled_vis": sampled_vis,
            "sampled_latent": sampled[0].detach().cpu(),
            "anchor": anchor,
            "c": c.detach(),
            "gru_feat": None if gru_feat is None else gru_feat.detach(),
            "used_tto": bool(use_tto and float(t_start) <= 0.0),
            "used_frecas": bool(getattr(self, "use_frecas", False)),
        }
        if lr_cond is not None:
            out["cascade_lr_vis"] = torch.clamp(lr_cond[0] * 0.5 + 0.5, 0.0, 1.0).detach().cpu()
        if return_latent:
            out["sampled"] = sampled.detach()
        return out


    def _build_undamaged_z1_ref(self, anchors, di_values):
        """Generate a DI≈0 sample (no guidance) and encode it as z1_ref.

        Uses a generated undamaged image instead of GT to avoid interferers.
        Returns (z1_ref[B=1,...], ref_item) or (None, None) if no near-zero DI.
        """
        if not anchors:
            return None, None
        # Prefer the DI target closest to 0 among the panel values.
        best_i = min(range(len(di_values)), key=lambda i: abs(float(di_values[i])))
        if abs(float(di_values[best_i])) > 0.05 and float(anchors[best_i]["gt_di"]) > 0.05:
            # Fall back: find any anchor with very low GT DI.
            best_i = min(range(len(anchors)), key=lambda i: float(anchors[i]["gt_di"]))
        ref_di = 0.0
        ref_item = self._run_inference_for_anchor(
            anchors[best_i],
            ref_di,
            z1_ref=None,
            guidance_weight=0.0,
            return_latent=True,
        )
        # Prefer the continuous latent from sampling (exact, no decode/encode drift).
        if "sampled" in ref_item:
            z1_ref = ref_item["sampled"]
        else:
            undamaged_img = ref_item["sampled_vis"].unsqueeze(0).to(self.device)
            undamaged_img = undamaged_img * 2.0 - 1.0
            z1_ref = self._encode_to_target_space(undamaged_img)
        return z1_ref, ref_item


    def _build_inverted_z0_from_undamaged(self, anchors, di_values):
        """DI=0 generate → backward Heun inversion → shared structural z0."""
        z1_ref, ref_item = self._build_undamaged_z1_ref(anchors, di_values)
        if z1_ref is None:
            return None, None
        c0 = ref_item["c"]
        gru0 = ref_item["gru_feat"]
        z0_inv = self.invert_with_ode(z1_ref, c0, gru_feat=gru0)
        print(
            f"[CFM] ODE inversion done: z1_ref→z0_inv "
            f"(||z0||={z0_inv.float().norm().item():.3f})",
            flush=True,
        )
        return z0_inv, ref_item


    def _run_di_sequence_chained(self, anchors, di_values, t_start=None):
        """Sequential Img2Img DI evolution (OT partial-noise chaining).

        1. First DI: full Heun from pure noise (t: 0 → 1).
        2. Later DIs: ``z_t = (1-t_start)·ε + t_start·z_prev``, then Heun
           from ``t_start → 1`` under the new DI / GRU condition.
        """
        if t_start is None:
            t_start = float(getattr(self, "chain_t_start", 0.5))
        t_start = float(t_start)
        if not (0.0 < t_start < 1.0):
            raise ValueError(f"chain t_start must be in (0,1), got {t_start}")

        print(
            f"[CFM] DI chain ON: t_start={t_start:.3f} · sequence={list(map(float, di_values))}",
            flush=True,
        )

        rows = []
        z_prev = None
        for step_i, (anchor, di_value) in enumerate(zip(anchors, di_values)):
            if step_i == 0:
                item = self._run_inference_for_anchor(
                    anchor,
                    di_value,
                    guidance_weight=0.0,
                    return_latent=True,
                    t_start=0.0,
                )
            else:
                z_t, _eps = self.ot_partial_noise(z_prev, t_start)
                item = self._run_inference_for_anchor(
                    anchor,
                    di_value,
                    guidance_weight=0.0,
                    return_latent=True,
                    t_start=t_start,
                    z_init=z_t,
                )
            z_prev = item["sampled"]
            rows.append(item)
        return rows


    def _run_fixed_di_inference_for_anchors(self, anchors, di_values):
        """Run ODE sampling per (target DI, matched anchor) pair.

        Each anchor carries the hidden_state at timestep t whose GT_DI is nearest
        to the target DI (DI → t → hidden_state[exp][t]).

        Reference structure reuse (``ref_guidance_mode``):
          - ``latent`` / ``velocity``: need ``ref_guidance_weight>0`` and z1_ref blend
          - ``inversion``: invert undamaged DI=0 → z0_inv, then forward all DI from z0_inv
          - ``chain``: sequential img2img — partial OT noise at ``chain_t_start``
        """
        if len(anchors) != len(di_values):
            raise ValueError(
                f"Anchor/DI count mismatch: {len(anchors)} anchors vs {len(di_values)} DI values"
            )

        mode = str(getattr(self, "ref_guidance_mode", "latent")).lower()
        if mode == "chain":
            return self._run_di_sequence_chained(anchors, di_values)

        alpha = float(getattr(self, "ref_guidance_weight", 0.0))
        z1_ref = None
        z0_shared = None

        if mode == "inversion":
            z0_shared, _ref_item = self._build_inverted_z0_from_undamaged(anchors, di_values)
            if z0_shared is None:
                print(
                    "[CFM] ODE inversion failed to build z0; falling back to randn z0",
                    flush=True,
                )
            else:
                print(
                    "[CFM] ODE inversion ON: shared z0_inv from undamaged DI=0 "
                    "(forward Heun under each target DI)",
                    flush=True,
                )
            alpha = 0.0  # no latent/velocity blend during forward
        elif alpha > 0.0:
            z1_ref, _ref_item = self._build_undamaged_z1_ref(anchors, di_values)
            if z1_ref is None:
                print(
                    "[CFM] ref_guidance_weight>0 but failed to build z1_ref; "
                    "falling back to unguided ODE",
                    flush=True,
                )
                alpha = 0.0
            else:
                print(
                    f"[CFM] Reference guidance ON: α={alpha:.3f} "
                    f"mode={mode} (undamaged DI=0 latent as z1_ref)",
                    flush=True,
                )

        rows = []
        for anchor, di_value in zip(anchors, di_values):
            rows.append(
                self._run_inference_for_anchor(
                    anchor,
                    di_value,
                    z0=z0_shared,
                    z1_ref=z1_ref,
                    guidance_weight=alpha,
                )
            )
        return rows


    def _save_fixed_di_inference_panel(self, train_items, val_items, di_values, step_num):
        """Save val fixed-DI panel: GT / Stage-1 LR / Stage-2 when cascade is on."""
        del train_items  # unused; panel is val GT / optional Stage-1 LR / val inference
        pair_count = len(di_values)
        show_lr = any(item.get("cascade_lr_vis") is not None for item in val_items)
        n_rows = 3 if show_lr else 2
        fig, axes = plt.subplots(n_rows, pair_count, figsize=(max(4, pair_count * 2.2), 2.2 * n_rows + 0.6))
        if pair_count == 1:
            axes = np.array([axes], dtype=object)
        if axes.ndim == 1:
            axes = axes.reshape(n_rows, 1)

        for col, (val_item, di_value) in enumerate(zip(val_items, di_values)):
            val_anchor = val_item["anchor"]
            gt = val_anchor["real"]
            axes[0, col].imshow(gt.permute(1, 2, 0).numpy())
            if self._condition_includes_di():
                axes[0, col].set_title(
                    f"GT DI={di_value:.2f}\nt={val_anchor['timestep']}",
                    fontsize=8,
                )
            else:
                axes[0, col].set_title(
                    f"GT {val_anchor['exp']} t={val_anchor['timestep']}",
                    fontsize=8,
                )
            axes[0, col].axis("off")

            row_inf = 1
            if show_lr:
                lr_vis = val_item.get("cascade_lr_vis")
                if lr_vis is not None:
                    axes[1, col].imshow(lr_vis.permute(1, 2, 0).clamp(0, 1).numpy())
                    axes[1, col].set_title("Stage-1 → 256 LR", fontsize=8)
                axes[1, col].axis("off")
                row_inf = 2

            axes[row_inf, col].imshow(val_item["sampled_vis"].permute(1, 2, 0).numpy())
            if self._condition_includes_di():
                axes[row_inf, col].set_title(
                    f"S2 DI={di_value:.2f}\nt={val_anchor['timestep']}",
                    fontsize=8,
                )
            else:
                axes[row_inf, col].set_title(
                    f"S2 {val_anchor['exp']} t={val_anchor['timestep']}",
                    fontsize=8,
                )
            axes[row_inf, col].axis("off")

        val_anchor = val_items[0]["anchor"]
        subtitle = (
            "GT (top) / Stage-1 LR (mid) / Stage-2 (bottom)"
            if show_lr
            else "GT (top) / Inference (bottom)"
        )
        fig.suptitle(
            f"step {step_num} · Val {val_anchor['exp']} · {subtitle}",
            fontsize=10,
        )
        plt.tight_layout()

        save_dir = os.path.join(self.result_dir, "inference_samples")
        os.makedirs(save_dir, exist_ok=True)
        fig_path = os.path.join(save_dir, f"fixed_di_step_{step_num:05d}.png")
        fig.savefig(fig_path, dpi=150, bbox_inches="tight")
        print(f"[CFM] Saved fixed-DI inference samples: {fig_path}")
        if wandb.run is not None:
            wandb.log(
                {f"CFM/FixedDI/step_{step_num:05d}": wandb.Image(fig)},
                step=step_num,
            )
        plt.close(fig)


    def _save_spall_sequence_panel(self, items, step_num, kind="spall sequence"):
        """Val GT (top of each band) and the chained sample (bottom), in time order."""
        if not items:
            return
        cols = 8
        n = len(items)
        bands = (n + cols - 1) // cols
        fig, axes = plt.subplots(
            bands * 2,
            cols,
            figsize=(cols * 1.6, bands * 3.4),
        )
        axes = np.atleast_2d(axes)
        for ax in axes.ravel():
            ax.axis("off")
        for i, item in enumerate(items):
            band, col = divmod(i, cols)
            anchor = item["anchor"]
            gt_ax = axes[band * 2, col]
            gen_ax = axes[band * 2 + 1, col]
            gt_ax.imshow(anchor["real"].permute(1, 2, 0).numpy())
            gt_ax.set_title(f"GT t={int(anchor['timestep'])}", fontsize=7)
            gt_ax.axis("off")
            gen_ax.imshow(item["sampled_vis"].permute(1, 2, 0).numpy())
            gen_ax.set_title(f"GM t={int(anchor['timestep'])}", fontsize=7)
            gen_ax.axis("off")
        anchor0 = items[0]["anchor"]
        fig.suptitle(
            f"step {step_num} · Val {anchor0['exp']} · {kind} · {n} frames",
            fontsize=10,
        )
        plt.tight_layout()
        save_dir = os.path.join(self.result_dir, "inference_samples")
        os.makedirs(save_dir, exist_ok=True)
        fig_path = os.path.join(save_dir, f"fixed_di_step_{step_num:05d}.png")
        fig.savefig(fig_path, dpi=120, bbox_inches="tight")
        print(f"[CFM] Saved spall-sequence inference samples: {fig_path}", flush=True)
        if wandb.run is not None:
            wandb.log(
                {f"CFM/SpallSeq/step_{step_num:05d}": wandb.Image(fig)},
                step=step_num,
            )
        plt.close(fig)
