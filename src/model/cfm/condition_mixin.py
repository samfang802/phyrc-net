"""Condition construction, augmentation, and batch unpacking for CFMADM."""

import torch
import torch.nn.functional as F

from src.model.cfm.condition import build_generation_condition, build_physics_condition


class CFMConditionMixin:
    def process_condition(self, info, hidden_state, direction, DI):
        return build_generation_condition(
            info,
            hidden_state,
            direction,
            DI,
            self.latent_conditioning,
            self.direction_conditioning,
            self.include_di_conditioning,
            self.condition_mode,
            zero_di_condition=self.zero_di_condition,
            keep_zero_di_slot=bool(getattr(self, "keep_zero_di_slot", False)),
        )


    def process_physics_condition(self, hidden_state):
        return build_physics_condition(hidden_state, self.condition_mode)


    def _align_spatial_to_target(self, tensor):
        """Nearest-resize masks to U-Net spatial size (VAE latent HW)."""
        if tensor is None:
            return None
        h = w = int(self.model_spatial_size)
        if tuple(tensor.shape[-2:]) == (h, w):
            return tensor
        return F.interpolate(tensor, size=(h, w), mode="nearest")


    def _apply_mafp_if_needed(self, images, condition_mask=None, valid_mask=None):
        if not self.use_mafp:
            return images, condition_mask, valid_mask, None
        from src.model.exp.mafp import apply_mafp
        return apply_mafp(
            images,
            condition_mask=condition_mask,
            valid_mask=valid_mask,
            p_diff=self.mafp_p_diff,
            p_ada=self.mafp_p_ada,
            tx_max=self.mafp_tx_max,
            ty_max=self.mafp_ty_max,
            training=bool(self.training),
        )


    def _ensure_undamaged_ref_pool(self):
        if not self.use_undamaged_ref or self._undamaged_ref_pool is not None:
            return
        from src.model.exp.undamaged_ref import build_undamaged_ref_pool
        images, masks, exps = build_undamaged_ref_pool(
            data_root="./Data",
            data_path=getattr(self, "_dmg_data_path", "id_list_val_R307.json"),
            image_size=int(self.image_size),
            di_max=self.undamaged_ref_di_max,
            use_grayscale_input=getattr(self, "_use_grayscale_input", True),
            use_clahe=getattr(self, "_use_clahe", True),
            clahe_clip_limit=getattr(self, "_clahe_clip_limit", 2.0),
            clahe_tile_grid_size=getattr(self, "_clahe_tile_grid_size", "8,8"),
        )
        self._undamaged_ref_pool = {
            "images": images,
            "masks": masks,
            "exps": exps,
        }


    @staticmethod
    def _batch_exp_ids(exp, batch_size):
        if exp is None:
            return ["?"] * int(batch_size)
        if isinstance(exp, (list, tuple)):
            return [str(x) for x in exp]
        if torch.is_tensor(exp):
            return [str(int(x)) for x in exp.detach().cpu().view(-1).tolist()]
        return [str(exp)] * int(batch_size)


    def _embed_undamaged_ref_for_exps(self, exp, batch_size, device, dtype):
        if not self.use_undamaged_ref:
            return None
        self._ensure_undamaged_ref_pool()
        dim = int(self.undamaged_ref_dim)
        if self.undamaged_ref_encoder is None or self._undamaged_ref_pool is None:
            return torch.zeros(batch_size, dim, device=device, dtype=dtype)
        pool = self._undamaged_ref_pool
        ids = self._batch_exp_ids(exp, batch_size)
        from src.model.exp.undamaged_ref import (
            jitter_undamaged_ref,
            sample_other_specimen_indices,
        )
        idxs = sample_other_specimen_indices(ids, pool["exps"])
        images = pool["images"][idxs].to(device=device, dtype=dtype)
        masks = pool["masks"][idxs].to(device=device, dtype=dtype)
        if images.shape[1] == 1:
            images = images.repeat(1, 3, 1, 1)
        if masks.ndim == 3:
            masks = masks.unsqueeze(1)
        images, masks, _, _ = jitter_undamaged_ref(
            images,
            masks,
            training=bool(self.training),
            tx_max=self.undamaged_ref_tx_max,
            ty_max=self.undamaged_ref_ty_max,
            p_diff=self.undamaged_ref_p_diff,
            p_ada=self.undamaged_ref_p_ada,
        )
        ref_in = torch.cat([images, masks], dim=1)
        return self.undamaged_ref_encoder(ref_in).to(dtype=dtype)


    def _unpack_conditions(self, info, hidden_state, direction, DI, mafp_params=None, exp=None):
        c = self.process_condition(info, hidden_state, direction, DI)
        gru_feat = self.process_physics_condition(hidden_state)
        if self.use_mafp:
            if mafp_params is None:
                mafp_params = torch.zeros(
                    c.shape[0],
                    self.mafp_param_dim,
                    device=c.device,
                    dtype=c.dtype,
                )
            else:
                mafp_params = mafp_params.to(device=c.device, dtype=c.dtype)
            c = torch.cat([c, mafp_params], dim=1)
        if self.use_undamaged_ref:
            self._undamaged_ref_embed = self._embed_undamaged_ref_for_exps(
                exp, c.shape[0], c.device, c.dtype
            )
        else:
            self._undamaged_ref_embed = None
        return c, gru_feat

    def _drop_gru_feat_cfg(self, gru_feat):
        """Train-only CFG dropout: per-sample Bernoulli zero of physics g. Coords unchanged."""
        p = float(getattr(self, "cfg_g_drop_prob", 0.0))
        if p <= 0.0 or gru_feat is None or not bool(self.training):
            return gru_feat
        if not (0.0 < p < 1.0):
            raise ValueError(f"cfg_g_drop_prob must be in (0,1), got {p}")
        b = gru_feat.shape[0]
        keep = (torch.rand(b, device=gru_feat.device) >= p).to(
            device=gru_feat.device, dtype=gru_feat.dtype
        )
        return gru_feat * keep.view(b, *([1] * (gru_feat.ndim - 1)))


    def _maybe_oracle_hidden_state(self, real_data, hidden_state):
        """oracle_dino: replace GRU g with the DINOv3 embedding of the GT frame."""
        encoder = getattr(self, "_oracle_dino_encoder", None)
        if encoder is None:
            return hidden_state
        feat = encoder.embed(real_data.to(self.device), training=bool(self.training))
        return feat.to(device=hidden_state.device, dtype=hidden_state.dtype)

    def _unpack_generation_batch(self, batch):
        """Unpack tuple (legacy) or AR dict batch into a common namespace."""
        packed = self._unpack_generation_batch_raw(batch)
        hidden_state = self._maybe_oracle_hidden_state(packed[0], packed[1])
        return (packed[0], hidden_state) + tuple(packed[2:])

    def _unpack_generation_batch_raw(self, batch):
        if isinstance(batch, dict):
            real_data = batch["image_target"] if "image_target" in batch else batch["image"]
            hidden_state = batch.get("hidden_state_target", batch["hidden_state"])
            direction = batch.get("direction_target", batch["direction"])
            exp = batch["exp"]
            timestep = batch.get("timestep_target", batch["timestep"])
            DI = batch.get("DI_target", batch["DI"])
            info = batch["info"]
            valid_mask = batch.get("loss_mask_target", batch.get("loss_mask"))
            if not self.use_object_loss_mask:
                valid_mask = None
            condition_mask = batch.get("mask_combined", batch.get("condition_mask"))
            if condition_mask is None and self.use_damage_mask_condition:
                condition_mask = torch.ones(
                    (real_data.shape[0], 1, real_data.shape[-2], real_data.shape[-1]),
                    device=real_data.device,
                    dtype=real_data.dtype,
                )
            ar = {
                "image_prev": batch.get("image_prev"),
                "hidden_state_prev": batch.get("hidden_state_prev"),
                "direction_prev": batch.get("direction_prev"),
                "DI_prev": batch.get("DI_prev"),
                "mask_combined": batch.get("mask_combined", condition_mask),
                "has_prev": batch.get("has_prev"),
                "image_prev2": batch.get("image_prev2"),
                "hidden_state_prev2": batch.get("hidden_state_prev2"),
                "direction_prev2": batch.get("direction_prev2"),
                "DI_prev2": batch.get("DI_prev2"),
                "has_prev2": batch.get("has_prev2"),
                "state_prev": batch.get("state_prev"),
                "state_prev2": batch.get("state_prev2"),
            }
            return (
                real_data,
                hidden_state,
                direction,
                exp,
                timestep,
                DI,
                info,
                valid_mask,
                condition_mask,
                ar,
            )

        real_data, hidden_state, direction, exp, timestep, DI, info = batch[:7]
        valid_mask = None
        condition_mask = None

        if self.use_damage_mask_condition:
            if len(batch) >= 9:
                valid_mask, condition_mask = batch[7], batch[8]
            elif len(batch) >= 8:
                condition_mask = batch[7]
            if not self.use_object_loss_mask:
                valid_mask = None
        elif len(batch) >= 8:
            valid_mask = batch[7]

        if condition_mask is None and self.use_damage_mask_condition:
            condition_mask = torch.ones(
                (real_data.shape[0], 1, real_data.shape[-2], real_data.shape[-1]),
                device=real_data.device,
                dtype=real_data.dtype,
            )
        return (
            real_data,
            hidden_state,
            direction,
            exp,
            timestep,
            DI,
            info,
            valid_mask,
            condition_mask,
            None,
        )
