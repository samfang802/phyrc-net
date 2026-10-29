"""Patch Diffusion / self-conditioning / velocity-net input assembly."""

import math

import torch
import torch.nn.functional as F
from torchvision.transforms.functional import gaussian_blur

from src.model.cfm.condition import (
    boost_half_by_cfm_t,
    patch_pixel_mass,
    patch_size_probs_at_step,
    patch_size_probs_pixel_mass_at_step,
)


class CFMPatchMixin:
    @staticmethod
    def _build_full_coordinate_maps(height, width, device, dtype):
        """XY grids normalized to [-1, 1] over the full image extent."""
        ys = torch.linspace(-1.0, 1.0, steps=height, device=device, dtype=dtype)
        xs = torch.linspace(-1.0, 1.0, steps=width, device=device, dtype=dtype)
        grid_y, grid_x = torch.meshgrid(ys, xs, indexing="ij")
        return grid_x, grid_y


    def _current_train_step(self):
        if getattr(self, "_trainer", None) is not None:
            return int(self.global_step)
        return 0

    def _patch_size_probs_at_step(self, step=None):
        if step is None:
            step = self._current_train_step()
        total_steps = max(1, int(getattr(self, "total_steps", 1)))
        enabled = bool(getattr(self, "use_patch_prob_schedule", False))
        mode = str(getattr(self, "patch_schedule_mode", "two_segment")).lower()
        if enabled and mode == "pixel_mass":
            probs = patch_size_probs_pixel_mass_at_step(
                step,
                mass_start=float(getattr(self, "patch_pixel_mass_start", 0.55)),
                mass_end=float(getattr(self, "patch_pixel_mass_end", 0.64)),
                quarter_start=float(getattr(self, "patch_quarter_prob_start", 0.15)),
                quarter_end=float(getattr(self, "patch_quarter_prob_end", 0.10)),
                total_steps=total_steps,
                enabled=True,
                fallback_probs=self.patch_size_probs,
            )
        else:
            mid_frac = float(getattr(self, "patch_schedule_mid_frac", 0.2))
            end_frac = float(getattr(self, "patch_schedule_end_frac", 0.7))
            probs = patch_size_probs_at_step(
                step,
                getattr(self, "patch_size_probs_start", self.patch_size_probs),
                self.patch_size_probs,
                getattr(self, "patch_size_probs_end", self.patch_size_probs),
                mid_step=int(round(mid_frac * total_steps)),
                end_start_step=int(round(end_frac * total_steps)),
                total_steps=total_steps,
                enabled=enabled,
            )
        self._last_patch_size_probs = list(probs)
        return probs

    def _patch_size_probs_for_batch(self, flow_t=None, step=None):
        """Iteration schedule, then optional CFM-t half boost averaged over sample t.

        Batch-mean t sits near 0.5 and would null the ramp. Average the per-sample
        boosted (full, half, quarter) triples, then sample one shared crop size.
        """
        probs = self._patch_size_probs_at_step(step=step)
        use_t = bool(getattr(self, "use_patch_t_schedule", False))
        extra = float(getattr(self, "patch_t_half_boost", 0.0))
        self._last_patch_t_mean = None
        if flow_t is None:
            return probs
        t_vals = flow_t.detach().float().reshape(-1)
        self._last_patch_t_mean = float(t_vals.mean().item())
        if not (use_t and extra > 0.0 and t_vals.numel() > 0):
            return probs
        t_min = float(getattr(self, "patch_t_crop_min", 0.5))
        acc = [0.0, 0.0, 0.0]
        for t_i in t_vals.tolist():
            boosted = boost_half_by_cfm_t(probs, t_i, t_min=t_min, extra_half=extra)
            acc[0] += boosted[0]
            acc[1] += boosted[1]
            acc[2] += boosted[2]
        n = float(t_vals.numel())
        probs = [a / n for a in acc]
        self._last_patch_size_probs = list(probs)
        return probs

    def _sample_stochastic_patch_size(self, full_size, flow_t=None):
        """Sample full / half / quarter from the optional patch-size schedule."""
        p_full, p_half, _p_quarter = self._patch_size_probs_for_batch(flow_t=flow_t)
        self._last_patch_pixel_mass = float(patch_pixel_mass((p_full, p_half, _p_quarter)))
        u = float(torch.rand((), device="cpu").item())
        if u < p_full:
            return int(full_size)
        if u < p_full + p_half:
            return int(full_size // 2)
        return int(full_size // 4)


    def _apply_patch_crop(
        self, z1, valid_mask=None, condition_mask=None, z0=None, z_aux=None, flow_t=None
    ):
        """
        Random-crop a mini-batch to a shared patch size and build matching XY coords.

        If ``z0`` / ``z_aux`` are provided, they are cropped with the same per-sample
        offsets so OT-coupled (z0, z1) pairs and AR self_cond latents stay aligned.

        Returns:
            z_crop, mask_crop, coords, condition_crop[, z0_crop][, z_aux_crop]
        """
        b, _c, height, width = z1.shape
        if height != width:
            raise ValueError(f"Patch Diffusion expects square inputs, got {height}x{width}")
        if z0 is not None and z0.shape != z1.shape:
            raise ValueError(f"z0 shape {tuple(z0.shape)} must match z1 {tuple(z1.shape)}")
        if z_aux is not None and z_aux.shape != z1.shape:
            raise ValueError(f"z_aux shape {tuple(z_aux.shape)} must match z1 {tuple(z1.shape)}")
        full_size = int(height)
        patch_size = (
            self._sample_stochastic_patch_size(full_size, flow_t=flow_t)
            if self.training
            else full_size
        )
        patch_size = max(1, min(int(patch_size), full_size))
        self._last_patch_size = int(patch_size)

        grid_x, grid_y = self._build_full_coordinate_maps(
            full_size, full_size, device=z1.device, dtype=z1.dtype
        )

        def _pack(z_out, mask_out, coords_out, cond_out, z0_out=None, z_aux_out=None):
            out = [z_out, mask_out, coords_out, cond_out]
            if z0 is not None:
                out.append(z0_out if z0_out is not None else z0)
            if z_aux is not None:
                out.append(z_aux_out if z_aux_out is not None else z_aux)
            return tuple(out)

        if patch_size == full_size:
            coords = torch.stack([grid_x, grid_y], dim=0).unsqueeze(0).expand(b, -1, -1, -1)
            return _pack(z1, valid_mask, coords.contiguous(), condition_mask, z0, z_aux)

        max_offset = full_size - patch_size
        tops = torch.randint(0, max_offset + 1, (b,), device=z1.device)
        lefts = torch.randint(0, max_offset + 1, (b,), device=z1.device)

        if valid_mask is not None and valid_mask.shape[-2:] != (height, width):
            raise ValueError(
                f"valid_mask spatial size {tuple(valid_mask.shape[-2:])} must match image {(height, width)}"
            )
        if condition_mask is not None and condition_mask.shape[-2:] != (height, width):
            raise ValueError(
                f"condition_mask spatial size {tuple(condition_mask.shape[-2:])} must match image {(height, width)}"
            )

        z_crops = []
        z0_crops = []
        z_aux_crops = []
        mask_crops = []
        condition_crops = []
        coord_crops = []
        for i in range(b):
            top = int(tops[i].item())
            left = int(lefts[i].item())
            z_crops.append(z1[i : i + 1, :, top : top + patch_size, left : left + patch_size])
            if z0 is not None:
                z0_crops.append(z0[i : i + 1, :, top : top + patch_size, left : left + patch_size])
            if z_aux is not None:
                z_aux_crops.append(z_aux[i : i + 1, :, top : top + patch_size, left : left + patch_size])
            if valid_mask is not None:
                mask_crops.append(
                    valid_mask[i : i + 1, :, top : top + patch_size, left : left + patch_size]
                )
            if condition_mask is not None:
                condition_crops.append(
                    condition_mask[i : i + 1, :, top : top + patch_size, left : left + patch_size]
                )
            xx = grid_x[top : top + patch_size, left : left + patch_size]
            yy = grid_y[top : top + patch_size, left : left + patch_size]
            coord_crops.append(torch.stack([xx, yy], dim=0))

        z_crop = torch.cat(z_crops, dim=0)
        coords = torch.stack(coord_crops, dim=0)
        mask_crop = torch.cat(mask_crops, dim=0) if valid_mask is not None else None
        condition_crop = torch.cat(condition_crops, dim=0) if condition_mask is not None else None
        z0_crop = torch.cat(z0_crops, dim=0) if z0 is not None else None
        z_aux_crop = torch.cat(z_aux_crops, dim=0) if z_aux is not None else None
        return _pack(z_crop, mask_crop, coords, condition_crop, z0_crop, z_aux_crop)


    def _self_cond_prob_at_step(self, step=None):
        """
        Self-conditioning trigger schedule (batch-level Bernoulli p):
          [0, warmup_start)           -> 0
          [warmup_start, warmup_end]  -> linear 0 -> self_cond_train_prob
          (warmup_end, +inf)          -> self_cond_train_prob
        Defaults: 0~2k: 0%, 2k~4k: ramp to 50%, 4k+: 50%.
        """
        if not self.use_self_conditioning:
            return 0.0
        if step is None:
            step = self._current_train_step()
        step = int(step)
        p_max = float(self.self_cond_train_prob)
        start = int(self.self_cond_warmup_start)
        end = int(self.self_cond_warmup_end)
        if step < start:
            return 0.0
        if step >= end or end == start:
            return p_max
        return p_max * float(step - start) / float(end - start)


    def _concat_coord_channels(self, image_tensor, coords):
        if bool(getattr(self, "zero_patch_coords", False)):
            coords = torch.zeros_like(coords)
        return torch.cat([image_tensor, coords], dim=1)


    def _image_channels_only(self, pred):
        """Drop predicted coordinate channels; CFM target/loss use image channels only."""
        return pred[:, : self.model_in_channels]


    def _cascade_lr_spatial(self, z_hr, blur_sigma=None):
        """Bicubic downsample to cascade_lr_size, optional Gaussian blur, upsample back."""
        if z_hr.ndim != 4:
            raise ValueError(f"cascade LR expects NCHW, got {tuple(z_hr.shape)}")
        lr_size = int(self.cascade_lr_size)
        hr_h, hr_w = int(z_hr.shape[-2]), int(z_hr.shape[-1])
        lr = F.interpolate(z_hr, size=(lr_size, lr_size), mode="bicubic", align_corners=False)
        if blur_sigma is not None:
            lr = self._gaussian_blur_nchw(lr, blur_sigma)
        return F.interpolate(lr, size=(hr_h, hr_w), mode="bicubic", align_corners=False)


    @staticmethod
    def _gaussian_blur_nchw(images, sigma):
        """Per-sample Gaussian blur. ``sigma`` is a float or (B,) tensor."""
        if not torch.is_tensor(sigma):
            sigma = torch.full(
                (images.shape[0],),
                float(sigma),
                device=images.device,
                dtype=torch.float32,
            )
        else:
            sigma = sigma.reshape(-1).to(device=images.device, dtype=torch.float32)
            if sigma.numel() == 1 and images.shape[0] > 1:
                sigma = sigma.expand(images.shape[0])
            if int(sigma.numel()) != int(images.shape[0]):
                raise ValueError(
                    f"blur sigma batch {int(sigma.numel())} must match images {images.shape[0]}"
                )
        blurred = []
        for i in range(images.shape[0]):
            s = float(sigma[i].item())
            if s <= 0.0:
                blurred.append(images[i : i + 1])
                continue
            k = int(2 * math.ceil(3.0 * s) + 1)
            k = max(3, min(k, 15))
            if k % 2 == 0:
                k += 1
            blurred.append(gaussian_blur(images[i : i + 1], kernel_size=[k, k], sigma=[s, s]))
        return torch.cat(blurred, dim=0)


    def _sample_cascade_blur_sigma(self, batch, device):
        lo = float(getattr(self, "cascade_blur_sigma_min", 0.0))
        hi = float(getattr(self, "cascade_blur_sigma_max", 0.0))
        if hi < 0.0 or lo < 0.0:
            raise ValueError(
                f"cascade blur sigma must be >= 0, got min={lo}, max={hi}"
            )
        if hi < lo:
            raise ValueError(f"cascade_blur_sigma_max ({hi}) must be >= min ({lo})")
        if hi <= 0.0:
            return None
        if not bool(self.training):
            return torch.full((batch,), 0.5 * (lo + hi), device=device, dtype=torch.float32)
        return torch.empty(batch, device=device, dtype=torch.float32).uniform_(lo, hi)


    def _prepare_cascade_target(self, z_hr, add_noise=None):
        """Split HR into (CFM target, LR cond). Residual is vs the degraded upsample (no pixel noise)."""
        blur_sigma = self._sample_cascade_blur_sigma(z_hr.shape[0], z_hr.device)
        lr_degraded = self._cascade_lr_spatial(z_hr, blur_sigma=blur_sigma)
        if add_noise is None:
            add_noise = bool(self.training) and float(self.cascade_noise_std) > 0.0
        lr_cond = lr_degraded
        if add_noise:
            b = z_hr.shape[0]
            std = torch.empty(b, device=z_hr.device, dtype=z_hr.dtype).uniform_(
                0.0, float(self.cascade_noise_std)
            )
            lr_cond = lr_degraded + torch.randn_like(lr_degraded) * std.view(b, 1, 1, 1)
        z_target = z_hr - lr_degraded if self.cascade_predict_residual else z_hr
        return z_target, lr_cond


    def _ensure_cascade_lr_cond(self, lr_cond, spatial_hw, batch, device, dtype):
        """Accept 128 or 256 LR images; return HR-sized cond matching the ODE state."""
        if lr_cond is None:
            raise ValueError("lr_cond is required when use_cascade_sr=True")
        lr_cond = lr_cond.to(device=device, dtype=dtype)
        if lr_cond.ndim != 4:
            raise ValueError(f"lr_cond must be NCHW, got {tuple(lr_cond.shape)}")
        if lr_cond.shape[0] == 1 and batch > 1:
            lr_cond = lr_cond.expand(batch, -1, -1, -1).contiguous()
        if lr_cond.shape[0] != batch:
            raise ValueError(
                f"lr_cond batch {lr_cond.shape[0]} must match condition batch {batch}"
            )
        if lr_cond.shape[1] != self.model_in_channels:
            raise ValueError(
                f"lr_cond channels {lr_cond.shape[1]} must match "
                f"model_in_channels={self.model_in_channels}"
            )
        hr_h, hr_w = int(spatial_hw[0]), int(spatial_hw[1])
        if tuple(lr_cond.shape[-2:]) != (hr_h, hr_w):
            lr_cond = F.interpolate(
                lr_cond, size=(hr_h, hr_w), mode="bicubic", align_corners=False
            )
        return lr_cond


    def _compose_cascade_sample(self, sampled, lr_cond):
        if getattr(self, "refiner_implicit", False) or getattr(self, "refiner_target_x1", False):
            # v10.5 / v10.6: the network generates the frame itself; S_{t-1} was only a condition.
            return sampled
        if self.use_refiner_residual:
            return (sampled + lr_cond).clamp(-1.0, 1.0)
        if not self.cascade_predict_residual:
            return sampled
        return (sampled + lr_cond).clamp(-1.0, 1.0)


    def _split_layout_channel(self, damage_mask, z_image):
        """layout_cond: the condition mask carries [object mask | layout SDF]. Returns (mask, layout).
        A 1-channel (or missing) mask means no layout: the layout input is 0, the dropped value."""
        b, h, w = z_image.shape[0], z_image.shape[-2], z_image.shape[-1]
        if damage_mask is None or damage_mask.shape[1] < 2:
            return damage_mask, z_image.new_zeros(b, 1, h, w)
        layout = damage_mask[:, 1:2]
        if layout.shape[-2:] != (h, w):
            layout = F.interpolate(layout, size=(h, w), mode="bilinear", align_corners=False)
        return damage_mask[:, :1], layout.to(dtype=z_image.dtype)

    def _build_velocity_net_input(
        self, z_image, coords=None, self_cond=None, lr_cond=None, spall_cond=None, layout=None
    ):
        """
        Assemble U-Net spatial input (mask is concatenated inside ADMUNetCFM.forward):
          [ z_image (C) | self_cond (C) | lr_cond (C) | coords_x | coords_y | pool(x_prev) | m | layout ]
        """
        if self.use_self_conditioning:
            if self_cond is None:
                self_cond = torch.zeros_like(z_image)
            elif self_cond.shape[1] != z_image.shape[1]:
                raise ValueError(
                    f"self_cond channels ({self_cond.shape[1]}) must match "
                    f"z_image channels ({z_image.shape[1]})"
                )
            z_in = torch.cat([z_image, self_cond], dim=1)
        else:
            if self_cond is not None:
                raise ValueError("self_cond provided but use_self_conditioning=False")
            z_in = z_image

        if self.use_cascade_sr or (self.use_refiner_residual and not getattr(self, "refiner_implicit", False)):
            if lr_cond is None:
                raise ValueError("lr_cond (state S_{t-1}) is required for cascade SR / refiner residual")
            if lr_cond.shape[1] != z_image.shape[1]:
                raise ValueError(
                    f"lr_cond channels ({lr_cond.shape[1]}) must match "
                    f"z_image channels ({z_image.shape[1]})"
                )
            if lr_cond.shape[-2:] != z_image.shape[-2:]:
                raise ValueError(
                    f"lr_cond spatial {tuple(lr_cond.shape[-2:])} must match "
                    f"z_image {tuple(z_image.shape[-2:])}"
                )
            z_in = torch.cat([z_in, lr_cond], dim=1)
        elif lr_cond is not None:
            raise ValueError("lr_cond provided but use_cascade_sr=False")

        if self.use_patch_diffusion:
            if coords is None:
                raise ValueError("coords are required when use_patch_diffusion=True")
            z_in = self._concat_coord_channels(z_in, coords)
        if self.use_spall_growth_mask:
            if spall_cond is None:
                spall_cond = z_image.new_zeros(
                    z_image.shape[0], 2, z_image.shape[-2], z_image.shape[-1]
                )
            if spall_cond.shape[1] != 2 or spall_cond.shape[-2:] != z_image.shape[-2:]:
                raise ValueError(
                    f"spall_cond must be [B,2,H,W] matching z_image, got {tuple(spall_cond.shape)}"
                )
            z_in = torch.cat([z_in, spall_cond.to(dtype=z_in.dtype)], dim=1)
        elif spall_cond is not None:
            raise ValueError("spall_cond provided but use_spall_growth_mask=False")
        if getattr(self, "layout_cond", False):
            if layout is None:
                layout = z_image.new_zeros(z_image.shape[0], 1, z_image.shape[-2], z_image.shape[-1])
            z_in = torch.cat([z_in, layout.to(dtype=z_in.dtype)], dim=1)
        elif layout is not None:
            raise ValueError("layout provided but layout_cond=False")
        return z_in


    def _predict_velocity(
        self,
        z_image,
        t,
        c,
        coords=None,
        damage_mask=None,
        gru_feat=None,
        self_cond=None,
        lr_cond=None,
        spall_cond=None,
        return_attn_map=False,
        ref_feat=None,
        gru_delta=None,
        control_scale=None,
    ):
        # v10.5: S_{t-1} (passed as lr_cond) is the ControlNet hint, not an input channel.
        control_hint = lr_cond if getattr(self, "refiner_implicit", False) else None
        layout = None
        if getattr(self, "layout_cond", False):
            damage_mask, layout = self._split_layout_channel(damage_mask, z_image)
        model_in = self._build_velocity_net_input(
            z_image,
            coords=coords,
            self_cond=self_cond,
            lr_cond=None if control_hint is not None else lr_cond,
            spall_cond=spall_cond,
            **({"layout": layout} if layout is not None else {}),
        )
        if ref_feat is None:
            ref_feat = getattr(self, "_undamaged_ref_embed", None)

        def _net(hint, scale):
            return self.velocity_net(
                model_in,
                t,
                c,
                gru_feat=gru_feat,
                damage_mask=damage_mask,
                return_attn_map=return_attn_map,
                ref_feat=ref_feat,
                **({"gru_delta": gru_delta} if gru_delta is not None else {}),
                **({"control_hint": hint, "control_scale": scale} if hint is not None else {}),
            )

        pred = _net(control_hint, control_scale)
        w = float(getattr(self, "control_guidance", 1.0))
        if control_hint is not None and not self.training and w != 1.0 and not return_attn_map:
            # Control guidance: v_off + w (v_on - v_off); v_off is the trunk alone.
            pred_off = _net(None, None)
            pred = pred_off + w * (pred - pred_off)
        attn_maps = None
        if return_attn_map:
            if not isinstance(pred, tuple) or len(pred) != 2:
                raise ValueError(
                    f"Expected (final_out, attn_maps) from velocity_net, got {type(pred)}"
                )
            final_out, attn_maps = pred
        else:
            final_out = pred

        if self.use_patch_diffusion:
            final_out = self._image_channels_only(final_out)

        if return_attn_map:
            return final_out, attn_maps
        return final_out
