"""Training, validation, optimization, and related Lightning hooks for CFMADM."""

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import r2_score


class CFMTrainingMixin:
    def _fm_training_loss_on_batch(
        self,
        real_data,
        hidden_state,
        direction,
        DI,
        info,
        valid_mask,
        condition_mask,
        ar_pack=None,
        exp=None,
    ):
        real_data, condition_mask, valid_mask, mafp_params = self._apply_mafp_if_needed(
            real_data, condition_mask=condition_mask, valid_mask=valid_mask
        )
        c, gru_feat = self._unpack_conditions(
            info, hidden_state, direction, DI, mafp_params=mafp_params, exp=exp
        )
        target_space_data = self._encode_to_target_space(real_data)
        condition_mask = self._align_spatial_to_target(condition_mask)
        valid_mask = self._align_spatial_to_target(valid_mask)
        loss_fm = self._flow_matching_loss_for_batch(
            target_space_data, c, gru_feat, valid_mask, condition_mask, ar_pack, info=info
        )
        if self.add_DI_loss:
            lr_cond = (
                self._cascade_lr_spatial(target_space_data) if self.use_cascade_sr else None
            )
            sampled = self.sample_with_ode(c, gru_feat=gru_feat, lr_cond=lr_cond)
            sampled_image = self._decode_from_target_space(sampled)
            sampled_vis = torch.clamp(sampled_image * 0.5 + 0.5, 0.0, 1.0)
            sampled_trans = self.custom_transform(sampled_vis)
            pred_di = self.regressor(sampled_trans, info)
            loss_di = nn.MSELoss()(pred_di, DI)
            return loss_fm + loss_di, loss_fm, loss_di
        return loss_fm, loss_fm, None


    def training_step(self, batch, batch_idx):
        (
            real_data,
            hidden_state,
            direction,
            exp,
            timestep,
            DI,
            info,
            valid_mask,
            condition_mask,
            ar_pack,
        ) = self._unpack_generation_batch(batch)

        loss, loss_fm, loss_di = self._fm_training_loss_on_batch(
            real_data,
            hidden_state,
            direction,
            DI,
            info,
            valid_mask,
            condition_mask,
            ar_pack=ar_pack,
            exp=exp,
        )
        trainer = getattr(self, "_trainer", None)
        if trainer is not None and trainer.optimizers:
            current_lr = float(trainer.optimizers[0].param_groups[0]["lr"])
            self.log("train/lr", current_lr, prog_bar=False, on_step=True, on_epoch=False)
        if loss_di is not None:
            self.log("train/errDI", loss_di, prog_bar=False, on_step=True, on_epoch=False)
        self.log("train/loss", loss, prog_bar=True, on_step=True, on_epoch=False)
        self.log("train/fm_loss", loss_fm, prog_bar=False, on_step=True, on_epoch=False)
        if bool(getattr(self, "use_hf_aux_loss", False)):
            self.log(
                "train/hf_aux_loss",
                float(getattr(self, "_last_hf_aux_loss", 0.0)),
                prog_bar=True,
                on_step=True,
                on_epoch=False,
            )
            # fm_loss log above is total when DI aux is off; expose pure FM if stored.
            if hasattr(self, "_last_fm_loss"):
                self.log(
                    "train/fm_vel_loss",
                    float(self._last_fm_loss),
                    prog_bar=False,
                    on_step=True,
                    on_epoch=False,
                )

        alpha_val = self.velocity_net.get_gated_fusion_alpha()
        if alpha_val is not None:
            self.log(
                "train/gated_fusion_alpha",
                alpha_val,
                prog_bar=True,
                on_step=True,
                on_epoch=False,
            )
        alpha_d_val = self.velocity_net.get_damage_alpha() if hasattr(self.velocity_net, "get_damage_alpha") else None
        if alpha_d_val is not None:
            self.log("train/damage_alpha", alpha_d_val, prog_bar=False, on_step=True, on_epoch=False)
        alpha_ref_val = self.velocity_net.get_undamaged_ref_alpha()
        if alpha_ref_val is not None:
            self.log(
                "train/gated_fusion_alpha_ref",
                alpha_ref_val,
                prog_bar=True,
                on_step=True,
                on_epoch=False,
            )
        if bool(getattr(self, "use_gamma_fm_loss", False)):
            self.log(
                "train/gamma_fm_std",
                float(getattr(self, "_last_gamma_fm_std", 0.0)),
                prog_bar=False,
                on_step=True,
                on_epoch=False,
            )
        if self.use_self_conditioning:
            self.log(
                "train/self_cond_prob",
                float(getattr(self, "_last_self_cond_prob", 0.0)),
                prog_bar=True,
                on_step=True,
                on_epoch=False,
            )
        if self.use_patch_diffusion:
            probs = getattr(self, "_last_patch_size_probs", self.patch_size_probs)
            self.log("train/patch_p_full", float(probs[0]), on_step=True, on_epoch=False)
            self.log("train/patch_p_half", float(probs[1]), on_step=True, on_epoch=False)
            self.log("train/patch_p_quarter", float(probs[2]), on_step=True, on_epoch=False)
            self.log(
                "train/patch_pixel_mass",
                float(getattr(self, "_last_patch_pixel_mass", 0.0)),
                on_step=True,
                on_epoch=False,
            )
            t_mean = getattr(self, "_last_patch_t_mean", None)
            if t_mean is not None:
                self.log("train/patch_t_mean", float(t_mean), on_step=True, on_epoch=False)
            self.log(
                "train/patch_size",
                float(getattr(self, "_last_patch_size", self.model_spatial_size)),
                on_step=True,
                on_epoch=False,
            )
        if self.use_ar_self_conditioning:
            self.log(
                "train/ar_drop_prob",
                float(getattr(self, "_last_ar_drop_prob", 0.0)),
                prog_bar=True,
                on_step=True,
                on_epoch=False,
            )
        if self.use_ar_chain_ot:
            self.log(
                "train/chain_t_start",
                float(getattr(self, "_last_chain_t_start", self.chain_t_start)),
                prog_bar=True,
                on_step=True,
                on_epoch=False,
            )
        if self.use_refiner_residual:
            self.log(
                "train/refiner_chain",
                float(getattr(self, "_last_refiner_chain", 0.0)),
                prog_bar=False,
                on_step=True,
                on_epoch=False,
            )
            if self.refiner_implicit or self.refiner_target_x1:
                if self.refiner_implicit and not self.refiner_target_x1:
                    self.log(
                        "train/refiner_control_frac",
                        float(getattr(self, "_last_refiner_control_frac", 0.0)),
                        prog_bar=False,
                        on_step=True,
                        on_epoch=False,
                    )
                self.log(
                    "train/refiner_fm_loss",
                    float(getattr(self, "_last_fm_loss", 0.0)),
                    prog_bar=False,
                    on_step=True,
                    on_epoch=False,
                )
                if float(self.refiner_tone_loss_weight) > 0.0:
                    self.log(
                        "train/refiner_tone_loss",
                        float(getattr(self, "_last_refiner_tone_loss", 0.0)),
                        prog_bar=False,
                        on_step=True,
                        on_epoch=False,
                    )
            else:
                self.log(
                    "train/refiner_resid_std",
                    float(getattr(self, "_last_refiner_resid_std", 0.0)),
                    prog_bar=False,
                    on_step=True,
                    on_epoch=False,
                )
        if self.use_paired_bridge:
            self.log(
                "train/bridge_frac",
                float(getattr(self, "_last_bridge_frac", 0.0)),
                prog_bar=False,
                on_step=True,
                on_epoch=False,
            )
        if self.use_spall_growth_mask:
            self.log(
                "train/spall_mask_loss",
                float(getattr(self, "_last_spall_mask_loss", 0.0)),
                prog_bar=True,
                on_step=True,
                on_epoch=False,
            )

        real_vis = torch.clamp(real_data * 0.5 + 0.5, 0.0, 1.0)
        self._collect_inference_anchor_candidates(
            "train_anchor_candidates",
            "train",
            real_vis,
            hidden_state,
            direction,
            exp,
            timestep,
            DI,
            info,
            condition_mask=condition_mask,
        )
        return loss


    def _flow_matching_loss_for_batch(
        self, target_space_data, c, gru_feat, valid_mask, condition_mask, ar_pack, info=None
    ):
        if self.use_refiner_residual:
            if ar_pack is None or ar_pack.get("image_prev") is None:
                raise RuntimeError("use_refiner_residual requires image_prev in the batch")
            mask_combined = ar_pack.get("mask_combined", condition_mask)
            mask_combined = self._align_spatial_to_target(mask_combined)
            if mask_combined is not None and condition_mask is not None:
                mask_combined = torch.maximum(mask_combined, condition_mask)
            elif mask_combined is None:
                mask_combined = condition_mask
            loss_fn = (
                self.refiner_implicit_flow_matching_loss
                if self.refiner_implicit or self.refiner_target_x1
                else self.refiner_residual_flow_matching_loss
            )
            return loss_fn(
                target_space_data,
                ar_pack,
                info,
                c,
                gru_feat_target=gru_feat,
                valid_mask=valid_mask,
                mask_combined=mask_combined,
            )
        if self.use_paired_bridge:
            if ar_pack is None or ar_pack.get("image_prev") is None:
                raise RuntimeError("use_paired_bridge requires image_prev in the batch")
            z_prev = self._encode_to_target_space(ar_pack["image_prev"])
            mask_combined = ar_pack.get("mask_combined", condition_mask)
            mask_combined = self._align_spatial_to_target(mask_combined)
            if mask_combined is not None and condition_mask is not None:
                mask_combined = torch.maximum(mask_combined, condition_mask)
            elif mask_combined is None:
                mask_combined = condition_mask
            if self.refiner_base_checkpoint:
                base_prev = self._refiner_base_prev_state(ar_pack, info, mask_combined)
                if base_prev is not None:
                    z_prev = base_prev
            return self.paired_bridge_flow_matching_loss(
                target_space_data,
                z_prev,
                c,
                gru_feat_target=gru_feat,
                valid_mask=valid_mask,
                mask_combined=mask_combined,
                has_prev=ar_pack.get("has_prev"),
            )
        if self.use_spall_growth_mask:
            if ar_pack is None or ar_pack.get("image_prev") is None:
                raise RuntimeError("use_spall_growth_mask requires image_prev in the batch")
            z_prev = self._encode_to_target_space(ar_pack["image_prev"])
            g_prev = self.process_physics_condition(ar_pack.get("hidden_state_prev"))
            return self.spall_growth_flow_matching_loss(
                target_space_data,
                z_prev,
                c,
                gru_feat,
                g_prev,
                valid_mask=valid_mask,
                condition_mask=condition_mask,
                has_prev=ar_pack.get("has_prev"),
            )
        if (
            (self.use_ar_chain_ot or self.use_ar_self_conditioning)
            and ar_pack is not None
            and ar_pack.get("image_prev") is not None
        ):
            z_prev = self._encode_to_target_space(ar_pack["image_prev"])
            mask_combined = ar_pack.get("mask_combined", condition_mask)
            mask_combined = self._align_spatial_to_target(mask_combined)
            if mask_combined is not None and condition_mask is not None:
                mask_combined = torch.maximum(mask_combined, condition_mask)
            elif mask_combined is None:
                mask_combined = condition_mask
            if self.use_ar_chain_ot:
                return self.ar_chain_flow_matching_loss(
                    target_space_data,
                    z_prev=z_prev,
                    c_target=c,
                    gru_feat_target=gru_feat,
                    valid_mask=valid_mask,
                    mask_combined=mask_combined,
                    has_prev=ar_pack.get("has_prev"),
                )
            return self.ar_temporal_flow_matching_loss(
                target_space_data,
                z_prev=z_prev,
                c_target=c,
                gru_feat_target=gru_feat,
                valid_mask=valid_mask,
                mask_combined=mask_combined,
                has_prev=ar_pack.get("has_prev"),
            )
        return self.flow_matching_loss(
            target_space_data,
            c,
            valid_mask=valid_mask,
            condition_mask=condition_mask,
            gru_feat=gru_feat,
        )


    def validation_step(self, batch, batch_idx):
        (
            real_data,
            hidden_state,
            direction,
            exp,
            timestep,
            DI,
            info,
            valid_mask,
            condition_mask,
            ar_pack,
        ) = self._unpack_generation_batch(batch)
        real_data, condition_mask, valid_mask, mafp_params = self._apply_mafp_if_needed(
            real_data, condition_mask=condition_mask, valid_mask=valid_mask
        )
        c, gru_feat = self._unpack_conditions(
            info, hidden_state, direction, DI, mafp_params=mafp_params, exp=exp
        )

        target_space_data = self._encode_to_target_space(real_data)
        condition_mask = self._align_spatial_to_target(condition_mask)
        valid_mask = self._align_spatial_to_target(valid_mask)
        val_fm_loss = self._flow_matching_loss_for_batch(
            target_space_data, c, gru_feat, valid_mask, condition_mask, ar_pack, info=info
        )

        real_vis = torch.clamp(real_data * 0.5 + 0.5, 0.0, 1.0)
        # FID/PSNR/SSIM removed from training; ODE sample only if DI aux loss is on.
        if self.add_DI_loss:
            lr_cond = (
                self._cascade_lr_spatial(target_space_data) if self.use_cascade_sr else None
            )
            sampled = self.sample_with_ode(c, gru_feat=gru_feat, lr_cond=lr_cond)
            sampled_image = self._decode_from_target_space(sampled)
            sampled_vis = torch.clamp(sampled_image * 0.5 + 0.5, 0.0, 1.0)
            sampled_trans = self.custom_transform(sampled_vis)
            pred_di = self.regressor(sampled_trans, info)
            self.validation_outputs["pred_DI"].append(pred_di.detach())
            self.validation_outputs["gt_DI"].append(DI.detach())

        self._collect_inference_anchor_candidates(
            "val_anchor_candidates",
            "val",
            real_vis,
            hidden_state,
            direction,
            exp,
            timestep,
            DI,
            info,
            condition_mask=condition_mask,
        )
        if getattr(self, "spall_mask_only", False) and self._should_run_inference() and ar_pack is not None:
            self._stash_spall_mask_examples(
                target_space_data, ar_pack, gru_feat, exp, timestep, DI, direction
            )

        self.log("val/loss", val_fm_loss, prog_bar=False, on_step=False, on_epoch=True)


    def on_validation_epoch_end(self):
        should_infer = self._should_run_inference()
        train_candidates = self.validation_outputs["train_anchor_candidates"]
        val_candidates = self.validation_outputs["val_anchor_candidates"]
        if getattr(self, "spall_mask_only", False):
            if should_infer:
                self._save_spall_mask_preview(int(self.global_step))
                self._has_logged_initial_inference = True
            self._spall_mask_preview = []
            self.validation_outputs = {
                "gt_DI": [],
                "pred_DI": [],
                "train_anchor_candidates": train_candidates,
                "val_anchor_candidates": [],
            }
            return
        # Panel is val GT (top) / val inference (bottom); train candidates optional.
        if should_infer and len(val_candidates) > 0:
            di_values = self._get_inference_di_values()
            step_num = int(self.global_step)

            if self.use_spall_growth_mask:
                # Same column and loading direction, a few frames spread in time.
                # Each frame still waits on the previous generated image, so this is not one batch.
                val_anchors = self._select_spall_sequence_anchors(
                    val_candidates,
                    split="val",
                    anchor_exp=self.inference_val_anchor_exp,
                )
            elif self.use_paired_bridge or self.use_refiner_residual:
                val_anchors = self._select_bridge_sequence_anchors(
                    val_candidates,
                    split="val",
                    anchor_exp=self.inference_val_anchor_exp,
                )
            else:
                # Per target DI: pick val anchor whose GT_DI ≈ target (DI → t → hidden_state[t]).
                val_anchors = self._select_anchors_for_di_values(
                    val_candidates,
                    di_values,
                    split="val",
                    anchor_exp=self.inference_val_anchor_exp,
                )
            if not val_anchors:
                self.validation_outputs = {
                    "gt_DI": [],
                    "pred_DI": [],
                    "train_anchor_candidates": train_candidates,
                    "val_anchor_candidates": val_candidates,
                }
                return
            if self.use_spall_growth_mask:
                val_items = self._run_spall_sequence_inference(val_anchors)
                self._save_spall_sequence_panel(val_items, step_num)
            elif self.use_paired_bridge or self.use_refiner_residual:
                val_items = self._run_bridge_sequence_inference(val_anchors)
                self._save_spall_sequence_panel(
                    val_items,
                    step_num,
                    kind="refiner residual" if self.use_refiner_residual else "paired bridge",
                )
            else:
                val_items = self._run_fixed_di_inference_for_anchors(val_anchors, di_values)
                self._save_fixed_di_inference_panel([], val_items, di_values, step_num)
            self._has_logged_initial_inference = True

        if self.add_DI_loss and len(self.validation_outputs["gt_DI"]) > 0:
            gt = torch.concatenate(self.validation_outputs["gt_DI"]).cpu().numpy()
            prediction = torch.concatenate(self.validation_outputs["pred_DI"]).cpu().numpy()
            self.log("val/r2", r2_score(gt, prediction))
        preserved_train = (
            []
            if self._fixed_train_inference_anchors is not None
            else train_candidates
        )
        self.validation_outputs = {
            "gt_DI": [],
            "pred_DI": [],
            "train_anchor_candidates": preserved_train,
            "val_anchor_candidates": [],
        }


    def configure_optimizers(self):
        optimizer_name = getattr(self, "optimizer_name", "adamw")
        param_groups = []
        if getattr(self, "refiner_implicit", False):
            # Control branch at --lr; an unfrozen trunk at trunk_lr_mult * lr. Frozen params are left out.
            named = [(n, p) for n, p in self.velocity_net.named_parameters() if p.requires_grad]
            control = [p for n, p in named if n.startswith("control.")]
            trunk = [p for n, p in named if not n.startswith("control.")]
            param_groups.append({"params": control})
            if trunk:
                param_groups.append({"params": trunk, "lr": self.lr * float(self.trunk_lr_mult)})
            print(
                f"[CFM] optimizer: control {sum(p.numel() for p in control) / 1e6:.2f}M @ lr={self.lr:g}, "
                f"trunk {sum(p.numel() for p in trunk) / 1e6:.2f}M @ lr={self.lr * float(self.trunk_lr_mult):g}"
                f"{' (frozen)' if not trunk else ''}",
                flush=True,
            )
        elif not getattr(self, "spall_mask_only", False):
            param_groups.append({"params": self.velocity_net.parameters()})
        if (
            getattr(self, "use_spall_growth_mask", False)
            and self.spall_mask_head is not None
            and not getattr(self, "freeze_spall_mask", False)
        ):
            param_groups.append({"params": self.spall_mask_head.parameters()})
        if not param_groups:
            param_groups = self.velocity_net.parameters()
        if optimizer_name == "adamw":
            optimizer = torch.optim.AdamW(param_groups, lr=self.lr)
        elif optimizer_name == "adam":
            optimizer = torch.optim.Adam(param_groups, lr=self.lr)
        else:
            raise ValueError(f"Unsupported optimizer: {optimizer_name}")
        if self.lr_schedule != "warmup_cosine":
            return optimizer

        def lr_lambda(step_idx):
            if self.warmup_steps > 0 and step_idx < self.warmup_steps:
                return float(step_idx + 1) / float(self.warmup_steps)
            remain = max(self.total_steps - self.warmup_steps, 1)
            progress = min(max((step_idx - self.warmup_steps) / remain, 0.0), 1.0)
            cosine = 0.5 * (1.0 + np.cos(np.pi * progress))
            return self.min_lr_ratio + (1.0 - self.min_lr_ratio) * cosine

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
                "frequency": 1,
            },
        }


    def on_before_optimizer_step(self, optimizer):
        # Log pre-clip L2 so the W&B y-axis shows actual magnitude (not capped at 1).
        grad_norm_sq = torch.zeros(1, device=self.device)
        grad_params = list(self.velocity_net.parameters())
        if getattr(self, "spall_mask_head", None) is not None:
            grad_params.extend(self.spall_mask_head.parameters())
        for param in grad_params:
            if param.grad is not None:
                grad_norm_sq += torch.sum(param.grad.detach() ** 2)
        grad_norm_l2 = torch.sqrt(grad_norm_sq).squeeze(0)
        self.log(
            "train/grad_norm_L2",
            grad_norm_l2,
            on_step=True,
            on_epoch=False,
            prog_bar=False,
            logger=True,
        )
        if self.grad_clip_max_norm > 0:
            torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=self.grad_clip_max_norm)
