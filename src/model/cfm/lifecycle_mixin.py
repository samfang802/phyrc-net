"""Lifecycle, target-space conversion, and setup helpers for CFMADM."""

import os
from contextlib import nullcontext

import torch
import matplotlib.pyplot as plt
import wandb


class CFMLifecycleMixin:
    def _vae_context(self):
        if self.vae is None:
            return nullcontext()
        vae_dtype = next(self.vae.parameters()).dtype
        if self.device.type == "cuda" and vae_dtype == torch.float16:
            return torch.autocast(device_type="cuda", dtype=torch.float16)
        return nullcontext()


    def on_fit_start(self):
        self._save_damage_mask_preview()
        self._attach_cascade_stage1()
        self._ensure_undamaged_ref_pool()
        if wandb.run is not None:
            wandb.define_metric("trainer/global_step")
            wandb.define_metric("*", step_metric="trainer/global_step")
            wandb.define_metric("train/loss", step_metric="trainer/global_step")
            wandb.define_metric("train/fm_loss", step_metric="trainer/global_step")
            wandb.define_metric("train/grad_norm_L2", step_metric="trainer/global_step")
            wandb.define_metric("train/lr", step_metric="trainer/global_step")
            wandb.define_metric("val/loss", step_metric="trainer/global_step")
            wandb.define_metric("CFM/FixedDI/*", step_metric="trainer/global_step")


    def _save_damage_mask_preview(self):
        if not self.use_damage_mask_condition:
            return
        height = width = int(self.model_spatial_size)
        ones = torch.ones((1, 1, height, width), dtype=torch.float32)
        zeros = torch.zeros((1, 1, height, width), dtype=torch.float32)
        panels = [("val/test condition (all-ones)", ones[0, 0]), ("contrast (zeros)", zeros[0, 0])]
        fig, axes = plt.subplots(1, 2, figsize=(6, 2.8))
        for ax, (title, panel) in zip(axes, panels):
            ax.imshow(panel.numpy(), cmap="gray", vmin=0.0, vmax=1.0)
            ax.set_title(title, fontsize=9)
            ax.axis("off")
        fig.suptitle("Mask-as-condition: train uses Data/DamagePattern/mask; val/test use all-ones", fontsize=10)
        plt.tight_layout()
        out_dir = os.path.join(self.result_dir, "data_check")
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, "object_mask_condition_preview.png")
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(
            f"[CFM] Mask-as-condition enabled (object mask from Data/DamagePattern/mask). "
            f"Preview: {out_path}",
            flush=True,
        )


    def _attach_cascade_stage1(self):
        """Load frozen Stage-1 128 CFM for two-stage inference. Not an nn.Module child."""
        if not self.use_cascade_sr:
            return
        if self._get_cascade_stage1() is not None:
            return
        ckpt = str(getattr(self, "cascade_stage1_checkpoint", "") or "").strip()
        if not ckpt:
            raise ValueError(
                "use_cascade_sr=True requires --cascade_stage1_checkpoint so train/val "
                "inference panels run Stage-1 → Stage-2 (not GT teacher-force)."
            )
        ckpt = os.path.abspath(ckpt)
        if not os.path.isfile(ckpt):
            raise FileNotFoundError(f"Stage-1 checkpoint not found: {ckpt}")
        stage1_dir = str(getattr(self, "cascade_stage1_result_dir", "") or "").strip()
        if not stage1_dir:
            stage1_dir = os.path.dirname(ckpt)
        stage1_dir = os.path.abspath(stage1_dir)
        args_path = os.path.join(stage1_dir, "args.json")
        if not os.path.isfile(args_path):
            raise FileNotFoundError(f"Stage-1 args.json not found: {args_path}")
        import argparse
        import importlib
        import json

        stage1_args = argparse.Namespace(**json.load(open(args_path, "r")))
        stage1_args.result_dir = stage1_dir
        stage1_args.use_cascade_sr = False
        stage1_args.cascade_stage1_checkpoint = ""
        os.environ.setdefault("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD", "1")
        payload = torch.load(ckpt, map_location=self.device, weights_only=False)
        state = payload["state_dict"]
        drop = [
            k
            for k in list(state.keys())
            if str(k).startswith("fid.") or ("u2_vss" in str(k))
        ]
        for k in drop:
            del state[k]
        w = state.get("velocity_net.scalar_mlp.0.weight")
        if (
            w is not None
            and int(w.shape[1]) == 3
            and bool(getattr(stage1_args, "zero_di_condition", False))
        ):
            stage1_args.keep_zero_di_slot = True
        stage1 = getattr(importlib.import_module("src.model.ImageGeneration"), stage1_args.architecture)(regressor=None, **vars(stage1_args))
        status = stage1.load_state_dict(state, strict=False)
        stage1.to(self.device)
        stage1.eval()
        for param in stage1.parameters():
            param.requires_grad = False
        # Hold in a list so Lightning/PyTorch does not register Stage-1 as a child module.
        self._cascade_stage1_ref = [stage1]
        print(
            f"[CFM] Cascade Stage-1 attached (frozen): {ckpt} "
            f"image_size={getattr(stage1_args, 'image_size', '?')} "
            f"missing={len(status.missing_keys)} unexpected={len(status.unexpected_keys)}",
            flush=True,
        )


    def _encode_to_target_space(self, image_tensor):
        if self.vae is None:
            return image_tensor
        with torch.no_grad():
            with self._vae_context():
                posterior = self.vae.encode(image_tensor).latent_dist
                latent = posterior.mean
                latent = (latent - self.vae_shift_factor) * self.vae_scaling_factor
        latent = latent.float()
        if latent.shape[1] != self.model_in_channels:
            raise ValueError(
                f"VAE latent channels={latent.shape[1]} != model_in_channels={self.model_in_channels}"
            )
        return latent


    def _decode_from_target_space(self, target_tensor):
        if self.vae is None:
            return target_tensor
        with torch.no_grad():
            with self._vae_context():
                model_input = target_tensor / self.vae_scaling_factor + self.vae_shift_factor
                decoded = self.vae.decode(model_input.to(next(self.vae.parameters()).dtype)).sample
        return decoded.float()


    def on_train_epoch_start(self):
        self.log("Learning rate", self.trainer.optimizers[0].param_groups[0]["lr"])
