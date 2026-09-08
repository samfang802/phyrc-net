"""LightningModule for ADM-UNet Conditional Flow Matching (CFMADM)."""

import os
from contextlib import nullcontext

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import lightning.pytorch as pl
import matplotlib.pyplot as plt
import wandb
from sklearn.metrics import r2_score
from torchvision.transforms import v2

from src.model.Component.cfm_adm import ADMUNetCFM
from src.model.Component.wavelet_unet import WaveletUNet
from src.model.cfm.condition import (
    build_generation_condition,
    build_physics_condition,
    parse_fixed_di_values,
    parse_patch_size_probs,
)
from src.model.cfm.inference_mixin import CFMInferenceMixin
from src.model.cfm.patch_mixin import CFMPatchMixin
from src.model.cfm.sampling_mixin import CFMSamplingMixin
from src.model.cfm.mafp import MAFP_PARAM_DIM, apply_mafp
from src.model.cfm.undamaged_ref import (
    UndamagedRefEncoder,
    build_undamaged_ref_pool,
    jitter_undamaged_ref,
    sample_other_specimen_indices,
)

try:
    from diffusers import AutoencoderKL
except ImportError:
    AutoencoderKL = None


class CFMADM(CFMPatchMixin, CFMSamplingMixin, CFMInferenceMixin, pl.LightningModule):
    def __init__(
        self,
        architecture,
        condition_dim,
        image_size,
        lr,
        result_dir,
        latent_conditioning,
        direction_conditioning,
        add_DI_loss,
        cfm_model_channels,
        cfm_num_res_blocks,
        cfm_cond_dim,
        cfm_dropout,
        cfm_ode_steps,
        cfm_solver,
        **kwargs,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=["kwargs", "result_dir"])
        self.condition_dim = condition_dim
        self.image_size = image_size
        self.lr = lr
        self.result_dir = result_dir
        self.latent_conditioning = latent_conditioning
        self.direction_conditioning = direction_conditioning
        self.include_di_conditioning = kwargs.get("include_di_conditioning", False)
        self.condition_mode = str(kwargs.get("condition_mode", "full")).lower()
        self.zero_di_condition = bool(kwargs.get("zero_di_condition", False))
        self.gated_fusion_alpha_dim = str(kwargs.get("gated_fusion_alpha_dim", "scalar")).lower()
        self.grad_clip_max_norm = float(kwargs.get("grad_clip_max_norm", 2.0))
        self.add_DI_loss = add_DI_loss
        self.cfm_ode_steps = cfm_ode_steps
        self.cfm_solver = cfm_solver.lower()
        self.lr_schedule = kwargs.get("lr_schedule", "constant")
        self.warmup_epochs = kwargs.get("warmup_epochs", 5)
        self.warmup_steps = int(kwargs.get("warmup_steps", 800))
        self.min_lr_ratio = kwargs.get("min_lr_ratio", 0.1)
        self.total_epochs = kwargs.get("epoch", 10) + 1
        self.total_steps = int(kwargs.get("max_steps", 15000))
        self.optimizer_name = str(kwargs.get("optimizer", "adamw")).lower()
        self.inference_every_n_epochs = max(1, int(kwargs.get("inference_every_n_epochs", 10)))
        self.inference_num_images = max(1, int(kwargs.get("inference_num_images", 6)))
        self.metrics_every_n_epochs = max(1, int(kwargs.get("metrics_every_n_epochs", 100)))
        self.use_velocity_min_snr_weighting = bool(kwargs.get("use_velocity_min_snr_weighting", True))
        self.min_snr_gamma = float(kwargs.get("min_snr_gamma", 5.0))
        self.use_wavelet_unet = bool(kwargs.get("use_wavelet_unet", False))
        self.wavelet_hf_boost = float(kwargs.get("wavelet_hf_boost", 0.0))
        if self.wavelet_hf_boost < 0.0:
            raise ValueError(f"wavelet_hf_boost must be >= 0, got {self.wavelet_hf_boost}")
        self.use_gamma_fm_loss = bool(kwargs.get("use_gamma_fm_loss", False))
        self.gamma_fm_eps = float(kwargs.get("gamma_fm_eps", 0.1))
        if self.use_gamma_fm_loss and not (0.0 <= self.gamma_fm_eps <= 1.0):
            raise ValueError(f"gamma_fm_eps must be in [0, 1], got {self.gamma_fm_eps}")
        self.inference_anchor_exp = str(kwargs.get("inference_anchor_exp", "")).strip()
        self.inference_train_anchor_exp = str(kwargs.get("inference_train_anchor_exp", "R615")).strip()
        self.inference_val_anchor_exp = str(
            kwargs.get("inference_val_anchor_exp", kwargs.get("inference_anchor_exp", "R307"))
        ).strip()
        self.inference_anchor_timestep = int(kwargs.get("inference_anchor_timestep", -1))
        self.fixed_di_values = parse_fixed_di_values(
            kwargs.get("fixed_di_values", "0.0,0.4,0.75,1.0"),
            self.inference_num_images,
        )
        self.use_sd3_vae_latent = kwargs.get("use_sd3_vae_latent", False)
        self.sd3_vae_model_id = kwargs.get("sd3_vae_model_id", "stabilityai/stable-diffusion-3-medium-diffusers")
        self.sd3_vae_subfolder = kwargs.get("sd3_vae_subfolder", "vae")
        self.sd3_vae_use_fp16 = kwargs.get("sd3_vae_use_fp16", True)
        # Late-t high-frequency / edge aux on x̂₁ (OT velocity target u=x1-ε unchanged).
        self.use_hf_aux_loss = bool(kwargs.get("use_hf_aux_loss", False))
        self.hf_aux_lambda = float(kwargs.get("hf_aux_lambda", 0.02))
        self.hf_aux_t_min = float(kwargs.get("hf_aux_t_min", 0.7))
        self.hf_aux_edge_weight = float(kwargs.get("hf_aux_edge_weight", 1.0))
        self.hf_aux_hp_weight = float(kwargs.get("hf_aux_hp_weight", 0.5))
        self.hf_aux_hp_kernel = int(kwargs.get("hf_aux_hp_kernel", 3))
        if self.use_hf_aux_loss:
            if self.hf_aux_lambda < 0.0:
                raise ValueError(f"hf_aux_lambda must be >= 0, got {self.hf_aux_lambda}")
            if not (0.0 <= self.hf_aux_t_min < 1.0):
                raise ValueError(f"hf_aux_t_min must be in [0,1), got {self.hf_aux_t_min}")
            if self.use_sd3_vae_latent:
                print(
                    "[CFM] WARNING: use_hf_aux_loss is a no-op in VAE latent mode "
                    "(needs pixel-space x̂₁).",
                    flush=True,
                )
        self.use_patch_diffusion = bool(kwargs.get("use_patch_diffusion", False))
        self.patch_size_probs = parse_patch_size_probs(kwargs.get("patch_size_probs", "0.5,0.3,0.2"))
        self.coord_channels = 2 if self.use_patch_diffusion else 0
        self.use_self_conditioning = bool(kwargs.get("use_self_conditioning", False))
        self.self_cond_train_prob = float(kwargs.get("self_cond_train_prob", 0.5))
        self.self_cond_warmup_start = int(kwargs.get("self_cond_warmup_start", 2000))
        self.self_cond_warmup_end = int(kwargs.get("self_cond_warmup_end", 4000))
        # Auto-regressive two-stage self-conditioning (temporal prev → draft → target).
        self.use_ar_self_conditioning = bool(kwargs.get("use_ar_self_conditioning", False))
        self.use_ar_chain_ot = bool(kwargs.get("use_ar_chain_ot", False))
        self.ar_drop_prob_start = float(kwargs.get("ar_drop_prob_start", 0.5))
        self.ar_drop_prob_end = float(kwargs.get("ar_drop_prob_end", 0.15))
        self.ar_drop_prob_end_step = int(kwargs.get("ar_drop_prob_end_step", 80000))
        self.ar_noise_std_min = float(kwargs.get("ar_noise_std_min", 0.05))
        self.ar_noise_std_max = float(kwargs.get("ar_noise_std_max", 0.10))
        self.use_cascade_sr = bool(kwargs.get("use_cascade_sr", False))
        self.cascade_lr_size = int(kwargs.get("cascade_lr_size", 128))
        self.cascade_noise_std = float(kwargs.get("cascade_noise_std", 0.05))
        self.cascade_blur_sigma_min = float(kwargs.get("cascade_blur_sigma_min", 0.8))
        self.cascade_blur_sigma_max = float(kwargs.get("cascade_blur_sigma_max", 2.0))
        self.cascade_predict_residual = bool(kwargs.get("cascade_predict_residual", True))
        self.cascade_stage1_checkpoint = str(kwargs.get("cascade_stage1_checkpoint", "") or "").strip()
        self.cascade_stage1_result_dir = str(kwargs.get("cascade_stage1_result_dir", "") or "").strip()
        self._cascade_stage1_ref = []
        self.use_frecas = bool(kwargs.get("use_frecas", False))
        self.frecas_stage1_size = int(kwargs.get("frecas_stage1_size", 64))
        self.frecas_stage2_size = int(kwargs.get("frecas_stage2_size", 128))
        self.frecas_stage1_stop_index = int(kwargs.get("frecas_stage1_stop_index", -1))
        self.frecas_gamma = float(kwargs.get("frecas_gamma", 2.0))
        self.frecas_w_h = float(kwargs.get("frecas_w_h", 2.0))
        self.frecas_w_c = float(kwargs.get("frecas_w_c", 0.6))
        self.frecas_pool_kernel = int(kwargs.get("frecas_pool_kernel", 3))
        self.frecas_single_model = bool(kwargs.get("frecas_single_model", False))
        self.use_mafp = bool(kwargs.get("use_mafp", False))
        self.mafp_p_diff = float(kwargs.get("mafp_p_diff", 0.1))
        self.mafp_p_ada = float(kwargs.get("mafp_p_ada", 0.1))
        self.mafp_tx_max = float(kwargs.get("mafp_tx_max", 0.125))
        self.mafp_ty_max = float(kwargs.get("mafp_ty_max", 0.02))
        self.mafp_param_dim = MAFP_PARAM_DIM if self.use_mafp else 0
        if self.use_mafp:
            self.condition_dim = int(self.condition_dim) + int(self.mafp_param_dim)
            if self.use_cascade_sr:
                raise ValueError("use_mafp is for 128 VAE CFM; disable cascade SR.")
            if not (0.0 <= self.mafp_p_diff <= 1.0 and 0.0 <= self.mafp_p_ada <= 1.0):
                raise ValueError("mafp_p_diff / mafp_p_ada must be in [0, 1]")
        self.use_undamaged_ref = bool(kwargs.get("use_undamaged_ref", False))
        self.undamaged_ref_dim = int(kwargs.get("undamaged_ref_dim", 128))
        self.undamaged_ref_di_max = float(kwargs.get("undamaged_ref_di_max", 0.05))
        self.undamaged_ref_tx_max = float(kwargs.get("undamaged_ref_tx_max", 0.08))
        self.undamaged_ref_ty_max = float(kwargs.get("undamaged_ref_ty_max", 0.02))
        self.undamaged_ref_p_diff = float(kwargs.get("undamaged_ref_p_diff", 0.5))
        self.undamaged_ref_p_ada = float(kwargs.get("undamaged_ref_p_ada", 0.3))
        self._undamaged_ref_pool = None
        self.undamaged_ref_encoder = None
        if self.use_undamaged_ref:
            if self.use_cascade_sr or self.use_sd3_vae_latent:
                raise ValueError("use_undamaged_ref is pixel-space Patch CFM only (no cascade / VAE).")
            self.undamaged_ref_encoder = UndamagedRefEncoder(in_ch=4, out_dim=self.undamaged_ref_dim)
            self._dmg_data_path = str(kwargs.get("data_path", "id_list_val_R307.json"))
            self._use_grayscale_input = bool(kwargs.get("use_grayscale_input", True))
            self._use_clahe = bool(kwargs.get("use_clahe", True))
            self._clahe_clip_limit = float(kwargs.get("clahe_clip_limit", 2.0))
            self._clahe_tile_grid_size = kwargs.get("clahe_tile_grid_size", "8,8")
        if self.use_cascade_sr:
            if self.cascade_lr_size < 8:
                raise ValueError(f"cascade_lr_size must be >= 8, got {self.cascade_lr_size}")
            if self.cascade_noise_std < 0.0:
                raise ValueError(f"cascade_noise_std must be >= 0, got {self.cascade_noise_std}")
            if self.cascade_blur_sigma_min < 0.0 or self.cascade_blur_sigma_max < self.cascade_blur_sigma_min:
                raise ValueError(
                    "Need 0 <= cascade_blur_sigma_min <= cascade_blur_sigma_max, "
                    f"got min={self.cascade_blur_sigma_min}, max={self.cascade_blur_sigma_max}"
                )
            if self.use_ar_chain_ot or self.use_ar_self_conditioning:
                raise ValueError(
                    "use_cascade_sr is mutually exclusive with use_ar_chain_ot / "
                    "use_ar_self_conditioning."
                )
        if self.use_ar_chain_ot and self.use_ar_self_conditioning:
            raise ValueError(
                "use_ar_chain_ot and use_ar_self_conditioning are mutually exclusive; "
                "pick prev→target chain OT or noisy-z_prev self_cond, not both."
            )
        if self.use_ar_self_conditioning:
            # Noisy-z_prev path needs the self_cond channel in in_conv.
            self.use_self_conditioning = True
        if not (0.0 <= self.ar_drop_prob_end <= self.ar_drop_prob_start <= 1.0):
            raise ValueError(
                "Expected 0 <= ar_drop_prob_end <= ar_drop_prob_start <= 1, "
                f"got start={self.ar_drop_prob_start}, end={self.ar_drop_prob_end}"
            )
        if self.ar_drop_prob_end_step < 1:
            raise ValueError(f"ar_drop_prob_end_step must be >= 1, got {self.ar_drop_prob_end_step}")
        if self.ar_noise_std_min < 0.0 or self.ar_noise_std_max < self.ar_noise_std_min:
            raise ValueError(
                f"Need 0 <= ar_noise_std_min <= ar_noise_std_max, "
                f"got min={self.ar_noise_std_min}, max={self.ar_noise_std_max}"
            )
        # Training-free OT reference guidance (latent or velocity blend toward undamaged z1_ref).
        self.ref_guidance_weight = float(kwargs.get("ref_guidance_weight", 0.0))
        self.ref_guidance_mode = str(kwargs.get("ref_guidance_mode", "latent")).lower()
        if self.ref_guidance_weight < 0.0:
            raise ValueError(
                f"ref_guidance_weight must be >= 0, got {self.ref_guidance_weight}"
            )
        if self.ref_guidance_mode not in ("latent", "velocity", "inversion", "chain", "none"):
            raise ValueError(
                f"ref_guidance_mode must be latent|velocity|inversion|chain|none, "
                f"got {self.ref_guidance_mode!r}"
            )
        self.chain_t_start = float(kwargs.get("chain_t_start", 0.5))
        if not (0.0 < self.chain_t_start < 1.0):
            raise ValueError(
                f"chain_t_start must be in (0,1), got {self.chain_t_start}"
            )
        if self.self_cond_warmup_end < self.self_cond_warmup_start:
            raise ValueError(
                f"self_cond_warmup_end ({self.self_cond_warmup_end}) must be >= "
                f"self_cond_warmup_start ({self.self_cond_warmup_start})"
            )
        self.use_damage_mask_condition = bool(kwargs.get("use_damage_mask_condition", False))
        self.use_object_loss_mask = bool(kwargs.get("use_object_loss_mask", True))
        self.physics_dim = int(kwargs.get("physics_dim", 1024))
        self.cfm_physics_num_tokens = int(kwargs.get("cfm_physics_num_tokens", 16))
        self.cfm_physics_token_dim = int(kwargs.get("cfm_physics_token_dim", cfm_cond_dim))
        self.cfm_physics_adagn_dim = int(kwargs.get("cfm_physics_adagn_dim", cfm_cond_dim))
        self.use_split_conditioning = self.condition_mode == "split"
        self.use_physics_adagn = self.condition_mode == "adagn"
        self.use_physics_cross_attn = bool(kwargs.get("use_physics_cross_attn", False))
        if self.use_physics_cross_attn and not self.use_split_conditioning:
            raise ValueError(
                "use_physics_cross_attn requires condition_mode=split "
                "(Dual-Track AdaGN + Cross-Attn)."
            )

        self.vae = None
        self.vae_scaling_factor = 1.0
        self.vae_shift_factor = 0.0
        self.model_in_channels = 3
        self.model_spatial_size = self.image_size
        self.vae_downsample_factor = 1

        if self.use_sd3_vae_latent:
            if AutoencoderKL is None:
                raise ImportError(
                    "diffusers is required for --use_sd3_vae_latent. "
                    "Please install it with: pip install diffusers transformers"
                )

            vae_dtype = torch.float16 if (torch.cuda.is_available() and self.sd3_vae_use_fp16) else torch.float32
            self.vae = AutoencoderKL.from_pretrained(
                self.sd3_vae_model_id,
                subfolder=self.sd3_vae_subfolder,
                torch_dtype=vae_dtype,
            )
            self.vae.eval()
            for p in self.vae.parameters():
                p.requires_grad = False

            self.vae_scaling_factor = float(getattr(self.vae.config, "scaling_factor", 1.0))
            self.vae_shift_factor = float(getattr(self.vae.config, "shift_factor", 0.0))
            # SD3 AutoencoderKL is 16-channel; do not force 4 (encode still returns 16).
            self.model_in_channels = int(getattr(self.vae.config, "latent_channels", 16))
            # AutoencoderKL downsamples once per block after the first:
            # 2 ** (len(block_out_channels) - 1). Do not use 2 ** len(...),
            # and do not read pipeline-only `vae_scale_factor` from VAE config.
            block_out = list(getattr(self.vae.config, "block_out_channels", (128, 256, 512, 512)))
            n_down = max(1, len(block_out) - 1)
            self.vae_downsample_factor = 2 ** n_down
            if self.image_size % self.vae_downsample_factor != 0:
                raise ValueError(
                    f"image_size={self.image_size} is not divisible by VAE downsample factor={self.vae_downsample_factor}"
                )
            self.model_spatial_size = self.image_size // self.vae_downsample_factor
            print(
                f"[CFM] SD3 VAE latent {self.model_in_channels}ch "
                f"{self.model_spatial_size}x{self.model_spatial_size} "
                f"(pixel {self.image_size}, downsample x{self.vae_downsample_factor}, "
                f"scale={self.vae_scaling_factor}, shift={self.vae_shift_factor})",
                flush=True,
            )

        if self.use_cascade_sr and self.use_sd3_vae_latent:
            raise ValueError(
                "use_cascade_sr is only supported in pixel-space CFM "
                "(disable --use_sd3_vae_latent)."
            )
        if self.use_cascade_sr and self.model_spatial_size <= self.cascade_lr_size:
            raise ValueError(
                f"use_cascade_sr requires image_size > cascade_lr_size "
                f"(got {self.model_spatial_size} <= {self.cascade_lr_size})."
            )
        if self.use_patch_diffusion and self.use_sd3_vae_latent:
            raise ValueError(
                "use_patch_diffusion is only supported in pixel-space CFM "
                "(disable --use_sd3_vae_latent)."
            )
        if self.use_patch_diffusion and self.model_spatial_size % 8 != 0:
            raise ValueError(
                f"Patch Diffusion requires model_spatial_size divisible by 8 "
                f"(got {self.model_spatial_size}) so 1/2 and 1/4 crops remain U-Net compatible."
            )

        net_in_channels = (
            self.model_in_channels
            + (self.model_in_channels if self.use_self_conditioning else 0)
            + (self.model_in_channels if self.use_cascade_sr else 0)
            + self.coord_channels
        )
        net_out_channels = self.model_in_channels + self.coord_channels
        cfm_mid_attn_heads = max(1, int(kwargs.get("cfm_mid_attn_heads", 4)))
        use_vss_l3_u3 = bool(kwargs.get("use_vss_l3_u3", True))
        use_vss_u2 = bool(kwargs.get("use_vss_u2", False))
        use_vss_l2 = bool(kwargs.get("use_vss_l2", False))
        use_u1_extra_resblock = bool(kwargs.get("use_u1_extra_resblock", False))
        net_cls = WaveletUNet if self.use_wavelet_unet else ADMUNetCFM
        self.velocity_net = net_cls(
            in_channels=net_in_channels,
            out_channels=net_out_channels,
            model_channels=cfm_model_channels,
            channel_mult=(1, 2, 4, 4),
            num_res_blocks=cfm_num_res_blocks,
            cond_dim=cfm_cond_dim,
            latent_dim=self.condition_dim,
            dropout=cfm_dropout,
            use_mid_attention=True,
            mid_attn_heads=cfm_mid_attn_heads,
            use_vss_l3_u3=use_vss_l3_u3,
            use_vss_u2=use_vss_u2,
            use_vss_l2=use_vss_l2,
            use_u1_extra_resblock=use_u1_extra_resblock,
            use_damage_mask_condition=self.use_damage_mask_condition,
            use_split_conditioning=self.use_split_conditioning,
            use_physics_adagn=self.use_physics_adagn,
            use_physics_cross_attn=self.use_physics_cross_attn,
            physics_dim=self.physics_dim,
            physics_num_tokens=self.cfm_physics_num_tokens,
            physics_token_dim=self.cfm_physics_token_dim,
            physics_adagn_dim=self.cfm_physics_adagn_dim,
            gated_fusion_alpha_dim=self.gated_fusion_alpha_dim,
            image_channels=self.model_in_channels,
            self_cond_channels=(self.model_in_channels if self.use_self_conditioning else 0),
            use_undamaged_ref=self.use_undamaged_ref,
            undamaged_ref_dim=self.undamaged_ref_dim,
            use_freeu=bool(kwargs.get("use_freeu", False)),
            freeu_b1=float(kwargs.get("freeu_b1", 1.2)),
            freeu_b2=float(kwargs.get("freeu_b2", 1.4)),
            freeu_s1=float(kwargs.get("freeu_s1", 0.9)),
            freeu_s2=float(kwargs.get("freeu_s2", 0.2)),
            freeu_r_thresh=float(kwargs.get("freeu_r_thresh", 0.25)),
        )
        if self.use_wavelet_unet:
            print("[CFM] velocity_net=WaveletUNet (Haar DWT/IWT, LL bottleneck, HF skip)", flush=True)
        if self.use_physics_adagn:
            cond_mlp_in = int(self.velocity_net.scalar_mlp[0].in_features)
            expected_cond_in = int(self.cfm_physics_adagn_dim) + int(self.condition_dim)
        elif self.use_split_conditioning:
            cond_mlp_in = int(self.velocity_net.scalar_mlp[0].in_features)
            expected_cond_in = int(self.condition_dim)
        else:
            cond_mlp_in = int(self.velocity_net.cond_mlp[0].in_features)
            expected_cond_in = int(self.condition_dim)
        if cond_mlp_in != expected_cond_in:
            raise ValueError(
                f"Condition dimension mismatch: expected_in={expected_cond_in}, "
                f"cond_mlp.in_features={cond_mlp_in} (condition_dim={self.condition_dim})"
            )
        if self.use_split_conditioning:
            alpha = self.velocity_net.alpha
            if not isinstance(alpha, nn.Parameter):
                raise TypeError("velocity_net.alpha must be nn.Parameter for gated fusion")
            if float(alpha.detach().abs().max().item()) != 0.0:
                raise ValueError("gated fusion alpha must be zero-initialized")
            extra = ""
            if self.use_undamaged_ref:
                alpha_ref = self.velocity_net.alpha_ref
                if not isinstance(alpha_ref, nn.Parameter):
                    raise TypeError("velocity_net.alpha_ref must be nn.Parameter")
                if float(alpha_ref.detach().abs().max().item()) != 0.0:
                    raise ValueError("undamaged-ref fusion alpha_ref must be zero-initialized")
                extra = (
                    f" alpha_ref={float(alpha_ref.detach().mean().item()):.4f}"
                )
            print(
                f"[CFM] gated AdaGN: alpha is nn.Parameter shape={tuple(alpha.shape)} "
                f"init={float(alpha.detach().mean().item()):.4f} requires_grad={alpha.requires_grad}"
                f"{extra}",
                flush=True,
            )
        print(
            f"[CFM] condition_mode={self.condition_mode} condition_dim={self.condition_dim} "
            f"-> cond_mlp.in_features={cond_mlp_in} cond_dim={cfm_cond_dim} "
            f"use_split_conditioning={self.use_split_conditioning} "
            f"use_physics_adagn={self.use_physics_adagn} "
            f"use_physics_cross_attn={self.use_physics_cross_attn} "
            f"physics_dim={self.physics_dim} "
            f"physics_tokens={self.cfm_physics_num_tokens}x{self.cfm_physics_token_dim} "
            f"physics_adagn_dim={self.cfm_physics_adagn_dim} "
            f"zero_di_condition={self.zero_di_condition} "
            f"gated_fusion_alpha_dim={self.gated_fusion_alpha_dim} "
            f"in_conv.in_channels={self.velocity_net.in_conv.in_channels} "
            f"out_channels={net_out_channels} use_patch_diffusion={self.use_patch_diffusion} "
            f"use_damage_mask_condition={self.use_damage_mask_condition} "
            f"use_object_loss_mask={self.use_object_loss_mask} "
            f"use_self_conditioning={self.use_self_conditioning} "
            f"self_cond_train_prob={self.self_cond_train_prob} "
            f"self_cond_warmup=[{self.self_cond_warmup_start},{self.self_cond_warmup_end}] "
            f"use_cascade_sr={self.use_cascade_sr} "
            f"cascade_lr_size={self.cascade_lr_size} "
            f"cascade_noise_std={self.cascade_noise_std} "
            f"cascade_blur_sigma=[{self.cascade_blur_sigma_min},{self.cascade_blur_sigma_max}] "
            f"cascade_predict_residual={self.cascade_predict_residual} "
            f"use_mafp={self.use_mafp} mafp_p=[{self.mafp_p_diff},{self.mafp_p_ada}] "
            f"use_undamaged_ref={self.use_undamaged_ref} "
            f"use_gamma_fm_loss={self.use_gamma_fm_loss} "
            f"use_wavelet_unet={self.use_wavelet_unet} "
            f"wavelet_hf_boost={self.wavelet_hf_boost} "
            f"mid_attn_heads={cfm_mid_attn_heads} use_vss_l3_u3={use_vss_l3_u3} "
            f"use_vss_l2={use_vss_l2} use_vss_u2={use_vss_u2} "
            f"use_u1_extra_resblock={use_u1_extra_resblock} "
            f"use_hf_aux_loss={self.use_hf_aux_loss} "
            f"hf_aux_lambda={self.hf_aux_lambda} hf_aux_t_min={self.hf_aux_t_min} "
            f"(VSS: L3+U3"
            f"{'+L2' if use_vss_l2 else ''}"
            f"{'+U2' if use_vss_u2 else ''}; Dual-Track AdaGN"
            f"{'+CrossAttn' if self.use_physics_cross_attn else ''})",
            flush=True,
        )
        if self.use_patch_diffusion:
            full_size = int(self.model_spatial_size)
            print(
                f"[CFM] Patch Diffusion schedule: "
                f"{full_size}x{full_size}={self.patch_size_probs[0]:.2f}, "
                f"{full_size // 2}x{full_size // 2}={self.patch_size_probs[1]:.2f}, "
                f"{full_size // 4}x{full_size // 4}={self.patch_size_probs[2]:.2f}",
                flush=True,
            )

        # FID / PSNR / SSIM disabled during training (compute offline if needed).
        self.validation_outputs = {"gt_DI": [], "pred_DI": [], "train_anchor_candidates": [], "val_anchor_candidates": []}
        self._has_logged_initial_inference = False
        self._fixed_train_inference_anchor = None
        self._fixed_val_inference_anchor = None
        self._fixed_train_inference_anchors = None
        self._fixed_val_inference_anchors = None

        self.regressor = kwargs.get("regressor", None)
        if self.add_DI_loss:
            if self.regressor is None:
                raise ValueError("CFMADM with --add_DI_loss requires a loaded regressor.")
            self.custom_transform = v2.Compose([
                v2.Resize((224, 224)),
                v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])


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
        stage1 = importlib.import_module("src.model.ImageGeneration").__dict__[
            stage1_args.architecture
        ](regressor=None, **vars(stage1_args))
        os.environ.setdefault("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD", "1")
        payload = torch.load(ckpt, map_location=self.device, weights_only=False)
        state = payload["state_dict"]
        drop = [k for k in list(state.keys()) if str(k).startswith("fid.")]
        for k in drop:
            del state[k]
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


    def _unpack_generation_batch(self, batch):
        """Unpack tuple (legacy) or AR dict batch into a common namespace."""
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
                loss_fm = self.ar_chain_flow_matching_loss(
                    target_space_data,
                    z_prev=z_prev,
                    c_target=c,
                    gru_feat_target=gru_feat,
                    valid_mask=valid_mask,
                    mask_combined=mask_combined,
                    has_prev=ar_pack.get("has_prev"),
                )
            else:
                loss_fm = self.ar_temporal_flow_matching_loss(
                    target_space_data,
                    z_prev=z_prev,
                    c_target=c,
                    gru_feat_target=gru_feat,
                    valid_mask=valid_mask,
                    mask_combined=mask_combined,
                    has_prev=ar_pack.get("has_prev"),
                )
        else:
            loss_fm = self.flow_matching_loss(
                target_space_data,
                c,
                valid_mask=valid_mask,
                condition_mask=condition_mask,
                gru_feat=gru_feat,
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
        if (
            (self.use_ar_chain_ot or self.use_ar_self_conditioning)
            and ar_pack is not None
            and ar_pack.get("image_prev") is not None
        ):
            z_prev = self._encode_to_target_space(ar_pack["image_prev"])
            mask_combined = self._align_spatial_to_target(
                ar_pack.get("mask_combined", condition_mask)
            )
            if mask_combined is not None and condition_mask is not None:
                mask_combined = torch.maximum(mask_combined, condition_mask)
            elif mask_combined is None:
                mask_combined = condition_mask
            if self.use_ar_chain_ot:
                val_fm_loss = self.ar_chain_flow_matching_loss(
                    target_space_data,
                    z_prev=z_prev,
                    c_target=c,
                    gru_feat_target=gru_feat,
                    valid_mask=valid_mask,
                    mask_combined=mask_combined,
                    has_prev=ar_pack.get("has_prev"),
                )
            else:
                val_fm_loss = self.ar_temporal_flow_matching_loss(
                    target_space_data,
                    z_prev=z_prev,
                    c_target=c,
                    gru_feat_target=gru_feat,
                    valid_mask=valid_mask,
                    mask_combined=mask_combined,
                    has_prev=ar_pack.get("has_prev"),
                )
        else:
            val_fm_loss = self.flow_matching_loss(
                target_space_data,
                c,
                valid_mask=valid_mask,
                condition_mask=condition_mask,
                gru_feat=gru_feat,
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

        self.log("val/loss", val_fm_loss, prog_bar=False, on_step=False, on_epoch=True)


    def on_validation_epoch_end(self):
        should_infer = self._should_run_inference()
        train_candidates = self.validation_outputs["train_anchor_candidates"]
        val_candidates = self.validation_outputs["val_anchor_candidates"]
        # Panel is val GT (top) / val inference (bottom); train candidates optional.
        if should_infer and len(val_candidates) > 0:
            di_values = self._get_inference_di_values()
            step_num = int(self.global_step)

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
        if optimizer_name == "adamw":
            optimizer = torch.optim.AdamW(self.velocity_net.parameters(), lr=self.lr)
        elif optimizer_name == "adam":
            optimizer = torch.optim.Adam(self.velocity_net.parameters(), lr=self.lr)
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
        for param in self.velocity_net.parameters():
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
