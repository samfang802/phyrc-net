"""Bind CFMADM constructor flags. The Lightning class stays in module.py."""

import torch
import torch.nn as nn
from torchvision.transforms import v2

from src.model.adm import ADMUNetCFM
from src.model.cfm.condition import (
    parse_fixed_di_values,
    parse_patch_size_probs,
    patch_pixel_mass,
    probs_from_pixel_mass,
)

try:
    from diffusers import AutoencoderKL
except ImportError:
    AutoencoderKL = None


def bind_cfm_options(
    self,
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
    kwargs,
):
    self.condition_dim = condition_dim
    self.image_size = image_size
    self.lr = lr
    self.result_dir = result_dir
    self.latent_conditioning = latent_conditioning
    self.direction_conditioning = direction_conditioning
    self.include_di_conditioning = kwargs.get("include_di_conditioning", False)
    self.condition_mode = str(kwargs.get("condition_mode", "full")).lower()
    self.zero_di_condition = bool(kwargs.get("zero_di_condition", False))
    self.cfg_g_drop_prob = float(kwargs.get("cfg_g_drop_prob", 0.0))
    if not (0.0 <= self.cfg_g_drop_prob < 1.0):
        raise ValueError(
            f"cfg_g_drop_prob must be in [0,1), got {self.cfg_g_drop_prob}"
        )
    self.keep_zero_di_slot = bool(kwargs.get("keep_zero_di_slot", False))
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
    self.use_patch_prob_schedule = bool(kwargs.get("use_patch_prob_schedule", False))
    self.patch_schedule_mode = str(kwargs.get("patch_schedule_mode", "two_segment")).lower()
    self.patch_schedule_mid_frac = float(kwargs.get("patch_schedule_mid_frac", 0.2))
    self.patch_schedule_end_frac = float(kwargs.get("patch_schedule_end_frac", 0.7))
    self.patch_pixel_mass_start = float(kwargs.get("patch_pixel_mass_start", 0.55))
    self.patch_pixel_mass_end = float(kwargs.get("patch_pixel_mass_end", 0.64))
    self.patch_quarter_prob_start = float(kwargs.get("patch_quarter_prob_start", 0.15))
    self.patch_quarter_prob_end = float(kwargs.get("patch_quarter_prob_end", 0.10))
    self.use_patch_t_schedule = bool(kwargs.get("use_patch_t_schedule", False))
    self.patch_t_crop_min = float(kwargs.get("patch_t_crop_min", 0.5))
    self.patch_t_half_boost = float(kwargs.get("patch_t_half_boost", 0.20))
    if self.patch_schedule_mode not in ("two_segment", "pixel_mass"):
        raise ValueError(
            f"patch_schedule_mode must be two_segment|pixel_mass, got {self.patch_schedule_mode!r}"
        )
    if self.use_patch_prob_schedule and self.patch_schedule_mode == "pixel_mass":
        for name, value in (
            ("patch_pixel_mass_start", self.patch_pixel_mass_start),
            ("patch_pixel_mass_end", self.patch_pixel_mass_end),
        ):
            if not (0.25 <= float(value) <= 1.0):
                raise ValueError(f"{name} must be in [0.25, 1], got {value}")
        for name, value in (
            ("patch_quarter_prob_start", self.patch_quarter_prob_start),
            ("patch_quarter_prob_end", self.patch_quarter_prob_end),
        ):
            if not (0.0 <= float(value) <= 1.0):
                raise ValueError(f"{name} must be in [0, 1], got {value}")
        self.patch_size_probs_start = list(self.patch_size_probs)
        self.patch_size_probs_end = list(self.patch_size_probs)
    elif self.use_patch_prob_schedule:
        self.patch_size_probs_start = parse_patch_size_probs(
            kwargs.get("patch_size_probs_start", "0.8,0.2,0.0")
        )
        self.patch_size_probs_end = parse_patch_size_probs(
            kwargs.get("patch_size_probs_end", "0.7,0.25,0.05")
        )
        if not (0.0 <= self.patch_schedule_mid_frac <= 1.0):
            raise ValueError(
                f"patch_schedule_mid_frac must be in [0, 1], got {self.patch_schedule_mid_frac}"
            )
        if not (self.patch_schedule_mid_frac <= self.patch_schedule_end_frac <= 1.0):
            raise ValueError(
                "patch_schedule_end_frac must be in "
                f"[{self.patch_schedule_mid_frac}, 1], got {self.patch_schedule_end_frac}"
            )
    else:
        self.patch_size_probs_start = list(self.patch_size_probs)
        self.patch_size_probs_end = list(self.patch_size_probs)
    if not (0.0 <= self.patch_t_crop_min <= 1.0):
        raise ValueError(f"patch_t_crop_min must be in [0, 1], got {self.patch_t_crop_min}")
    if self.patch_t_half_boost < 0.0:
        raise ValueError(f"patch_t_half_boost must be >= 0, got {self.patch_t_half_boost}")
    self.coord_channels = 2 if self.use_patch_diffusion else 0
    self.zero_patch_coords = bool(kwargs.get("zero_patch_coords", False))
    self.use_self_conditioning = bool(kwargs.get("use_self_conditioning", False))
    self.self_cond_train_prob = float(kwargs.get("self_cond_train_prob", 0.5))
    self.self_cond_warmup_start = int(kwargs.get("self_cond_warmup_start", 2000))
    self.self_cond_warmup_end = int(kwargs.get("self_cond_warmup_end", 4000))
    # Auto-regressive two-stage self-conditioning (temporal prev → draft → target).
    self.use_ar_self_conditioning = bool(kwargs.get("use_ar_self_conditioning", False))
    self.use_ar_chain_ot = bool(kwargs.get("use_ar_chain_ot", False))
    self.use_paired_bridge = bool(kwargs.get("use_paired_bridge", False))
    self.paired_bridge_noise_max = float(kwargs.get("paired_bridge_noise_max", 0.15))
    if not (0.0 <= self.paired_bridge_noise_max < 1.0):
        raise ValueError(
            f"paired_bridge_noise_max must be in [0, 1), got {self.paired_bridge_noise_max}"
        )
    self.use_spall_growth_mask = bool(kwargs.get("use_spall_growth_mask", False))
    self.spall_mask_grid = int(kwargs.get("spall_mask_grid", 16))
    self.spall_shift_px = int(kwargs.get("spall_shift_px", 4))
    self.spall_mask_threshold = float(kwargs.get("spall_mask_threshold", 0.5))
    self.spall_dark_level = float(kwargs.get("spall_dark_level", 0.0))
    self.spall_bg_quantile = float(kwargs.get("spall_bg_quantile", 0.70))
    self.spall_bg_target = float(kwargs.get("spall_bg_target", 0.30))
    self.spall_border_gap = float(kwargs.get("spall_border_gap", 0.55))
    self.spall_seed_margin = float(kwargs.get("spall_seed_margin", 0.2))
    self.spall_dilate_k = int(kwargs.get("spall_dilate_k", 1))
    self.spall_close_k = int(kwargs.get("spall_close_k", 3))
    self.spall_tv_weight = float(kwargs.get("spall_tv_weight", 0.1))
    self.spall_new_cell_weight = float(kwargs.get("spall_new_cell_weight", 8.0))
    self.spall_feather_px = float(kwargs.get("spall_feather_px", 4.0))
    self.spall_pool_noise_std = float(kwargs.get("spall_pool_noise_std", 0.05))
    self.spall_loss_weight = float(kwargs.get("spall_loss_weight", 1.0))
    self.spall_mask_only = bool(kwargs.get("spall_mask_only", False))
    self.freeze_spall_mask = bool(kwargs.get("freeze_spall_mask", False))
    self.spall_cond_prob = bool(kwargs.get("spall_cond_prob", False))
    self.spall_soft_paint = bool(kwargs.get("spall_soft_paint", False))
    self.spall_healthy_margin = float(kwargs.get("spall_healthy_margin", 0.10))
    self.spall_lock_healthy = bool(kwargs.get("spall_lock_healthy", True))
    self.spall_ctx_noise_max = float(kwargs.get("spall_ctx_noise_max", 0.15))
    if self.spall_tv_weight < 0.0:
        raise ValueError(f"spall_tv_weight must be >= 0, got {self.spall_tv_weight}")
    if not (0.5 <= self.spall_bg_quantile < 1.0):
        raise ValueError(
            f"spall_bg_quantile must be in [0.5, 1), got {self.spall_bg_quantile}"
        )
    if self.spall_border_gap < 0.0:
        raise ValueError(f"spall_border_gap must be >= 0, got {self.spall_border_gap}")
    if self.spall_seed_margin < 0.0:
        raise ValueError(f"spall_seed_margin must be >= 0, got {self.spall_seed_margin}")
    if self.spall_dilate_k < 1 or self.spall_dilate_k % 2 == 0:
        raise ValueError(f"spall_dilate_k must be an odd integer >= 1, got {self.spall_dilate_k}")
    if self.spall_close_k < 1 or self.spall_close_k % 2 == 0:
        raise ValueError(f"spall_close_k must be an odd integer >= 1, got {self.spall_close_k}")
    if self.spall_soft_paint and self.spall_cond_prob:
        raise ValueError("spall_soft_paint and spall_cond_prob cannot both be true")
    if self.spall_healthy_margin < 0.0:
        raise ValueError(
            f"spall_healthy_margin must be >= 0, got {self.spall_healthy_margin}"
        )
    if not (0.0 <= self.spall_ctx_noise_max < 1.0):
        raise ValueError(
            f"spall_ctx_noise_max must be in [0, 1), got {self.spall_ctx_noise_max}"
        )
    if self.spall_new_cell_weight < 1.0:
        raise ValueError(
            f"spall_new_cell_weight must be >= 1, got {self.spall_new_cell_weight}"
        )
    if self.spall_mask_grid < 2:
        raise ValueError(f"spall_mask_grid must be >= 2, got {self.spall_mask_grid}")
    if not (0.0 <= self.spall_mask_threshold < 1.0):
        raise ValueError(
            f"spall_mask_threshold must be in [0,1), got {self.spall_mask_threshold}"
        )
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
    if self.use_spall_growth_mask and (
        self.use_ar_self_conditioning
        or self.use_ar_chain_ot
        or self.use_self_conditioning
        or self.use_cascade_sr
    ):
        raise ValueError(
            "use_spall_growth_mask cannot be combined with AR, self-conditioning, or cascade SR"
        )
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
    if self.use_mafp:
        from src.model.exp.mafp import MAFP_PARAM_DIM
        self.mafp_param_dim = MAFP_PARAM_DIM
    else:
        self.mafp_param_dim = 0
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
        from src.model.exp.undamaged_ref import UndamagedRefEncoder
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
    if self.use_paired_bridge and (
        self.use_spall_growth_mask
        or self.use_ar_chain_ot
        or self.use_ar_self_conditioning
        or self.use_self_conditioning
        or self.use_cascade_sr
    ):
        raise ValueError(
            "use_paired_bridge cannot be combined with spall, AR, self-conditioning, or cascade SR"
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
    self.feat_guidance_scale = float(kwargs.get("feat_guidance_scale", 0.0))
    self.feat_guidance_t_power = float(kwargs.get("feat_guidance_t_power", 1.0))
    self.feat_guidance_fn = kwargs.get("feat_guidance_fn", None)
    self.sag_scale = float(kwargs.get("sag_scale", 0.0))
    self.sag_mask_frac = float(kwargs.get("sag_mask_frac", 0.3))
    self.sag_t_min = float(kwargs.get("sag_t_min", 0.2))
    self.sag_fill = str(kwargs.get("sag_fill", "zero")).lower()
    self.sag_reduce = str(kwargs.get("sag_reduce", "mean")).lower()
    self.sag_axis = str(kwargs.get("sag_axis", "key")).lower()
    if self.sag_fill not in ("zero", "local_mean"):
        raise ValueError(f"sag_fill must be zero|local_mean, got {self.sag_fill!r}")
    if self.sag_reduce not in ("mean", "max"):
        raise ValueError(f"sag_reduce must be mean|max, got {self.sag_reduce!r}")
    if self.sag_axis not in ("key", "query"):
        raise ValueError(f"sag_axis must be key|query, got {self.sag_axis!r}")
    if not (0.0 < self.sag_mask_frac < 1.0):
        raise ValueError(f"sag_mask_frac must be in (0,1), got {self.sag_mask_frac}")
    if not (0.0 <= self.sag_t_min < 1.0):
        raise ValueError(f"sag_t_min must be in [0,1), got {self.sag_t_min}")
    self.coord_guidance_scale = float(kwargs.get("coord_guidance_scale", 0.0))
    self.coord_guidance_hipass_sigma = float(
        kwargs.get("coord_guidance_hipass_sigma", 0.0)
    )
    self.coord_guidance_t_min = float(kwargs.get("coord_guidance_t_min", 0.0))
    self.coord_guidance_t_max = float(kwargs.get("coord_guidance_t_max", 1.0))
    if not (0.0 <= self.coord_guidance_t_min < self.coord_guidance_t_max <= 1.0):
        raise ValueError(
            "coord guidance window must satisfy "
            f"0 <= t_min < t_max <= 1, got "
            f"[{self.coord_guidance_t_min}, {self.coord_guidance_t_max}]"
        )
    self.apg_guidance_scale = float(kwargs.get("apg_guidance_scale", 0.0))
    self.apg_eta = float(kwargs.get("apg_eta", 0.0))
    self.apg_norm_threshold = float(kwargs.get("apg_norm_threshold", 15.0))
    self.apg_momentum = float(kwargs.get("apg_momentum", -0.5))
    self.coord_g0 = str(kwargs.get("coord_g0", "") or "").lower()
    if self.coord_g0 not in ("", "zero", "undamaged"):
        raise ValueError(f"coord_g0 must be '', 'zero' or 'undamaged', got {self.coord_g0!r}")
    self.apg_g0 = str(kwargs.get("apg_g0", "") or "").lower()
    if self.apg_g0 not in ("", "zero", "undamaged"):
        raise ValueError(f"apg_g0 must be '', 'zero' or 'undamaged', got {self.apg_g0!r}")
    self.autoguide_checkpoint = str(kwargs.get("autoguide_checkpoint", "") or "").strip()
    self.autoguide_result_dir = str(kwargs.get("autoguide_result_dir", "") or "").strip()
    self.autoguide_weight = float(kwargs.get("autoguide_weight", 0.0))
    self.autoguide_eta = float(kwargs.get("autoguide_eta", 1.0))
    self.autoguide_gamma = float(kwargs.get("autoguide_gamma", 0.0))
    self.autoguide_w_min = float(kwargs.get("autoguide_w_min", 0.0))
    self.autoguide_wl = float(kwargs.get("autoguide_wl", 1.0))
    self.autoguide_wh = float(kwargs.get("autoguide_wh", 1.0))
    self._autoguide_ref = []
    self.use_delta_g = bool(kwargs.get("use_delta_g", False))
    self.use_refiner_residual = bool(kwargs.get("use_refiner_residual", False))
    self.refiner_chain_prob = float(kwargs.get("refiner_chain_prob", 0.5))
    self.refiner_state_noise_max = float(kwargs.get("refiner_state_noise_max", 0.05))
    if not (0.0 <= self.refiner_chain_prob <= 1.0):
        raise ValueError(f"refiner_chain_prob must be in [0, 1], got {self.refiner_chain_prob}")
    if self.refiner_state_noise_max < 0.0:
        raise ValueError(
            f"refiner_state_noise_max must be >= 0, got {self.refiner_state_noise_max}"
        )
    self.refiner_base_checkpoint = str(kwargs.get("refiner_base_checkpoint", "") or "").strip()
    self.refiner_base_result_dir = str(kwargs.get("refiner_base_result_dir", "") or "").strip()
    self.refiner_roll_prob = float(kwargs.get("refiner_roll_prob", 0.15))
    self.refiner_roll_nfe = int(kwargs.get("refiner_roll_nfe", 10))
    if not (0.0 <= self.refiner_roll_prob <= 1.0):
        raise ValueError(f"refiner_roll_prob must be in [0, 1], got {self.refiner_roll_prob}")
    if self.refiner_roll_nfe < 2:
        raise ValueError(f"refiner_roll_nfe must be >= 2, got {self.refiner_roll_nfe}")
    if self.refiner_base_checkpoint and not (self.use_paired_bridge or self.use_refiner_residual):
        raise ValueError(
            "--refiner_base_checkpoint requires --use_paired_bridge or --use_refiner_residual"
        )
    if self.use_refiner_residual:
        if not self.refiner_base_checkpoint:
            raise ValueError("--use_refiner_residual requires --refiner_base_checkpoint")
        if (
            self.use_paired_bridge
            or self.use_spall_growth_mask
            or self.use_ar_chain_ot
            or self.use_ar_self_conditioning
            or self.use_self_conditioning
            or self.use_cascade_sr
        ):
            raise ValueError(
                "use_refiner_residual cannot be combined with paired bridge, spall, AR, "
                "self-conditioning, or cascade SR"
            )
    self._refiner_base_ref = []
    self.repel_prev_scale = float(kwargs.get("repel_prev_scale", 0.0))
    self.repel_prev_mode = str(kwargs.get("repel_prev_mode", "x1")).lower()
    if self.repel_prev_mode not in ("x1", "state"):
        raise ValueError(
            f"repel_prev_mode must be x1|state, got {self.repel_prev_mode!r}"
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
    self.g_pca_dim = int(kwargs.get("g_pca_dim", 0))
    if self.g_pca_dim < 0:
        raise ValueError(f"g_pca_dim must be >= 0, got {self.g_pca_dim}")
    if self.g_pca_dim > 0:
        self.physics_dim = int(self.g_pca_dim)
    self.physics_bottleneck_dim = int(kwargs.get("physics_bottleneck_dim", 0))
    if self.physics_bottleneck_dim < 0:
        raise ValueError(
            f"physics_bottleneck_dim must be >= 0, got {self.physics_bottleneck_dim}"
        )
    if self.g_pca_dim > 0 and self.physics_bottleneck_dim > 0:
        raise ValueError("Use g_pca_dim or physics_bottleneck_dim, not both")
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
        + (self.model_in_channels if (self.use_cascade_sr or self.use_refiner_residual) else 0)
        + self.coord_channels
        + (2 if self.use_spall_growth_mask else 0)
    )
    net_out_channels = self.model_in_channels + self.coord_channels
    cfm_mid_attn_heads = max(1, int(kwargs.get("cfm_mid_attn_heads", 4)))
    use_vss_l3_u3 = bool(kwargs.get("use_vss_l3_u3", True))
    use_vss_u2 = bool(kwargs.get("use_vss_u2", False))
    use_vss_l2 = bool(kwargs.get("use_vss_l2", False))
    use_u1_extra_resblock = bool(kwargs.get("use_u1_extra_resblock", False))
    if self.use_wavelet_unet:
        from src.model.exp.wavelet_unet import WaveletUNet
        net_cls = WaveletUNet
    else:
        net_cls = ADMUNetCFM
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
        physics_bottleneck_dim=self.physics_bottleneck_dim,
        physics_num_tokens=self.cfm_physics_num_tokens,
        physics_token_dim=self.cfm_physics_token_dim,
        physics_adagn_dim=self.cfm_physics_adagn_dim,
        gated_fusion_alpha_dim=self.gated_fusion_alpha_dim,
        **({"use_delta_g": True} if self.use_delta_g else {}),
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
    self.velocity_net.spall_cond_channels = 2 if self.use_spall_growth_mask else 0
    if self.use_spall_growth_mask:
        with torch.no_grad():
            weight = self.velocity_net.in_conv.weight
            end = weight.shape[1] - (1 if self.use_damage_mask_condition else 0)
            weight[:, end - 2:end].zero_()
        from src.model.cfm.spall_mask import SpallMaskHead
        self.spall_mask_head = SpallMaskHead(self.physics_dim, grid=self.spall_mask_grid)
        if self.freeze_spall_mask:
            if self.spall_mask_only:
                raise ValueError("freeze_spall_mask cannot be combined with spall_mask_only")
            for param in self.spall_mask_head.parameters():
                param.requires_grad = False
    else:
        self.spall_mask_head = None
        if self.freeze_spall_mask:
            raise ValueError("freeze_spall_mask requires use_spall_growth_mask")
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
        f"g_pca_dim={self.g_pca_dim} "
        f"physics_bottleneck_dim={self.physics_bottleneck_dim} "
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
        if self.use_patch_prob_schedule and self.patch_schedule_mode == "pixel_mass":
            p0 = probs_from_pixel_mass(self.patch_pixel_mass_start, self.patch_quarter_prob_start)
            p1 = probs_from_pixel_mass(self.patch_pixel_mass_end, self.patch_quarter_prob_end)
            print(
                f"[CFM] Patch Diffusion pixel-mass schedule "
                f"(steps={self.total_steps}): "
                f"M {self.patch_pixel_mass_start:.2f}→{self.patch_pixel_mass_end:.2f} "
                f"q {self.patch_quarter_prob_start:.2f}→{self.patch_quarter_prob_end:.2f} "
                f"start={p0[0]:.2f}/{p0[1]:.2f}/{p0[2]:.2f} "
                f"end={p1[0]:.2f}/{p1[1]:.2f}/{p1[2]:.2f} "
                f"(M_end={patch_pixel_mass(p1):.2f}, {full_size}/{full_size // 2}/{full_size // 4})",
                flush=True,
            )
        elif self.use_patch_prob_schedule:
            mid_step = int(round(self.patch_schedule_mid_frac * max(1, self.total_steps)))
            end_start = int(round(self.patch_schedule_end_frac * max(1, self.total_steps)))
            print(
                f"[CFM] Patch Diffusion two-segment schedule "
                f"(steps={self.total_steps}, mid@{mid_step}, end_ramp@{end_start}): "
                f"start={self.patch_size_probs_start[0]:.2f}/{self.patch_size_probs_start[1]:.2f}/{self.patch_size_probs_start[2]:.2f} "
                f"mid={self.patch_size_probs[0]:.2f}/{self.patch_size_probs[1]:.2f}/{self.patch_size_probs[2]:.2f} "
                f"end={self.patch_size_probs_end[0]:.2f}/{self.patch_size_probs_end[1]:.2f}/{self.patch_size_probs_end[2]:.2f} "
                f"({full_size}/{full_size // 2}/{full_size // 4})",
                flush=True,
            )
        else:
            print(
                f"[CFM] Patch Diffusion schedule: "
                f"{full_size}x{full_size}={self.patch_size_probs[0]:.2f}, "
                f"{full_size // 2}x{full_size // 2}={self.patch_size_probs[1]:.2f}, "
                f"{full_size // 4}x{full_size // 4}={self.patch_size_probs[2]:.2f}",
                flush=True,
            )
        if self.use_patch_t_schedule:
            print(
                f"[CFM] Patch t-schedule (CFM t): extra_half={self.patch_t_half_boost:.2f} "
                f"for t>{self.patch_t_crop_min:.2f} (avg over sample t; never boosts quarter)",
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
