#!/bin/bash
#PBS -l select=1:ncpus=2:gpu_id=7
#PBS -l place=shared
#PBS -o log/train_selfcondnoDI.out
#PBS -e log/train_selfcondnoDI.err
#PBS -N CFM_scndi
#
# V9 / selfcondnoDI: pixel-space ADMUNetCFM + split gated AdaGN + zero DI
# + classic self-conditioning + VSS L3/U3 + Patch Diffusion.
# Matches results/DamagePatternGeneration/CFMADM/selfcondnoDI/args.json

cd "${PBS_O_WORKDIR:-$HOME/intership-project}"

set +u
source ~/.bashrc
set -u
conda activate internship

module load cuda-12.4

# Process priority toggle.
# - USE_NICE=False  -> original behavior (default OS priority, no nice wrapper)
# - USE_NICE=True   -> run training with nice -n ${NICE}
# Override at submit: qsub -v USE_NICE=False ...  or  qsub -v NICE=15 ...
USE_NICE=False
NICE=10
if [[ "${USE_NICE}" == "True" || "${USE_NICE}" == "true" || "${USE_NICE}" == "1" ]]; then
    NICE_CMD=(nice -n "${NICE}")
else
    NICE_CMD=()
fi

# Centralized augmentation config (single source in this script).
AUG_PROB=0.5
AUG_BLUR_PROB=0.15
AUG_BRIGHTNESS=0.2
AUG_CONTRAST=0.2
AUG_SATURATION=0.0
AUG_HUE=0.0
USE_GRAYSCALE_INPUT=True
USE_CLAHE=True
CLAHE_CLIP_LIMIT=2.0
CLAHE_TILE_GRID_SIZE=8,8
AUG_HFLIP_PROB=0.5
GRAD_CLIP_MAX_NORM=1.0
USE_VELOCITY_MIN_SNR_WEIGHTING=True
MIN_SNR_GAMMA=5.0
AUG_BLUR_KERNEL=3
AUG_BLUR_SIGMA_MIN=0.1
AUG_BLUR_SIGMA_MAX=0.5
AUG_ERASE_PROB=0.0
AUG_ERASE_SCALE_MIN=0.05
AUG_ERASE_SCALE_MAX=0.15
AUG_ERASE_RATIO_MIN=0.5
AUG_ERASE_RATIO_MAX=2.5

NUM_WORKERS=2
COND_PERTURB_DIST=uniform
COND_PERTURB_MIN=0.01
COND_PERTURB_MAX=0.03

# Split gated AdaGN: scalar = (zeroed DI) + direction; GRU g fused via learnable alpha.
CONDITION_MODE=split
ZERO_DI_CONDITION=True
GATED_FUSION_ALPHA_DIM=scalar
USE_PHYSICS_CROSS_ATTN=False
CFM_PHYSICS_NUM_TOKENS=16
CFM_PHYSICS_TOKEN_DIM=128

# Architecture bool: extra in_conv self-cond channels + train Bernoulli + ODE z1-hat carry.
# False → no self-cond channels (incompatible with selfcondnoDI checkpoints).
SELF_CONDITION=True
SELF_COND_TRAIN_PROB=0.5
SELF_COND_WARMUP_START=2000
SELF_COND_WARMUP_END=4000

# V9 backbone: ADM U-Net (not Wavelet). VSS at L3/U3 only.
USE_WAVELET_UNET=False
USE_VSS_L3_U3=True
USE_VSS_L2=False
USE_VSS_U2=False
USE_U1_EXTRA_RESBLOCK=False
USE_PATCH_DIFFUSION=True
PATCH_SIZE_PROBS=0.5,0.3,0.2

IMAGE_SIZE=128
BATCH_SIZE=128
CFM_MODEL_CHANNELS=32
CFM_COND_DIM=128
MAX_STEPS=15000
WARMUP_STEPS=1000
VAL_CHECK_INTERVAL=5

DATA_PATH=id_list_val_R307.json
INFERENCE_TRAIN_ANCHOR_EXP=R615
INFERENCE_VAL_ANCHOR_EXP=R307
INFERENCE_ANCHOR_TIMESTEP=-1
FIXED_DI_VALUES=0.0,0.4,0.75,1.0
EARLY_STOP_PATIENCE=15000
EARLY_STOP_MIN_DELTA=0.0
USE_EARLY_STOPPING=False
RESET_EARLY_STOP_PATIENCE=True
RESET_EARLY_STOP_BEST_SCORE=False
CHECKPOINT_EVERY_N_STEPS=2000
CHECKPOINT_START_STEP=10000

EXTRACTOR_PATH="./phase1_extractor"
if [[ ! -f "${EXTRACTOR_PATH}/best.ckpt" || ! -f "${EXTRACTOR_PATH}/args.json" ]]; then
    echo "[train_ADMCFM] Missing ${EXTRACTOR_PATH}/best.ckpt or args.json"
    echo "[train_ADMCFM] Run: bash job/DamagePatternGeneration/prepare_extractor_from_hf.sh"
    exit 1
fi

RESUME_CKPT=""
if [[ -n "${RESUME_CKPT}" ]]; then
    if [[ ! -f "${RESUME_CKPT}" ]]; then
        echo "[train_ADMCFM] Resume checkpoint not found: ${RESUME_CKPT}"
        exit 1
    fi
    RESUME_ARGS=(--resume "${RESUME_CKPT}")
    echo "[train_ADMCFM] Resuming from: ${RESUME_CKPT}"
else
    RESUME_ARGS=()
    echo "[train_ADMCFM] Resume disabled; training from scratch."
fi

if ((${#NICE_CMD[@]})); then
    echo "[train_ADMCFM] USE_NICE=True NICE=${NICE}"
else
    echo "[train_ADMCFM] USE_NICE=False (default priority, no nice)"
fi
echo "[train_ADMCFM] CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
echo "[train_ADMCFM] V9 selfcondnoDI: ADM U-Net + split gated AdaGN + zero DI"
echo "[train_ADMCFM] CONDITION_MODE=${CONDITION_MODE} ZERO_DI_CONDITION=${ZERO_DI_CONDITION} DATA_PATH=${DATA_PATH}"
echo "[train_ADMCFM] gated_fusion_alpha_dim=${GATED_FUSION_ALPHA_DIM} USE_PHYSICS_CROSS_ATTN=${USE_PHYSICS_CROSS_ATTN}"
echo "[train_ADMCFM] SELF_CONDITION=${SELF_CONDITION} p=${SELF_COND_TRAIN_PROB} warmup=[${SELF_COND_WARMUP_START},${SELF_COND_WARMUP_END}]"
echo "[train_ADMCFM] USE_WAVELET_UNET=${USE_WAVELET_UNET} USE_VSS_L3_U3=${USE_VSS_L3_U3} USE_PATCH_DIFFUSION=${USE_PATCH_DIFFUSION}"
echo "[train_ADMCFM] IMAGE_SIZE=${IMAGE_SIZE} BATCH_SIZE=${BATCH_SIZE} CFM_MODEL_CHANNELS=${CFM_MODEL_CHANNELS} CFM_COND_DIM=${CFM_COND_DIM}"
echo "[train_ADMCFM] MAX_STEPS=${MAX_STEPS} WARMUP_STEPS=${WARMUP_STEPS} VAL_CHECK_INTERVAL=${VAL_CHECK_INTERVAL}"
echo "[train_ADMCFM] val_anchor=${INFERENCE_VAL_ANCHOR_EXP} train_anchor=${INFERENCE_TRAIN_ANCHOR_EXP}"
echo "[train_ADMCFM] FIXED_DI_VALUES=${FIXED_DI_VALUES}"
python3 - <<'PY'
import os
import subprocess

visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
print(f"[train_ADMCFM] Visible GPU list (physical IDs): {visible if visible else 'all'}")
try:
    out = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=index,name,uuid", "--format=csv,noheader"],
        text=True,
    ).strip()
    print("[train_ADMCFM] Host GPU inventory:")
    for line in out.splitlines():
        print(f"[train_ADMCFM]   {line}")
except Exception as exc:
    print(f"[train_ADMCFM] Failed to query nvidia-smi: {exc}")
PY

if [[ -n "${WANDB_API_KEY:-}" ]]; then
    echo "[train_ADMCFM] Logging in to Weights & Biases with WANDB_API_KEY"
    wandb login "${WANDB_API_KEY}" --relogin >/dev/null 2>&1
elif [[ -f "${HOME}/.netrc" ]] && grep -q "api\.wandb\.ai" "${HOME}/.netrc"; then
    echo "[train_ADMCFM] Using existing wandb login from ~/.netrc"
else
    echo "[train_ADMCFM] No wandb credential found."
    echo "[train_ADMCFM] Please run one-time login: wandb login"
    echo "[train_ADMCFM] or submit with: qsub -v WANDB_API_KEY=<your_key> ..."
    exit 1
fi

"${NICE_CMD[@]}" python3 -u train_mixed.py --lr 1e-4 --epoch 3000 --batch_size "${BATCH_SIZE}" --num_workers "${NUM_WORKERS}" --train_virtual_repeat 100 --max_steps "${MAX_STEPS}" --warmup_steps "${WARMUP_STEPS}" --val_check_interval "${VAL_CHECK_INTERVAL}" --architecture CFMADM --loss_type CFMADM --data_path "${DATA_PATH}" \
        --extractor_path "${EXTRACTOR_PATH}" --use_wandb \
        "${RESUME_ARGS[@]}" \
        --img_aug \
        --use_grayscale_input "${USE_GRAYSCALE_INPUT}" \
        --use_clahe "${USE_CLAHE}" --clahe_clip_limit "${CLAHE_CLIP_LIMIT}" \
        --clahe_tile_grid_size "${CLAHE_TILE_GRID_SIZE}" \
        --aug_prob "${AUG_PROB}" --aug_blur_prob "${AUG_BLUR_PROB}" \
        --aug_brightness "${AUG_BRIGHTNESS}" --aug_contrast "${AUG_CONTRAST}" --aug_saturation "${AUG_SATURATION}" --aug_hue "${AUG_HUE}" \
        --aug_hflip_prob "${AUG_HFLIP_PROB}" \
        --aug_blur_kernel "${AUG_BLUR_KERNEL}" --aug_blur_sigma_min "${AUG_BLUR_SIGMA_MIN}" --aug_blur_sigma_max "${AUG_BLUR_SIGMA_MAX}" \
        --aug_erase_prob "${AUG_ERASE_PROB}" --aug_erase_scale_min "${AUG_ERASE_SCALE_MIN}" --aug_erase_scale_max "${AUG_ERASE_SCALE_MAX}" \
        --aug_erase_ratio_min "${AUG_ERASE_RATIO_MIN}" --aug_erase_ratio_max "${AUG_ERASE_RATIO_MAX}" \
        --condition_perturb --condition_perturb_dist "${COND_PERTURB_DIST}" \
        --condition_perturb_min "${COND_PERTURB_MIN}" --condition_perturb_max "${COND_PERTURB_MAX}" \
        --condition_mode "${CONDITION_MODE}" \
        --zero_di_condition "${ZERO_DI_CONDITION}" \
        --gated_fusion_alpha_dim "${GATED_FUSION_ALPHA_DIM}" \
        --use_physics_cross_attn "${USE_PHYSICS_CROSS_ATTN}" \
        --cfm_physics_num_tokens "${CFM_PHYSICS_NUM_TOKENS}" \
        --cfm_physics_token_dim "${CFM_PHYSICS_TOKEN_DIM}" \
        --self_condition "${SELF_CONDITION}" \
        --self_cond_train_prob "${SELF_COND_TRAIN_PROB}" \
        --self_cond_warmup_start "${SELF_COND_WARMUP_START}" \
        --self_cond_warmup_end "${SELF_COND_WARMUP_END}" \
        --use_u1_extra_resblock "${USE_U1_EXTRA_RESBLOCK}" \
        --use_wavelet_unet "${USE_WAVELET_UNET}" \
        --use_vss_l3_u3 "${USE_VSS_L3_U3}" \
        --use_vss_l2 "${USE_VSS_L2}" \
        --use_vss_u2 "${USE_VSS_U2}" \
        --cfm_physics_adagn_dim 128 --physics_dim 1024 \
        --use_early_stopping "${USE_EARLY_STOPPING}" \
        --early_stop_patience "${EARLY_STOP_PATIENCE}" --early_stop_min_delta "${EARLY_STOP_MIN_DELTA}" \
        --reset_early_stop_patience "${RESET_EARLY_STOP_PATIENCE}" \
        --reset_early_stop_best_score "${RESET_EARLY_STOP_BEST_SCORE}" \
        --checkpoint_every_n_steps "${CHECKPOINT_EVERY_N_STEPS}" \
        --checkpoint_start_step "${CHECKPOINT_START_STEP}" \
        --latent_conditioning True --direction_conditioning True --include_di_conditioning True --add_DI_loss False \
        --use_damage_mask_condition True --use_object_loss_mask False \
        --image_size "${IMAGE_SIZE}" --cfm_model_channels "${CFM_MODEL_CHANNELS}" --cfm_num_res_blocks 1 --cfm_cond_dim "${CFM_COND_DIM}" --cfm_dropout 0.1 \
        --cfm_mid_attn_heads 4 \
        --use_patch_diffusion "${USE_PATCH_DIFFUSION}" --patch_size_probs "${PATCH_SIZE_PROBS}" \
        --cfm_solver heun --cfm_ode_steps 30 --lr_schedule warmup_cosine --min_lr_ratio 0.1 \
        --grad_clip_max_norm "${GRAD_CLIP_MAX_NORM}" \
        --use_velocity_min_snr_weighting "${USE_VELOCITY_MIN_SNR_WEIGHTING}" --min_snr_gamma "${MIN_SNR_GAMMA}" \
        --inference_every_n_epochs 500 --inference_num_images 4 \
        --inference_train_anchor_exp "${INFERENCE_TRAIN_ANCHOR_EXP}" \
        --inference_val_anchor_exp "${INFERENCE_VAL_ANCHOR_EXP}" \
        --inference_anchor_timestep "${INFERENCE_ANCHOR_TIMESTEP}" \
        --fixed_di_values "${FIXED_DI_VALUES}"
