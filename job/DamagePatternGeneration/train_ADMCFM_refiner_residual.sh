#!/bin/bash
#PBS -l select=1:ncpus=4:mem=15gb:gpu_id=5
#PBS -l place=shared
#PBS -o log/train_refiner_residual.out
#PBS -e log/train_refiner_residual.err
#PBS -N CFM_refres

# Same Linear(1024→128) split AdaGN, plus train-only 10% g=0 CFG dropout.
# Does not drop patch XY. Resume-compatible with no-drop Linear ckpt (same arch).

cd "${PBS_O_WORKDIR:-$HOME/intership-project}"
source job/DamagePatternGeneration/common.sh
adm_env

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

# v11 refiner (docs/ARCHITECTURE_UPDATE_V11.md §8 next step): 10.6-C with the v11 condition
# ([encoder z | 4 damage scalars], loading sign, flip keeps the face) and a v11 base. The base is the
# warm start, the state cache source and frame 0 at inference. Δ[z | scalars] goes to the control branch.
V11="${V11:-False}"
V11_ARGS=()
if [[ "${V11}" == "True" ]]; then
    V11_ENCODER="${V11_ENCODER:-results/DamageStateEncoder/A_info/2026-10-23_22-34-47}"
    REFINER_BASE_DIR="${REFINER_BASE_DIR:-results/DamagePatternGeneration/CFMADM/2026-10-24_12-36-59_v11}"
    REFINER_BASE_CKPT="${REFINER_BASE_CKPT:-${REFINER_BASE_DIR}/checkpoint_step_012000.pt}"
    VELOCITY_INIT="${VELOCITY_INIT:-${REFINER_BASE_CKPT}}"
    REFINER_TARGET_X1="${REFINER_TARGET_X1:-True}"
    REFINER_IMPLICIT="${REFINER_IMPLICIT:-True}"
    CONTROL_HINT_ONLY="${CONTROL_HINT_ONLY:-True}"
    CONTROL_INJECT_ENCODER="${CONTROL_INJECT_ENCODER:-True}"
    TRUNK_LR_MULT="${TRUNK_LR_MULT:-1.0}"
    RUN_TAG="${RUN_TAG:-v11_refiner}"
    export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}" MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"   # 2 CPUs per job
    V11_ARGS=(--damage_condition_encoder "${V11_ENCODER}" --damage_scalar_dim 4 \
              --load_sign_condition True --flip_keeps_direction True)
    # Appearance-code base (ARCHITECTURE_UPDATE_V11.md §9.4): APP_CODE_DIM must match the base; unseen
    # columns (val KID during training) get the mean code (-2) unless APP_UNSEEN_CODE says otherwise.
    APP_CODE_DIM="${APP_CODE_DIM:-0}"
    if [[ "${APP_CODE_DIM}" != "0" ]]; then
        V11_ARGS+=(--app_code_dim "${APP_CODE_DIM}" --app_drop_prob "${APP_DROP_PROB:-0.2}" \
                   --app_noise_std "${APP_NOISE_STD:-0.05}" --app_mix_prob "${APP_MIX_PROB:-0.0}" \
                   --app_unseen_code "${APP_UNSEEN_CODE:--2}")
    fi
    echo "[train_ADMCFM] V11 refiner: encoder=${V11_ENCODER} base=${REFINER_BASE_CKPT} APP_CODE_DIM=${APP_CODE_DIM}"
fi

NUM_WORKERS="${NUM_WORKERS:-2}"
# 4 CPUs: main process + 2 loader workers (1 thread each). Keep BLAS/OpenMP small.
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-2}" MKL_NUM_THREADS="${MKL_NUM_THREADS:-2}"
RUN_TAG="${RUN_TAG:-refiner_res}"
COND_PERTURB_DIST=uniform
COND_PERTURB_MIN=0.01
COND_PERTURB_MAX=0.03

# Split gated AdaGN: scalar = direction only (DI channel dropped); GRU g fused via alpha.
CONDITION_MODE=split
ZERO_DI_CONDITION=True
CFG_G_DROP_PROB=0.0
USE_SPALL_GROWTH_MASK="${USE_SPALL_GROWTH_MASK:-False}"
# Single-frame G. M stays off unless USE_SPALL_GROWTH_MASK=True.
# SPALL_SOFT_PAINT=True: feathered paint weight, same path in train and ODE.
# SPALL_COND_PROB=True is the full-sigmoid ablation and cannot be combined with it.
# Mask-only retrains stop at 2000 steps.
SPALL_MASK_ONLY="${SPALL_MASK_ONLY:-False}"
FREEZE_SPALL_MASK=True
SPALL_COND_PROB=False
SPALL_SOFT_PAINT=True
SPALL_LOCK_HEALTHY="${SPALL_LOCK_HEALTHY:-True}"
SPALL_HEALTHY_MARGIN="${SPALL_HEALTHY_MARGIN:-0.10}"
SPALL_CTX_NOISE_MAX="${SPALL_CTX_NOISE_MAX:-0.15}"
SPALL_MASK_CKPT="results/DamagePatternGeneration/CFMADM/2026-10-01_12-04-28/Opensees Based damage Patterns Generation/ixmlkm6l/checkpoints/last.ckpt"
VELOCITY_INIT="${VELOCITY_INIT:-}"
# Paired bridge: t=0 is the previous photo, t=1 is this photo. No M.
USE_PAIRED_BRIDGE="${USE_PAIRED_BRIDGE:-False}"
USE_REFINER_RESIDUAL="${USE_REFINER_RESIDUAL:-True}"
CACHE_DATA="${CACHE_DATA:-True}"
USE_DELTA_G="${USE_DELTA_G:-True}"
REFINER_CHAIN_PROB="${REFINER_CHAIN_PROB:-0.5}"
REFINER_STATE_NOISE_MAX="${REFINER_STATE_NOISE_MAX:-0.05}"
PAIRED_BRIDGE_NOISE_MAX="${PAIRED_BRIDGE_NOISE_MAX:-0.15}"
# Refiner: S_{t-1} is the frozen single-frame model's own sample (1 Euler step, or a
# REFINER_ROLL_NFE-NFE Heun rollout with prob REFINER_ROLL_PROB), not the real photo.
REFINER_BASE_DIR="${REFINER_BASE_DIR:-results/DamagePatternGeneration/CFMADM/1024-dadaGNsplit}"
REFINER_BASE_CKPT="${REFINER_BASE_CKPT:-${REFINER_BASE_DIR}/Opensees Based damage Patterns Generation/chjqbp9m/checkpoints/last.ckpt}"
REFINER_ROLL_PROB="${REFINER_ROLL_PROB:-0.15}"
REFINER_ROLL_NFE="${REFINER_ROLL_NFE:-10}"
# v10.5 implicit refiner (docs/ARCHITECTURE_UPDATE_V10_5.md): S_{t-1} through a ControlNet branch,
# the net generates x_t. Same script for both versions, only two switches differ:
#   10.5-A: CONTROL_INJECT_ENCODER=False FREEZE_TRUNK=True   (decoder skips, frozen gn trunk)
#   10.5-B: CONTROL_INJECT_ENCODER=True  FREEZE_TRUNK=False  (encoder features, trunk at TRUNK_LR_MULT x lr)
# Everything else (state cache, perturbation, shared augmentation, tone loss, low-frequency noise
# sharing, roll prob, steps) follows REFINER_IMPLICIT so A and B stay matched.
# v10.6 (docs/ARCHITECTURE_UPDATE_V10_6.md): REFINER_TARGET_X1=True keeps S_{t-1} concatenated
# (V10 layout) but generates x_t, with the same exposure package and no single-frame task.
#   10.6-F: REFINER_TARGET_X1=True                         (gn weights, zeroed out_conv)
#   10.6-S: REFINER_TARGET_X1=True REFINER_FROM_SCRATCH=True (random init)
#   10.6-C: REFINER_TARGET_X1=True REFINER_IMPLICIT=True CONTROL_HINT_ONLY=True CONTROL_INJECT_ENCODER=True TRUNK_LR_MULT=1.0
#           (F, but S_{t-1} goes through a fresh nycu-project control branch instead of the input channels)
REFINER_IMPLICIT="${REFINER_IMPLICIT:-False}"
REFINER_TARGET_X1="${REFINER_TARGET_X1:-False}"
REFINER_FROM_SCRATCH="${REFINER_FROM_SCRATCH:-False}"
REFINER_X1=False
if [[ "${REFINER_IMPLICIT}" == "True" || "${REFINER_TARGET_X1}" == "True" ]]; then
    REFINER_X1=True
fi
CONTROL_INJECT_ENCODER="${CONTROL_INJECT_ENCODER:-False}"
CONTROL_HINT_ONLY="${CONTROL_HINT_ONLY:-False}"
FREEZE_TRUNK="${FREEZE_TRUNK:-False}"
TRUNK_LR_MULT="${TRUNK_LR_MULT:-0.2}"
REFINER_CONTROL_DROP_PROB="${REFINER_CONTROL_DROP_PROB:-0.15}"
REFINER_STATE_CACHE="${REFINER_STATE_CACHE:-${REFINER_X1}}"
REFINER_STATE_CACHE_N="${REFINER_STATE_CACHE_N:-4}"
REFINER_STATE_PERTURB="${REFINER_STATE_PERTURB:-${REFINER_X1}}"
SHARE_PHOTOMETRIC_AUG="${SHARE_PHOTOMETRIC_AUG:-${REFINER_X1}}"
REFINER_SHARED_LOWFREQ_NOISE="${REFINER_SHARED_LOWFREQ_NOISE:-${REFINER_X1}}"
KID_EVERY_N_STEPS="${KID_EVERY_N_STEPS:-3000}"
KID_K="${KID_K:-8}"
if [[ "${REFINER_X1}" == "True" ]]; then
    REFINER_ROLL_PROB="${REFINER_ROLL_PROB_IMPLICIT:-0.3}"
    REFINER_TONE_LOSS_WEIGHT="${REFINER_TONE_LOSS_WEIGHT:-1.0}"
    MAX_STEPS="${MAX_STEPS:-15000}"
else
    REFINER_TONE_LOSS_WEIGHT="${REFINER_TONE_LOSS_WEIGHT:-0.0}"
fi
if [[ "${USE_PAIRED_BRIDGE}" == "True" || "${USE_PAIRED_BRIDGE}" == "true" || "${USE_PAIRED_BRIDGE}" == "1" || "${USE_REFINER_RESIDUAL}" == "True" ]]; then
    USE_SPALL_GROWTH_MASK=False
    FREEZE_SPALL_MASK=False
    SPALL_MASK_CKPT=""
    SPALL_SOFT_PAINT=False
    if [[ -z "${VELOCITY_INIT}" ]]; then
        VELOCITY_INIT="results/DamagePatternGeneration/CFMADM/1024-dadaGNsplit/chjqbp9m_last.ckpt"
    fi
    if [[ "${REFINER_FROM_SCRATCH}" == "True" ]]; then
        VELOCITY_INIT=""
    fi
fi
if [[ "${USE_SPALL_GROWTH_MASK}" != "True" && "${USE_SPALL_GROWTH_MASK}" != "true" && "${USE_SPALL_GROWTH_MASK}" != "1" ]]; then
    FREEZE_SPALL_MASK=False
    SPALL_MASK_CKPT=""
fi
SPALL_MASK_GRID=16
SPALL_SHIFT_PX=4
SPALL_MASK_THRESHOLD=0.5
SPALL_DARK_LEVEL=0.0
SPALL_BG_QUANTILE=0.70
SPALL_BG_TARGET=0.30
SPALL_BORDER_GAP=0.55
SPALL_SEED_MARGIN=0.2
SPALL_DILATE_K=1
SPALL_CLOSE_K=3
SPALL_TV_WEIGHT=0.0
SPALL_NEW_CELL_WEIGHT=8
SPALL_FEATHER_PX=4
SPALL_POOL_NOISE_STD=0.05
SPALL_LOSS_WEIGHT=1.0
GATED_FUSION_ALPHA_DIM=scalar
USE_PHYSICS_CROSS_ATTN=False
CFM_PHYSICS_NUM_TOKENS=16
CFM_PHYSICS_TOKEN_DIM=128

# Architecture bool: extra in_conv self-cond channels + train Bernoulli + ODE z1-hat carry.
# False → no self-cond channels (incompatible with selfcondnoDI checkpoints).
SELF_CONDITION=False
SELF_COND_TRAIN_PROB=0.5
SELF_COND_WARMUP_START=2000
SELF_COND_WARMUP_END=4000

# V9 backbone: ADM U-Net (not Wavelet). VSS at L3/U3 only.
USE_WAVELET_UNET=False
USE_VSS_L3_U3=True
USE_VSS_L2=False
USE_VSS_U2=False
USE_U1_EXTRA_RESBLOCK=False
USE_PATCH_DIFFUSION="${USE_PATCH_DIFFUSION:-True}"
USE_PATCH_PROB_SCHEDULE=False
PATCH_SCHEDULE_MODE=pixel_mass
PATCH_PIXEL_MASS_START=0.55
PATCH_PIXEL_MASS_END=0.64
PATCH_QUARTER_PROB_START=0.15
PATCH_QUARTER_PROB_END=0.10
USE_PATCH_T_SCHEDULE=False
PATCH_T_CROP_MIN=0.5
PATCH_T_HALF_BOOST=0.20
PATCH_SIZE_PROBS=0.5,0.3,0.2
PHYSICS_BOTTLENECK_DIM=0
G_TRAJ_MIX="${G_TRAJ_MIX:-False}"
G_TRAJ_MIX_DELTA="${G_TRAJ_MIX_DELTA:-2}"
G_TRAJ_MIX_MAX_LAMBDA="${G_TRAJ_MIX_MAX_LAMBDA:-0.5}"
G_PCA_DIM=0

IMAGE_SIZE="${IMAGE_SIZE:-128}"
BATCH_SIZE=128
CFM_MODEL_CHANNELS=32
CFM_COND_DIM=128
MAX_STEPS="${MAX_STEPS:-3000}"
if [[ "${SPALL_MASK_ONLY}" == "True" || "${SPALL_MASK_ONLY}" == "true" || "${SPALL_MASK_ONLY}" == "1" ]]; then
    MAX_STEPS=2000
fi
WARMUP_STEPS=1000
VAL_CHECK_INTERVAL="${VAL_CHECK_INTERVAL:-1000}"

DATA_PATH=id_list_val_R307.json
INFERENCE_TRAIN_ANCHOR_EXP=R615
INFERENCE_VAL_ANCHOR_EXP=R307
INFERENCE_ANCHOR_TIMESTEP=-1
FIXED_DI_VALUES=0.0,0.4,0.75,1.0
EARLY_STOP_PATIENCE=3000
EARLY_STOP_MIN_DELTA=0.0
USE_EARLY_STOPPING=False
RESET_EARLY_STOP_PATIENCE=True
RESET_EARLY_STOP_BEST_SCORE=False
CHECKPOINT_EVERY_N_STEPS=1000
CHECKPOINT_START_STEP=1000

EXTRACTOR_PATH="./phase1_extractor"
if [[ ! -f "${EXTRACTOR_PATH}/best.ckpt" || ! -f "${EXTRACTOR_PATH}/args.json" ]]; then
    echo "[train_ADMCFM] Missing ${EXTRACTOR_PATH}/best.ckpt or args.json"
    echo "[train_ADMCFM] Run: bash job/DamagePatternGeneration/prepare_extractor_from_hf.sh"
    exit 1
fi

# Fresh run: M now predicts accumulated extent, so the old incremental head is the wrong function.
RESUME_CKPT="${RESUME_CKPT:-}"
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
echo "[train_ADMCFM] ADM U-Net + split gated AdaGN + zero DI + no self-cond + static patch + Linear g"
echo "[train_ADMCFM] CONDITION_MODE=${CONDITION_MODE} ZERO_DI_CONDITION=${ZERO_DI_CONDITION} CFG_G_DROP_PROB=${CFG_G_DROP_PROB} DATA_PATH=${DATA_PATH}"
echo "[train_ADMCFM] gated_fusion_alpha_dim=${GATED_FUSION_ALPHA_DIM} USE_PHYSICS_CROSS_ATTN=${USE_PHYSICS_CROSS_ATTN}"
echo "[train_ADMCFM] SELF_CONDITION=${SELF_CONDITION} p=${SELF_COND_TRAIN_PROB} warmup=[${SELF_COND_WARMUP_START},${SELF_COND_WARMUP_END}]"
echo "[train_ADMCFM] USE_WAVELET_UNET=${USE_WAVELET_UNET} USE_VSS_L3_U3=${USE_VSS_L3_U3} USE_PATCH_DIFFUSION=${USE_PATCH_DIFFUSION}"
echo "[train_ADMCFM] PATCH_SCHED=${USE_PATCH_PROB_SCHEDULE} PATCH_SIZE_PROBS=${PATCH_SIZE_PROBS} PATCH_T=${USE_PATCH_T_SCHEDULE}"
echo "[train_ADMCFM] PHYSICS_BOTTLENECK_DIM=${PHYSICS_BOTTLENECK_DIM} G_PCA_DIM=${G_PCA_DIM} G_TRAJ_MIX=${G_TRAJ_MIX}"
echo "[train_ADMCFM] IMAGE_SIZE=${IMAGE_SIZE} BATCH_SIZE=${BATCH_SIZE} CFM_MODEL_CHANNELS=${CFM_MODEL_CHANNELS} CFM_COND_DIM=${CFM_COND_DIM}"
echo "[train_ADMCFM] MAX_STEPS=${MAX_STEPS} WARMUP_STEPS=${WARMUP_STEPS} VAL_CHECK_INTERVAL=${VAL_CHECK_INTERVAL} NUM_WORKERS=${NUM_WORKERS}"
echo "[train_ADMCFM] RUN_TAG=${RUN_TAG:-<none>} NUM_WORKERS=${NUM_WORKERS} PAIRED_BRIDGE=${USE_PAIRED_BRIDGE} BRIDGE_NOISE_MAX=${PAIRED_BRIDGE_NOISE_MAX} SPALL_COND_PROB=${SPALL_COND_PROB} SPALL_SOFT_PAINT=${SPALL_SOFT_PAINT} LOCK_HEALTHY=${SPALL_LOCK_HEALTHY} HEALTHY_MARGIN=${SPALL_HEALTHY_MARGIN} CTX_NOISE_MAX=${SPALL_CTX_NOISE_MAX} SPALL_MASK_ONLY=${SPALL_MASK_ONLY} FREEZE_SPALL_MASK=${FREEZE_SPALL_MASK} VELOCITY_INIT=${VELOCITY_INIT:-<empty>}"
echo "[train_ADMCFM] REFINER_IMPLICIT=${REFINER_IMPLICIT} REFINER_TARGET_X1=${REFINER_TARGET_X1} FROM_SCRATCH=${REFINER_FROM_SCRATCH} CONTROL_INJECT_ENCODER=${CONTROL_INJECT_ENCODER} HINT_ONLY=${CONTROL_HINT_ONLY} FREEZE_TRUNK=${FREEZE_TRUNK} TRUNK_LR_MULT=${TRUNK_LR_MULT} CONTROL_DROP=${REFINER_CONTROL_DROP_PROB} STATE_CACHE=${REFINER_STATE_CACHE} STATE_PERTURB=${REFINER_STATE_PERTURB} SHARE_PHOTO_AUG=${SHARE_PHOTOMETRIC_AUG} LOWFREQ_NOISE=${REFINER_SHARED_LOWFREQ_NOISE} TONE_W=${REFINER_TONE_LOSS_WEIGHT} ROLL_PROB=${REFINER_ROLL_PROB} KID_EVERY=${KID_EVERY_N_STEPS} KID_K=${KID_K}"
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

"${NICE_CMD[@]}" python3 -u train_mixed.py --lr 1e-4 --epoch 3000 --batch_size "${BATCH_SIZE}" --num_workers "${NUM_WORKERS}" --run_tag "${RUN_TAG}" --train_virtual_repeat 100 --max_steps "${MAX_STEPS}" --warmup_steps "${WARMUP_STEPS}" --val_check_interval "${VAL_CHECK_INTERVAL}" --architecture CFMADM --loss_type CFMADM --data_path "${DATA_PATH}" \
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
        --cfg_g_drop_prob "${CFG_G_DROP_PROB}" \
        --use_paired_bridge "${USE_PAIRED_BRIDGE}" \
        --paired_bridge_noise_max "${PAIRED_BRIDGE_NOISE_MAX}" \
        --refiner_base_checkpoint "${REFINER_BASE_CKPT}" --refiner_base_result_dir "${REFINER_BASE_DIR}" \
        --refiner_roll_prob "${REFINER_ROLL_PROB}" --refiner_roll_nfe "${REFINER_ROLL_NFE}" \
        --use_refiner_residual "${USE_REFINER_RESIDUAL}" --refiner_chain_prob "${REFINER_CHAIN_PROB}" \
        --refiner_state_noise_max "${REFINER_STATE_NOISE_MAX}" --cache_data "${CACHE_DATA}" --use_delta_g "${USE_DELTA_G}" \
        --refiner_implicit "${REFINER_IMPLICIT}" --refiner_target_x1 "${REFINER_TARGET_X1}" --control_inject_encoder "${CONTROL_INJECT_ENCODER}" \
        --control_hint_only "${CONTROL_HINT_ONLY}" \
        --freeze_trunk "${FREEZE_TRUNK}" --trunk_lr_mult "${TRUNK_LR_MULT}" \
        --refiner_control_drop_prob "${REFINER_CONTROL_DROP_PROB}" \
        --refiner_state_cache "${REFINER_STATE_CACHE}" --refiner_state_cache_n "${REFINER_STATE_CACHE_N}" \
        --refiner_state_perturb "${REFINER_STATE_PERTURB}" --refiner_tone_loss_weight "${REFINER_TONE_LOSS_WEIGHT}" \
        --share_photometric_aug "${SHARE_PHOTOMETRIC_AUG}" --refiner_shared_lowfreq_noise "${REFINER_SHARED_LOWFREQ_NOISE}" \
        --kid_every_n_steps "${KID_EVERY_N_STEPS}" --kid_k "${KID_K}" \
        --use_spall_growth_mask "${USE_SPALL_GROWTH_MASK}" \
        --spall_mask_grid "${SPALL_MASK_GRID}" \
        --spall_shift_px "${SPALL_SHIFT_PX}" \
        --spall_mask_threshold "${SPALL_MASK_THRESHOLD}" \
        --spall_dark_level "${SPALL_DARK_LEVEL}" \
        --spall_bg_quantile "${SPALL_BG_QUANTILE}" \
        --spall_bg_target "${SPALL_BG_TARGET}" \
        --spall_border_gap "${SPALL_BORDER_GAP}" \
        --spall_seed_margin "${SPALL_SEED_MARGIN}" \
        --spall_dilate_k "${SPALL_DILATE_K}" \
        --spall_close_k "${SPALL_CLOSE_K}" \
        --spall_tv_weight "${SPALL_TV_WEIGHT}" \
        --spall_new_cell_weight "${SPALL_NEW_CELL_WEIGHT}" \
        --spall_feather_px "${SPALL_FEATHER_PX}" \
        --spall_pool_noise_std "${SPALL_POOL_NOISE_STD}" \
        --spall_loss_weight "${SPALL_LOSS_WEIGHT}" \
        --spall_mask_only "${SPALL_MASK_ONLY}" \
        --freeze_spall_mask "${FREEZE_SPALL_MASK}" \
        --spall_cond_prob "${SPALL_COND_PROB}" \
        --spall_soft_paint "${SPALL_SOFT_PAINT}" \
        --spall_lock_healthy "${SPALL_LOCK_HEALTHY}" \
        --spall_healthy_margin "${SPALL_HEALTHY_MARGIN}" \
        --spall_ctx_noise_max "${SPALL_CTX_NOISE_MAX}" \
        --spall_mask_ckpt "${SPALL_MASK_CKPT}" \
        --velocity_init "${VELOCITY_INIT}" \
        "${V11_ARGS[@]}" \
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
        --cfm_physics_adagn_dim 128 --physics_dim 1024 --physics_bottleneck_dim "${PHYSICS_BOTTLENECK_DIM}" \
        --g_traj_mix "${G_TRAJ_MIX}" --g_traj_mix_delta "${G_TRAJ_MIX_DELTA}" \
        --g_traj_mix_max_lambda "${G_TRAJ_MIX_MAX_LAMBDA}" --g_pca_dim "${G_PCA_DIM}" \
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
        --use_patch_diffusion "${USE_PATCH_DIFFUSION}" --use_patch_prob_schedule "${USE_PATCH_PROB_SCHEDULE}" \
        --patch_schedule_mode "${PATCH_SCHEDULE_MODE}" --patch_size_probs "${PATCH_SIZE_PROBS}" \
        --patch_pixel_mass_start "${PATCH_PIXEL_MASS_START}" --patch_pixel_mass_end "${PATCH_PIXEL_MASS_END}" \
        --patch_quarter_prob_start "${PATCH_QUARTER_PROB_START}" --patch_quarter_prob_end "${PATCH_QUARTER_PROB_END}" \
        --use_patch_t_schedule "${USE_PATCH_T_SCHEDULE}" --patch_t_crop_min "${PATCH_T_CROP_MIN}" \
        --patch_t_half_boost "${PATCH_T_HALF_BOOST}" \
        --cfm_solver heun --cfm_ode_steps 30 --lr_schedule warmup_cosine --min_lr_ratio 0.1 \
        --grad_clip_max_norm "${GRAD_CLIP_MAX_NORM}" \
        --use_velocity_min_snr_weighting "${USE_VELOCITY_MIN_SNR_WEIGHTING}" --min_snr_gamma "${MIN_SNR_GAMMA}" \
        --inference_every_n_epochs 1000 --inference_num_images 4 \
        --inference_train_anchor_exp "${INFERENCE_TRAIN_ANCHOR_EXP}" \
        --inference_val_anchor_exp "${INFERENCE_VAL_ANCHOR_EXP}" \
        --inference_anchor_timestep "${INFERENCE_ANCHOR_TIMESTEP}" \
        --fixed_di_values "${FIXED_DI_VALUES}"
