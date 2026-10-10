#!/bin/bash
#PBS -l select=1:ncpus=2:gpu_id=3
#PBS -l place=shared
#PBS -o log/train_cfm_cascade_64to128.out
#PBS -e log/train_cfm_cascade_64to128.err
#PBS -N CFM_c64_128

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
AUG_TRANSLATE=0.0
AUG_SCALE_MIN=1.0
AUG_SCALE_MAX=1.0
USE_RANDOM_AFFINE=False
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

# AR doubles image/mask I/O per sample; keep 1–2 cores for training process.
# With ncpus=4, NUM_WORKERS=2 is the stable choice (workers=3 can starve the main process).
NUM_WORKERS=2
# Condition perturbation on physical conditioning features.
COND_PERTURB_DIST=uniform
COND_PERTURB_MIN=0.01
COND_PERTURB_MAX=0.03
# Stage-2 residual SR: 128 CFM conditioned on Stage-1 64.
# Train LR = bicubic 128→64 + blur then upsample; residual target.
# Patch kept (128/64/32). No undamaged-ref.
# Fill CASCADE_STAGE1_* after Stage-1 job finishes.
CONDITION_MODE=split
ZERO_DI_CONDITION=True
GATED_FUSION_ALPHA_DIM=scalar
USE_PHYSICS_CROSS_ATTN=False
CFM_PHYSICS_NUM_TOKENS=16
CFM_PHYSICS_TOKEN_DIM=128
USE_HF_AUX_LOSS=False
HF_AUX_LAMBDA=0.02
HF_AUX_T_MIN=0.7
HF_AUX_EDGE_WEIGHT=1.0
HF_AUX_HP_WEIGHT=0.5
HF_AUX_HP_KERNEL=3
USE_SELF_CONDITIONING="${USE_SELF_CONDITIONING:-True}"
SELF_COND_TRAIN_PROB=0.5
SELF_COND_WARMUP_START=2000
SELF_COND_WARMUP_END=4000
USE_AR_SELF_CONDITIONING=False
AR_DROP_PROB_START=0.5
AR_DROP_PROB_END=0.15
AR_DROP_PROB_END_STEP=20000
AR_NOISE_STD_MIN=0.05
AR_NOISE_STD_MAX=0.10
USE_AR_CHAIN_OT=False
CHAIN_T_START=0.5
USE_U1_EXTRA_RESBLOCK=False
USE_VSS_L3_U3=True
USE_VSS_L2=False
USE_VSS_U2=False
IMAGE_SIZE=128
BATCH_SIZE=128
CFM_MODEL_CHANNELS=32
CFM_COND_DIM=128
MAX_STEPS=15000
WARMUP_STEPS=2000
USE_PATCH_DIFFUSION=True
PATCH_SIZE_PROBS=0.5,0.3,0.2
USE_CASCADE_SR=True
CASCADE_LR_SIZE=64
CASCADE_NOISE_STD=0.05
CASCADE_BLUR_SIGMA_MIN=0.5
CASCADE_BLUR_SIGMA_MAX=1.5
CASCADE_PREDICT_RESIDUAL="${CASCADE_PREDICT_RESIDUAL:-True}"
CASCADE_STAGE1_CKPT="${CASCADE_STAGE1_CKPT:-}"
CASCADE_STAGE1_RESULT_DIR="${CASCADE_STAGE1_RESULT_DIR:-}"
if [[ "${CASCADE_STAGE1_CKPT}" == "auto" || -z "${CASCADE_STAGE1_CKPT}" ]]; then
    CASCADE_STAGE1_RESULT_DIR="$(ls -dt results/DamagePatternGeneration/CFMADM/*_cascade64 2>/dev/null | head -1 || true)"
    if [[ -n "${CASCADE_STAGE1_RESULT_DIR}" ]]; then
        CASCADE_STAGE1_CKPT="$(find "${CASCADE_STAGE1_RESULT_DIR}" -name last.ckpt -print -quit)"
    fi
fi
USE_UNDAMAGED_REF=False
USE_GAMMA_FM_LOSS="${USE_GAMMA_FM_LOSS:-True}"
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
CHECKPOINT_START_STEP=8000

EXTRACTOR_PATH="./phase1_extractor"
if [[ ! -f "${EXTRACTOR_PATH}/best.ckpt" || ! -f "${EXTRACTOR_PATH}/args.json" ]]; then
    echo "[train_ADMCFM] Missing ${EXTRACTOR_PATH}/best.ckpt or args.json"
    echo "[train_ADMCFM] Run: bash job/DamagePatternGeneration/prepare_extractor_from_hf.sh"
    exit 1
fi

if [[ -z "${CASCADE_STAGE1_CKPT}" || ! -f "${CASCADE_STAGE1_CKPT}" ]]; then
    echo "[train_ADMCFM] Stage-2 needs Stage-1 checkpoint."
    echo "[train_ADMCFM] Finish train_ADMCFM_stage1_64.sh then e.g.:"
    echo "  qsub -v CASCADE_STAGE1_CKPT=.../checkpoint_step_015000.pt,CASCADE_STAGE1_RESULT_DIR=.../run_dir job/DamagePatternGeneration/train_ADMCFM_cascade_64to128.sh"
    exit 1
fi
if [[ -z "${CASCADE_STAGE1_RESULT_DIR}" ]]; then
    CASCADE_STAGE1_RESULT_DIR="$(dirname "${CASCADE_STAGE1_CKPT}")"
fi
if [[ ! -f "${CASCADE_STAGE1_RESULT_DIR}/args.json" ]]; then
    echo "[train_ADMCFM] Stage-1 args.json missing in ${CASCADE_STAGE1_RESULT_DIR}"
    exit 1
fi

# Width-48 cannot load ch32 weights.
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

# Print allocated GPU mapping for debugging/resource tracking.
if ((${#NICE_CMD[@]})); then
    echo "[train_ADMCFM] USE_NICE=True NICE=${NICE}"
else
    echo "[train_ADMCFM] USE_NICE=False (default priority, no nice)"
fi
echo "[train_ADMCFM] CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
echo "[train_ADMCFM] CONDITION_MODE=${CONDITION_MODE} ZERO_DI_CONDITION=${ZERO_DI_CONDITION} DATA_PATH=${DATA_PATH}"
echo "[train_ADMCFM] gated_fusion_alpha_dim=${GATED_FUSION_ALPHA_DIM}"
echo "[train_ADMCFM] USE_PHYSICS_CROSS_ATTN=${USE_PHYSICS_CROSS_ATTN} tokens=${CFM_PHYSICS_NUM_TOKENS}x${CFM_PHYSICS_TOKEN_DIM}"
echo "[train_ADMCFM] USE_HF_AUX_LOSS=${USE_HF_AUX_LOSS} lambda=${HF_AUX_LAMBDA} t_min=${HF_AUX_T_MIN} edge=${HF_AUX_EDGE_WEIGHT} hp=${HF_AUX_HP_WEIGHT}"
echo "[train_ADMCFM] IMAGE_SIZE=${IMAGE_SIZE} BATCH_SIZE=${BATCH_SIZE} CFM_MODEL_CHANNELS=${CFM_MODEL_CHANNELS} CFM_COND_DIM=${CFM_COND_DIM}"
echo "[train_ADMCFM] MAX_STEPS=${MAX_STEPS} WARMUP_STEPS=${WARMUP_STEPS}"
echo "[train_ADMCFM] USE_SELF_CONDITIONING=${USE_SELF_CONDITIONING} SELF_COND_TRAIN_PROB=${SELF_COND_TRAIN_PROB}"
echo "[train_ADMCFM] self_cond_warmup=[${SELF_COND_WARMUP_START},${SELF_COND_WARMUP_END}]"
echo "[train_ADMCFM] USE_AR_SELF_CONDITIONING=${USE_AR_SELF_CONDITIONING} ar_drop=[${AR_DROP_PROB_START}->${AR_DROP_PROB_END}@${AR_DROP_PROB_END_STEP}] ar_noise=[${AR_NOISE_STD_MIN},${AR_NOISE_STD_MAX}]"
echo "[train_ADMCFM] USE_AR_CHAIN_OT=${USE_AR_CHAIN_OT} CHAIN_T_START=${CHAIN_T_START}"
echo "[train_ADMCFM] USE_U1_EXTRA_RESBLOCK=${USE_U1_EXTRA_RESBLOCK}"
echo "[train_ADMCFM] USE_VSS_L3_U3=${USE_VSS_L3_U3} USE_VSS_L2=${USE_VSS_L2} USE_VSS_U2=${USE_VSS_U2}"
echo "[train_ADMCFM] USE_CASCADE_SR=${USE_CASCADE_SR} lr=${CASCADE_LR_SIZE} noise=${CASCADE_NOISE_STD} blur=[${CASCADE_BLUR_SIGMA_MIN},${CASCADE_BLUR_SIGMA_MAX}] residual=${CASCADE_PREDICT_RESIDUAL}"
echo "[train_ADMCFM] USE_PATCH_DIFFUSION=${USE_PATCH_DIFFUSION} PATCH_SIZE_PROBS=${PATCH_SIZE_PROBS}"
echo "[train_ADMCFM] USE_UNDAMAGED_REF=${USE_UNDAMAGED_REF}"
echo "[train_ADMCFM] CASCADE_STAGE1_CKPT=${CASCADE_STAGE1_CKPT}"
echo "[train_ADMCFM] CASCADE_STAGE1_RESULT_DIR=${CASCADE_STAGE1_RESULT_DIR}"
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

# Always use wandb. Login with WANDB_API_KEY if provided,
# otherwise reuse existing credential from ~/.netrc.
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

RUN_TAG="${RUN_TAG:-}"
"${NICE_CMD[@]}" python3 -u train_mixed.py --lr 1e-4 --epoch 3000 --batch_size "${BATCH_SIZE}" --num_workers "${NUM_WORKERS}" --run_tag "${RUN_TAG}" --train_virtual_repeat 100 --max_steps "${MAX_STEPS}" --warmup_steps "${WARMUP_STEPS}" --val_check_interval 10 --architecture CFMADM --loss_type CFMADM --data_path "${DATA_PATH}" \
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
        --use_hf_aux_loss "${USE_HF_AUX_LOSS}" \
        --hf_aux_lambda "${HF_AUX_LAMBDA}" \
        --hf_aux_t_min "${HF_AUX_T_MIN}" \
        --hf_aux_edge_weight "${HF_AUX_EDGE_WEIGHT}" \
        --hf_aux_hp_weight "${HF_AUX_HP_WEIGHT}" \
        --hf_aux_hp_kernel "${HF_AUX_HP_KERNEL}" \
        --use_self_conditioning "${USE_SELF_CONDITIONING}" \
        --self_cond_train_prob "${SELF_COND_TRAIN_PROB}" \
        --self_cond_warmup_start "${SELF_COND_WARMUP_START}" \
        --self_cond_warmup_end "${SELF_COND_WARMUP_END}" \
        --use_ar_self_conditioning "${USE_AR_SELF_CONDITIONING}" \
        --use_ar_chain_ot "${USE_AR_CHAIN_OT}" \
        --chain_t_start "${CHAIN_T_START}" \
        --ar_drop_prob_start "${AR_DROP_PROB_START}" \
        --ar_drop_prob_end "${AR_DROP_PROB_END}" \
        --ar_drop_prob_end_step "${AR_DROP_PROB_END_STEP}" \
        --ar_noise_std_min "${AR_NOISE_STD_MIN}" \
        --ar_noise_std_max "${AR_NOISE_STD_MAX}" \
        --use_u1_extra_resblock "${USE_U1_EXTRA_RESBLOCK}" \
        --use_vss_l3_u3 "${USE_VSS_L3_U3}" \
        --use_vss_l2 "${USE_VSS_L2}" \
        --use_vss_u2 "${USE_VSS_U2}" \
        --use_cascade_sr "${USE_CASCADE_SR}" \
        --cascade_lr_size "${CASCADE_LR_SIZE}" \
        --cascade_noise_std "${CASCADE_NOISE_STD}" \
        --cascade_blur_sigma_min "${CASCADE_BLUR_SIGMA_MIN}" \
        --cascade_blur_sigma_max "${CASCADE_BLUR_SIGMA_MAX}" \
        --cascade_predict_residual "${CASCADE_PREDICT_RESIDUAL}" \
        --cascade_stage1_checkpoint "${CASCADE_STAGE1_CKPT}" \
        --cascade_stage1_result_dir "${CASCADE_STAGE1_RESULT_DIR}" \
        --use_undamaged_ref "${USE_UNDAMAGED_REF}" \
        --use_gamma_fm_loss "${USE_GAMMA_FM_LOSS}" \
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
        --inference_every_n_epochs 1000 --inference_num_images 4 \
        --inference_train_anchor_exp "${INFERENCE_TRAIN_ANCHOR_EXP}" \
        --inference_val_anchor_exp "${INFERENCE_VAL_ANCHOR_EXP}" \
        --inference_anchor_timestep "${INFERENCE_ANCHOR_TIMESTEP}" \
        --fixed_di_values "${FIXED_DI_VALUES}"
