#!/bin/bash
#PBS -l select=1:ncpus=1:gpu_id=3
#PBS -l place=shared
#PBS -o log/out.out
#PBS -e log/err.out
#PBS -N ADMCFM_32_latent

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
USE_NICE=True
NICE=10
if [[ "${USE_NICE}" == "True" || "${USE_NICE}" == "true" || "${USE_NICE}" == "1" ]]; then
    NICE_CMD=(nice -n "${NICE}")
else
    NICE_CMD=()
fi

# Centralized augmentation config (single source in this script).
AUG_PROB=0.5
AUG_BLUR_PROB=0.15
AUG_BRIGHTNESS=0.12
AUG_CONTRAST=0.12
AUG_SATURATION=0.0
AUG_HUE=0.0
USE_GRAYSCALE_INPUT=True
USE_CLAHE=True
CLAHE_CLIP_LIMIT=2.0
CLAHE_TILE_GRID_SIZE=8,8
AUG_TRANSLATE=0.0
AUG_SCALE_MIN=1.0
AUG_SCALE_MAX=1.0
USE_RANDOM_AFFINE=True
AUG_HFLIP_PROB=0.5
GRAD_CLIP_MAX_NORM=2.0
CFG_SCALE=1.0
USE_VELOCITY_MIN_SNR_WEIGHTING=True
MIN_SNR_GAMMA=5.0
AUG_BLUR_KERNEL=3
AUG_BLUR_SIGMA_MIN=0.1
AUG_BLUR_SIGMA_MAX=0.5
AUG_ERASE_PROB=0.5
AUG_ERASE_SCALE_MIN=0.05
AUG_ERASE_SCALE_MAX=0.15
AUG_ERASE_RATIO_MIN=0.5
AUG_ERASE_RATIO_MAX=2.5
# Condition perturbation on physical conditioning features.
COND_PERTURB_DIST=uniform
COND_PERTURB_MIN=0.01
COND_PERTURB_MAX=0.03
CONDITION_MODE=di_only
INFERENCE_ANCHOR_EXP=R615
INFERENCE_ANCHOR_TIMESTEP=-1
EARLY_STOP_PATIENCE=300
EARLY_STOP_MIN_DELTA=0.0

EXTRACTOR_PATH="./phase1_extractor"
if [[ ! -f "${EXTRACTOR_PATH}/best.ckpt" || ! -f "${EXTRACTOR_PATH}/args.json" ]]; then
    echo "[train_ADMCFM] Missing ${EXTRACTOR_PATH}/best.ckpt or args.json"
    echo "[train_ADMCFM] Run: bash job/DamagePatternGeneration/prepare_extractor_from_hf.sh"
    exit 1
fi

# Leave empty to train from scratch. Set a .ckpt path to resume.
RESUME_CKPT=""
RESUME_ARGS=()
if [[ -n "${RESUME_CKPT}" ]]; then
    if [[ ! -f "${RESUME_CKPT}" ]]; then
        echo "[train_ADMCFM] Resume checkpoint not found: ${RESUME_CKPT}"
        exit 1
    fi
    RESUME_ARGS=(--resume "${RESUME_CKPT}")
    echo "[train_ADMCFM] Resuming from: ${RESUME_CKPT}"
else
    echo "[train_ADMCFM] Resume disabled; training from scratch."
fi

# Print allocated GPU mapping for debugging/resource tracking.
if ((${#NICE_CMD[@]})); then
    echo "[train_ADMCFM] USE_NICE=True NICE=${NICE}"
else
    echo "[train_ADMCFM] USE_NICE=False (default priority, no nice)"
fi
echo "[train_ADMCFM] CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
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

"${NICE_CMD[@]}" python3 -u train_mixed.py --lr 4e-4 --epoch 3000 --batch_size 128 --num_workers 1 --train_virtual_repeat 100 --max_steps 8000 --warmup_steps 500 --val_check_interval 5 --architecture CFMADM --loss_type CFMADM --data_path id_list_no_leak.json \
        --extractor_path "${EXTRACTOR_PATH}" --use_wandb \
        "${RESUME_ARGS[@]}" \
        --img_aug \
        --use_grayscale_input "${USE_GRAYSCALE_INPUT}" \
        --use_clahe "${USE_CLAHE}" --clahe_clip_limit "${CLAHE_CLIP_LIMIT}" \
        --clahe_tile_grid_size "${CLAHE_TILE_GRID_SIZE}" \
        --aug_prob "${AUG_PROB}" --aug_blur_prob "${AUG_BLUR_PROB}" \
        --aug_brightness "${AUG_BRIGHTNESS}" --aug_contrast "${AUG_CONTRAST}" --aug_saturation "${AUG_SATURATION}" --aug_hue "${AUG_HUE}" \
        --aug_translate "${AUG_TRANSLATE}" --aug_scale_min "${AUG_SCALE_MIN}" --aug_scale_max "${AUG_SCALE_MAX}" \
        --use_random_affine "${USE_RANDOM_AFFINE}" --aug_hflip_prob "${AUG_HFLIP_PROB}" \
        --aug_blur_kernel "${AUG_BLUR_KERNEL}" --aug_blur_sigma_min "${AUG_BLUR_SIGMA_MIN}" --aug_blur_sigma_max "${AUG_BLUR_SIGMA_MAX}" \
        --aug_erase_prob "${AUG_ERASE_PROB}" --aug_erase_scale_min "${AUG_ERASE_SCALE_MIN}" --aug_erase_scale_max "${AUG_ERASE_SCALE_MAX}" \
        --aug_erase_ratio_min "${AUG_ERASE_RATIO_MIN}" --aug_erase_ratio_max "${AUG_ERASE_RATIO_MAX}" \
        --condition_perturb --condition_perturb_dist "${COND_PERTURB_DIST}" \
        --condition_perturb_min "${COND_PERTURB_MIN}" --condition_perturb_max "${COND_PERTURB_MAX}" \
        --condition_mode "${CONDITION_MODE}" --use_damage_mask_condition False \
        --early_stop_patience "${EARLY_STOP_PATIENCE}" --early_stop_min_delta "${EARLY_STOP_MIN_DELTA}" \
        --latent_conditioning True --direction_conditioning True --include_di_conditioning True --add_DI_loss False \
        --image_size 128 --cfm_model_channels 32 --cfm_num_res_blocks 1 --cfm_cond_dim 128 --cfm_dropout 0.1 \
        --cfm_solver heun --cfm_ode_steps 30 --lr_schedule warmup_cosine --min_lr_ratio 0.1 \
        --grad_clip_max_norm "${GRAD_CLIP_MAX_NORM}" --cfg_scale "${CFG_SCALE}" \
        --use_velocity_min_snr_weighting "${USE_VELOCITY_MIN_SNR_WEIGHTING}" --min_snr_gamma "${MIN_SNR_GAMMA}" \
        --inference_every_n_epochs 500 --inference_num_images 4 --metrics_every_n_epochs 500 \
        --inference_anchor_exp "${INFERENCE_ANCHOR_EXP}" --inference_anchor_timestep "${INFERENCE_ANCHOR_TIMESTEP}" \
        --fixed_di_values 0.0,0.4,0.75,0.95
