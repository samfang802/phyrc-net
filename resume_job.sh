#!/bin/bash
#PBS -l select=1:ncpus=1:gpu_id=2
#PBS -l place=shared
#PBS -o log/out.out
#PBS -e log/err.out
#PBS -N ADMCFM_32_resume

cd "${PBS_O_WORKDIR:-$HOME/intership-project}"

set +u
source ~/.bashrc
set -u
conda activate internship

module load cuda-12.4
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1

# Centralized augmentation config (single source in this script).
AUG_PROB=0.5
AUG_BLUR_PROB=0.15
AUG_BRIGHTNESS=0.12
AUG_CONTRAST=0.12
AUG_SATURATION=0.08
AUG_HUE=0.02
AUG_TRANSLATE=0.02
AUG_SCALE_MIN=0.98
AUG_SCALE_MAX=1.02
AUG_BLUR_KERNEL=3
AUG_BLUR_SIGMA_MIN=0.1
AUG_BLUR_SIGMA_MAX=0.5
AUG_ERASE_PROB=0.0
AUG_ERASE_SCALE_MIN=0.01
AUG_ERASE_SCALE_MAX=0.08
AUG_ERASE_RATIO_MIN=0.3
AUG_ERASE_RATIO_MAX=3.3
EARLY_STOP_PATIENCE=50
EARLY_STOP_MIN_DELTA=0.0
USE_EARLY_STOPPING=True
RESET_EARLY_STOP_PATIENCE=True
RESET_EARLY_STOP_BEST_SCORE=False

EXTRACTOR_PATH="./phase1_extractor"
CKPT_PATH="./results/DamagePatternGeneration/CFMADM/2026-07-17_23-01-43/Opensees Based damage Patterns Generation/ry2oq2xs/checkpoints/best_val_loss_val/loss=0.0227.ckpt"

if [[ ! -f "${EXTRACTOR_PATH}/best.ckpt" || ! -f "${EXTRACTOR_PATH}/args.json" ]]; then
    echo "[resume_job] Missing ${EXTRACTOR_PATH}/best.ckpt or args.json"
    echo "[resume_job] Run: bash job/DamagePatternGeneration/prepare_extractor_from_hf.sh"
    exit 1
fi

if [[ ! -f "${CKPT_PATH}" ]]; then
    echo "[resume_job] CKPT_PATH not found: ${CKPT_PATH}"
    echo "[resume_job] Please set CKPT_PATH to your loss=0.0227.ckpt path."
    exit 1
fi

echo "[resume_job] CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
python3 - <<'PY'
import os
import subprocess

visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
print(f"[resume_job] Visible GPU list (physical IDs): {visible if visible else 'all'}")
try:
    out = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=index,name,uuid", "--format=csv,noheader"],
        text=True,
    ).strip()
    print("[resume_job] Host GPU inventory:")
    for line in out.splitlines():
        print(f"[resume_job]   {line}")
except Exception as exc:
    print(f"[resume_job] Failed to query nvidia-smi: {exc}")
PY

if [[ -n "${WANDB_API_KEY:-}" ]]; then
    echo "[resume_job] Logging in to Weights & Biases with WANDB_API_KEY"
    wandb login "${WANDB_API_KEY}" --relogin >/dev/null 2>&1
elif [[ -f "${HOME}/.netrc" ]] && grep -q "api\.wandb\.ai" "${HOME}/.netrc"; then
    echo "[resume_job] Using existing wandb login from ~/.netrc"
else
    echo "[resume_job] No wandb credential found."
    echo "[resume_job] Please run one-time login: wandb login"
    echo "[resume_job] or submit with: qsub -v WANDB_API_KEY=<your_key> ..."
    exit 1
fi

python3 -u train_mixed.py --lr 4e-4 --epoch 1800 --batch_size 32 --num_workers 1 --architecture CFMADM --loss_type CFMADM --optimizer adamw --data_path id_list_no_leak.json \
        --extractor_path "${EXTRACTOR_PATH}" --use_wandb --resume "${CKPT_PATH}" \
        --img_aug \
        --aug_prob "${AUG_PROB}" --aug_blur_prob "${AUG_BLUR_PROB}" \
        --aug_brightness "${AUG_BRIGHTNESS}" --aug_contrast "${AUG_CONTRAST}" --aug_saturation "${AUG_SATURATION}" --aug_hue "${AUG_HUE}" \
        --aug_translate "${AUG_TRANSLATE}" --aug_scale_min "${AUG_SCALE_MIN}" --aug_scale_max "${AUG_SCALE_MAX}" \
        --aug_blur_kernel "${AUG_BLUR_KERNEL}" --aug_blur_sigma_min "${AUG_BLUR_SIGMA_MIN}" --aug_blur_sigma_max "${AUG_BLUR_SIGMA_MAX}" \
        --aug_erase_prob "${AUG_ERASE_PROB}" --aug_erase_scale_min "${AUG_ERASE_SCALE_MIN}" --aug_erase_scale_max "${AUG_ERASE_SCALE_MAX}" \
        --aug_erase_ratio_min "${AUG_ERASE_RATIO_MIN}" --aug_erase_ratio_max "${AUG_ERASE_RATIO_MAX}" \
        --use_early_stopping "${USE_EARLY_STOPPING}" \
        --early_stop_patience "${EARLY_STOP_PATIENCE}" --early_stop_min_delta "${EARLY_STOP_MIN_DELTA}" \
        --reset_early_stop_patience "${RESET_EARLY_STOP_PATIENCE}" \
        --reset_early_stop_best_score "${RESET_EARLY_STOP_BEST_SCORE}" \
        --latent_conditioning True --direction_conditioning True --include_di_conditioning False --add_DI_loss False \
        --image_size 128 --cfm_model_channels 32 --cfm_num_res_blocks 1 --cfm_cond_dim 128 --cfm_dropout 0.1 \
        --cfm_mid_attn_heads 4 \
        --use_patch_diffusion True --patch_size_probs 0.5,0.3,0.2 \
        --cfm_solver heun --cfm_ode_steps 30 --lr_schedule warmup_cosine --warmup_epochs 180 --min_lr_ratio 0.1 \
        --val_every_n_epoch 10 --inference_every_n_epochs 50 --inference_num_images 4 --metrics_every_n_epochs 100 \
        --fixed_di_values 0.0,0.4,0.75,0.95
