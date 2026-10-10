#!/bin/bash
# Independent Laplacian-band CFM. Submit three jobs (d1 / d2 / d3); they do not share weights.
# Set GPU yourself (cluster is often full):  qsub -v BAND=d1 -l select=1:ncpus=2:gpu_id=N ...
#
#PBS -l select=1:ncpus=2:gpu_id=7
#PBS -l place=shared
#PBS -o log/train_laplacian_cfm.out
#PBS -e log/train_laplacian_cfm.err
#PBS -N CFM_lap

cd "${PBS_O_WORKDIR:-$HOME/intership-project}"
source job/DamagePatternGeneration/common.sh
adm_env

BAND="${BAND:-d1}"

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
NUM_WORKERS=2
CONDITION_MODE=split
ZERO_DI_CONDITION=True
IMAGE_SIZE=128
BATCH_SIZE=128
MAX_STEPS=15000
WARMUP_STEPS=2000
USE_LAPLACIAN_PATCH=False
USE_SELF_CONDITIONING=False
DATA_PATH=id_list_val_R307.json
CHECKPOINT_EVERY_N_STEPS=2000
CHECKPOINT_START_STEP=10000

EXTRACTOR_PATH="./phase1_extractor"
if [[ ! -f "${EXTRACTOR_PATH}/best.ckpt" || ! -f "${EXTRACTOR_PATH}/args.json" ]]; then
    echo "[train_laplacian_cfm] Missing ${EXTRACTOR_PATH}/best.ckpt or args.json"
    exit 1
fi

echo "[train_laplacian_cfm] BAND=${BAND} CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
echo "[train_laplacian_cfm] BATCH_SIZE=${BATCH_SIZE} MAX_STEPS=${MAX_STEPS} patch=${USE_LAPLACIAN_PATCH} self_cond=${USE_SELF_CONDITIONING}"

if [[ -n "${WANDB_API_KEY:-}" ]]; then
    echo "[train_laplacian_cfm] wandb login via WANDB_API_KEY"
    wandb login "${WANDB_API_KEY}" --relogin >/dev/null 2>&1
elif [[ -f "${HOME}/.netrc" ]] && grep -q "api\.wandb\.ai" "${HOME}/.netrc"; then
    echo "[train_laplacian_cfm] Using existing wandb login from ~/.netrc"
else
    echo "[train_laplacian_cfm] No wandb credential. Run: wandb login"
    echo "[train_laplacian_cfm] or qsub -v WANDB_API_KEY=<key>,BAND=${BAND} ..."
    exit 1
fi

mkdir -p log

"${NICE_CMD[@]}" python3 -u train_laplacian_cfm.py \
        --lr 1e-4 --epoch 3000 --batch_size "${BATCH_SIZE}" --num_workers "${NUM_WORKERS}" \
        --train_virtual_repeat 100 --max_steps "${MAX_STEPS}" --warmup_steps "${WARMUP_STEPS}" \
        --val_check_interval 10 \
        --laplacian_band "${BAND}" --laplacian_dummy False \
        --use_laplacian_patch "${USE_LAPLACIAN_PATCH}" \
        --use_self_conditioning "${USE_SELF_CONDITIONING}" \
        --use_ar_self_conditioning False --use_ar_chain_ot False \
        --use_patch_diffusion False \
        --architecture LaplacianBandCFM --loss_type "LaplacianCFM_${BAND}" \
        --data_path "${DATA_PATH}" --extractor_path "${EXTRACTOR_PATH}" --use_wandb \
        --img_aug \
        --use_grayscale_input "${USE_GRAYSCALE_INPUT}" \
        --use_clahe "${USE_CLAHE}" --clahe_clip_limit "${CLAHE_CLIP_LIMIT}" \
        --clahe_tile_grid_size "${CLAHE_TILE_GRID_SIZE}" \
        --aug_prob "${AUG_PROB}" --aug_blur_prob "${AUG_BLUR_PROB}" \
        --aug_brightness "${AUG_BRIGHTNESS}" --aug_contrast "${AUG_CONTRAST}" \
        --aug_saturation "${AUG_SATURATION}" --aug_hue "${AUG_HUE}" \
        --aug_hflip_prob "${AUG_HFLIP_PROB}" \
        --aug_blur_kernel "${AUG_BLUR_KERNEL}" \
        --aug_blur_sigma_min "${AUG_BLUR_SIGMA_MIN}" --aug_blur_sigma_max "${AUG_BLUR_SIGMA_MAX}" \
        --aug_erase_prob "${AUG_ERASE_PROB}" \
        --condition_mode "${CONDITION_MODE}" \
        --zero_di_condition "${ZERO_DI_CONDITION}" \
        --latent_conditioning True --direction_conditioning True --include_di_conditioning True \
        --add_DI_loss False \
        --use_damage_mask_condition False --use_object_loss_mask False \
        --image_size "${IMAGE_SIZE}" --cfm_cond_dim 128 --cfm_dropout 0.1 \
        --cfm_solver heun --cfm_ode_steps 30 --lr_schedule warmup_cosine --min_lr_ratio 0.1 \
        --grad_clip_max_norm "${GRAD_CLIP_MAX_NORM}" \
        --use_velocity_min_snr_weighting "${USE_VELOCITY_MIN_SNR_WEIGHTING}" \
        --min_snr_gamma "${MIN_SNR_GAMMA}" \
        --use_early_stopping False \
        --checkpoint_every_n_steps "${CHECKPOINT_EVERY_N_STEPS}" \
        --checkpoint_start_step "${CHECKPOINT_START_STEP}" \
        --inference_every_n_epochs 1000 --inference_num_images 4
