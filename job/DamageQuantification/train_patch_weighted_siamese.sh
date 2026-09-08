#!/bin/bash
#PBS -l select=1:ncpus=1:gpu_id=7
#PBS -l place=shared
#PBS -o log/out.out
#PBS -e log/err.out
#PBS -N patch_w_siamese

cd "${PBS_O_WORKDIR:-$HOME/intership-project}"

set +u
source ~/.bashrc
set -u
conda activate internship

module load cuda-12.4

USE_NICE=False
NICE=10
if [[ "${USE_NICE}" == "True" || "${USE_NICE}" == "true" || "${USE_NICE}" == "1" ]]; then
    NICE_CMD=(nice -n "${NICE}")
else
    NICE_CMD=()
fi

# Same protocol as train_resnet18_siamese.sh: independent shared backbone,
# random same-specimen pairs, gray 128, train+val merged, Focal-R + λ_rank=1.
LR=1e-4
EPOCH=80
BATCH_SIZE=16
NUM_WORKERS=0
IMAGE_SIZE=128
PRETRAINED=True
IMG_AUG=True
REDUCTION=16
DATA_PATH=id_list_val_R307.json
PATTERN_DIRNAME=DamagePattern_inpaint_selfcondnoDI
USE_WANDB=True
FIGURE_LAST_N_EPOCHS=1
LAMBDA_RANK=1.0
FOCAL_GAMMA=2.0
FOCAL_BETA=1.0
LOSS_SCALE=100
SEED=42
TARGET_MODE=di
INCLUDE_VAL_IN_TRAIN=False
RESULT_ROOT=./results/DamageQuantification/patch_weighted_siamese

mkdir -p log

echo "[train_patch_weighted_siamese] LR=${LR} EPOCH=${EPOCH} lambda_rank=${LAMBDA_RANK} include_val=${INCLUDE_VAL_IN_TRAIN}"

EXTRA_ARGS=()
if [[ "${PRETRAINED}" == "True" || "${PRETRAINED}" == "true" || "${PRETRAINED}" == "1" ]]; then
    EXTRA_ARGS+=(--pretrained)
fi
if [[ "${IMG_AUG}" == "True" || "${IMG_AUG}" == "true" || "${IMG_AUG}" == "1" ]]; then
    EXTRA_ARGS+=(--img_aug)
fi
if [[ "${USE_WANDB}" == "True" || "${USE_WANDB}" == "true" || "${USE_WANDB}" == "1" ]]; then
    EXTRA_ARGS+=(--use_wandb)
fi
if [[ "${INCLUDE_VAL_IN_TRAIN}" == "True" || "${INCLUDE_VAL_IN_TRAIN}" == "true" || "${INCLUDE_VAL_IN_TRAIN}" == "1" ]]; then
    EXTRA_ARGS+=(--include_val_in_train)
else
    EXTRA_ARGS+=(--no-include_val_in_train)
fi
EXTRA_ARGS+=(--no-use_mask)

"${NICE_CMD[@]}" python3 -u train_patch_weighted_siamese.py \
  --lr "${LR}" \
  --epoch "${EPOCH}" \
  --batch_size "${BATCH_SIZE}" \
  --num_workers "${NUM_WORKERS}" \
  --image_size "${IMAGE_SIZE}" \
  --reduction "${REDUCTION}" \
  --data_path "${DATA_PATH}" \
  --pattern_dirname "${PATTERN_DIRNAME}" \
  --figure_last_n_epochs "${FIGURE_LAST_N_EPOCHS}" \
  --lambda_rank "${LAMBDA_RANK}" \
  --focal_gamma "${FOCAL_GAMMA}" \
  --focal_beta "${FOCAL_BETA}" \
  --loss_scale "${LOSS_SCALE}" \
  --seed "${SEED}" \
  --target_mode "${TARGET_MODE}" \
  --result_root "${RESULT_ROOT}" \
  "${EXTRA_ARGS[@]}"
