#!/bin/bash
#PBS -l select=1:ncpus=1:gpu_id=7
#PBS -l place=shared
#PBS -o log/train_resnet18_siamese_rgb.out
#PBS -e log/train_resnet18_siamese_rgb.err
#PBS -N r18sia_rgb

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

# DIresnet18 recipe: ImageNet pretrained, 128, DI-only. Data = original RGB photos, no CLAHE.
LR=1e-4
EPOCH=50
BATCH_SIZE=16
NUM_WORKERS=1
IMAGE_SIZE=128
PRETRAINED=True
IMG_AUG=True
REDUCTION=16
DATA_PATH=id_list_val_R307.json
PATTERN_DIRNAME=DamagePattern
USE_MASK=False
USE_RGB=False
USE_CLAHE=True
USE_WANDB=True
FIGURE_LAST_N_EPOCHS=1
DROPBLOCK_PROB=0.0
DROPBLOCK_SIZE_L3=3
DROPBLOCK_SIZE_L4=2
LAMBDA_RANK=1.0
FOCAL_GAMMA=2.0
FOCAL_BETA=1.0
LOSS_SCALE=100
SEED=42

TARGET_MODE=di
MULTI_TASK=False
INCLUDE_VAL_IN_TRAIN=False
DI_LOW_THRESH=0.3
DI_LOW_WEIGHT=1.0
RS0_WEIGHT=1.0

RS_LOSS_WEIGHT=1.0
RS_LOSS_SCALE=100
RS_POS_WEIGHT=1.0
UNCERTAINTY_WEIGHTING=False
RS_WARMUP_EPOCHS=0

BETA=0.3
R_BAR_METHOD=poly
POLY_DEGREE=2
MA_WINDOW=51
R_SPALL_CSV=./Data/DamagePattern/anylabel_r_spall.csv
RESULT_ROOT=./results/DamageQuantification/resnet18_siamese_rgb_orig

mkdir -p log

echo "[train_resnet18_siamese] PATTERN_DIRNAME=${PATTERN_DIRNAME} IMAGE_SIZE=${IMAGE_SIZE}"
echo "[train_resnet18_siamese] RGB=${USE_RGB} CLAHE=${USE_CLAHE} pretrained=${PRETRAINED}"
echo "[train_resnet18_siamese] LR=${LR} EPOCH=${EPOCH} include_val=${INCLUDE_VAL_IN_TRAIN}"
echo "[train_resnet18_siamese] target=${TARGET_MODE} multi_task=${MULTI_TASK}"

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
if [[ "${USE_MASK}" == "True" || "${USE_MASK}" == "true" || "${USE_MASK}" == "1" ]]; then
    EXTRA_ARGS+=(--use_mask)
else
    EXTRA_ARGS+=(--no-use_mask)
fi
if [[ "${USE_RGB}" == "True" || "${USE_RGB}" == "true" || "${USE_RGB}" == "1" ]]; then
    EXTRA_ARGS+=(--use_rgb)
else
    EXTRA_ARGS+=(--no-use_rgb)
fi
if [[ "${USE_CLAHE}" == "True" || "${USE_CLAHE}" == "true" || "${USE_CLAHE}" == "1" ]]; then
    EXTRA_ARGS+=(--use_clahe)
else
    EXTRA_ARGS+=(--no-use_clahe)
fi
if [[ "${MULTI_TASK}" == "True" || "${MULTI_TASK}" == "true" || "${MULTI_TASK}" == "1" ]]; then
    EXTRA_ARGS+=(--multi_task)
else
    EXTRA_ARGS+=(--no-multi_task)
fi
if [[ "${INCLUDE_VAL_IN_TRAIN}" == "True" || "${INCLUDE_VAL_IN_TRAIN}" == "true" || "${INCLUDE_VAL_IN_TRAIN}" == "1" ]]; then
    EXTRA_ARGS+=(--include_val_in_train)
else
    EXTRA_ARGS+=(--no-include_val_in_train)
fi
if [[ "${UNCERTAINTY_WEIGHTING}" == "True" || "${UNCERTAINTY_WEIGHTING}" == "true" || "${UNCERTAINTY_WEIGHTING}" == "1" ]]; then
    EXTRA_ARGS+=(--uncertainty_weighting)
else
    EXTRA_ARGS+=(--no-uncertainty_weighting)
fi
if [[ -n "${R_SPALL_CSV}" ]]; then
    EXTRA_ARGS+=(--r_spall_csv "${R_SPALL_CSV}")
fi
if [[ -n "${RS_LOSS_SCALE}" ]]; then
    EXTRA_ARGS+=(--rs_loss_scale "${RS_LOSS_SCALE}")
fi

"${NICE_CMD[@]}" python3 -u train_resnet18_siamese.py \
  --lr "${LR}" \
  --epoch "${EPOCH}" \
  --batch_size "${BATCH_SIZE}" \
  --num_workers "${NUM_WORKERS}" \
  --image_size "${IMAGE_SIZE}" \
  --reduction "${REDUCTION}" \
  --data_path "${DATA_PATH}" \
  --pattern_dirname "${PATTERN_DIRNAME}" \
  --figure_last_n_epochs "${FIGURE_LAST_N_EPOCHS}" \
  --dropblock_prob "${DROPBLOCK_PROB}" \
  --dropblock_size_l3 "${DROPBLOCK_SIZE_L3}" \
  --dropblock_size_l4 "${DROPBLOCK_SIZE_L4}" \
  --lambda_rank "${LAMBDA_RANK}" \
  --focal_gamma "${FOCAL_GAMMA}" \
  --focal_beta "${FOCAL_BETA}" \
  --loss_scale "${LOSS_SCALE}" \
  --seed "${SEED}" \
  --target_mode "${TARGET_MODE}" \
  --di_low_thresh "${DI_LOW_THRESH}" \
  --di_low_weight "${DI_LOW_WEIGHT}" \
  --rs0_weight "${RS0_WEIGHT}" \
  --rs_loss_weight "${RS_LOSS_WEIGHT}" \
  --rs_pos_weight "${RS_POS_WEIGHT}" \
  --rs_warmup_epochs "${RS_WARMUP_EPOCHS}" \
  --beta "${BETA}" \
  --r_bar_method "${R_BAR_METHOD}" \
  --poly_degree "${POLY_DEGREE}" \
  --ma_window "${MA_WINDOW}" \
  --result_root "${RESULT_ROOT}" \
  "${EXTRA_ARGS[@]}"
