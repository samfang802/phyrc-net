#!/bin/bash
#PBS -l select=1:ncpus=1:gpu_id=4
#PBS -l place=shared
#PBS -o log/out.out
#PBS -e log/err.out
#PBS -N resnet18_cbam_di

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

# -------- ResNet18 + CBAM DI regressor config --------
LR=1e-4
EPOCH=100
BATCH_SIZE=16
NUM_WORKERS=1
IMAGE_SIZE=128
PRETRAINED=True
IMG_AUG=True
REDUCTION=16
# Same split as CFM (val=R307, test=CTR1). Images = CFM-inpainted.
DATA_PATH=id_list_val_R307.json
PATTERN_DIRNAME=DamagePattern_inpaint_selfcondnoDI
# Gray-only ablations: set USE_MASK=False (→ --no-use_mask).
USE_MASK=False
USE_WANDB=True
# Only draw scatter figures in the last N epochs; earlier epochs → numbers only.
FIGURE_LAST_N_EPOCHS=100
# DropBlock / Manifold Mixup / Focal-R
DROPBLOCK_PROB=0.0
DROPBLOCK_SIZE_L3=3
DROPBLOCK_SIZE_L4=2
MIXUP_ALPHA=0.2
MIXUP_PROB=1.0
FOCAL_GAMMA=2.0
FOCAL_BETA=1.0
RESULT_ROOT=./results/DamageQuantification/resnet18_cbam_inpaint_nomask

mkdir -p log

if ((${#NICE_CMD[@]})); then
    echo "[train_resnet18_cbam] USE_NICE=True NICE=${NICE}"
else
    echo "[train_resnet18_cbam] USE_NICE=False (default priority, no nice)"
fi
echo "[train_resnet18_cbam] CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
echo "[train_resnet18_cbam] LR=${LR} EPOCH=${EPOCH} BATCH_SIZE=${BATCH_SIZE} IMAGE_SIZE=${IMAGE_SIZE}"
echo "[train_resnet18_cbam] PRETRAINED=${PRETRAINED} IMG_AUG=${IMG_AUG} DATA_PATH=${DATA_PATH} PATTERN=${PATTERN_DIRNAME} USE_MASK=${USE_MASK}"
echo "[train_resnet18_cbam] USE_WANDB=${USE_WANDB} FIGURE_LAST_N_EPOCHS=${FIGURE_LAST_N_EPOCHS}"
echo "[train_resnet18_cbam] DropBlock p=${DROPBLOCK_PROB} bs_l3=${DROPBLOCK_SIZE_L3} bs_l4=${DROPBLOCK_SIZE_L4} Mixup α=${MIXUP_ALPHA} p=${MIXUP_PROB}"
echo "[train_resnet18_cbam] Focal-R γ=${FOCAL_GAMMA} β=${FOCAL_BETA}"
echo "[train_resnet18_cbam] RESULT_ROOT=${RESULT_ROOT}"

python3 - <<'PY'
import os
import subprocess

visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
print(f"[train_resnet18_cbam] Visible GPU list (physical IDs): {visible if visible else 'all'}")
try:
    out = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=index,name,uuid", "--format=csv,noheader"],
        text=True,
    ).strip()
    print("[train_resnet18_cbam] nvidia-smi GPUs:")
    print(out)
except Exception as exc:
    print(f"[train_resnet18_cbam] nvidia-smi failed: {exc}")
PY

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

"${NICE_CMD[@]}" python3 -u train_resnet18_cbam.py \
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
  --mixup_alpha "${MIXUP_ALPHA}" \
  --mixup_prob "${MIXUP_PROB}" \
  --focal_gamma "${FOCAL_GAMMA}" \
  --focal_beta "${FOCAL_BETA}" \
  --result_root "${RESULT_ROOT}" \
  "${EXTRA_ARGS[@]}"
