#!/bin/bash
#PBS -l select=1:ncpus=1:gpu_id=7
#PBS -l place=shared
#PBS -o log/test_resnet18_siamese.out
#PBS -e log/test_resnet18_siamese.err
#PBS -N test_r18sia

cd "${PBS_O_WORKDIR:-$HOME/intership-project}"

set +u
source ~/.bashrc
set -u
conda activate internship

module load cuda-12.4
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1

# Default: latest cascade-256 dual-head run (same RESULT_ROOT as train_resnet18_siamese.sh).
# Override: qsub -v CKPT_PATH=...,RESULT_DIR=...,SPLIT=test ...
CKPT_PATH="${CKPT_PATH:-./results/DamageQuantification/resnet18_siamese_rgb_orig/2026-08-24_21-51-23/checkpoints/last.ckpt}"
RESULT_DIR="${RESULT_DIR:-$(cd "$(dirname "${CKPT_PATH}")/.." && pwd)}"
SPLIT="${SPLIT:-all}"
BATCH_SIZE="${BATCH_SIZE:-32}"

mkdir -p log

if [[ ! -f "${CKPT_PATH}" ]]; then
    echo "[test_resnet18_siamese] Checkpoint not found: ${CKPT_PATH}"
    exit 1
fi
if [[ ! -f "${RESULT_DIR}/args.json" ]]; then
    echo "[test_resnet18_siamese] args.json not found in: ${RESULT_DIR}"
    exit 1
fi

echo "[test_resnet18_siamese] CKPT_PATH=${CKPT_PATH}"
echo "[test_resnet18_siamese] RESULT_DIR=${RESULT_DIR}"
echo "[test_resnet18_siamese] SPLIT=${SPLIT} BATCH_SIZE=${BATCH_SIZE}"
echo "[test_resnet18_siamese] CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"

python3 -u test_resnet18_siamese.py \
  --ckpt "${CKPT_PATH}" \
  --result_dir "${RESULT_DIR}" \
  --split "${SPLIT}" \
  --batch_size "${BATCH_SIZE}" \
  --gpu 0

echo "[test_resnet18_siamese] Outputs under ${RESULT_DIR}/prediction/"
ls -la "${RESULT_DIR}/prediction/" 2>/dev/null || true
