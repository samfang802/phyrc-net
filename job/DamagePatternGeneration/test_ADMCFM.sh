#!/bin/bash
#PBS -l select=1:ncpus=1:gpu_id=5
#PBS -l place=shared
#PBS -o log/test_admcfm.out
#PBS -e log/test_admcfm.err
#PBS -N test_ADMCFM_s12000

cd "${PBS_O_WORKDIR:-$HOME/intership-project}"
source job/DamagePatternGeneration/common.sh
adm_env
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1

CKPT_PATH="${CKPT_PATH:-./results/DamagePatternGeneration/CFMADM/2026-09-18_02-15-49/checkpoint_step_014000.pt}"
RESULT_DIR="${RESULT_DIR:-$(dirname "${CKPT_PATH}")}"
EXTRACTOR_PATH="./phase1_extractor"
BATCH_SIZE="${BATCH_SIZE:-32}"
NUM_WORKERS="${NUM_WORKERS:-1}"
FIXED_DI_VALUES="${FIXED_DI_VALUES:-0.0,0.4,0.75,1.0}"
SPLIT="${SPLIT:-test}"
ANCHOR_EXP="${ANCHOR_EXP:-CTR1}"

mkdir -p log

if [[ ! -f "${CKPT_PATH}" ]]; then
    echo "[test_ADMCFM] Checkpoint not found: ${CKPT_PATH}"
    exit 1
fi
if [[ ! -f "${RESULT_DIR}/args.json" ]]; then
    echo "[test_ADMCFM] args.json not found in: ${RESULT_DIR}"
    exit 1
fi
if [[ ! -f "${EXTRACTOR_PATH}/best.ckpt" || ! -f "${EXTRACTOR_PATH}/args.json" ]]; then
    echo "[test_ADMCFM] Missing ${EXTRACTOR_PATH}/best.ckpt or args.json"
    echo "[test_ADMCFM] Run: bash job/DamagePatternGeneration/prepare_extractor_from_hf.sh"
    exit 1
fi

echo "[test_ADMCFM] CKPT_PATH=${CKPT_PATH}"
echo "[test_ADMCFM] RESULT_DIR=${RESULT_DIR}"
echo "[test_ADMCFM] FIXED_DI_VALUES=${FIXED_DI_VALUES}"
echo "[test_ADMCFM] SPLIT=${SPLIT} (test=held-out test set, val=validation set)"
METRICS_SPLIT="${METRICS_SPLIT:-test}"

echo "[test_ADMCFM] METRICS_SPLIT=${METRICS_SPLIT} (FID/PSNR/SSIM/loss)"
echo "[test_ADMCFM] CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"

python3 -u test_cfm_checkpoint.py \
    --checkpoint "${CKPT_PATH}" \
    --result_dir "${RESULT_DIR}" \
    --batch_size "${BATCH_SIZE}" \
    --num_workers "${NUM_WORKERS}" \
    --fixed_di_values "${FIXED_DI_VALUES}" \
    --split "${SPLIT}" \
    --anchor_exp "${ANCHOR_EXP}" \
    --metrics_split "${METRICS_SPLIT}"

echo "[test_ADMCFM] Outputs:"
echo "  - GT top / inference bottom panel: ${RESULT_DIR}/inference_samples/fixed_di_${SPLIT}_step_XXXXX.png"
echo "  - ${METRICS_SPLIT} metrics json: ${RESULT_DIR}/test_metrics/metrics_${METRICS_SPLIT}_step_XXXXX.json"
ls -la "${RESULT_DIR}/inference_samples/" 2>/dev/null || true
ls -la "${RESULT_DIR}/test_metrics/" 2>/dev/null || true
