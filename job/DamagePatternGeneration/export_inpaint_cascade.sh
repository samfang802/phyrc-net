#!/bin/bash
#PBS -l select=1:ncpus=2:gpu_id=5
#PBS -l place=shared
#PBS -o log/export_inpaint_cascade.out
#PBS -e log/export_inpaint_cascade.err
#PBS -N inpaintC256

cd "${PBS_O_WORKDIR:-$HOME/intership-project}"

set +u
source ~/.bashrc
set -u
conda activate internship

module load cuda-12.4
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1

mkdir -p log

STAGE1_CKPT="results/DamagePatternGeneration/CFMADM/γ-FM/Opensees Based damage Patterns Generation/69dvfbw9/checkpoints/last.ckpt"
STAGE1_DIR="results/DamagePatternGeneration/CFMADM/γ-FM"
CASCADE_CKPT="results/DamagePatternGeneration/CFMADM/Cascaded 256/Opensees Based damage Patterns Generation/5waniet2/checkpoints/last.ckpt"
CASCADE_DIR="results/DamagePatternGeneration/CFMADM/Cascaded 256"
OUT_DIR="Data/DamagePattern_inpaint_cascade256"
BATCH_SIZE="${BATCH_SIZE:-4}"
NUM_WORKERS="${NUM_WORKERS:-1}"
ODE_STEPS="${ODE_STEPS:-30}"

if [[ ! -f "${STAGE1_CKPT}" ]]; then
    echo "[export_inpaint] missing Stage-1: ${STAGE1_CKPT}"
    exit 1
fi
if [[ ! -f "${CASCADE_CKPT}" ]]; then
    echo "[export_inpaint] missing Stage-2: ${CASCADE_CKPT}"
    exit 1
fi
if [[ ! -f "${STAGE1_DIR}/args.json" || ! -f "${CASCADE_DIR}/args.json" ]]; then
    echo "[export_inpaint] missing args.json"
    exit 1
fi

echo "[export_inpaint] STAGE1=${STAGE1_CKPT}"
echo "[export_inpaint] CASCADE=${CASCADE_CKPT}"
echo "[export_inpaint] OUT=${OUT_DIR}"
echo "[export_inpaint] CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"

python3 -u scripts/export_inpaint_dataset.py \
    --stage1_checkpoint "${STAGE1_CKPT}" \
    --stage1_result_dir "${STAGE1_DIR}" \
    --cascade_checkpoint "${CASCADE_CKPT}" \
    --cascade_result_dir "${CASCADE_DIR}" \
    --out_dir "${OUT_DIR}" \
    --splits train,val,test \
    --ode_steps "${ODE_STEPS}" \
    --batch_size "${BATCH_SIZE}" \
    --num_workers "${NUM_WORKERS}" \
    --skip_existing

echo "[export_inpaint] done. ${OUT_DIR}/images $(ls "${OUT_DIR}/images" 2>/dev/null | wc -l) files"
