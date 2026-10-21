#!/bin/bash
#PBS -l select=1:ncpus=1:mem=8gb:gpu_id=6
#PBS -l place=shared
#PBS -o log/eval_diagnostics.out
#PBS -e log/eval_diagnostics.err
#PBS -N diagnostics

# Single-frame model diagnostics (not part of the evaluation suite). Replaces eval_base_vs_gt_shift.sh
# and eval_state_spectrum.sh.
#   WHAT=shift     samples vs real frames: translation and residual energy (measure_base_vs_gt_shift.py)
#   WHAT=spectrum  radial power spectrum at 1/10/30/60 NFE and the HF share of r = x_t - S (measure_state_spectrum.py)
#   WHAT=all       both
#   qsub -v WHAT=spectrum,CKPT=... job/DamagePatternGeneration/eval_diagnostics.sh
cd "${PBS_O_WORKDIR:-$HOME/intership-project}"
source job/DamagePatternGeneration/common.sh
adm_env
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1

WHAT="${WHAT:-all}"
RESULT_DIR="${RESULT_DIR:-results/DamagePatternGeneration/CFMADM/1024-dadaGNsplit}"
CKPT="${CKPT:-${RESULT_DIR}/Opensees Based damage Patterns Generation/chjqbp9m/checkpoints/last.ckpt}"
common=(--checkpoint "${CKPT}" --result_dir "${RESULT_DIR}" --batch_size 16 --num_workers 0)

if [[ "${WHAT}" == "shift" || "${WHAT}" == "all" ]]; then
    for split in val test train; do
        python3 -u scripts/eval/measure_base_vs_gt_shift.py "${common[@]}" --split "${split}"
    done
fi
if [[ "${WHAT}" == "spectrum" || "${WHAT}" == "all" ]]; then
    for split in val test; do
        python3 -u scripts/eval/measure_state_spectrum.py "${common[@]}" --split "${split}"
    done
fi
