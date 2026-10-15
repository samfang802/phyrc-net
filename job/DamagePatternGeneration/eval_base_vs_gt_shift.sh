#!/bin/bash
#PBS -l select=1:ncpus=1:mem=8gb:gpu_id=6
#PBS -l place=shared
#PBS -o log/eval_base_vs_gt_shift.out
#PBS -e log/eval_base_vs_gt_shift.err
#PBS -N base_vs_gt

# Single-frame samples vs real frames: translation and residual energy (see the script header).
cd "${PBS_O_WORKDIR:-$HOME/intership-project}"
source job/DamagePatternGeneration/common.sh
adm_env
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1

RESULT_DIR="results/DamagePatternGeneration/CFMADM/1024-dadaGNsplit"
CKPT="${CKPT:-${RESULT_DIR}/Opensees Based damage Patterns Generation/chjqbp9m/checkpoints/last.ckpt}"
for split in val test train; do
    python3 -u scripts/eval/measure_base_vs_gt_shift.py --checkpoint "${CKPT}" --result_dir "${RESULT_DIR}" --split "${split}" --batch_size 16 --num_workers 0
done
