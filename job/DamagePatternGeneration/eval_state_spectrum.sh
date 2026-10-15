#!/bin/bash
#PBS -l select=1:ncpus=1:mem=8gb:gpu_id=6
#PBS -l place=shared
#PBS -o log/eval_state_spectrum.out
#PBS -e log/eval_state_spectrum.err
#PBS -N state_spectrum

# Radial power spectrum of single-frame samples at 1 / 10 / 30 / 60 NFE, and the high-frequency share of r = x_t - S.
cd "${PBS_O_WORKDIR:-$HOME/intership-project}"
source job/DamagePatternGeneration/common.sh
adm_env
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1

RESULT_DIR="results/DamagePatternGeneration/CFMADM/1024-dadaGNsplit"
CKPT="${CKPT:-${RESULT_DIR}/Opensees Based damage Patterns Generation/chjqbp9m/checkpoints/last.ckpt}"
for split in val test; do
    python3 -u scripts/eval/measure_state_spectrum.py --checkpoint "${CKPT}" --result_dir "${RESULT_DIR}" --split "${split}" --batch_size 16 --num_workers 0
done
