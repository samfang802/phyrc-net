#!/bin/bash
#PBS -l select=1:ncpus=2:mem=15gb:gpu_id=2
#PBS -l place=shared
#PBS -o log/heun_hop_check.out
#PBS -e log/heun_hop_check.err
#PBS -N heunchk

# Leftover start noise of a Heun resampling hop, and TF32 sampling time (scripts/eval/heun_hop_check.py), rfmv2 6k.
#   qsub job/evaluation/heun_hop_check.sh
cd "${PBS_O_WORKDIR:-$HOME/intership-project}"
source job/DamagePatternGeneration/common.sh
adm_env
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
set -e
R="${RUN:-results/DamagePatternGeneration/CFMADM/2026-11-10_03-47-29_rfmv2}"
STEP="${STEP:-6000}"
mkdir -p results/heun_hop_check
python3 -u scripts/eval/heun_hop_check.py --model "m=${R}@${R}/checkpoint_step_$(printf %06d "${STEP}").pt" \
    --out "results/heun_hop_check/$(basename "${R}")_${STEP}.json"
