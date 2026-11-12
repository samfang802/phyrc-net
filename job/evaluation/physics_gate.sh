#!/bin/bash
#PBS -l select=1:ncpus=2:mem=15gb:gpu_id=2
#PBS -l place=shared
#PBS -o log/physics_gate.out
#PBS -e log/physics_gate.err
#PBS -N physics_gate

# Mechanics write-gate diagnostic (docs/redesign/phase1_ilf.md §12, scripts/eval/phase1_physics_gate.py): fiber-section
# curvature history kappa_t(z) per frame vs where the real memory and the v12memc 6k self memory write. CPU only.
#   qsub job/evaluation/physics_gate.sh
#   qsub -v NAME=v12memc,STEP=009000 job/evaluation/physics_gate.sh
cd "${PBS_O_WORKDIR:-$HOME/intership-project}"
source job/DamagePatternGeneration/common.sh
adm_env
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
set -e
NAME="${NAME:-v12memc}"
STEP="${STEP:-006000}"
OUT=results/phase1_physics_gate/${NAME}_${STEP}
mkdir -p log "${OUT}"
# v2 (step births) at two kappa-growth thresholds: the simulated - peak is ~2.7 % above the + peak of the same level,
# so delta 0.02 also opens env on the - peak; 0.05 does not.
for delta in 0.02 0.05; do
    for split in test val; do
        python3 -u scripts/eval/phase1_physics_gate.py --split "${split}" \
            --npz "results/eval_suite/${NAME}/${split}/samples/${NAME}_${STEP}_grav.npz" \
            --mm_per_px 6.25 --base_row 127 --window 1 --delta "${delta}" \
            --out "${OUT}/v2_d${delta}_${split}.json" | tee "${OUT}/v2_d${delta}_${split}.txt"
    done
done
