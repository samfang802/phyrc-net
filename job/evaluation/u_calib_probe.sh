#!/bin/bash
#PBS -l select=1:ncpus=2:mem=15gb:gpu_id=6
#PBS -l place=shared
#PBS -o log/u_calib_probe.out
#PBS -e log/u_calib_probe.err
#PBS -N u_calib

# Pre-check for the U channel (scripts/eval/u_calib_probe.py, no training): on real pairs, how well does the
# spread of two draws |x_a - x_b| predict the real error |x_a - GT_t|, and what k calibrates it.
# Model = v12gtrefiner 6k (the base of the planned U refiner). train / val read the dataset layout cache
# (train: GT iso32 as in training, val: grav); test reads the grav layouts.
#   qsub job/evaluation/u_calib_probe.sh
#   qsub -v SPLITS=val,EXTRA=--no_chain job/evaluation/u_calib_probe.sh
cd "${PBS_O_WORKDIR:-$HOME/intership-project}"
source job/DamagePatternGeneration/common.sh
adm_env
export HF_HUB_OFFLINE=1

R=results/DamagePatternGeneration/CFMADM
V12="${R}/2026-11-05_22-06-00_v12gtrefiner"
CKPT="${CKPT:-${V12}/checkpoint_step_006000.pt}"
OUT="${OUT:-results/u_calib_probe}"
SPLITS="${SPLITS:-val,test,train}"
for SPLIT in ${SPLITS//,/ }; do
    LAY=""
    [[ "${SPLIT}" == "test" ]] && LAY="@results/phase1_fiber_layout/layouts/grav_hspP_test.npz"
    MAXP=0
    [[ "${SPLIT}" == "train" ]] && MAXP="${TRAIN_PAIRS:-256}"
    python3 scripts/eval/u_calib_probe.py --split "${SPLIT}" --out_dir "${OUT}/${SPLIT}" --max_pairs "${MAXP}" \
        --model "v12gtr_6k=${V12}@${CKPT}${LAY}" ${EXTRA:-} || exit 1
done
