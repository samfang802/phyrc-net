#!/bin/bash
#PBS -l select=1:ncpus=1:mem=8gb:gpu_id=6
#PBS -l place=shared
#PBS -o log/viz_high_refiner.out
#PBS -e log/viz_high_refiner.err
#PBS -N viz_refiner

# 6x6 High-DI grid from a residual refiner checkpoint. Same cells as the single-frame grids
# (cell order locked to the single-frame run's JSON). Each column is walked from its first frame.
cd "${PBS_O_WORKDIR:-$HOME/intership-project}"
source job/DamagePatternGeneration/common.sh
adm_env
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1

RUN_DIR="${RUN_DIR:-results/DamagePatternGeneration/CFMADM/2026-10-13_03-02-40_refiner_res_15k}"
CKPT="${CKPT:-${RUN_DIR}/checkpoint_step_015000.pt}"
META="results/DamagePatternGeneration/CFMADM/1024-dadaGNsplit/inference_samples/overview_di_high_selfcondOFF_step_15000_ode30.json"

python3 -u scripts/viz/export_di_tertile_overview_grids.py \
    --checkpoint "${CKPT}" --result_dir "${RUN_DIR}" \
    --only_groups High --split test --gt_from_meta "${META}" \
    --batch_size 16 --num_workers 0 --seed 42 ${GRAY:+--refiner_gray_state}
ls -la "${RUN_DIR}/inference_samples" | grep overview
