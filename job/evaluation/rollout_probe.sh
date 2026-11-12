#!/bin/bash
#PBS -l select=1:ncpus=2:mem=15gb:gpu_id=1
#PBS -l place=shared
#PBS -o log/rollout_probe.out
#PBS -e log/rollout_probe.err
#PBS -N rollprb

# h-step self-rollout S_{t-1} vs one Resampling-Forcing hop at t_s (scripts/eval/rollout_vs_resample_probe.py).
#   qsub job/evaluation/rollout_probe.sh                         (v12rfm 9k, val)
#   qsub -v RUN=<result_dir>,STEP=6000,TAG=rfc job/evaluation/rollout_probe.sh
cd "${PBS_O_WORKDIR:-$HOME/intership-project}"
source job/DamagePatternGeneration/common.sh
adm_env
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
set -e
RUN="${RUN:-results/DamagePatternGeneration/CFMADM/2026-11-08_14-30-55_v12rfm}"
STEP="${STEP:-9000}"
TAG="${TAG:-rfm}"
SPLIT="${SPLIT:-val}"
mkdir -p results/rollout_probe
python3 -u scripts/eval/rollout_vs_resample_probe.py \
    --model "${TAG}=${RUN}@${RUN}/checkpoint_step_$(printf %06d "${STEP}").pt" --split "${SPLIT}" \
    --t_every "${T_EVERY:-3}" --chunk "${CHUNK:-12}" ${EXTRA:-} --out "results/rollout_probe/${TAG}_$((STEP / 1000))k_${SPLIT}.json"
