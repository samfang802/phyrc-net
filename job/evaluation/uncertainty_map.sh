#!/bin/bash
#PBS -l select=1:ncpus=2:mem=15gb:gpu_id=6
#PBS -l place=shared
#PBS -o log/uncertainty_map.out
#PBS -e log/uncertainty_map.err
#PBS -N umap

# Does a per-pixel uncertainty read off the sampling trajectory say where S is wrong
# (scripts/eval/uncertainty_map_probe.py)? val split, v12rfm 9k, S = own h-step rollout.
#   qsub job/evaluation/uncertainty_map.sh
#   qsub -v SIGNED=1,OUT_TAG=rfm_9k_signed job/evaluation/uncertainty_map.sh   (U+ / U-)
#   qsub -v MODELS="name=dir@ckpt ...",OUT_TAG=x,SPLIT=test job/evaluation/uncertainty_map.sh
cd "${PBS_O_WORKDIR:-$HOME/intership-project}"
source job/DamagePatternGeneration/common.sh
adm_env
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
set -e
R=results/DamagePatternGeneration/CFMADM
V1="${R}/2026-11-08_14-30-55_v12rfm"
mkdir -p results/uncertainty_map
MODELS="${MODELS:-rfm_9k=${V1}@${V1}/checkpoint_step_009000.pt}"
margs=(); for m in ${MODELS}; do margs+=(--model "${m}"); done
python3 -u scripts/eval/uncertainty_map_probe.py --split "${SPLIT:-val}" --hops "${HOPS:-1,4,8}" \
    --t_ref "${T_REF:-0.3,0.5,0.7}" --chunk "${CHUNK:-12}" ${SIGNED:+--signed} \
    "${margs[@]}" --out "results/uncertainty_map/${OUT_TAG:-rfm_9k}_${SPLIT:-val}.json"
