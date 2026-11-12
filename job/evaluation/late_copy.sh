#!/bin/bash
#PBS -l select=1:ncpus=1:mem=15gb:gpu_id=6
#PBS -l place=shared
#PBS -o log/late_copy.out
#PBS -e log/late_copy.err
#PBS -N latecopy

# Do the late ODE steps copy a washed-out S over cracks the early steps drew (scripts/eval/late_copy_probe.py)?
# S swapped mid-trajectory (self <-> real t-1) and the pull of the late steps towards S. val, v12rfm 9k.
#   qsub job/evaluation/late_copy.sh
#   qsub -v PRESET=hint job/evaluation/late_copy.sh   (S with the memory drawn in late; frequency-split S early)
cd "${PBS_O_WORKDIR:-$HOME/intership-project}"
source job/DamagePatternGeneration/common.sh
adm_env
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
set -e
R=results/DamagePatternGeneration/CFMADM
V1="${R}/2026-11-08_14-30-55_v12rfm"
mkdir -p results/late_copy
if [ "${PRESET:-}" = hint ]; then
    CONDS="${CONDS:-S,S>Am@0.6,S>Ar@0.6,Am,LrHs>S@0.6,LsHr>S@0.6}"; T_A="${T_A-}"; OUT_TAG="${OUT_TAG:-rfm_9k_hint}"
fi
MODELS="${MODELS:-rfm_9k=${V1}@${V1}/checkpoint_step_009000.pt}"
margs=(); for m in ${MODELS}; do margs+=(--model "${m}"); done
python3 -u scripts/eval/late_copy_probe.py --split "${SPLIT:-val}" --hops "${HOPS:-1,4,8}" \
    --conds "${CONDS:-S,R,S>R@0.4,R>S@0.4,S>R@0.6,R>S@0.6}" --t_a "${T_A-0.4}" --lp_sigma "${LP_SIGMA:-4}" --chunk "${CHUNK:-12}" \
    "${margs[@]}" --out "results/late_copy/${OUT_TAG:-rfm_9k}_${SPLIT:-val}.json"
