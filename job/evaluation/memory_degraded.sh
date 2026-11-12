#!/bin/bash
#PBS -l select=1:ncpus=2:mem=15gb:gpu_id=3
#PBS -l place=shared
#PBS -o log/memory_degraded.out
#PBS -e log/memory_degraded.err
#PBS -N memdeg

# Does the memory help once S has degraded (scripts/eval/memory_degraded_probe.py)? val split, S = the model's own
# h-step rollout, memory real / self / zero / swap. v12rfm 9k and rfmv2 6k.
#   qsub job/evaluation/memory_degraded.sh
#   qsub -v HOPS=8,SPLIT=test job/evaluation/memory_degraded.sh
# MODELS / OUT_TAG override the model list (space-separated name=dir@ckpt).
# CONFS: S confidences fed at sampling instead of the model's own (one run each, suffix _c<conf>), e.g.
#   qsub -v MODELS="rfm_9k=...",OUT_TAG=rfm_9k,CONFS="0.5 0.3" job/evaluation/memory_degraded.sh
cd "${PBS_O_WORKDIR:-$HOME/intership-project}"
source job/DamagePatternGeneration/common.sh
adm_env
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1 OMP_NUM_THREADS=${NCPUS:-2} MKL_NUM_THREADS=${NCPUS:-2}
set -e
R=results/DamagePatternGeneration/CFMADM
V1="${R}/2026-11-08_14-30-55_v12rfm"
V2="${R}/2026-11-10_03-47-29_rfmv2"
mkdir -p results/memory_degraded
MODELS="${MODELS:-rfm_9k=${V1}@${V1}/checkpoint_step_009000.pt rfmv2_6k=${V2}@${V2}/checkpoint_step_006000.pt}"
margs=(); for m in ${MODELS}; do margs+=(--model "${m}"); done
run() {  # $1 = extra args, $2 = output suffix
    python3 -u scripts/eval/memory_degraded_probe.py --split "${SPLIT:-val}" --hops "${HOPS:-1,4,8}" \
        --chunk "${CHUNK:-12}" $1 \
        "${margs[@]}" --out "results/memory_degraded/${OUT_TAG:-rfm_9k_rfmv2_6k}${2}_${SPLIT:-val}.json"
}
if [ -z "${CONFS:-}" ]; then
    run "" ""
else
    for c in ${CONFS}; do run "--conf ${c}" "_c${c}"; done
fi
