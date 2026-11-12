#!/bin/bash
#PBS -l select=1:ncpus=2:mem=15gb:gpu_id=3
#PBS -l place=shared
#PBS -o log/sm_ablation.out
#PBS -e log/sm_ablation.err
#PBS -N smabl

# S vs memory vs neither: where the refiner's cracks come from (scripts/eval/sm_ablation_probe.py), val split,
# rfmv2 6k against v12rfm 9k by default; MODELS / OUT_TAG override the list (space-separated name=dir@ckpt).
#   qsub job/evaluation/sm_ablation.sh
cd "${PBS_O_WORKDIR:-$HOME/intership-project}"
source job/DamagePatternGeneration/common.sh
adm_env
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1 OMP_NUM_THREADS=${NCPUS:-2} MKL_NUM_THREADS=${NCPUS:-2}
set -e
R=results/DamagePatternGeneration/CFMADM
V2="${R}/2026-11-10_03-47-29_rfmv2"
V1="${R}/2026-11-08_14-30-55_v12rfm"
mkdir -p results/sm_ablation
MODELS="${MODELS:-rfmv2_6k=${V2}@${V2}/checkpoint_step_006000.pt rfm_9k=${V1}@${V1}/checkpoint_step_009000.pt}"
margs=(); for m in ${MODELS}; do margs+=(--model "${m}"); done
python3 -u scripts/eval/sm_ablation_probe.py --split "${SPLIT:-val}" --chunk "${CHUNK:-12}" "${margs[@]}" \
    --out "results/sm_ablation/${OUT_TAG:-rfmv2_6k_vs_rfm_9k}_${SPLIT:-val}.json"
