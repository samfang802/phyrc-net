#!/bin/bash
#PBS -l select=1:ncpus=2:mem=15gb:gpu_id=6
#PBS -l place=shared
#PBS -o log/eval_suite.out
#PBS -e log/eval_suite.err
#PBS -N eval_suite

# Full evaluation (docs/eval/README.md): samples → interference removal → DINO / damage texture /
# temporal → summary.md, for every model in CONFIG.
#   qsub job/evaluation/eval_suite.sh
#   qsub -v CONFIG=job/evaluation/configs/eval_suite_final.json,SPLITS=test job/evaluation/eval_suite.sh
#   qsub -v CONFIG=...,QUICK=1 job/evaluation/eval_suite.sh     # test split, KID only
# An interrupted run continues where it stopped. The same model list is not scored twice.
# FORCE=1 redoes everything.
cd "${PBS_O_WORKDIR:-$HOME/intership-project}"
source job/DamagePatternGeneration/common.sh
adm_env
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"   # backbones from the local cache; set 0 for the first download

CONFIG="${CONFIG:-job/evaluation/configs/eval_suite_models.json}"
args=(--config "${CONFIG}")
[[ -n "${SPLITS:-}" ]] && args+=(--splits ${SPLITS})
[[ -n "${QUICK:-}" ]] && args+=(--quick)
[[ -n "${FORCE:-}" ]] && args+=(--force)
mkdir -p log
python3 -u scripts/eval/run_eval_suite.py "${args[@]}"
