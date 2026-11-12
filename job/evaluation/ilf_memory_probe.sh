#!/bin/bash
#PBS -l select=1:ncpus=2:mem=15gb:gpu_id=2
#PBS -l place=shared
#PBS -o log/ilf_memory_probe.out
#PBS -e log/ilf_memory_probe.err
#PBS -N ilf_mem

# Does the learned ILF memory (max over E of the earlier frames) collapse to the last frame?
# scripts/eval/ilf_memory_probe.py on real frames, no training.
#   qsub job/evaluation/ilf_memory_probe.sh
#   qsub -v RUN=<result_dir>,STEPS="2000 4000" job/evaluation/ilf_memory_probe.sh
cd "${PBS_O_WORKDIR:-$HOME/intership-project}"
source job/DamagePatternGeneration/common.sh
adm_env
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1 HF_HUB_OFFLINE=1
RUN="${RUN:-results/DamagePatternGeneration/CFMADM/2026-11-09_01-01-01_ILFnoK}"
STEPS="${STEPS:-1000 2000 3000 4000}"
args=()
for s in ${STEPS}; do args+=(--step "${s}"); done
python3 -u scripts/eval/ilf_memory_probe.py --run "${RUN}" "${args[@]}" --split train --split val \
    --out "results/ilf_memory_probe/$(basename "${RUN}").json"
