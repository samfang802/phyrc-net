#!/bin/bash
#PBS -l select=1:ncpus=2:mem=15gb:gpu_id=2
#PBS -l place=shared
#PBS -o log/memory_reliance.out
#PBS -e log/memory_reliance.err
#PBS -N memrel

# Memory vs S_{t-1} reliance per Resampling Forcing level (scripts/eval/memory_reliance_probe.py), val split:
# rule memory (v12rfm) against the learned-E ILF runs, all at STEP.
#   qsub job/evaluation/memory_reliance.sh            (STEP=9000, REPEATS=6)
cd "${PBS_O_WORKDIR:-$HOME/intership-project}"
source job/DamagePatternGeneration/common.sh
adm_env
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
set -e
STEP="${STEP:-9000}"
R=results/DamagePatternGeneration/CFMADM
args=()
for spec in v12rfm=2026-11-08_14-30-55_v12rfm ILF_hpnew=2026-11-09_11-35-07_ILF_hpnew \
            ILF_hpnew_aux=2026-11-09_12-41-41_ILF_hpnew_aux ILF_hpnew_bd=2026-11-09_17-25-16_ILF_hpnew_bd; do
    d="${R}/${spec#*=}"
    args+=(--model "${spec%%=*}=${d}@${d}/checkpoint_step_$(printf %06d "${STEP}").pt")
done
mkdir -p results/ilf_memory_probe
python3 -u scripts/eval/memory_reliance_probe.py "${args[@]}" --repeats "${REPEATS:-6}" \
    --out "results/ilf_memory_probe/reliance_${STEP}.json"
