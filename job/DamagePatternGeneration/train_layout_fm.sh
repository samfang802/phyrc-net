#!/bin/bash
#PBS -l select=1:ncpus=2:mem=15gb:gpu_id=3
#PBS -l place=shared
#PBS -o log/train_layout_fm.out
#PBS -e log/train_layout_fm.err
#PBS -N c2_layout
# C2 layout generator runs (docs/redesign/phase1_c2.md). Each line of RUNS: "name|args for train_layout_fm.py".
#   qsub -v RUNS="a|--cond tau --use_prev 0;b|--cond g" job/DamagePatternGeneration/train_layout_fm.sh
cd "${PBS_O_WORKDIR:-$HOME/intership-project}"
source job/DamagePatternGeneration/common.sh
adm_env
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
OUT=results/phase1_c2/runs
IFS=';' read -ra ITEMS <<< "${RUNS}"
for it in "${ITEMS[@]}"; do
    name="${it%%|*}"; args="${it#*|}"
    echo "=== ${name}: ${args}"
    python3 -u scripts/train_layout_fm.py --out_dir "${OUT}/${name}" ${args} 2>&1 | grep -v "^step" 
done
