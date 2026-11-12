#!/bin/bash
#PBS -l select=1:ncpus=2:mem=15gb:gpu_id=7
#PBS -l place=shared
#PBS -o log/state_source_probe.out
#PBS -e log/state_source_probe.err
#PBS -N state_probe

# Refiner state track: crack precision / recall vs GT_t and DINOv3 KID of S_{t-1} built from the base cache,
# GT_{t-1}, SDEdit from GT_{t-1} and DDS from GT_{t-1} (scripts/eval/state_source_probe.py), v12gt refiner.
#   qsub job/evaluation/state_source_probe.sh
#   qsub -v OUT=results/state_source_probe/v12gt_n4,EXTRA="--n 4" job/evaluation/state_source_probe.sh
cd "${PBS_O_WORKDIR:-$HOME/intership-project}"
source job/DamagePatternGeneration/common.sh
adm_env
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
set -e
mkdir -p log

OUT="${OUT:-results/state_source_probe/v12gt}"
python3 scripts/eval/state_source_probe.py --out_dir "${OUT}" ${EXTRA:-}
