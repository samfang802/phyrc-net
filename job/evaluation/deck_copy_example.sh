#!/bin/bash
#PBS -l select=1:ncpus=1:mem=15gb:gpu_id=6
#PBS -l place=shared
#PBS -N copy_example
# Copy example figures (scripts/viz/deck_copy_example.py) -> results/deck_20261110/figures
cd "${PBS_O_WORKDIR:-$HOME/intership-project}"
source job/DamagePatternGeneration/common.sh
adm_env
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
set -e
python3 -u scripts/viz/deck_copy_example.py --out results/deck_20261110/figures ${NPZ:+--npz ${NPZ}} ${TAG:+--tag ${TAG}}
