#!/bin/bash
#PBS -l select=1:ncpus=2:mem=15gb:gpu_id=1
#PBS -l place=shared
#PBS -o log/ilf_ae_probe.out
#PBS -e log/ilf_ae_probe.err
#PBS -N ilf_ae

# E + decoder alone on hpnew inputs: can a 1/4-resolution memory hold thin cracks, per loss / channels / input?
#   qsub job/evaluation/ilf_ae_probe.sh            (-v STEPS=600)
cd "${PBS_O_WORKDIR:-$HOME/intership-project}"
source job/DamagePatternGeneration/common.sh
adm_env
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
python3 -u scripts/eval/ilf_ae_probe.py --steps "${STEPS:-600}" --out results/ilf_memory_probe/ae_probe.json
