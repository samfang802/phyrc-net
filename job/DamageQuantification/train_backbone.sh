#!/bin/bash
#PBS -l select=1:ncpus=1:gpu_id=2
#PBS -l place=shared
#PBS -o log/train_backbone.out
#PBS -e log/train_backbone.err
#PBS -N DI_backbone

# DI regressor with a pretrained torchvision backbone (train_quantification.py).
# Replaces train_{resnet101,swin_b,vit_b_16,vit_b_32,vit_l_16,vit_l_32}.sh, which pointed at the old
# repo path and the removed `hysteresis` conda env. The learning rate and epochs below are the ones
# those scripts used; override with LR / EPOCH.
#   qsub -v BACKBONE=swin_b job/DamageQuantification/train_backbone.sh
cd "${PBS_O_WORKDIR:-$HOME/intership-project}"
set +u
source ~/.bashrc
set -u
conda activate internship
module load cuda-12.4

BACKBONE="${BACKBONE:-vit_b_16}"
case "${BACKBONE}" in
    swin_b)                     lr=1e-4; ep=3000 ;;
    vit_b_16)                   lr=1e-5; ep=500 ;;
    vit_b_32|vit_l_16|vit_l_32) lr=1e-6; ep=500 ;;
    resnet101)                  lr=1e-6; ep=500 ;;
    *) echo "[train_backbone] no preset for ${BACKBONE}; set LR and EPOCH"; lr=""; ep="" ;;
esac
LR="${LR:-${lr}}"
EPOCH="${EPOCH:-${ep}}"
mkdir -p log
python3 train_quantification.py --lr "${LR}" --epoch "${EPOCH}" --batch_size "${BATCH_SIZE:-8}" \
    --backbone "${BACKBONE}" --pretrained 2>&1 | tee "log/train_${BACKBONE}.log"
