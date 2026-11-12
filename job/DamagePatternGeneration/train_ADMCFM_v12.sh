#!/bin/bash
#PBS -l select=1:ncpus=2:mem=15gb:gpu_id=0
#PBS -l place=shared
#PBS -o log/train_v12.out
#PBS -e log/train_v12.err
#PBS -N CFM_v12

# v12 single-frame = v11_app (results/DamagePatternGeneration/CFMADM/2026-10-26_00-37-49_v11_app: v11 condition +
# 8-d appearance code, from scratch, 15k steps) + a spall-layout input channel, trained on the PREDICTED kind of
# layout used at test (LAYOUT_CACHE, scripts/eval/build_train_layout_cache.py; val R307 frames read their hspP
# prediction, CTR1 frames are not in the cache). Periodic val KID every 2000 steps (= checkpoint cadence) picks
# the base checkpoint of the v12 refiner (train_ADMCFM_v12refiner.sh, held on this job).
#   qsub -v LAYOUT_CACHE=<cache> job/DamagePatternGeneration/train_ADMCFM_v12.sh
#   gravity template (v12basis): qsub -N CFM_v12basis -o log/train_v12basis.out -e log/train_v12basis.err -v RUN_TAG=v12basis,LAYOUT_CACHE=\
#       results/phase1_fiber_layout/train_cache/grav_hspP_insample_trainval_sdf128.npz job/DamagePatternGeneration/train_ADMCFM_v12.sh
#   real layouts in training (v12gt): -N CFM_v12gt -o log/train_v12gt.out -e log/train_v12gt.err -v RUN_TAG=v12gt,LAYOUT_CACHE=\
#       results/phase1_fiber_layout/train_cache/gtiso32_train_gravval_sdf128.npz  (train = real iso32, val = gravity + hsp_P)
cd "${PBS_O_WORKDIR:-$HOME/intership-project}"
LAYOUT_CACHE="${LAYOUT_CACHE:?set LAYOUT_CACHE (results/phase1_fiber_layout/train_cache/*_trainval_sdf128.npz)}"
[[ -f "${LAYOUT_CACHE}" ]] || { echo "missing ${LAYOUT_CACHE}"; exit 1; }
export V11=True APP_CODE_DIM=8 RUN_TAG="${RUN_TAG:-v12}" MAX_STEPS=15000 \
    LAYOUT_COND=True LAYOUT_ZERO_INIT=False LAYOUT_DROP_PROB="${LAYOUT_DROP_PROB:-0.1}" LAYOUT_CACHE \
    KID_EVERY_N_STEPS=2000 KID_K=8
exec bash job/DamagePatternGeneration/train_ADMCFM.sh
