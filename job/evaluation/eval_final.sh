#!/bin/bash
#PBS -l select=1:ncpus=2:mem=15gb:gpu_id=7
#PBS -l place=shared
#PBS -o log/eval_final.out
#PBS -e log/eval_final.err
#PBS -N eval_final

# The results table of docs/eval/results.md: one checkpoint per version (job/evaluation/configs/eval_suite_final.json)
# through the suite (test + val; V9 samples are generated here), then the diversity / copy metrics on test.
# The cGAN npz comes from eval_cgan_clahe.sh (CLAHE'd + interference removed).
#   qsub job/evaluation/eval_final.sh
cd "${PBS_O_WORKDIR:-$HOME/intership-project}"
source job/DamagePatternGeneration/common.sh
adm_env
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
set -e
mkdir -p log

python3 -u scripts/eval/run_eval_suite.py --config job/evaluation/configs/eval_suite_final.json

S=results/eval_suite/final/test/samples
python3 -u scripts/eval/eval_diversity.py --split test --no_clahe \
    --image_dir Data/DamagePattern_inpaint_selfcondnoDI/images \
    --model V9="${S}/V9.npz" \
    --model V10=results/eval_suite/v105/test/samples/refiner.npz \
    --model V10.5=results/eval_suite/v105/test/samples/v105A.npz \
    --model V10.6=results/eval_suite/v106C/test/samples/v106C_6k.npz \
    --model V11=results/eval_suite/v11app_refiner/test/samples/app_r_6k.npz \
    --model V12=results/eval_suite/v12gtrefiner/test/samples/v12gtrefiner_006000_grav.npz \
    --model cGAN=results/eval_suite/cgan_clahe/test/samples/cGAN_clahe_eq.npz \
    --jpeg V9=75 V10=75 V10.5=75 V10.6=75 V11=75 V12=75 cGAN=75 \
    --out_dir results/eval_suite/final/test/diversity
