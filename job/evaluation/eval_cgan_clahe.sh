#!/bin/bash
#PBS -l select=1:ncpus=2:mem=15gb:gpu_id=7
#PBS -l place=shared
#PBS -o log/eval_cgan_clahe.out
#PBS -e log/eval_cgan_clahe.err
#PBS -N eval_cgan_clahe

# The cGAN in the reference domain. It was trained on the raw photos, so its archived images get the
# generator's CLAHE first, then the interference removal + GT-mask placebo (as cGAN_eq.npz, but in the
# CLAHE'd domain the detector and inpainter work in), then the test suite with our models
# (job/evaluation/configs/eval_suite_cgan_clahe.json) and the diversity metrics.
#   qsub job/evaluation/eval_cgan_clahe.sh
cd "${PBS_O_WORKDIR:-$HOME/intership-project}"
source job/DamagePatternGeneration/common.sh
adm_env
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
set -e

S=results/eval_suite/cgan_clahe/test/samples
mkdir -p log "${S}" results/eval_suite/cgan_clahe/logs
[[ -f "${S}/cGAN_clahe_src.npz" ]] || python3 -u scripts/eval/images_to_npz.py --image_dir cGANckpt/CTR1 --suffix _test.jpg \
    --clahe --split test --out "${S}/cGAN_clahe_src.npz"
[[ -f "${S}/cGAN_clahe_eq.npz" ]] || python3 -u scripts/eval/inpaint_generated_samples.py --samples "${S}/cGAN_clahe_src.npz" \
    --detector results/InterferenceDetector/v1/best.pt --split test --add_gt_mask --out "${S}/cGAN_clahe_eq.npz" \
    > results/eval_suite/cgan_clahe/logs/inpaint_cGAN.log 2>&1

python3 -u scripts/eval/run_eval_suite.py --config job/evaluation/configs/eval_suite_cgan_clahe.json

python3 -u scripts/eval/eval_diversity.py --split test --no_clahe \
    --image_dir Data/DamagePattern_inpaint_selfcondnoDI/images \
    --model cGAN="${S}/cGAN_clahe_eq.npz" \
    --model app_r_6k=results/eval_suite/v11app_refiner/test/samples/app_r_6k.npz \
    --model v12gt_grav=results/eval_suite/v12gtrefiner/test/samples/v12gtrefiner_006000_grav.npz \
    --model v12gt_gravnoise=results/eval_suite/v12gtrefiner/test/samples/v12gtrefiner_006000_gravnoise.npz \
    --jpeg cGAN=75 app_r_6k=75 v12gt_grav=75 v12gt_gravnoise=75 \
    --out_dir results/eval_suite/diversity
