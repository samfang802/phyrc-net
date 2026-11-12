#!/bin/bash
#PBS -l select=1:ncpus=1:mem=15gb:gpu_id=6
#PBS -l place=shared
#PBS -o log/eval_fid_panels.out
#PBS -e log/eval_fid_panels.err
#PBS -N eval_fid_panels

# K = 8 Inception FID (test + val) for V10 refiner, V10.6C, V11 (app_r_6k), V12 (v12gtrefiner,
# grav) and cGAN (test only, K = 1), same image domain as the KID of eval_suite_final.json; plus the
# drift × DI panels for V12 with the gravnoise layout (and GT, V11 for side by side).
#   qsub job/evaluation/eval_fid_panels.sh
cd "${PBS_O_WORKDIR:-$HOME/intership-project}"
source job/DamagePatternGeneration/common.sh
adm_env
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
set -e
mkdir -p log
OUT=results/eval_suite/fid_panels

python3 -u scripts/eval/eval_fid.py --split test \
    --samples V10=results/eval_suite/v105/test/samples/refiner.npz \
              V10.6=results/eval_suite/v106C/test/samples/v106C_6k.npz \
              V11=results/eval_suite/v11app_refiner/test/samples/app_r_6k.npz \
              V12=results/eval_suite/v12gtrefiner/test/samples/v12gtrefiner_006000_grav.npz \
              cGAN=results/eval_suite/cgan_clahe/test/samples/cGAN_clahe_eq.npz \
    --jpeg V10=75 V10.6=75 V11=75 V12=75 cGAN=75 \
    --other_npz results/eval_suite/v106/val/samples/refiner.npz \
    --out ${OUT}/fid_test.json

python3 -u scripts/eval/eval_fid.py --split val \
    --samples V10=results/eval_suite/v106/val/samples/refiner.npz \
              V10.6=results/eval_suite/v106C/val/samples/v106C_6k.npz \
              V11=results/eval_suite/v11app_refiner/val/samples/app_r_6k.npz \
              V12=results/eval_suite/v12gtrefiner/val/samples/v12gtrefiner_006000_grav.npz \
    --jpeg V10=75 V10.6=75 V11=75 V12=75 \
    --out ${OUT}/fid_val.json

P=${OUT}/drift_panels
python3 -u scripts/viz/export_drift_view_panel_from_images.py --gt \
    --gt_image_dir Data/DamagePattern_inpaint_selfcondnoDI/images --gt_no_clahe --tag gt --out_dir ${P}
python3 -u scripts/viz/export_drift_view_panel_from_images.py \
    --npz results/eval_suite/v11app_refiner/test/samples/app_r_6k.npz --k 0 --tag v11_app_r_6k --out_dir ${P}
for k in 0 1; do
  python3 -u scripts/viz/export_drift_view_panel_from_images.py \
      --npz results/eval_suite/v12gtrefiner/test/samples/v12gtrefiner_006000_gravnoise.npz --k ${k} \
      --tag v12gtrefiner_gravnoise_k${k} --out_dir ${P}
done
