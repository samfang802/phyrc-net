#!/bin/bash
#PBS -l select=1:ncpus=1:mem=15gb:gpu_id=6
#PBS -l place=shared
#PBS -o log/eval_gravnoise.out
#PBS -e log/eval_gravnoise.err
#PBS -N eval_gravnoise

# V12 (v12gtrefiner 6k) with the gravnoise layout (gravity template + boundary jitter, 8 layouts per
# condition, same hsp_P area) against grav and V11 (app_r_6k). STAGE picks one part so they run in parallel:
#   gen        val samples with layouts/gravnoise_hspP_val.npz (test samples already exist)
#   diversity  test copy / spread / mask IoU (grav, gravnoise, V11)
#   suite      eval_suite_gravnoise.json (test + val) + Inception FID for gravnoise   (after gen)
#   J=$(qsub -v STAGE=gen job/evaluation/eval_gravnoise.sh); qsub -v STAGE=diversity job/evaluation/eval_gravnoise.sh
#   qsub -W depend=afterok:$J -v STAGE=suite job/evaluation/eval_gravnoise.sh
cd "${PBS_O_WORKDIR:-$HOME/intership-project}"
source job/DamagePatternGeneration/common.sh
adm_env
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
set -e
mkdir -p log
RUN=results/DamagePatternGeneration/CFMADM/2026-11-05_22-06-00_v12gtrefiner
S=results/eval_suite/v12gtrefiner
OUT=results/eval_suite/gravnoise
mkdir -p ${OUT}

case "${STAGE}" in
gen)
    npz=${S}/val/samples/v12gtrefiner_006000_gravnoise.npz
    [[ -f ${npz} ]] || python3 -u scripts/eval/generate_eval_samples.py --cfm_result_dir ${RUN} \
        --cfm_checkpoint ${RUN}/checkpoint_step_006000.pt --split val --k 8 --batch_size 32 --seed 0 \
        --app_code mean --layouts results/phase1_fiber_layout/layouts/gravnoise_hspP_val.npz --out ${npz}
    ;;
diversity)
    python3 -u scripts/eval/eval_diversity.py --split test --no_clahe \
        --image_dir Data/DamagePattern_inpaint_selfcondnoDI/images \
        --model V11=results/eval_suite/v11app_refiner/test/samples/app_r_6k.npz \
        --model V12=${S}/test/samples/v12gtrefiner_006000_grav.npz \
        --model V12_gravnoise=${S}/test/samples/v12gtrefiner_006000_gravnoise.npz \
        --jpeg V11=75 V12=75 V12_gravnoise=75 \
        --out_dir ${OUT}/test/diversity
    ;;
suite)
    python3 -u scripts/eval/run_eval_suite.py --config job/evaluation/configs/eval_suite_gravnoise.json
    for split in test val; do
        python3 -u scripts/eval/eval_fid.py --split ${split} \
            --samples V12=${S}/${split}/samples/v12gtrefiner_006000_grav.npz \
                      V12_gravnoise=${S}/${split}/samples/v12gtrefiner_006000_gravnoise.npz \
            --jpeg V12=75 V12_gravnoise=75 --out ${OUT}/fid_${split}.json
    done
    ;;
*) echo "STAGE must be gen | diversity | suite"; exit 1 ;;
esac
