#!/bin/bash
#PBS -l select=1:ncpus=2:mem=15gb:gpu_id=0
#PBS -l place=shared
#PBS -o log/eval_v12.out
#PBS -e log/eval_v12.err
#PBS -N eval_v12

# v12 (layout-conditioned v11app refiner) (train_ADMCFM_v12refiner.sh) vs app_r_6k (docs/redesign/phase1_fiber_clock.md §6).
# Per checkpoint step, four layout inputs on the same seed / app code (mean) as app_r_6k:
#   none   layout 0 (dropped): should match app_r_6k if the extra channel cost nothing
#   drift  OFP shape + max_drift area     (clock control)
#   hspP   OFP shape + hsp_P area         (the v12 method)
#   grav   gravity template + hsp_P area  (the v12basis method; build_clock_layouts.py --shape template)
#   iso32  real monotone 32x32 layout     (ceiling)
# VARIANTS picks the subset (default all five).
#   qsub job/evaluation/eval_v12.sh                 (RUN = newest *_v12refiner, STEPS="006000")
#   qsub -v QUICK=1 job/evaluation/eval_v12.sh      # test split only, KID only
#   qsub -v FORCE=1 ...                             # recompute a step that was already scored
#   qsub -v RUN=<dir>,STEPS="006000 009000" job/evaluation/eval_v12.sh
#   qsub -v NAME=v12hspP,RUN_GLOB=v12refiner_hspP job/evaluation/eval_v12.sh   (the train=test hspP refiner)
#   qsub -v NAME=v12basis,RUN_GLOB=v12basisrefiner job/evaluation/eval_v12.sh           (gravity-template refiner)
cd "${PBS_O_WORKDIR:-$HOME/intership-project}"
source job/DamagePatternGeneration/common.sh
adm_env
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
set -e

NAME="${NAME:-v12}"   # sample / output prefix; v12refiner_hspP run: NAME=v12hspP RUN_GLOB=v12refiner_hspP
RUN="${RUN:-$(ls -d results/DamagePatternGeneration/CFMADM/*_${RUN_GLOB:-v12refiner} | sort | tail -1)}"
STEPS="${STEPS:-006000}"
VARIANTS="${VARIANTS:-none drift hspP grav iso32}"
SPLITS="${SPLITS:-test val}"
[[ -n "${QUICK:-}" ]] && SPLITS="test"
OUT=results/eval_suite/${NAME}
TEMPLATE=job/evaluation/configs/eval_suite_v12.template.json
mkdir -p log "${OUT}/logs"
echo "[${NAME}] run ${RUN} steps ${STEPS}"

declare -A LAY
for split in test val; do
    LAY[none_${split}]=""
    LAY[drift_${split}]="results/phase1_fiber_layout/layouts/ofp_drift_${split}.npz"
    LAY[hspP_${split}]="results/phase1_fiber_layout/layouts/ofp_hspP_${split}.npz"
    LAY[grav_${split}]="results/phase1_fiber_layout/layouts/grav_hspP_${split}.npz"
    LAY[iso32_${split}]="results/phase1_layout_harness/layouts/real_iso32_${split}.npz"
done

for step in ${STEPS}; do
    ckpt="${RUN}/checkpoint_step_${step}.pt"
    [[ -f "${ckpt}" ]] || { echo "[v12] missing ${ckpt}"; exit 1; }
    for split in ${SPLITS}; do
        mkdir -p "${OUT}/${split}/samples"
        for v in ${VARIANTS}; do
            npz="${OUT}/${split}/samples/${NAME}_${step}_${v}.npz"
            [[ -f "${npz}" ]] && continue
            lay="${LAY[${v}_${split}]}"
            python3 -u scripts/eval/generate_eval_samples.py --cfm_result_dir "${RUN}" --cfm_checkpoint "${ckpt}" \
                --split "${split}" --k 8 --batch_size 32 --seed 0 --app_code mean ${lay:+--layouts "${lay}"} \
                --out "${npz}" > "${OUT}/logs/gen_${step}_${v}_${split}.log" 2>&1
        done
    done
    cfg="job/evaluation/configs/eval_suite_${NAME}_${step}.json"
    # baseline entry from the template, one entry per variant
    python3 - "${TEMPLATE}" "${cfg}" "${OUT}" "${OUT}/step_${step}" "${NAME}_${step}" ${VARIANTS} <<'PY'
import json, sys
tpl, cfg, samples, table, prefix, *variants = sys.argv[1:]
c = json.load(open(tpl))
c["out_dir"] = table
c["models"] = [c["models"][0]] + [
    {"name": f"{prefix}_{v}", "npz": {s: f"{samples}/{s}/samples/{prefix}_{v}.npz" for s in c["splits"]},
     "jpeg_quality": 75}
    for v in variants]
json.dump(c, open(cfg, "w"), indent=2)
PY
    suite=()
    [[ -n "${QUICK:-}" ]] && suite+=(--quick)
    [[ -n "${FORCE:-}" ]] && suite+=(--force)
    python3 -u scripts/eval/run_eval_suite.py --config "${cfg}" "${suite[@]}"
    for split in ${SPLITS}; do
        args=(--model "app_r_6k=results/eval_suite/v11app_refiner/${split}/samples/app_r_6k.npz")
        for v in ${VARIANTS}; do args+=(--model "${NAME}_${step}_${v}=${OUT}/${split}/samples/${NAME}_${step}_${v}.npz"); done
        python3 -u scripts/eval/phase0_layout_oracle_metrics.py --split "${split}" "${args[@]}" --baseline app_r_6k \
            --out "${OUT}/spall_metrics_${step}_${split}.json"
    done
done
