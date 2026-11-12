#!/bin/bash
#PBS -l select=1:ncpus=2:mem=15gb:gpu_id=6
#PBS -l place=shared
#PBS -o log/phase1_fiber_layout.out
#PBS -e log/phase1_fiber_layout.err
#PBS -N fiber_layout

# Fiber-clock follow-up (docs/redesign/phase1_fiber_clock.md):
#   1. robustness sweep of the mechanics constants (CPU)            -> results/phase1_fiber_clock/sweep/
#   2. layouts: OFP c_iou shape + area from a clock (hsp_P, max_drift) -> results/phase1_fiber_layout/layouts/
#   3. render with layout_scratch (same seed as the iso32 ceiling), eval suite, image-side spall metrics
#   qsub job/evaluation/phase1_fiber_layout.sh      (STAGES="sweep layouts render eval" to pick)
# Existing outputs are reused.
cd "${PBS_O_WORKDIR:-$HOME/intership-project}"
source job/DamagePatternGeneration/common.sh
adm_env
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
set -e

STAGES="${STAGES:-sweep layouts render eval}"
FC=results/phase1_fiber_clock
D=results/phase1_fiber_layout
CONFIG=job/evaluation/configs/eval_suite_fiber_layout.json
mkdir -p log "${FC}/sweep" "${D}/layouts" "${D}/samples" "${D}/logs"
has() { [[ " ${STAGES} " == *" $1 "* ]]; }

if has sweep && [[ ! -f "${FC}/sweep/sweep_summary.json" ]]; then
    python3 -u scripts/eval/phase1_fiber_clock.py --sweep --out "${FC}/sweep" 2>"${D}/logs/sweep.err" | tee "${FC}/sweep/report.txt"
fi

if has layouts; then
    for split in test val; do
        cp -n "results/phase1_ofp/layouts_${split}_c_iou.npz" "${D}/layouts/ofp_c_iou_${split}.npz"
        for pair in ofp_hspP:hsp_P ofp_drift:max_drift; do
            out="${D}/layouts/${pair%%:*}_${split}.npz"
            [[ -f "${out}" ]] || python3 -u scripts/eval/build_clock_layouts.py --frames "${FC}/frames_with_clocks.csv" \
                --clock "${pair##*:}" --ofp "results/phase1_ofp/layouts_${split}_c_iou.npz" --out "${out}" 2>>"${D}/logs/layouts.err"
        done
    done
fi

if has render; then
    for split in test val; do
        for name in ofp_c_iou ofp_drift ofp_hspP; do
            out="${D}/samples/${name}_${split}.npz"
            [[ -f "${out}" ]] || python3 -u scripts/eval/render_layouts.py --layouts "${D}/layouts/${name}_${split}.npz" \
                --out "${out}" > "${D}/logs/render_${name}_${split}.log" 2>&1
        done
    done
fi

if has eval; then
    python3 -u scripts/eval/run_eval_suite.py --config "${CONFIG}"
    for split in test val; do
        args=()
        for name in gn iso32_gt ofp_c_iou ofp_drift ofp_hspP; do
            npz=$(python3 -c "import json;m={x['name']:x for x in json.load(open('${CONFIG}'))['models']};print(m['${name}']['npz']['${split}'])")
            args+=(--model "${name}=${npz}")
        done
        python3 -u scripts/eval/phase0_layout_oracle_metrics.py --split "${split}" "${args[@]}" --baseline ofp_c_iou \
            --out "${D}/spall_metrics_${split}.json"
    done
fi
