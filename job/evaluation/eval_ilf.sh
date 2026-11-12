#!/bin/bash
#PBS -l select=1:ncpus=2:mem=15gb:gpu_id=1
#PBS -l place=shared
#PBS -o log/eval_ilf.out
#PBS -e log/eval_ilf.err
#PBS -N eval_ilf

# Learned-memory ILF runs at STEP against v12gtrefiner 6k / v12rfc 6k / v12rfm 9k (grav and g0.5-0), same seed /
# app code / grav layouts as eval_v12rfm.sh. Split in jobs so generation runs on several GPUs:
#   PART=gen,TAG=<run tag>   samples of one run (test + val) into results/eval_suite/ilf/<split>/samples/
#   PART=suite               the suite + irreversibility over every TAGS run (after the gen jobs)
#   qsub -l select=1:ncpus=2:mem=15gb:gpu_id=1 -v PART=gen,TAG=ILF_hpnew job/evaluation/eval_ilf.sh
#   qsub -l select=1:ncpus=2:mem=15gb:gpu_id=2 -v PART=gen,TAG=ILF_hpnew_aux job/evaluation/eval_ilf.sh
#   qsub -W depend=afterok:<gen1>:<gen2> -v PART=suite job/evaluation/eval_ilf.sh
cd "${PBS_O_WORKDIR:-$HOME/intership-project}"
source job/DamagePatternGeneration/common.sh
adm_env
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
set -e

STEP="${STEP:-009000}"
TAGS="${TAGS:-ILF_hpnew ILF_hpnew_aux}"
OUT=results/eval_suite/ilf
CFG=job/evaluation/configs/eval_suite_ilf.json
mkdir -p log "${OUT}/logs"
run_of() { ls -d results/DamagePatternGeneration/CFMADM/*_"$1" | sort | while read -r d; do
    [[ -f "${d}/checkpoint_step_${STEP}.pt" ]] && echo "${d}"; done | tail -1; }

case "${PART:-gen}" in
gen)
    RUN="$(run_of "${TAG:?set TAG}")"
    [[ -n "${RUN}" ]] || { echo "[eval_ilf] no *_${TAG} run with checkpoint_step_${STEP}.pt"; exit 1; }
    echo "[eval_ilf] ${TAG}: ${RUN} step ${STEP}"
    for split in test val; do
        npz="${OUT}/${split}/samples/${TAG}_${STEP}_grav.npz"
        mkdir -p "$(dirname "${npz}")"
        [[ -f "${npz}" ]] && continue
        python3 -u scripts/eval/generate_eval_samples.py --cfm_result_dir "${RUN}" \
            --cfm_checkpoint "${RUN}/checkpoint_step_${STEP}.pt" \
            --split "${split}" --k 8 --batch_size 32 --seed 0 --app_code mean \
            --layouts "results/phase1_fiber_layout/layouts/grav_hspP_${split}.npz" \
            --out "${npz}" > "${OUT}/logs/gen_${TAG}_${STEP}_${split}.log" 2>&1
    done
    ;;
suite)
    ENTRIES=(
        "v12gtrefiner_6k=results/eval_suite/v12gtrefiner/{s}/samples/v12gtrefiner_006000_grav.npz"
        "rfc_6k=results/eval_suite/v12rfc/{s}/samples/v12rfc_006000_grav.npz"
        "rfm_9k_grav=results/eval_suite/v12rfm/{s}/samples/v12rfm_009000_grav.npz"
        "rfm_9k_g0.5-0=results/eval_suite/v12rfm/{s}/samples/v12rfm_009000_g0.5-0.npz"
    )
    for tag in ${TAGS}; do
        ENTRIES+=("${tag}_$((10#${STEP} / 1000))k=${OUT}/{s}/samples/${tag}_${STEP}_grav.npz")
    done
    python3 - "${CFG}" "${OUT}" "${ENTRIES[@]}" <<'PY'
import json, sys
cfg, out, *entries = sys.argv[1:]
models = [{"name": e.split("=", 1)[0], "npz": {s: e.split("=", 1)[1].format(s=s) for s in ("test", "val")},
           "jpeg_quality": 75} for e in entries]
c = json.load(open("job/evaluation/configs/eval_suite_v12.template.json"))
c.update(out_dir=out, baseline=models[0]["name"], models=models)
json.dump(c, open(cfg, "w"), indent=2)
PY
    suite=()
    [[ -n "${QUICK:-}" ]] && suite+=(--quick)
    [[ -n "${FORCE:-}" ]] && suite+=(--force)
    python3 -u scripts/eval/run_eval_suite.py --config "${CFG}" "${suite[@]}"
    for split in test val; do
        args=()
        for e in "${ENTRIES[@]}"; do args+=(--model "${e%%=*}=$(echo "${e#*=}" | sed "s/{s}/${split}/")"); done
        python3 -u scripts/eval/irreversibility_curve.py --split "${split}" "${args[@]}" --thr 0.05 0.08 \
            --out "${OUT}/irreversibility_${split}.json" | tee "${OUT}/irreversibility_${split}.txt"
    done
    ;;
*) echo "[eval_ilf] PART must be gen or suite" >&2; exit 1 ;;
esac
