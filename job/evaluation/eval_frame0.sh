#!/bin/bash
#PBS -l select=1:ncpus=2:mem=15gb:gpu_id=1
#PBS -l place=shared
#PBS -o log/eval_frame0.out
#PBS -e log/eval_frame0.err
#PBS -N evf0

# Refiner walks started from the real first photo (--set refiner_frame0_source=real) instead of the base draw;
# same seed / app code / grav layouts as the existing grav samples, which are the frame-0 = base controls.
#   PART=gen,TAGS="v12rfm"      samples into results/eval_suite/frame0_real/<split>/samples/<tag>_<step>_grav_f0real.npz
#   PART=suite                  suite + irreversibility: each model with base vs real frame 0
#   qsub -v PART=gen,TAGS="v12rfm v12rfc" job/evaluation/eval_frame0.sh
#   qsub -W depend=afterok:<gen jobs> -v PART=suite job/evaluation/eval_frame0.sh
cd "${PBS_O_WORKDIR:-$HOME/intership-project}"
source job/DamagePatternGeneration/common.sh
adm_env
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
set -e

R=results/DamagePatternGeneration/CFMADM
declare -A RUN=([v12gtrefiner]="${R}/2026-11-05_22-06-00_v12gtrefiner" [v12rfc]="${R}/2026-11-08_06-40-24_v12rfc"
                [v12rfm]="${R}/2026-11-08_14-30-55_v12rfm")
declare -A STEP=([v12gtrefiner]=006000 [v12rfc]=006000 [v12rfm]=009000)
TAGS="${TAGS:-v12gtrefiner v12rfc v12rfm}"
OUT=results/eval_suite/frame0_real
CFG=job/evaluation/configs/eval_suite_frame0_real.json
mkdir -p log "${OUT}/logs"

case "${PART:-gen}" in
gen)
    for tag in ${TAGS}; do
        run="${RUN[$tag]}"; step="${STEP[$tag]}"
        for split in test val; do
            npz="${OUT}/${split}/samples/${tag}_${step}_grav_f0real.npz"
            mkdir -p "$(dirname "${npz}")"
            [[ -f "${npz}" ]] && continue
            echo "[eval_frame0] ${tag} ${step} ${split}"
            python3 -u scripts/eval/generate_eval_samples.py --cfm_result_dir "${run}" \
                --cfm_checkpoint "${run}/checkpoint_step_${step}.pt" \
                --split "${split}" --k 8 --batch_size 32 --seed 0 --app_code mean \
                --layouts "results/phase1_fiber_layout/layouts/grav_hspP_${split}.npz" \
                --set refiner_frame0_source=real \
                --out "${npz}" > "${OUT}/logs/gen_${tag}_${step}_${split}.log" 2>&1
        done
    done
    ;;
suite)
    ENTRIES=()
    for tag in ${TAGS}; do
        step="${STEP[$tag]}"; k="$((10#${step} / 1000))k"
        ENTRIES+=("${tag}_${k}=results/eval_suite/${tag}/{s}/samples/${tag}_${step}_grav.npz")
        ENTRIES+=("${tag}_${k}_f0real=${OUT}/{s}/samples/${tag}_${step}_grav_f0real.npz")
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
    python3 -u scripts/eval/run_eval_suite.py --config "${CFG}"
    for split in test val; do
        args=()
        for e in "${ENTRIES[@]}"; do args+=(--model "${e%%=*}=$(echo "${e#*=}" | sed "s/{s}/${split}/")"); done
        python3 -u scripts/eval/irreversibility_curve.py --split "${split}" "${args[@]}" --thr 0.05 0.08 \
            --out "${OUT}/irreversibility_${split}.json" | tee "${OUT}/irreversibility_${split}.txt"
    done
    ;;
*) echo "[eval_frame0] PART must be gen or suite" >&2; exit 1 ;;
esac
