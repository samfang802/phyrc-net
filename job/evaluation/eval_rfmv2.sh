#!/bin/bash
#PBS -l select=1:ncpus=2:mem=15gb:gpu_id=2
#PBS -l place=shared
#PBS -o log/eval_rfmv2.out
#PBS -e log/eval_rfmv2.err
#PBS -N eval_rfmv2

# rfmv2 (docs/RFMV2_ARCHITECTURE.md) at STEP, sampled with several S-confidence values on the same seed / app code /
# grav layouts as eval_v12rfm.sh. c0.1 is the trained default (refiner_state_conf_value); a higher c tells the
# model S is more trustworthy, which should keep more of S's cracks (keep@6 in the irreversibility table).
# Against v12gtrefiner 6k, v12rfc 6k and v12rfm 9k. Suite plus irreversibility / step births.
#   qsub job/evaluation/eval_rfmv2.sh
#   qsub -v CONFS="0.1 0.5",STEP=009000 job/evaluation/eval_rfmv2.sh
# NAME=rfmv2e: Euler-resample retrain (newest *_rfmv2 run) into its own results/eval_suite/rfmv2e, with the Heun
# run's 6k c0.1 added to the comparison.
#   qsub -v NAME=rfmv2e,STEP=009000 job/evaluation/eval_rfmv2.sh
# NAME=rfmv2x: crack-erasure retrain (RUN=<its dir>), rfmv2e 9k c0.1 added to the comparison.
# NAME=rfmv2b: own-rollout bank retrain (RUN=<its dir>), rfmv2e and rfmv2x 9k c0.1 added to the comparison.
# OUT_TAG: write into results/eval_suite/<OUT_TAG> instead of <NAME> (e.g. a second STEP of the same run).
cd "${PBS_O_WORKDIR:-$HOME/intership-project}"
source job/DamagePatternGeneration/common.sh
adm_env
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1 OMP_NUM_THREADS=${NCPUS:-2} MKL_NUM_THREADS=${NCPUS:-2}
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
set -e

NAME="${NAME:-rfmv2}"
STEP="${STEP:-006000}"
CONFS="${CONFS:-0.1 0.3 0.5 0.7}"
RUN="${RUN:-$(ls -d results/DamagePatternGeneration/CFMADM/*_rfmv2 | sort | tail -1)}"
OUT=results/eval_suite/${OUT_TAG:-${NAME}}
CFG=job/evaluation/configs/eval_suite_${OUT_TAG:-${NAME}}.json
ckpt="${RUN}/checkpoint_step_${STEP}.pt"
[[ -f "${ckpt}" ]] || { echo "[${NAME}] missing ${ckpt}"; exit 1; }
mkdir -p log "${OUT}/logs"
echo "[${NAME}] ${RUN} step ${STEP}, confs ${CONFS}"

ENTRIES=(
    "v12gtrefiner_6k=results/eval_suite/v12gtrefiner/{s}/samples/v12gtrefiner_006000_grav.npz"
    "rfc_6k=results/eval_suite/v12rfc/{s}/samples/v12rfc_006000_grav.npz"
    "rfm_9k=results/eval_suite/v12rfm/{s}/samples/v12rfm_009000_grav.npz"
)
[[ "${NAME}" != rfmv2 ]] && ENTRIES+=("rfmv2heun_6k_c0.1=results/eval_suite/rfmv2/{s}/samples/rfmv2_006000_c0.1.npz")
[[ "${NAME}" == rfmv2x || "${NAME}" == rfmv2b ]] && ENTRIES+=("rfmv2e_9k_c0.1=results/eval_suite/rfmv2e/{s}/samples/rfmv2e_009000_c0.1.npz")
[[ "${NAME}" == rfmv2b ]] && ENTRIES+=("rfmv2x_9k_c0.1=results/eval_suite/rfmv2x/{s}/samples/rfmv2x_009000_c0.1.npz")
k="$((10#${STEP} / 1000))k"
# One process: both splits, every conf (dataset / model built once per split, the walks of a split batched).
variants=()
for c in ${CONFS}; do
    variants+=(--variant "c${c}:refiner_state_conf_value=${c}")
    ENTRIES+=("${NAME}_${k}_c${c}=${OUT}/{s}/samples/${NAME}_${STEP}_c${c}.npz")
done
mkdir -p "${OUT}/test/samples" "${OUT}/val/samples"
python3 -u scripts/eval/generate_eval_samples.py --cfm_result_dir "${RUN}" --cfm_checkpoint "${ckpt}" \
    --split test,val --k 8 --batch_size 32 --seed 0 --app_code mean \
    --layouts "results/phase1_fiber_layout/layouts/grav_hspP_{split}.npz" \
    "${variants[@]}" --skip_existing \
    --out "${OUT}/{split}/samples/${NAME}_${STEP}_{name}.npz" > "${OUT}/logs/gen_${STEP}.log" 2>&1

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
