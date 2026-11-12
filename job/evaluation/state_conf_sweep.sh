#!/bin/bash
#PBS -l select=1:ncpus=2:mem=15gb:gpu_id=2
#PBS -l place=shared
#PBS -o log/state_conf_sweep.out
#PBS -e log/state_conf_sweep.err
#PBS -N state_conf_sweep

# S confidence at sampling (docs/redesign/phase1_ilf.md §11.2): a refiner trained with --refiner_state_conf
# (v12rfc: t_s for resampled S, 1 for real frames) sampled with a constant --set refiner_state_conf_value=c on every
# hop. c = 1 (the eval_v12.sh samples) tells it S is a real frame; lower c says S is a noisier draw.
# Same seed / app code / grav layouts as eval_v12.sh; the c = 1 samples are reused as the reference.
#   qsub job/evaluation/state_conf_sweep.sh
#   qsub -v NAME=v12rfm,STEP=006000,CONFS="0.5 0.7" job/evaluation/state_conf_sweep.sh
cd "${PBS_O_WORKDIR:-$HOME/intership-project}"
source job/DamagePatternGeneration/common.sh
adm_env
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
set -e

NAME="${NAME:-v12rfc}"
STEP="${STEP:-006000}"
CONFS="${CONFS:-0.3 0.5 0.7}"
EXTRA_SET="${EXTRA_SET:-}"   # more --set overrides for every conf, e.g. "refiner_memory_guidance=0.5 refiner_memory_guidance_late=0"
RUN="${RUN:-$(ls -d results/DamagePatternGeneration/CFMADM/*_${NAME} | sort | tail -1)}"
OUT=results/eval_suite/state_conf/${NAME}_${STEP}
CFG=job/evaluation/configs/eval_suite_state_conf_${NAME}_${STEP}.json
mkdir -p log "${OUT}/logs"
echo "[state_conf] ${RUN} @ ${STEP}, confs ${CONFS}"

ENTRIES=("v12gtrefiner_6k=results/eval_suite/v12gtrefiner/{s}/samples/v12gtrefiner_006000_grav.npz"
         "${NAME}_c1=results/eval_suite/${NAME}/{s}/samples/${NAME}_${STEP}_grav.npz")
for c in ${CONFS}; do
    for split in test val; do
        npz="${OUT}/${split}/samples/${NAME}_${STEP}_c${c}.npz"
        mkdir -p "$(dirname "${npz}")"
        [[ -f "${npz}" ]] && continue
        python3 -u scripts/eval/generate_eval_samples.py --cfm_result_dir "${RUN}" \
            --cfm_checkpoint "${RUN}/checkpoint_step_${STEP}.pt" --split "${split}" --k 8 --batch_size 32 --seed 0 \
            --app_code mean --layouts "results/phase1_fiber_layout/layouts/grav_hspP_${split}.npz" \
            --set "refiner_state_conf_value=${c}" ${EXTRA_SET} --out "${npz}" \
            > "${OUT}/logs/gen_c${c}_${split}.log" 2>&1
    done
    ENTRIES+=("${NAME}_c${c}=${OUT}/{s}/samples/${NAME}_${STEP}_c${c}.npz")
done

python3 - "${CFG}" "${OUT}" "${ENTRIES[@]}" <<'PY'
import json, sys
cfg, out, *entries = sys.argv[1:]
S = ("test", "val")
models = []
for e in entries:
    n, path = e.split("=", 1)
    models.append({"name": n, "npz": {s: path.format(s=s) for s in S}, "jpeg_quality": 75})
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
        --out "${OUT}/irreversibility_${split}.json"
done
