#!/bin/bash
#PBS -l select=1:ncpus=2:mem=15gb:gpu_id=3
#PBS -l place=shared
#PBS -o log/memory_guidance.out
#PBS -e log/memory_guidance.err
#PBS -N mem_guidance

# Memory guidance (docs/redesign/phase1_ilf.md §9.2): one ILF-lite checkpoint sampled with its own memory and
#   v = v_zero + w (v_mem - v_zero)        (--set refiner_memory_guidance=w)
# where v_zero reads the dropped (all-zero) memory; w < 1 trusts the memory less, w > 1 more. w = 1 is the normal
# sampler (eval_v12.sh grav samples), w = 0 the zero memory (ilf_oracle.sh samples); both are reused.
# Same seed / app code / grav layouts as eval_v12.sh.
#   qsub -v NAME=v12memc,STEP=006000,WEIGHTS="0.25 0.5 0.75" job/evaluation/memory_guidance.sh
#   qsub -v NAME=v12memcr,STEP=009000,WEIGHTS="1.5 2" job/evaluation/memory_guidance.sh
# Noise-level schedule (t = 0 noise, 1 data): SCHEDULES="early:late:t ..." samples w = early for t < t, late after,
# e.g. SCHEDULES="0:1:0.5 0:0.5:0.5 0.5:0:0.5" WEIGHTS="" (tag w<early>-<late>t<t>).
cd "${PBS_O_WORKDIR:-$HOME/intership-project}"
source job/DamagePatternGeneration/common.sh
adm_env
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
set -e

NAME="${NAME:-v12memc}"
STEP="${STEP:-006000}"
WEIGHTS="${WEIGHTS-0.25 0.5 0.75}"
SCHEDULES="${SCHEDULES:-}"
# tag -> --set args
TAGS=(); SETS=()
for w in ${WEIGHTS}; do TAGS+=("${w}"); SETS+=("refiner_memory_guidance=${w}"); done
for sc in ${SCHEDULES}; do
    IFS=: read -r we wl ts <<< "${sc}"
    TAGS+=("${we}-${wl}t${ts}")
    SETS+=("refiner_memory_guidance=${we} refiner_memory_guidance_late=${wl} refiner_memory_guidance_t=${ts}")
done
RUN="${RUN:-$(ls -d results/DamagePatternGeneration/CFMADM/*_${NAME} | sort | tail -1)}"
OUT=results/eval_suite/memory_guidance/${NAME}_${STEP}
CFG=job/evaluation/configs/eval_suite_memory_guidance_${NAME}_${STEP}.json
SELF=results/eval_suite/${NAME}/{s}/samples/${NAME}_${STEP}_grav.npz
ZERO=results/eval_suite/ilf_oracle/${NAME}_${STEP}/{s}/samples/${NAME}_${STEP}_zero.npz
mkdir -p log "${OUT}/logs"
echo "[mem_guidance] ${RUN} @ ${STEP} w = ${TAGS[*]}"
for split in test val; do
    for f in "${SELF}" "${ZERO}"; do
        [[ -f "${f//\{s\}/${split}}" ]] || { echo "[mem_guidance] missing ${f//\{s\}/${split}}"; exit 1; }
    done
done

for split in test val; do
    mkdir -p "${OUT}/${split}/samples"
    for i in "${!TAGS[@]}"; do
        w="${TAGS[$i]}"
        npz="${OUT}/${split}/samples/${NAME}_${STEP}_w${w}.npz"
        [[ -f "${npz}" ]] && continue
        python3 -u scripts/eval/generate_eval_samples.py --cfm_result_dir "${RUN}" \
            --cfm_checkpoint "${RUN}/checkpoint_step_${STEP}.pt" --split "${split}" --k 8 --batch_size 32 --seed 0 \
            --app_code mean --layouts "results/phase1_fiber_layout/layouts/grav_hspP_${split}.npz" \
            --set ${SETS[$i]} --out "${npz}" > "${OUT}/logs/gen_w${w}_${split}.log" 2>&1
    done
done

python3 - "${CFG}" "${OUT}" "${NAME}" "${STEP}" "${SELF}" "${ZERO}" "${TAGS[@]}" <<'PY'
import json, sys
cfg, out, name, step, self_path, zero_path, *weights = sys.argv[1:]
S = ("test", "val")
def entry(n, path):
    return {"name": n, "npz": {s: path.format(s=s) for s in S}, "jpeg_quality": 75}
models = [entry("v12gtrefiner_6k", "results/eval_suite/v12gtrefiner/{s}/samples/v12gtrefiner_006000_grav.npz"),
          entry(f"{name}_w0", zero_path), entry(f"{name}_w1", self_path)]
models += [entry(f"{name}_w{w}", f"{out}/{{s}}/samples/{name}_{step}_w{w}.npz") for w in weights]
c = json.load(open("job/evaluation/configs/eval_suite_v12.template.json"))
c.update(out_dir=out, baseline=models[0]["name"], models=models)
json.dump(c, open(cfg, "w"), indent=2)
PY
suite=()
[[ -n "${QUICK:-}" ]] && suite+=(--quick)
[[ -n "${FORCE:-}" ]] && suite+=(--force)
python3 -u scripts/eval/run_eval_suite.py --config "${CFG}" "${suite[@]}"

for split in test val; do
    args=(--model "v12gtrefiner_6k=results/eval_suite/v12gtrefiner/${split}/samples/v12gtrefiner_006000_grav.npz"
          --model "${NAME}_w0=${ZERO//\{s\}/${split}}" --model "${NAME}_w1=${SELF//\{s\}/${split}}")
    for w in "${TAGS[@]}"; do args+=(--model "${NAME}_w${w}=${OUT}/${split}/samples/${NAME}_${STEP}_w${w}.npz"); done
    python3 -u scripts/eval/irreversibility_curve.py --split "${split}" "${args[@]}" --thr 0.05 0.08 \
        --out "${OUT}/irreversibility_${split}.json"
done
