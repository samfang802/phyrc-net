#!/bin/bash
#PBS -l select=1:ncpus=2:mem=15gb:gpu_id=2
#PBS -l place=shared
#PBS -o log/frame0_hop.out
#PBS -e log/frame0_hop.err
#PBS -N frame0_hop

# Frame 0 refined (docs/redesign/phase1_ilf.md §10): the frozen base draws frame 0, then the refiner does one hop
# on it (S = the base draw, frame 0's own conditions, zero delta g): --set refiner_frame0_hop=1.
# Per model NAME:STEP, the normal grav samples (eval_v12.sh) are reused as the no-hop reference.
# Same seed / app code / grav layouts as eval_v12.sh.
#   qsub job/evaluation/frame0_hop.sh
#   qsub -v MODELS="v12rf:006000" job/evaluation/frame0_hop.sh
#   one model into its own folder (an unchanged model list is not rescored; FORCE=1 recomputes it):
#     qsub -v MODELS="v12rfm:009000",OUT=results/eval_suite/frame0_hop_v12rfm job/evaluation/frame0_hop.sh
#   qsub -v QUICK=1 ...    # test split, KID only
cd "${PBS_O_WORKDIR:-$HOME/intership-project}"
source job/DamagePatternGeneration/common.sh
adm_env
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
set -e

MODELS="${MODELS:-v12gtrefiner:006000 v12memc:006000}"
OUT="${OUT:-results/eval_suite/frame0_hop}"
CFG="job/evaluation/configs/eval_suite_$(basename "${OUT}").json"
HOPS=results/eval_suite/frame0_hop  # hop samples are always generated / reused here
mkdir -p log "${OUT}/logs" "${HOPS}/logs"

ENTRIES=()
for m in ${MODELS}; do
    name="${m%%:*}"; step="${m##*:}"
    run="$(ls -d results/DamagePatternGeneration/CFMADM/*_${name} | sort | tail -1)"
    ref="results/eval_suite/${name}/{s}/samples/${name}_${step}_grav.npz"
    hop="${HOPS}/{s}/samples/${name}_${step}_hop.npz"
    echo "[frame0_hop] ${run} @ ${step}"
    for split in test val; do
        [[ -f "${ref//\{s\}/${split}}" ]] || { echo "[frame0_hop] missing ${ref//\{s\}/${split}}"; exit 1; }
        npz="${hop//\{s\}/${split}}"
        mkdir -p "$(dirname "${npz}")"
        [[ -f "${npz}" ]] && continue
        python3 -u scripts/eval/generate_eval_samples.py --cfm_result_dir "${run}" \
            --cfm_checkpoint "${run}/checkpoint_step_${step}.pt" --split "${split}" --k 8 --batch_size 32 --seed 0 \
            --app_code mean --layouts "results/phase1_fiber_layout/layouts/grav_hspP_${split}.npz" \
            --set refiner_frame0_hop=1 --out "${npz}" > "${HOPS}/logs/gen_${name}_${step}_${split}.log" 2>&1
    done
    ENTRIES+=("${name}_${step/#00/}=${ref}" "${name}_${step/#00/}_hop=${hop}")
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
