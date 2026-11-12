#!/bin/bash
#PBS -l select=1:ncpus=2:mem=15gb:gpu_id=1
#PBS -l place=shared
#PBS -o log/eval_v12rf.out
#PBS -e log/eval_v12rf.err
#PBS -N eval_v12rf

# Resampling Forcing (v12rf, v12rfc) and the time-gated memory (v12memg) (docs/redesign/phase1_ilf.md §10-11)
# against the original v12gtrefiner 6k, v12memc 6k and v12memc 6k with early-only guidance (0.5 -> 0 @ 0.5).
# Grav samples come from eval_v12.sh (NAME = run tag, VARIANTS=grav), same seed / app code / layouts.
# One suite table plus the irreversibility curves.
#   qsub -W depend=afterok:<gen jobs> job/evaluation/eval_v12rf.sh
#   qsub -v QUICK=1 job/evaluation/eval_v12rf.sh    # test split, KID only
#   qsub -v FORCE=1 ...                             # recompute steps that already match this model list
cd "${PBS_O_WORKDIR:-$HOME/intership-project}"
source job/DamagePatternGeneration/common.sh
adm_env
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
set -e

OUT=results/eval_suite/v12rf_compare
CFG=job/evaluation/configs/eval_suite_v12rf_compare.json
mkdir -p log "${OUT}"
ENTRIES=(
    "v12gtrefiner_6k=results/eval_suite/v12gtrefiner/{s}/samples/v12gtrefiner_006000_grav.npz"
    "memc_6k=results/eval_suite/v12memc/{s}/samples/v12memc_006000_grav.npz"
    "memc_6k_g0.5-0=results/eval_suite/memory_guidance/v12memc_006000/{s}/samples/v12memc_006000_w0.5-0t0.5.npz"
    "rf_6k=results/eval_suite/v12rf/{s}/samples/v12rf_006000_grav.npz"
    "rf_7k=results/eval_suite/v12rf/{s}/samples/v12rf_007000_grav.npz"
    "rfc_6k=results/eval_suite/v12rfc/{s}/samples/v12rfc_006000_grav.npz"
    "memg_6k=results/eval_suite/v12memg/{s}/samples/v12memg_006000_grav.npz"
    "memg_8k=results/eval_suite/v12memg/{s}/samples/v12memg_008000_grav.npz"
)
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
