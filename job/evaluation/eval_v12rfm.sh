#!/bin/bash
#PBS -l select=1:ncpus=2:mem=15gb:gpu_id=3
#PBS -l place=shared
#PBS -o log/eval_v12rfm.out
#PBS -e log/eval_v12rfm.err
#PBS -N eval_v12rfm

# v12rfm = Resampling Forcing + S confidence + ILF-lite memory (docs/redesign/phase1_ilf.md §11.2) at STEPS,
# each sampled two ways on the same seed / app code / grav layouts as eval_v12.sh:
#   grav     plain memory (w = 1)
#   g0.5-0   memory guidance 0.5 for t < 0.5, 0 after (the best v12memc schedule, §9.2)
# against v12gtrefiner 6k, v12memc 6k (+ g0.5-0) and v12rfc 6k. Suite plus irreversibility / step births.
#   qsub job/evaluation/eval_v12rfm.sh
#   qsub -v STEPS="006000 009000" job/evaluation/eval_v12rfm.sh
cd "${PBS_O_WORKDIR:-$HOME/intership-project}"
source job/DamagePatternGeneration/common.sh
adm_env
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
set -e

NAME=v12rfm
STEPS="${STEPS:-006000 009000}"
RUN="${RUN:-$(ls -d results/DamagePatternGeneration/CFMADM/*_${NAME} | sort | tail -1)}"
OUT=results/eval_suite/${NAME}
CFG=job/evaluation/configs/eval_suite_${NAME}.json
mkdir -p log "${OUT}/logs"
echo "[${NAME}] ${RUN} steps ${STEPS}"

declare -A SET=([grav]="" [g0.5-0]="refiner_memory_guidance=0.5 refiner_memory_guidance_late=0 refiner_memory_guidance_t=0.5")
ENTRIES=(
    "v12gtrefiner_6k=results/eval_suite/v12gtrefiner/{s}/samples/v12gtrefiner_006000_grav.npz"
    "memc_6k=results/eval_suite/v12memc/{s}/samples/v12memc_006000_grav.npz"
    "memc_6k_g0.5-0=results/eval_suite/memory_guidance/v12memc_006000/{s}/samples/v12memc_006000_w0.5-0t0.5.npz"
    "rfc_6k=results/eval_suite/v12rfc/{s}/samples/v12rfc_006000_grav.npz"
)
for step in ${STEPS}; do
    ckpt="${RUN}/checkpoint_step_${step}.pt"
    [[ -f "${ckpt}" ]] || { echo "[${NAME}] missing ${ckpt}"; exit 1; }
    for v in grav g0.5-0; do
        for split in test val; do
            npz="${OUT}/${split}/samples/${NAME}_${step}_${v}.npz"
            mkdir -p "$(dirname "${npz}")"
            [[ -f "${npz}" ]] && continue
            python3 -u scripts/eval/generate_eval_samples.py --cfm_result_dir "${RUN}" --cfm_checkpoint "${ckpt}" \
                --split "${split}" --k 8 --batch_size 32 --seed 0 --app_code mean \
                --layouts "results/phase1_fiber_layout/layouts/grav_hspP_${split}.npz" \
                ${SET[$v]:+--set ${SET[$v]}} --out "${npz}" > "${OUT}/logs/gen_${step}_${v}_${split}.log" 2>&1
        done
        ENTRIES+=("rfm_${step:0:3}_${v}=${OUT}/{s}/samples/${NAME}_${step}_${v}.npz")
    done
done

python3 - "${CFG}" "${OUT}" "${ENTRIES[@]}" <<'PY'
import json, sys
cfg, out, *entries = sys.argv[1:]
S = ("test", "val")
models = []
for e in entries:
    n, path = e.split("=", 1)
    n = n.replace("rfm_006", "rfm_6k").replace("rfm_009", "rfm_9k")
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
    for e in "${ENTRIES[@]}"; do
        n="${e%%=*}"; n="${n/rfm_006/rfm_6k}"; n="${n/rfm_009/rfm_9k}"
        args+=(--model "${n}=$(echo "${e#*=}" | sed "s/{s}/${split}/")")
    done
    python3 -u scripts/eval/irreversibility_curve.py --split "${split}" "${args[@]}" --thr 0.05 0.08 \
        --out "${OUT}/irreversibility_${split}.json" | tee "${OUT}/irreversibility_${split}.txt"
done
