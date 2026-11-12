#!/bin/bash
#PBS -l select=1:ncpus=2:mem=15gb:gpu_id=3
#PBS -l place=shared
#PBS -o log/ilf_oracle.out
#PBS -e log/ilf_oracle.err
#PBS -N ilf_oracle

# Ceiling of the ILF-lite memory (docs/redesign/phase1_ilf.md §8): one ILF-lite checkpoint sampled with its memory
#   self    accumulated from its own generated frames (the normal sampler; samples from eval_v12.sh)
#   oracle  accumulated from the real frames with the same rule (what a perfect memory could give)
#   zero    always empty (the memory-dropped value)
#   swap    accumulated from the real frames of the other face of the same column (real memory, wrong positions)
# SOURCES picks the given memories (default "oracle zero"); existing sample files are reused.
# grav layouts, same seed / app code as eval_v12.sh, against the original v12gtrefiner 6k and its controls.
# The oracle reads the real crack positions, so crack-F1 style columns are not comparable for it.
#   qsub job/evaluation/ilf_oracle.sh                                   (v12memc 6k)
#   qsub -v NAME=v12memo,STEP=006000,CONTROLS="..." job/evaluation/ilf_oracle.sh
#   qsub -v SOURCES="oracle zero swap" job/evaluation/ilf_oracle.sh      (adds the wrong-trajectory memory)
cd "${PBS_O_WORKDIR:-$HOME/intership-project}"
source job/DamagePatternGeneration/common.sh
adm_env
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
set -e

NAME="${NAME:-v12memc}"
STEP="${STEP:-006000}"
SOURCES="${SOURCES:-oracle zero}"
RUN="${RUN:-$(ls -d results/DamagePatternGeneration/CFMADM/*_${NAME} | sort | tail -1)}"
# name=sample prefix under results/eval_suite/<dir>/<split>/samples/
CONTROLS="${CONTROLS:-v12gtrefiner_6k=v12gtrefiner/v12gtrefiner_006000_grav fixA_6k=v12gtrefiner_fixA/v12gtrefiner_fixA_006000_grav fixB_6k=v12gtrefiner_fixB/v12gtrefiner_fixB_006000_grav}"
OUT=results/eval_suite/ilf_oracle/${NAME}_${STEP}
CFG=job/evaluation/configs/eval_suite_ilf_oracle_${NAME}_${STEP}.json
mkdir -p log "${OUT}/logs"
echo "[ilf_oracle] ${RUN} @ ${STEP}"

for split in test val; do
    mkdir -p "${OUT}/${split}/samples"
    for src in ${SOURCES}; do
        npz="${OUT}/${split}/samples/${NAME}_${STEP}_${src}.npz"
        [[ -f "${npz}" ]] && continue
        python3 -u scripts/eval/generate_eval_samples.py --cfm_result_dir "${RUN}" \
            --cfm_checkpoint "${RUN}/checkpoint_step_${STEP}.pt" --split "${split}" --k 8 --batch_size 32 --seed 0 \
            --app_code mean --layouts "results/phase1_fiber_layout/layouts/grav_hspP_${split}.npz" \
            --set refiner_memory_source=${src} --out "${npz}" > "${OUT}/logs/gen_${src}_${split}.log" 2>&1
    done
done

python3 - "${CFG}" "${OUT}" "${NAME}" "${STEP}" "${SOURCES}" ${CONTROLS} <<'PY'
import json, sys
cfg, out, name, step, sources, *controls = sys.argv[1:]
S = ("test", "val")
def entry(n, path):
    return {"name": n, "npz": {s: path.format(s=s) for s in S}, "jpeg_quality": 75}
models = [entry(n, f"results/eval_suite/{p.split('/')[0]}/{{s}}/samples/{p.split('/')[1]}.npz")
          for n, p in (c.split("=", 1) for c in controls)]
models.append(entry(f"{name}_self", f"results/eval_suite/{name}/{{s}}/samples/{name}_{step}_grav.npz"))
for src in sources.split():
    models.append(entry(f"{name}_{src}", f"{out}/{{s}}/samples/{name}_{step}_{src}.npz"))
c = json.load(open("job/evaluation/configs/eval_suite_v12.template.json"))
c.update(out_dir=out, baseline=models[0]["name"], models=models)
json.dump(c, open(cfg, "w"), indent=2)
PY
# A changed SOURCES / CONTROLS list reruns the metrics; an unchanged one is skipped.
suite=()
[[ -n "${QUICK:-}" ]] && suite+=(--quick)
[[ -n "${FORCE:-}" ]] && suite+=(--force)
python3 -u scripts/eval/run_eval_suite.py --config "${CFG}" "${suite[@]}"

for split in test val; do
    args=()
    for c in ${CONTROLS}; do
        n="${c%%=*}"; p="${c#*=}"
        args+=(--model "${n}=results/eval_suite/${p%%/*}/${split}/samples/${p#*/}.npz")
    done
    args+=(--model "${NAME}_self=results/eval_suite/${NAME}/${split}/samples/${NAME}_${STEP}_grav.npz")
    for src in ${SOURCES}; do args+=(--model "${NAME}_${src}=${OUT}/${split}/samples/${NAME}_${STEP}_${src}.npz"); done
    python3 -u scripts/eval/irreversibility_curve.py --split "${split}" "${args[@]}" --thr 0.05 0.08 \
        --out "${OUT}/irreversibility_${split}.json"
done
