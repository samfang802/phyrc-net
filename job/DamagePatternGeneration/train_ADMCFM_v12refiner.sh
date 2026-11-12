#!/bin/bash
#PBS -l select=1:ncpus=2:mem=15gb:gpu_id=2
#PBS -l place=shared
#PBS -o log/train_v12refiner.out
#PBS -e log/train_v12refiner.err
#PBS -N CFM_rfmv2

# v12 refiner on the v12 single-frame base. RECIPE picks the variant (docs/redesign/phase1_ilf.md);
# anything unset below can still be overridde with qsub -v.
#   refiner  base = newest *_v12 run, lowest periodic val KID at step >= MIN_STEP (default 10000).
#            qsub -v LAYOUT_CACHE=<cache> job/DamagePatternGeneration/train_ADMCFM_v12refiner.sh
#            hand-picked base: -v BASE_CKPT=<run>/checkpoint_step_NNNNNN.pt
#            gravity template: -v BASE_TAG=v12basis,RUN_TAG=v12basisrefiner,LAYOUT_CACHE=<grav cache>
#   hspP     predicted layouts, base = v11_app.  -v RECIPE=hspP
#   mem      ILF-lite memory on v12gt 10k.       -v RECIPE=mem
#            early-only memory: also -v REFINER_MEMORY_TMAX=0.5,RUN_TAG=v12memg
#   memr     fine-tune v12memc 6k with recycled errors.  -v RECIPE=memr
#   nbr      bidirectional neighbors, no base at sampling.  -v RECIPE=nbr
#   rf       resampling forcing.  -v RECIPE=rf
#            confidence channel: also -v REFINER_STATE_CONF=True,RUN_TAG=v12rfc
#            rfc + memory: also -v REFINER_STATE_CONF=True,REFINER_MEMORY=True,RUN_TAG=v12rfm
#            full ILF, no Kaiser gate (learned encoder E + frozen trunk; phase1_ilf.md §2):
#              -v RECIPE=rf,REFINER_STATE_CONF=True,REFINER_MEMORY=True,REFINER_MEMORY_ENCODER=True,FREEZE_TRUNK=True,REFINER_MEMORY_GATE=none,RUN_TAG=ILFnoK
#              Kaiser variant: REFINER_MEMORY_GATE=kaiser,RUN_TAG=ILFK
#              trunk trained (= v12rfm + learned E): drop FREEZE_TRUNK, RUN_TAG=ILFnoK_tr
#              E input: REFINER_MEMORY_INPUT=raw|hp|delta|hpnew (hpnew = thin dark structure new since frame 0)
#              decoder aux loss (field -> accumulated darkening): REFINER_MEMORY_AUX_WEIGHT=0.1
#                e.g. -v ...,REFINER_MEMORY_INPUT=hpnew,REFINER_MEMORY_AUX_WEIGHT=0.1,RUN_TAG=ILF_hpnew_aux
#   rfmv2    window self-resampling + rule memory written from the resampled frames (docs/RFMV2_ARCHITECTURE.md).
#            -v RECIPE=rfmv2   (resampling hops: Euler; the first run's Heun hops left start noise in S, §3.3)
#            crack erasure in S (Ring-Forcing A-D-R, §9): -v RECIPE=rfmv2,REFINER_ERASE_PROB=0.3,RUN_TAG=rfmv2x
#            own-rollout bank pasted into S + its errors in the memory (src/model/cfm/rollout_bank.py):
#              -v RECIPE=rfmv2,RUN_TAG=rfmv2b,REFINER_ERASE_SOURCE=bank,REFINER_ERASE_PROB=0.3,REFINER_BANK_MEM_ERR_PROB=0.3
#   smoke: add STOP_STEP=20,KID_EVERY_N_STEPS=20,VAL_CHECK_INTERVAL=10,RUN_TAG=<name>_smoke
cd "${PBS_O_WORKDIR:-$HOME/intership-project}"
source job/DamagePatternGeneration/common.sh
adm_env

GT10="results/DamagePatternGeneration/CFMADM/2026-11-05_12-51-53_v12gt/checkpoint_step_010000.pt"
GT_CACHE="results/phase1_fiber_layout/train_cache/gtiso32_train_gravval_sdf128.npz"
RECIPE="${RECIPE:-refiner}"
case "${RECIPE}" in
refiner) ;;
hspP)
    CACHE=results/phase1_fiber_layout/train_cache/hspP_loco_sdf128.npz
    [[ -f "${CACHE}" ]] || { echo "missing ${CACHE}: python scripts/eval/build_train_layout_cache.py"; exit 1; }
    BASE_DIR=results/DamagePatternGeneration/CFMADM/2026-10-26_00-37-49_v11_app
    export V11=True APP_CODE_DIM=8 \
        REFINER_BASE_DIR="${BASE_DIR}" \
        REFINER_BASE_CKPT="${BASE_DIR}/Opensees Based damage Patterns Generation/q1sz4bss/checkpoints/last.ckpt" \
        RUN_TAG="${RUN_TAG:-v12refiner_hspP}" MAX_STEPS=15000 STOP_STEP="${STOP_STEP:-9000}" \
        LAYOUT_COND=True LAYOUT_ZERO_INIT=False LAYOUT_DROP_PROB="${LAYOUT_DROP_PROB:-0.1}" LAYOUT_CACHE="${CACHE}"
    export NUM_WORKERS="${NUM_WORKERS:-1}"
    exec bash job/DamagePatternGeneration/train_ADMCFM_refiner_residual.sh
    ;;
mem)
    export BASE_CKPT="${BASE_CKPT:-${GT10}}" LAYOUT_CACHE="${LAYOUT_CACHE:-${GT_CACHE}}" \
        REFINER_STATE_CACHE="${REFINER_STATE_CACHE:-True}" \
        REFINER_MEMORY="${REFINER_MEMORY:-True}" REFINER_MEMORY_SIGMA="${REFINER_MEMORY_SIGMA:-2.0}" \
        REFINER_MEMORY_DROP_PROB="${REFINER_MEMORY_DROP_PROB:-0.1}" \
        RUN_TAG="${RUN_TAG:-v12memc}" STOP_STEP="${STOP_STEP:-9000}"
    ;;
memr)
    export VELOCITY_INIT="${VELOCITY_INIT:-results/DamagePatternGeneration/CFMADM/2026-11-07_05-14-25_v12memc/checkpoint_step_006000.pt}" \
        REFINER_MEMORY_RECYCLE="${REFINER_MEMORY_RECYCLE:-True}" \
        REFINER_MEMORY_RECYCLE_CLEAN="${REFINER_MEMORY_RECYCLE_CLEAN:-0.3}" \
        REFINER_MEMORY_RECYCLE_TRANSPLANT="${REFINER_MEMORY_RECYCLE_TRANSPLANT:-0.5}" \
        REFINER_MEMORY_RECYCLE_N="${REFINER_MEMORY_RECYCLE_N:-8}" \
        REFINER_MEMORY_RECYCLE_TMIN="${REFINER_MEMORY_RECYCLE_TMIN:-0.5}" \
        KID_EVERY_N_STEPS="${KID_EVERY_N_STEPS:-1000}" \
        BASE_CKPT="${BASE_CKPT:-${GT10}}" LAYOUT_CACHE="${LAYOUT_CACHE:-${GT_CACHE}}" \
        REFINER_STATE_CACHE="${REFINER_STATE_CACHE:-True}" \
        REFINER_MEMORY="${REFINER_MEMORY:-True}" REFINER_MEMORY_SIGMA="${REFINER_MEMORY_SIGMA:-2.0}" \
        REFINER_MEMORY_DROP_PROB="${REFINER_MEMORY_DROP_PROB:-0.1}" \
        RUN_TAG="${RUN_TAG:-v12memr}" STOP_STEP="${STOP_STEP:-6000}"
    ;;
nbr)
    export BASE_CKPT="${BASE_CKPT:-${GT10}}" LAYOUT_CACHE="${LAYOUT_CACHE:-${GT_CACHE}}" \
        REFINER_NEIGHBORS=True REFINER_JOINT_MODE="${REFINER_JOINT_MODE:-sweep}" \
        REFINER_CHAIN_PROB=0 REFINER_ROLL_PROB_IMPLICIT=0 \
        RUN_TAG="${RUN_TAG:-v12nbrA2}" STOP_STEP="${STOP_STEP:-9000}"
    ;;
rf)
    export BASE_CKPT="${BASE_CKPT:-${GT10}}" LAYOUT_CACHE="${LAYOUT_CACHE:-${GT_CACHE}}" \
        REFINER_STATE_SOURCE=resample REFINER_CHAIN_PROB=0 REFINER_ROLL_PROB_IMPLICIT=0 \
        REFINER_RESAMPLE_MU="${REFINER_RESAMPLE_MU:-0.5}" REFINER_RESAMPLE_SIGMA="${REFINER_RESAMPLE_SIGMA:-1.0}" \
        REFINER_RESAMPLE_STEPS="${REFINER_RESAMPLE_STEPS:-8}" \
        REFINER_RESAMPLE_WARMUP="${REFINER_RESAMPLE_WARMUP:-1000}" REFINER_RESAMPLE_RAMP="${REFINER_RESAMPLE_RAMP:-1000}" \
        REFINER_STATE_CONF="${REFINER_STATE_CONF:-False}" \
        KID_EVERY_N_STEPS="${KID_EVERY_N_STEPS:-1000}" \
        RUN_TAG="${RUN_TAG:-v12rf}" STOP_STEP="${STOP_STEP:-9000}"
    ;;
rfmv2)
    export BASE_CKPT="${BASE_CKPT:-${GT10}}" LAYOUT_CACHE="${LAYOUT_CACHE:-${GT_CACHE}}" \
        REFINER_STATE_SOURCE=resample REFINER_CHAIN_PROB=0 REFINER_ROLL_PROB_IMPLICIT=0 \
        REFINER_RESAMPLE_MU="${REFINER_RESAMPLE_MU:--0.5}" REFINER_RESAMPLE_SIGMA="${REFINER_RESAMPLE_SIGMA:-1.2}" \
        REFINER_RESAMPLE_STEPS="${REFINER_RESAMPLE_STEPS:-8}" REFINER_RESAMPLE_SOLVER="${REFINER_RESAMPLE_SOLVER:-euler}" \
        REFINER_RESAMPLE_WARMUP="${REFINER_RESAMPLE_WARMUP:-1000}" REFINER_RESAMPLE_RAMP="${REFINER_RESAMPLE_RAMP:-1000}" \
        REFINER_STATE_CONF=True REFINER_STATE_CONF_HOPS=True REFINER_STATE_CONF_VALUE="${REFINER_STATE_CONF_VALUE:-0.1}" \
        REFINER_WINDOW="${REFINER_WINDOW:-8}" REFINER_FALSE_MEM_WEIGHT="${REFINER_FALSE_MEM_WEIGHT:-0.1}" \
        REFINER_MEMORY=True REFINER_MEMORY_GAIN="${REFINER_MEMORY_GAIN:-True}" \
        KID_EVERY_N_STEPS="${KID_EVERY_N_STEPS:-1000}" \
        RUN_TAG="${RUN_TAG:-rfmv2}" STOP_STEP="${STOP_STEP:-9000}"
    ;;
*)
    echo "[v12refiner] unknown RECIPE ${RECIPE} (refiner, hspP, mem, memr, nbr, rf, rfmv2)" >&2
    exit 1
    ;;
esac

LAYOUT_CACHE="${LAYOUT_CACHE:?set LAYOUT_CACHE (same as the base)}"
if [[ -n "${BASE_CKPT:-}" ]]; then
    # Hand-picked base: BASE_CKPT=<run>/checkpoint_step_NNNNNN.pt (its run dir is the base dir).
    [[ -f "${BASE_CKPT}" ]] || { echo "[v12refiner] missing BASE_CKPT ${BASE_CKPT}"; exit 1; }
    BASE_DIR="$(dirname "${BASE_CKPT}")"
    echo "[v12refiner] base checkpoint (given) ${BASE_CKPT}"
else
BASE_DIR="${BASE_DIR:-$(ls -d results/DamagePatternGeneration/CFMADM/*_${BASE_TAG:-v12} | sort | tail -1)}"
BASE_CKPT=$(python3 - "${BASE_DIR}" "${MIN_STEP:-10000}" <<'PY'
import json, os, sys
d, min_step = sys.argv[1], int(sys.argv[2])
rows = [json.loads(l) for l in open(os.path.join(d, "kid_log.jsonl")) if l.strip()]
rows = [r for r in rows if int(r["step"]) >= min_step
        and os.path.isfile(os.path.join(d, f"checkpoint_step_{int(r['step']):06d}.pt"))]
if not rows:
    sys.exit(f"no periodic KID row at step >= {min_step} with a checkpoint in {d}")
best = min(rows, key=lambda r: r["kid_global"])
print(os.path.join(d, f"checkpoint_step_{int(best['step']):06d}.pt"))
print(f"[v12refiner] base {d}: best val KID {best['kid_global']:.4f} at step {best['step']} of "
      + ", ".join(f"{r['step']}:{r['kid_global']:.4f}" for r in rows), file=sys.stderr)
PY
) || exit 1
echo "[v12refiner] base checkpoint (lowest periodic val KID, step >= ${MIN_STEP:-10000}) ${BASE_CKPT}"
fi
# Train = inference: the base must have been trained on the same layout cache as this refiner.
python3 - "${BASE_DIR}/args.json" "${LAYOUT_CACHE}" <<'PY' || exit 1
import json, os, sys
base = json.load(open(sys.argv[1])).get("layout_cache", "")
if os.path.abspath(base) != os.path.abspath(sys.argv[2]):
    sys.exit(f"[v12refiner] base layout_cache {base!r} != LAYOUT_CACHE {sys.argv[2]!r}")
print(f"[v12refiner] layout cache matches the base: {sys.argv[2]}")
PY
export V11=True APP_CODE_DIM=8 \
    REFINER_BASE_DIR="${BASE_DIR}" REFINER_BASE_CKPT="${BASE_CKPT}" \
    RUN_TAG="${RUN_TAG:-v12refiner}" MAX_STEPS=15000 STOP_STEP="${STOP_STEP:-0}" \
    LAYOUT_COND=True LAYOUT_ZERO_INIT=False LAYOUT_DROP_PROB="${LAYOUT_DROP_PROB:-0.1}" LAYOUT_CACHE
export NUM_WORKERS="${NUM_WORKERS:-1}"   # 2 CPUs: main process + 1 loader worker
exec bash job/DamagePatternGeneration/train_ADMCFM_refiner_residual.sh
