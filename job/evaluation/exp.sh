#!/bin/bash
#PBS -l select=1:ncpus=2:mem=15gb:gpu_id=6
#PBS -l place=shared
#PBS -o log/exp.out
#PBS -e log/exp.err
#PBS -N exp

# 2026-11-11 literature follow-up experiments (scripts/eval/exp_probe.py).
#   MODE=probe  MODELS="name=dir@ckpt ..." VARIANTS="base: sg1:refiner_state_guidance=1 ..." OUT=<json> [FULL=1]
#   MODE=ft     fine-tune rfmv2e 9k for STOP_STEP (1000) steps: RUN_TAG=<tag>, extra env (e.g.
#               REFINER_RESAMPLE_MAX_P=0.7,REFINER_RESAMPLE_GROUPS=4) passed through to the rfmv2 recipe.
cd "${PBS_O_WORKDIR:-$HOME/intership-project}"
source job/DamagePatternGeneration/common.sh
adm_env
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1 OMP_NUM_THREADS=${NCPUS:-2} MKL_NUM_THREADS=${NCPUS:-2}
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
set -e
mkdir -p log results/exp
[[ -n "${ENVFILE:-}" ]] && { set -a; source "${ENVFILE}"; set +a; }

case "${MODE:-probe}" in
probe)
    margs=()
    for m in ${MODELS}; do
        if [[ "${m#*=}" == TAG:* ]]; then  # name=TAG:<run_tag>:<step>: newest run of that tag
            IFS=: read -r _ tag st <<< "${m#*=}"
            d=$(ls -d results/DamagePatternGeneration/CFMADM/*_"${tag}" | sort | tail -1)
            m="${m%%=*}=${d}@${d}/checkpoint_step_${st}.pt"
        fi
        margs+=(--model "${m}")
    done
    vargs=(); for v in ${VARIANTS:-base:}; do vargs+=(--variant "${v}"); done
    python3 -u scripts/eval/exp_probe.py "${margs[@]}" "${vargs[@]}" ${FULL:+--full} \
        --hops "${HOPS:-8}" --n "${N:-4}" --t_every "${T_EVERY:-4}" --out "${OUT:?}"
    ;;
evalcg)
    # val samples of rfmv2e 9k with confidence guidance W (grav layouts, same seed / app code as eval_rfmv2.sh),
    # suite on the val split against the existing rfmv2e / rfmv2b / v12rfm samples.
    E=results/DamagePatternGeneration/CFMADM/2026-11-10_11-20-40_rfmv2
    OUTD=results/exp/eval_cg; W="${W:-1.5}"
    mkdir -p "${OUTD}/val/samples" "${OUTD}/logs"
    python3 -u scripts/eval/generate_eval_samples.py --cfm_result_dir "${E}" --cfm_checkpoint "${E}/checkpoint_step_009000.pt" \
        --split val --k 8 --batch_size 32 --seed 0 --app_code mean \
        --layouts "results/phase1_fiber_layout/layouts/grav_hspP_{split}.npz" \
        --variant "cg${W}:refiner_conf_guidance=${W}" --skip_existing \
        --out "${OUTD}/{split}/samples/rfmv2e_009000_{name}.npz" > "${OUTD}/logs/gen.log" 2>&1
    python3 - "${OUTD}" "${W}" <<'PY'
import json, sys
out, w = sys.argv[1:]
S = "results/eval_suite"
ent = [("rfm_9k", f"{S}/v12rfm/val/samples/v12rfm_009000_grav.npz"),
       ("rfmv2e_9k_c0.1", f"{S}/rfmv2e/val/samples/rfmv2e_009000_c0.1.npz"),
       (f"rfmv2e_9k_cg{w}", f"{out}/val/samples/rfmv2e_009000_cg{w}.npz"),
       ("rfmv2b_9k_c0.1", f"{S}/rfmv2b/val/samples/rfmv2b_009000_c0.1.npz")]
c = json.load(open("job/evaluation/configs/eval_suite_v12.template.json"))
c.update(out_dir=out, splits=["val"], baseline=ent[0][0],
         models=[{"name": n, "npz": {"val": p}, "jpeg_quality": 75} for n, p in ent])
json.dump(c, open(f"{out}/suite.json", "w"), indent=2)
PY
    python3 -u scripts/eval/run_eval_suite.py --config "${OUTD}/suite.json" --splits val
    ;;
evalv)
    [[ "${RUNDIR}" == TOPO ]] && RUNDIR=$(ls -d results/DamagePatternGeneration/CFMADM/*_exp_topo | sort | tail -1)
    # val samples of RUNDIR@STEP with GEN_VARIANTS (NAME:attr=v,...), suite against rfmv2e / v12rfm / extra REF npz.
    OUTD="results/exp/${EVAL_TAG:?}"; ck="${RUNDIR:?}/checkpoint_step_${STEP:?}.pt"
    mkdir -p "${OUTD}/val/samples" "${OUTD}/logs"
    gv=(); for v in ${GEN_VARIANTS:?}; do gv+=(--variant "${v}"); done
    python3 -u scripts/eval/generate_eval_samples.py --cfm_result_dir "${RUNDIR}" --cfm_checkpoint "${ck}" \
        --split val --k 8 --batch_size 32 --seed 0 --app_code mean \
        --layouts "results/phase1_fiber_layout/layouts/grav_hspP_{split}.npz" "${gv[@]}" --skip_existing \
        --out "${OUTD}/{split}/samples/${EVAL_TAG}_{name}.npz" > "${OUTD}/logs/gen.log" 2>&1
    python3 - "${OUTD}" "${EVAL_TAG}" ${GEN_VARIANTS} <<'PY'
import json, sys
out, tag, *vs = sys.argv[1:]
S = "results/eval_suite"
ent = [("rfm_9k", f"{S}/v12rfm/val/samples/v12rfm_009000_grav.npz"),
       ("rfmv2e_9k_c0.1", f"{S}/rfmv2e/val/samples/rfmv2e_009000_c0.1.npz"),
       ("rfmv2e_9k_cg1.5", "results/exp/eval_cg/val/samples/rfmv2e_009000_cg1.5.npz")]
ent += [(f"{tag}_{v.split(':')[0]}", f"{out}/val/samples/{tag}_{v.split(':')[0]}.npz") for v in vs]
c = json.load(open("job/evaluation/configs/eval_suite_v12.template.json"))
c.update(out_dir=out, splits=["val"], baseline=ent[0][0],
         models=[{"name": n, "npz": {"val": p}, "jpeg_quality": 75} for n, p in ent])
json.dump(c, open(f"{out}/suite.json", "w"), indent=2)
PY
    python3 -u scripts/eval/run_eval_suite.py --config "${OUTD}/suite.json" --splits val
    ;;
ft)
    export RECIPE=rfmv2 \
        VELOCITY_INIT="${VELOCITY_INIT:-results/DamagePatternGeneration/CFMADM/2026-11-10_11-20-40_rfmv2/checkpoint_step_009000.pt}" \
        REFINER_RESAMPLE_WARMUP=0 REFINER_RESAMPLE_RAMP=0 WARMUP_STEPS="${WARMUP_STEPS:-100}" \
        KID_EVERY_N_STEPS=100000 STOP_STEP="${STOP_STEP:-1000}" RUN_TAG="${RUN_TAG:?}"
    exec bash job/DamagePatternGeneration/train_ADMCFM_v12refiner.sh
    ;;
esac
