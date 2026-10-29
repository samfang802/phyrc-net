#!/bin/bash
#PBS -l select=1:ncpus=2:mem=15gb:gpu_id=6
#PBS -l place=shared
#PBS -o log/eval_facfg.out
#PBS -e log/eval_facfg.err
#PBS -N eval_facfg

# FA-CFG samples (wl=1, wh=2; needs a per-sample g0, scripts/eval/phase0_facfg_samples.py) for each run in RUNS,
# then the eval suite. Replaces phase0_guidance.sh (EXP=guidance) and eval_opaflow.sh (EXP=opaflow). Resumable.
#   EXP=guidance  gn without guidance vs FA-CFG, autoguidance, APG, coord high-pass (docs/redesign/phase0_guidance.md);
#                 the non-FA-CFG samples come from run_eval_suite.py itself
#   EXP=opaflow   OPA-Flow runs, RUNS="A=<result_dir> B=<result_dir> [C=...]" (docs/redesign/phase1_opaflow.md);
#                 the config is written by scripts/eval/make_opaflow_suite.py, gn / gn_facfg reuse the phase-0 samples
#   qsub -v EXP=guidance job/DamagePatternGeneration/eval_facfg.sh
#   qsub -v EXP=opaflow,RUNS="A=results/...opa_A B=results/...opa_B" job/DamagePatternGeneration/eval_facfg.sh
cd "${PBS_O_WORKDIR:-$HOME/intership-project}"
source job/DamagePatternGeneration/common.sh
adm_env
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
set -e

EXP="${EXP:?set EXP=guidance or EXP=opaflow}"
case "${EXP}" in
    guidance)
        OUT=results/eval_suite/phase0_guidance
        CONFIG=job/DamagePatternGeneration/configs/eval_suite_phase0_guidance.json
        RUNS="gn=results/DamagePatternGeneration/CFMADM/1024-dadaGNsplit"
        ;;
    opaflow)
        OUT=results/eval_suite/phase1_opaflow
        CONFIG=${OUT}/suite_models.json
        mkdir -p "${OUT}"
        python3 scripts/eval/make_opaflow_suite.py ${RUNS:?RUNS is required} > "${CONFIG}"
        ;;
    *) echo "EXP must be guidance or opaflow"; exit 1 ;;
esac
mkdir -p log "${OUT}/logs"

for split in ${SPLITS:-test val}; do
    for kv in ${RUNS}; do
        name="${kv%%=*}"; dir="${kv#*=}"
        npz="${OUT}/${split}/samples/${name}_facfg.npz"
        [[ -f "${npz}" ]] && continue
        ck=$(ls "${dir}"/*/*/checkpoints/last.ckpt | tail -n 1)
        mkdir -p "$(dirname "${npz}")"
        echo "[facfg] ${name} ${split}: ${ck}"
        python3 -u scripts/eval/phase0_facfg_samples.py --cfm_result_dir "${dir}" --cfm_checkpoint "${ck}" \
            --split "${split}" --k 8 --batch_size 32 --seed 0 --out "${npz}" \
            --set facfg_wl=1.0 facfg_wh=2.0 > "${OUT}/logs/${split}_samples_${name}_facfg.log" 2>&1
    done
    python3 -u scripts/eval/run_eval_suite.py --config "${CONFIG}" --splits "${split}"
done
