#!/bin/bash
#PBS -l select=1:ncpus=2:mem=15gb:gpu_id=2
#PBS -l place=shared
#PBS -o log/irreversibility_curve.out
#PBS -e log/irreversibility_curve.err
#PBS -N irrev_curve

# Irreversibility curve (scripts/eval/irreversibility_curve.py): born/died of thin dark structure vs frame lag,
# keep vs the cross-trajectory control, Kaiser split (lag 2, cycles 1-2 vs 5-6), real vs generated sequences.
# Gravity-template layouts (grav_hspP) for every model, test + val, k 8, seed 0, app code mean.
#   v12gt_grav           single frame, independent noise per frame (no memory)
#   v12gt_grav_shared    single frame, one start noise per face and k (noise-only coherence)
#   v12gtrefiner_grav    V12 refiner (base cache), sequence walk
#   <A>_<mode>_grav      bidirectional refiner A, joint sampler modes sweep / parallel / causal / none
#   qsub job/evaluation/irreversibility_curve.sh                      (STEP_A=006000)
#   qsub -v STEP_A=009000 job/evaluation/irreversibility_curve.sh     (existing npz are reused)
cd "${PBS_O_WORKDIR:-$HOME/intership-project}"
source job/DamagePatternGeneration/common.sh
adm_env
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
set -e

RUN_GT="${RUN_GT:-results/DamagePatternGeneration/CFMADM/2026-11-05_12-51-53_v12gt}"
RUN_A="${RUN_A:-results/DamagePatternGeneration/CFMADM/2026-11-07_02-47-55_v12nbrA}"
STEP_A="${STEP_A:-006000}"
TAG_A="${TAG_A:-v12nbrA}"
MODES="${MODES:-sweep parallel causal none}"
OUT=results/eval_suite/irreversibility
mkdir -p log "${OUT}/logs"

gen() {  # name split run step [extra args...]
    local name=$1 split=$2 run=$3 step=$4; shift 4
    local npz="${OUT}/${split}/samples/${name}.npz"
    mkdir -p "${OUT}/${split}/samples"
    [[ -f "${npz}" ]] && return 0
    python3 -u scripts/eval/generate_eval_samples.py --cfm_result_dir "${run}" \
        --cfm_checkpoint "${run}/checkpoint_step_${step}.pt" --split "${split}" --k 8 --batch_size 32 --seed 0 \
        --app_code mean --layouts "results/phase1_fiber_layout/layouts/grav_hspP_${split}.npz" --out "${npz}" "$@" \
        > "${OUT}/logs/gen_${name}_${split}.log" 2>&1
    echo "[irrev] ${name} ${split} done"
}

for split in test val; do
    gen v12gt_010000_grav_shared "${split}" "${RUN_GT}" 010000 --shared_noise
    for mode in ${MODES}; do
        gen "${TAG_A}_${STEP_A}_${mode}_grav" "${split}" "${RUN_A}" "${STEP_A}" --set "refiner_joint_mode=${mode}"
    done
done

for split in test val; do
    args=(--model "v12gt_grav=results/eval_suite/v12_compare/${split}/samples/v12gt_010000_grav.npz"
          --model "v12gt_grav_shared=${OUT}/${split}/samples/v12gt_010000_grav_shared.npz"
          --model "v12gtrefiner_6k_grav=results/eval_suite/v12gtrefiner/${split}/samples/v12gtrefiner_006000_grav.npz")
    for f in "${OUT}/${split}/samples/${TAG_A}"_*_grav.npz; do
        n=$(basename "${f}" .npz); args+=(--model "${n}=${f}")
    done
    python3 -u scripts/eval/irreversibility_curve.py --split "${split}" "${args[@]}" --thr 0.03 0.05 0.08 \
        --out "${OUT}/curve_${split}.json"
done
echo "[irrev] done"
