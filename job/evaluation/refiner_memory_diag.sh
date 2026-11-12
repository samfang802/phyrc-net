#!/bin/bash
#PBS -l select=1:ncpus=2:mem=15gb:gpu_id=2
#PBS -l place=shared
#PBS -o log/refiner_memory_diag.out
#PBS -e log/refiner_memory_diag.err
#PBS -N refiner_memdiag

# Where does the V12 refiner's frame-to-frame persistence come from? 2 x 2 on v12gtrefiner (grav layouts):
#   shared low-frequency start noise  on / off   (refiner_shared_lowfreq_noise)
#   S_{t-1} from its own trajectory / from trajectory k+1   (refiner_state_roll=-1)
# Read with scripts/eval/irreversibility_curve.py: keep above the cross-trajectory control is memory; with the
# rolled state, a refiner that reads S moves that memory onto the cross-trajectory row.
#   qsub job/evaluation/refiner_memory_diag.sh
cd "${PBS_O_WORKDIR:-$HOME/intership-project}"
source job/DamagePatternGeneration/common.sh
adm_env
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
set -e

RUN="${RUN:-results/DamagePatternGeneration/CFMADM/2026-11-05_22-06-00_v12gtrefiner}"
STEP="${STEP:-006000}"
OUT=results/eval_suite/irreversibility
mkdir -p log "${OUT}/logs"

declare -A SETS=(
    [noise0]="refiner_shared_lowfreq_noise=0"
    [roll]="refiner_state_roll=-1"
    [noise0_roll]="refiner_shared_lowfreq_noise=0 refiner_state_roll=-1"
)
for split in test val; do
    mkdir -p "${OUT}/${split}/samples"
    for v in noise0 roll noise0_roll; do
        npz="${OUT}/${split}/samples/v12gtrefiner_${STEP}_grav_${v}.npz"
        [[ -f "${npz}" ]] && continue
        # shellcheck disable=SC2086
        python3 -u scripts/eval/generate_eval_samples.py --cfm_result_dir "${RUN}" \
            --cfm_checkpoint "${RUN}/checkpoint_step_${STEP}.pt" --split "${split}" --k 8 --batch_size 32 --seed 0 \
            --app_code mean --layouts "results/phase1_fiber_layout/layouts/grav_hspP_${split}.npz" --out "${npz}" \
            --set ${SETS[$v]} > "${OUT}/logs/gen_v12gtrefiner_${STEP}_grav_${v}_${split}.log" 2>&1
        echo "[memdiag] ${v} ${split} done"
    done
    python3 -u scripts/eval/irreversibility_curve.py --split "${split}" --thr 0.05 0.08 \
        --model "refiner=results/eval_suite/v12gtrefiner/${split}/samples/v12gtrefiner_${STEP}_grav.npz" \
        --model "refiner_noise0=${OUT}/${split}/samples/v12gtrefiner_${STEP}_grav_noise0.npz" \
        --model "refiner_roll=${OUT}/${split}/samples/v12gtrefiner_${STEP}_grav_roll.npz" \
        --model "refiner_noise0_roll=${OUT}/${split}/samples/v12gtrefiner_${STEP}_grav_noise0_roll.npz" \
        --out "${OUT}/memdiag_${split}.json"
done
echo "[memdiag] done"
