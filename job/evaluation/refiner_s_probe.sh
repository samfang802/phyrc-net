#!/bin/bash
#PBS -l select=1:ncpus=2:mem=15gb:gpu_id=7
#PBS -l place=shared
#PBS -o log/refiner_s_probe.out
#PBS -e log/refiner_s_probe.err
#PBS -N s_probe

# What each refiner takes from S_{t-1} (scripts/eval/refiner_s_probe.py): V11 base-state refiner
# and the V12 refiner with the gravity layouts. The teacher-forced S1 run was removed after its
# probe was recorded in results/refiner_s_probe.
#   qsub job/evaluation/refiner_s_probe.sh
#   qsub -v SPLIT=train,OUT=results/refiner_s_probe/train job/evaluation/refiner_s_probe.sh
cd "${PBS_O_WORKDIR:-$HOME/intership-project}"
source job/DamagePatternGeneration/common.sh
adm_env
export HF_HUB_OFFLINE=1

R=results/DamagePatternGeneration/CFMADM
V11="${R}/2026-10-26_12-45-16_v11app_refiner"
V12="${R}/2026-11-05_22-06-00_v12gtrefiner"
SPLIT="${SPLIT:-test}"
OUT="${OUT:-results/refiner_s_probe/${SPLIT}}"
LAY="results/phase1_fiber_layout/layouts/grav_hspP_${SPLIT}.npz"

python3 scripts/eval/refiner_s_probe.py --split "${SPLIT}" --out_dir "${OUT}" \
    --model "full_6k=${V11}@${V11}/checkpoint_step_006000.pt" \
    --model "full_15k=${V11}@${V11}/checkpoint_step_015000.pt" \
    --model "v12_6k=${V12}@${V12}/checkpoint_step_006000.pt@${LAY}" \
    ${EXTRA:-}
