#!/bin/bash
#PBS -l select=1:ncpus=2:mem=15gb:gpu_id=2
#PBS -l place=shared
#PBS -o log/damage_rate.out
#PBS -e log/damage_rate.err
#PBS -N damage_rate

# Per-frame damage-rate calibration (docs/redesign/phase1_ilf.md §13, scripts/eval/phase1_damage_rate.py) and the
# step-birth metrics (irreversibility_curve.py) rerun on the v12rf_compare and state_conf model sets. CPU only.
#   qsub job/evaluation/damage_rate.sh
cd "${PBS_O_WORKDIR:-$HOME/intership-project}"
source job/DamagePatternGeneration/common.sh
adm_env
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
set -e
mkdir -p log results/phase1_damage_rate
python3 -u scripts/eval/phase1_damage_rate.py --out results/phase1_damage_rate/rates.json | tee results/phase1_damage_rate/rates.txt

for split in test val; do
    args=()
    for m in "v12gtrefiner_6k=results/eval_suite/v12gtrefiner/{s}/samples/v12gtrefiner_006000_grav.npz" \
             "memc_6k=results/eval_suite/v12memc/{s}/samples/v12memc_006000_grav.npz" \
             "memc_6k_g0.5-0=results/eval_suite/memory_guidance/v12memc_006000/{s}/samples/v12memc_006000_w0.5-0t0.5.npz" \
             "rf_6k=results/eval_suite/v12rf/{s}/samples/v12rf_006000_grav.npz" \
             "rfc_6k=results/eval_suite/v12rfc/{s}/samples/v12rfc_006000_grav.npz" \
             "rfc_6k_c0.3=results/eval_suite/state_conf/v12rfc_006000/{s}/samples/v12rfc_006000_c0.3.npz" \
             "memg_8k=results/eval_suite/v12memg/{s}/samples/v12memg_008000_grav.npz" \
             "v12gt_single=results/eval_suite/v12_compare/{s}/samples/v12gt_010000_grav.npz"; do
        args+=(--model "${m%%=*}=$(echo "${m#*=}" | sed "s/{s}/${split}/")")
    done
    python3 -u scripts/eval/irreversibility_curve.py --split "${split}" "${args[@]}" --thr 0.05 \
        --out "results/phase1_damage_rate/step_births_${split}.json" | tee "results/phase1_damage_rate/step_births_${split}.txt"
done
