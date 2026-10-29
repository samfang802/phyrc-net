#!/bin/bash
#PBS -l select=1:ncpus=2:mem=15gb:gpu_id=6
#PBS -l place=shared
#PBS -o log/eval_diagnostics.out
#PBS -e log/eval_diagnostics.err
#PBS -N diagnostics

# Model diagnostics (not part of the evaluation suite).
#   WHAT=shift     samples vs real frames: translation and residual energy (measure_base_vs_gt_shift.py)
#   WHAT=spectrum  radial power spectrum at 1/10/30/60 NFE and the HF share of r = x_t - S (measure_state_spectrum.py)
#   WHAT=all       shift + spectrum
#   WHAT=train_area       v11appr (app_r_6k): does it under-draw spalling on its own training columns? Train-split
#                         samples (own appearance codes), then texture + per-column spall area vs GT, also for test/val
#   WHAT=velocity_fusion  v11appr: x̂1 error of base / refiner-off / refiner-on along τ, can they be fused (KF)?
#                         SMOKE=1 for a short run
# Replaces eval_base_vs_gt_shift.sh, eval_state_spectrum.sh, v11appr_train_area.sh, velocity_fusion_stats.sh.
#   qsub -v WHAT=spectrum,CKPT=... job/DamagePatternGeneration/eval_diagnostics.sh
cd "${PBS_O_WORKDIR:-$HOME/intership-project}"
source job/DamagePatternGeneration/common.sh
adm_env
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"

WHAT="${WHAT:-all}"
RESULT_DIR="${RESULT_DIR:-results/DamagePatternGeneration/CFMADM/1024-dadaGNsplit}"
CKPT="${CKPT:-${RESULT_DIR}/Opensees Based damage Patterns Generation/chjqbp9m/checkpoints/last.ckpt}"
common=(--checkpoint "${CKPT}" --result_dir "${RESULT_DIR}" --batch_size 16 --num_workers 0)

if [[ "${WHAT}" == "shift" || "${WHAT}" == "all" ]]; then
    for split in val test train; do
        python3 -u scripts/eval/measure_base_vs_gt_shift.py "${common[@]}" --split "${split}"
    done
fi
if [[ "${WHAT}" == "spectrum" || "${WHAT}" == "all" ]]; then
    for split in val test; do
        python3 -u scripts/eval/measure_state_spectrum.py "${common[@]}" --split "${split}"
    done
fi

# v11appr diagnostics: a different run and checkpoint than the gn defaults above
V11APPR=results/DamagePatternGeneration/CFMADM/2026-10-26_12-45-16_v11app_refiner
V11APPR_CKPT="${V11APPR}/checkpoint_step_006000.pt"
V11APPR_SUITE=results/eval_suite/v11app_refiner
if [[ "${WHAT}" == "train_area" ]]; then
    OUT="${V11APPR_SUITE}/train"
    NPZ="${OUT}/samples/app_r_6k.npz"
    mkdir -p "${OUT}/samples"
    if [[ ! -f "${NPZ}" ]]; then
        python3 -u scripts/eval/generate_eval_samples.py --cfm_result_dir "${V11APPR}" \
            --cfm_checkpoint "${V11APPR_CKPT}" --split train --k "${K:-4}" --batch_size 16 \
            --seed 0 --app_code mean --out "${NPZ}" || exit 1
    fi
    python3 -u scripts/eval/eval_damage_texture.py --samples "v11appr=${NPZ}" --no_clahe --out_dir "${OUT}/texture" || exit 1
    python3 -u scripts/eval/area_by_column.py --samples "${NPZ}" --out "${OUT}/area_by_column.csv" || exit 1
    for split in test val; do
        python3 -u scripts/eval/area_by_column.py --samples "${V11APPR_SUITE}/${split}/samples/app_r_6k.npz" \
            --out "${V11APPR_SUITE}/${split}/area_by_column.csv" || exit 1
    done
fi
if [[ "${WHAT}" == "velocity_fusion" ]]; then
    EXTRA=()
    OUT=results/velocity_fusion/v11appr
    if [[ -n "${SMOKE:-}" ]]; then EXTRA=(--splits val --taus 0.25 0.85 --draws 1); OUT=${OUT}_smoke; fi
    python3 -u scripts/eval/velocity_fusion_stats.py --cfm_result_dir "${V11APPR}" \
        --cfm_checkpoint "${V11APPR_CKPT}" \
        --samples val=${V11APPR_SUITE}/val/samples/app_r_6k.npz \
                  test=${V11APPR_SUITE}/test/samples/app_r_6k.npz \
        --out "${OUT}" "${EXTRA[@]}" || exit 1
fi
