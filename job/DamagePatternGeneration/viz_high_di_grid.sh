#!/bin/bash
#PBS -l select=1:ncpus=2:mem=15gb:gpu_id=6
#PBS -l place=shared
#PBS -o log/viz_high_di_grid.out
#PBS -e log/viz_high_di_grid.err
#PBS -N viz_high

# 6x6 High-DI overview grids (scripts/viz/export_di_tertile_overview_grids.py), one MODE per run.
# Replaces viz_high_{apg_g0,apg_facfg,hipass_g0,autoguide_fdg,refiner}.sh; same commands and outputs.
#
#   MODE=apg_g0         APG whose "unconditional" branch swaps only g (SCALES, G0S)
#   MODE=apg_facfg      APG + FA-CFG, g0 = lowest-DI frame of the column (BANDS="wl,wh ...")
#   MODE=hipass_g0      high-pass guide on (v_cond, v_g0) (SIGMAS, GAMMA, TMAX)
#   MODE=autoguide_fdg  no guidance vs autoguidance vs autoguidance + FDG (BAD_DIR, BAD_STEP, AG_*, FDG_WH)
#   MODE=refiner        residual refiner checkpoint, cells locked to the single-frame grid (RUN_DIR, CKPT, GRAY)
#   MODE=plain          no guidance (CKPT, RESULT_DIR)
#
#   qsub -v MODE=apg_g0,SCALES="1.5 2.0" job/DamagePatternGeneration/viz_high_di_grid.sh
# Guidance grids are moved to <RESULT_DIR>/inference_samples/other_guidance/.
cd "${PBS_O_WORKDIR:-$HOME/intership-project}"
source job/DamagePatternGeneration/common.sh
adm_env
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1

MODE="${MODE:-plain}"
RESULT_DIR="${RESULT_DIR:-results/DamagePatternGeneration/CFMADM/1024-dadaGNsplit}"
CKPT="${CKPT:-${RESULT_DIR}/Opensees Based damage Patterns Generation/chjqbp9m/checkpoints/last.ckpt}"
VIZ=(python3 -u scripts/viz/export_di_tertile_overview_grids.py)
common=(--checkpoint "${CKPT}" --result_dir "${RESULT_DIR}" --only_groups High --split test --seed 42)
OTHER="${RESULT_DIR}/inference_samples/other_guidance"

case "${MODE}" in
plain)
    "${VIZ[@]}" "${common[@]}"
    ;;
apg_g0)
    for g0 in ${G0S:-undamaged}; do
        for s in ${SCALES:-2.0 4.0}; do
            echo "[viz] APG s=${s} g0=${g0}"
            "${VIZ[@]}" "${common[@]}" --apg_guidance_scale "${s}" --apg_g0 "${g0}"
        done
    done
    mkdir -p "${OTHER}"
    mv "${RESULT_DIR}"/inference_samples/overview_di_3_high_6x6_*_apg*_g0*.png "${OTHER}/" 2>/dev/null
    ;;
apg_facfg)
    # The FA-CFG band weights set the strength; --apg_guidance_scale only needs to be non-zero.
    for b in ${BANDS:-1.0,2.0 1.5,4.0}; do
        wl="${b%,*}"; wh="${b#*,}"
        echo "[viz] APG + FA-CFG wl=${wl} wh=${wh}"
        "${VIZ[@]}" "${common[@]}" --apg_guidance_scale 2.0 --apg_g0 undamaged \
            --facfg_wl "${wl}" --facfg_wh "${wh}" --facfg_g0 undamaged
    done
    mkdir -p "${OTHER}"
    mv "${RESULT_DIR}"/inference_samples/overview_di_3_high_6x6_*_apg*_facfg*.png "${OTHER}/" 2>/dev/null
    ;;
hipass_g0)
    GAMMA="${GAMMA:-2.0}"; TMAX="${TMAX:-0.55}"
    for sg in ${SIGMAS:-2.0 6.0}; do
        echo "[viz] high-pass g0=undamaged gamma=${GAMMA} sigma=${sg} t<${TMAX}"
        "${VIZ[@]}" "${common[@]}" --coord_guidance_scale "${GAMMA}" --coord_guidance_hipass_sigma "${sg}" \
            --coord_guidance_t_max "${TMAX}" --coord_g0 undamaged --facfg_g0 ""
    done
    mkdir -p "${OTHER}"
    mv "${RESULT_DIR}"/inference_samples/overview_di_3_high_6x6_*_coordg*_g0undamaged.png "${OTHER}/" 2>/dev/null
    ;;
autoguide_fdg)
    BAD_DIR="${BAD_DIR:-results/DamagePatternGeneration/CFMADM/2026-10-12_03-59-39_autoguide_bad}"
    BAD_CKPT="${BAD_DIR}/checkpoint_step_${BAD_STEP:-006000}.pt"
    ag=(--autoguide_checkpoint "${BAD_CKPT}" --autoguide_result_dir "${BAD_DIR}" \
        --autoguide_weight "${AG_W:-0.5}" --autoguide_gamma "${AG_GAMMA:-4.0}" --autoguide_eta "${AG_ETA:-1.0}")
    echo "[viz] 1/3 no guidance"
    "${VIZ[@]}" "${common[@]}"
    echo "[viz] 2/3 autoguidance"
    "${VIZ[@]}" "${common[@]}" "${ag[@]}"
    echo "[viz] 3/3 autoguidance + FDG (wl=1 wh=${FDG_WH:-0.5})"
    "${VIZ[@]}" "${common[@]}" "${ag[@]}" --autoguide_wl 1.0 --autoguide_wh "${FDG_WH:-0.5}"
    ;;
refiner)
    # Same cells as the single-frame grid (order locked to its JSON); each column is walked from frame 0.
    RUN_DIR="${RUN_DIR:-results/DamagePatternGeneration/CFMADM/2026-10-13_03-02-40_refiner_res_15k}"
    RCKPT="${RCKPT:-${RUN_DIR}/checkpoint_step_015000.pt}"
    META="${META:-results/DamagePatternGeneration/CFMADM/1024-dadaGNsplit/inference_samples/overview_di_high_selfcondOFF_step_15000_ode30.json}"
    "${VIZ[@]}" --checkpoint "${RCKPT}" --result_dir "${RUN_DIR}" --only_groups High --split test \
        --gt_from_meta "${META}" --batch_size 16 --num_workers 0 --seed 42 ${GRAY:+--refiner_gray_state}
    RESULT_DIR="${RUN_DIR}"
    ;;
*)
    echo "[viz] unknown MODE=${MODE}"; exit 1
    ;;
esac
ls -la "${RESULT_DIR}/inference_samples" "${OTHER}" 2>/dev/null | grep overview
