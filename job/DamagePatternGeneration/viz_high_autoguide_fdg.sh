#!/bin/bash
#PBS -l select=1:ncpus=2:mem=15gb:gpu_id=5
#PBS -l place=shared
#PBS -o log/viz_high_autoguide_fdg.out
#PBS -e log/viz_high_autoguide_fdg.err
#PBS -N viz_ag_fdg

# 6x6 High-DI grid: no guidance vs autoguidance vs autoguidance + FDG.
cd "${PBS_O_WORKDIR:-$HOME/intership-project}"
source job/DamagePatternGeneration/common.sh
adm_env
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1

MAIN_CKPT="results/DamagePatternGeneration/CFMADM/1024-dadaGNsplit/Opensees Based damage Patterns Generation/chjqbp9m/checkpoints/last.ckpt"
RESULT_DIR="results/DamagePatternGeneration/CFMADM/1024-dadaGNsplit"
BAD_DIR="results/DamagePatternGeneration/CFMADM/2026-10-12_03-59-39_autoguide_bad"
BAD_STEP="${BAD_STEP:-006000}"
BAD_CKPT="${BAD_DIR}/checkpoint_step_${BAD_STEP}.pt"
AG_W="${AG_W:-0.5}"
AG_GAMMA="${AG_GAMMA:-4.0}"
AG_ETA="${AG_ETA:-1.0}"
FDG_WH="${FDG_WH:-0.5}"

common=(--checkpoint "${MAIN_CKPT}" --result_dir "${RESULT_DIR}" --only_groups High --split test --seed 42)
ag=(--autoguide_checkpoint "${BAD_CKPT}" --autoguide_result_dir "${BAD_DIR}" \
    --autoguide_weight "${AG_W}" --autoguide_gamma "${AG_GAMMA}" --autoguide_eta "${AG_ETA}")

echo "[viz] 1/3 no guidance"
python3 -u scripts/viz/export_di_tertile_overview_grids.py "${common[@]}"
echo "[viz] 2/3 autoguidance"
python3 -u scripts/viz/export_di_tertile_overview_grids.py "${common[@]}" "${ag[@]}"
echo "[viz] 3/3 autoguidance + FDG (wl=1 wh=${FDG_WH})"
python3 -u scripts/viz/export_di_tertile_overview_grids.py "${common[@]}" "${ag[@]}" \
    --autoguide_wl 1.0 --autoguide_wh "${FDG_WH}"
ls -la "${RESULT_DIR}/inference_samples" | grep overview_di_high
