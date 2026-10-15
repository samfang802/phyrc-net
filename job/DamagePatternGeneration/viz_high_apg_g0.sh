#!/bin/bash
#PBS -l select=1:ncpus=2:mem=15gb:gpu_id=6
#PBS -l place=shared
#PBS -o log/viz_high_apg_g0.out
#PBS -e log/viz_high_apg_g0.err
#PBS -N viz_apg_g0

# 6x6 High-DI grids from the single-frame model: APG whose "unconditional" branch swaps only g
# (g = 0, or the lowest-DI frame of the same column and direction).
cd "${PBS_O_WORKDIR:-$HOME/intership-project}"
source job/DamagePatternGeneration/common.sh
adm_env
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1

MAIN_CKPT="results/DamagePatternGeneration/CFMADM/1024-dadaGNsplit/Opensees Based damage Patterns Generation/chjqbp9m/checkpoints/last.ckpt"
RESULT_DIR="results/DamagePatternGeneration/CFMADM/1024-dadaGNsplit"
SCALES="${SCALES:-2.0 4.0}"
G0S="${G0S:-undamaged}"

common=(--checkpoint "${MAIN_CKPT}" --result_dir "${RESULT_DIR}" --only_groups High --split test --seed 42)
for g0 in ${G0S}; do
  for s in ${SCALES}; do
    echo "[viz] APG s=${s} g0=${g0}"
    python3 -u scripts/viz/export_di_tertile_overview_grids.py "${common[@]}" \
        --apg_guidance_scale "${s}" --apg_g0 "${g0}"
  done
done
mkdir -p "${RESULT_DIR}/inference_samples/other_guidance"
mv "${RESULT_DIR}"/inference_samples/overview_di_3_high_6x6_*_apg*_g0*.png "${RESULT_DIR}/inference_samples/other_guidance/" 2>/dev/null
ls -la "${RESULT_DIR}/inference_samples/other_guidance" | grep g0
