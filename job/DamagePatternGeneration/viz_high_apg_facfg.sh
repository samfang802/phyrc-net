#!/bin/bash
#PBS -l select=1:ncpus=2:mem=15gb:gpu_id=6
#PBS -l place=shared
#PBS -o log/viz_high_apg_facfg.out
#PBS -e log/viz_high_apg_facfg.err
#PBS -N viz_apg_facfg

# 6x6 High-DI grids, single-frame model: APG + FA-CFG with g0 = lowest-DI frame of the same column.
# The FA-CFG band weights set the strength; --apg_guidance_scale only needs to be non-zero.
cd "${PBS_O_WORKDIR:-$HOME/intership-project}"
source job/DamagePatternGeneration/common.sh
adm_env
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1

MAIN_CKPT="results/DamagePatternGeneration/CFMADM/1024-dadaGNsplit/Opensees Based damage Patterns Generation/chjqbp9m/checkpoints/last.ckpt"
RESULT_DIR="results/DamagePatternGeneration/CFMADM/1024-dadaGNsplit"
BANDS="${BANDS:-1.0,2.0 1.5,4.0}"

common=(--checkpoint "${MAIN_CKPT}" --result_dir "${RESULT_DIR}" --only_groups High --split test --seed 42)
for b in ${BANDS}; do
  wl="${b%,*}"; wh="${b#*,}"
  echo "[viz] APG + FA-CFG wl=${wl} wh=${wh}"
  python3 -u scripts/viz/export_di_tertile_overview_grids.py "${common[@]}" \
      --apg_guidance_scale 2.0 --apg_g0 undamaged \
      --facfg_wl "${wl}" --facfg_wh "${wh}" --facfg_g0 undamaged
done
mkdir -p "${RESULT_DIR}/inference_samples/other_guidance"
mv "${RESULT_DIR}"/inference_samples/overview_di_3_high_6x6_*_apg*_facfg*.png "${RESULT_DIR}/inference_samples/other_guidance/" 2>/dev/null
ls -la "${RESULT_DIR}/inference_samples/other_guidance" | grep apg | grep facfg
