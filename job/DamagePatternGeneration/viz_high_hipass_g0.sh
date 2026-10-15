#!/bin/bash
#PBS -l select=1:ncpus=2:mem=15gb:gpu_id=6
#PBS -l place=shared
#PBS -o log/viz_high_hipass_g0.out
#PBS -e log/viz_high_hipass_g0.err
#PBS -N viz_hp_g0

# 6x6 High-DI grids, single-frame model: high-pass guide on (v_cond, v_g0), g0 = lowest-DI frame
# of the same column and direction. Same gamma / t window as the earlier coord-pair runs.
cd "${PBS_O_WORKDIR:-$HOME/intership-project}"
source job/DamagePatternGeneration/common.sh
adm_env
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1

MAIN_CKPT="results/DamagePatternGeneration/CFMADM/1024-dadaGNsplit/Opensees Based damage Patterns Generation/chjqbp9m/checkpoints/last.ckpt"
RESULT_DIR="results/DamagePatternGeneration/CFMADM/1024-dadaGNsplit"
SIGMAS="${SIGMAS:-2.0 6.0}"
GAMMA="${GAMMA:-2.0}"
TMAX="${TMAX:-0.55}"

common=(--checkpoint "${MAIN_CKPT}" --result_dir "${RESULT_DIR}" --only_groups High --split test --seed 42)
for sg in ${SIGMAS}; do
  echo "[viz] high-pass g0=undamaged gamma=${GAMMA} sigma=${sg} t<${TMAX}"
  python3 -u scripts/viz/export_di_tertile_overview_grids.py "${common[@]}" \
      --coord_guidance_scale "${GAMMA}" --coord_guidance_hipass_sigma "${sg}" \
      --coord_guidance_t_max "${TMAX}" --coord_g0 undamaged --facfg_g0 ""
done
mkdir -p "${RESULT_DIR}/inference_samples/other_guidance"
mv "${RESULT_DIR}"/inference_samples/overview_di_3_high_6x6_*_coordg*_g0undamaged.png "${RESULT_DIR}/inference_samples/other_guidance/" 2>/dev/null
ls -la "${RESULT_DIR}/inference_samples/other_guidance" | grep coordg | grep g0undamaged
