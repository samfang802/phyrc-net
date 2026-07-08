#!/usr/bin/env bash
set -euo pipefail

# Colab one-click runner (run from project root).
# Examples:
#   bash scripts/colab_quickstart.sh
#   bash scripts/colab_quickstart.sh --with-wandb --backend cfm
#   bash scripts/colab_quickstart.sh --mode overfit

WITH_WANDB=0
MODE="smoke"
BACKEND="gan"
for arg in "$@"; do
  case "$arg" in
    --with-wandb) WITH_WANDB=1 ;;
    --mode=*) MODE="${arg#*=}" ;;
    --backend=*) BACKEND="${arg#*=}" ;;
  esac
done

if [[ "$WITH_WANDB" -eq 1 ]]; then
  WANDB_FLAG="--use_wandb"
else
  WANDB_FLAG=""
fi

if [[ "$BACKEND" != "gan" && "$BACKEND" != "cfm" ]]; then
  echo "Unsupported --backend value: ${BACKEND}. Use gan or cfm."
  exit 1
fi

python -m pip install --upgrade pip
python -m pip install -r requirements.txt

AUTHOR_EXTRACTOR="results/HystereticPrediction/HystereticGRU/2024-06-12_18-13-47"
AUTHOR_REGRESSOR="results/DamageQuantification/vit_b_16/2024-06-12_14-01-03"

EPOCH_STAGE1=5
EPOCH_STAGE2=5
EPOCH_STAGE3=5
if [[ "$MODE" == "overfit" ]]; then
  EPOCH_STAGE3=100
fi

if [[ -d "${AUTHOR_EXTRACTOR}" && -d "${AUTHOR_REGRESSOR}" ]]; then
  echo "Found author checkpoints; skipping Stage 1/2."
  EXTRACTOR_PATH="${AUTHOR_EXTRACTOR}"
  REGRESSOR_PATH="${AUTHOR_REGRESSOR}"
else
  echo "Author checkpoints not found; running Stage 1/2 smoke training."
  echo "Expected:"
  echo "  ${AUTHOR_EXTRACTOR}"
  echo "  ${AUTHOR_REGRESSOR}"

  echo "Starting Stage 1 (Hysteretic prediction)..."
  python train_hysteresis.py --architecture HystereticGRU --epoch ${EPOCH_STAGE1} --batch_size 4 --num_workers 1 ${WANDB_FLAG}

  echo "Starting Stage 2 (Damage quantification)..."
  python train_quantification.py --backbone vit_b_16 --epoch ${EPOCH_STAGE2} --batch_size 4 --num_workers 1 ${WANDB_FLAG}

  EXTRACTOR_PATH="$(ls -td results/HystereticPrediction/HystereticGRU/* | head -1)"
  REGRESSOR_PATH="$(ls -td results/DamageQuantification/vit_b_16/* | head -1)"
fi

echo "extractor_path=${EXTRACTOR_PATH}"
echo "regressor_path=${REGRESSOR_PATH}"

CONDITION_ARGS="--latent_conditioning True --direction_conditioning True --include_di_conditioning True"

if [[ "$BACKEND" == "cfm" ]]; then
  echo "Starting Stage 3 (CFM backend)..."
  CFM_ARGS="--architecture CFMADM --cfm_solver heun --cfm_ode_steps 20 --lr_schedule warmup_cosine --warmup_epochs 10 --min_lr_ratio 0.1"
else
  echo "Starting Stage 3 (GAN backend)..."
  CFM_ARGS="--architecture HystereticCGAN --loss_type SGAN --z_dim 100 --G_h_size 128 --D_h_size 128 --SELU False --spectral_D False --no_batch_norm False"
fi

OVERFIT_ARGS=""
if [[ "$MODE" == "overfit" ]]; then
  echo "Overfit mode: training Stage 3 on one sample for 100 epochs."
  OVERFIT_ARGS="--overfit_single_sample --overfit_sample_index 0 --overfit_repeat_steps 32"
fi

python train_mixed.py \
  ${CFM_ARGS} \
  --extractor_path "${EXTRACTOR_PATH}" \
  --regressor_path "${REGRESSOR_PATH}" \
  --data_path id_list.json \
  ${CONDITION_ARGS} \
  --add_DI_loss False \
  --image_size 128 \
  --epoch ${EPOCH_STAGE3} \
  --batch_size 4 \
  --num_workers 1 \
  ${OVERFIT_ARGS} \
  ${WANDB_FLAG}

echo "Done."
