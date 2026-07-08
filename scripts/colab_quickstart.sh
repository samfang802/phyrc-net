#!/usr/bin/env bash
set -euo pipefail

# Colab quickstart (run from project root)
# Example:
#   bash scripts/colab_quickstart.sh
#   bash scripts/colab_quickstart.sh --with-wandb

WITH_WANDB=0
for arg in "$@"; do
  if [[ "$arg" == "--with-wandb" ]]; then
    WITH_WANDB=1
  fi
done

if [[ "$WITH_WANDB" -eq 1 ]]; then
  WANDB_FLAG="--use_wandb"
else
  WANDB_FLAG=""
fi

python -m pip install --upgrade pip
python -m pip install -r requirements.txt

echo "Starting Stage 1 (Hysteretic prediction)..."
python train_hysteresis.py --architecture HystereticGRU --epoch 5 --batch_size 4 --num_workers 1 ${WANDB_FLAG}

echo "Starting Stage 2 (Damage quantification)..."
python train_quantification.py --backbone vit_b_16 --epoch 5 --batch_size 4 --num_workers 1 ${WANDB_FLAG}

echo "Done. Update checkpoint paths before running train_mixed.py."
