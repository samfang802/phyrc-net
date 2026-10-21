#!/bin/bash
#PBS -l select=1:ncpus=1:gpu_id=5
#PBS -l place=shared
#PBS -o log/train_hysteresis.out
#PBS -e log/train_hysteresis.err
#PBS -N hysteresis

# Hysteresis predictor (train_hysteresis.py). ARCH=HystereticGRU with these settings is the recipe of
# ./phase1_extractor (the GRU whose hidden state conditions every generator; see its args.json).
# Replaces train_GRU.sh / train_LSTM.sh, which pointed at the old repo path and the removed
# `hysteresis` conda env.
#   qsub job/HysteresisPrediction/train_hysteresis.sh
#   qsub -v ARCH=HystereticLSTM job/HysteresisPrediction/train_hysteresis.sh
cd "${PBS_O_WORKDIR:-$HOME/intership-project}"
set +u
source ~/.bashrc
set -u
conda activate internship
module load cuda-12.4

ARCH="${ARCH:-HystereticGRU}"
mkdir -p log
python3 train_hysteresis.py --lr 1e-4 --epoch 5000 --batch_size 8 --architecture "${ARCH}" --data_size big \
    --hidden_state_dim 512 --layers 4 --MLP_activation softsign --bidirectional --simulate --scale \
    2>&1 | tee "log/train_${ARCH}.log"
