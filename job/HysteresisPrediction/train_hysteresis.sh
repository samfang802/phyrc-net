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
# MODEL=damage_state: the damage-state encoder instead (train_damage_encoder.py, docs/DAMAGE_STATE_ENCODER.md).
#   qsub -v MODEL=damage_state job/HysteresisPrediction/train_hysteresis.sh
# Sweeps: SEEDS=0:1:2 trains one run per seed (each also screened); TAG names the result folder.
if [[ "${MODEL:-gru}" == "damage_state" ]]; then
    export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
    TAG="${TAG:-default}"; SEEDS="${SEEDS:-0}"
    for SEED in ${SEEDS//:/ }; do
        python3 -u train_damage_encoder.py --input_mode "${INPUT_MODE:-points}" --hidden "${HIDDEN:-256}" \
            --dropout "${DROPOUT:-0.3}" --weight_decay "${WEIGHT_DECAY:-1e-2}" --lr "${LR:-3e-4}" \
            --static_info "${STATIC_INFO:-1}" --force_weight "${FORCE_WEIGHT:-1.0}" \
            --seq_info "${SEQ_INFO:-raw}" --force_target "${FORCE_TARGET:-signed}" --seed "${SEED}" \
            --result_dir "results/DamageStateEncoder/${TAG}" 2>&1 | tee "log/train_damage_state_${TAG}_s${SEED}.log"
        RUN=$(ls -d "results/DamageStateEncoder/${TAG}"/2026-* | tail -1)
        python3 -u scripts/eval/screen_condition_encoder.py "${RUN}" > "${RUN}/screen.log" 2>&1
    done
    exit 0
fi
python3 train_hysteresis.py --lr 1e-4 --epoch 5000 --batch_size 8 --architecture "${ARCH}" --data_size big \
    --hidden_state_dim 512 --layers 4 --MLP_activation softsign --bidirectional --simulate --scale \
    2>&1 | tee "log/train_${ARCH}.log"
