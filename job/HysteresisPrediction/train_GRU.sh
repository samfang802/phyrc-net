#!/bin/bash
#PBS -l select=1:ncpus=1:gpu_id=5
#PBS -l place=shared
#PBS -o log/out.out				
#PBS -e log/err.out		
#PBS -N GRU

cd ~/Opensees_based_damage_pattern_generation		

source ~/.bashrc			
conda activate hysteresis	

module load cuda-12.4


python3 train_hysteresis.py --lr 1e-4 --epoch 5000 --batch_size 8 --architecture HystereticGRU --data_size big \
        --hidden_state_dim 512 --layers 4 --MLP_activation softsign --bidirectional --simulate --scale  2>&1 | tee log/train_HystereticGRU_softsign2.log

