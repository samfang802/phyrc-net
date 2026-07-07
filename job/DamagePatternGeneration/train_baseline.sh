#!/bin/bash
#PBS -l select=1:ncpus=1:gpu_id=6
#PBS -l place=shared
#PBS -o log/out.out				
#PBS -e log/err.out		
#PBS -N Baseline_DI

cd ~/Opensees_based_damage_pattern_generation		

source ~/.bashrc			
conda activate hysteresis	

module load cuda-12.4


python3 train_mixed.py --lr 1e-4 --epoch 3000 --batch_size 4 --architecture Baseline --loss_type Baseline\
        --latent_condition False  --extractor_path ./results/HystereticPrediction/HystereticGRU/2024-06-12_18-13-47 \
        --image_size 512 --z_dim 100 --G_h_size 128 
