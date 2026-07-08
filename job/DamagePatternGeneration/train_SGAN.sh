#!/bin/bash
#PBS -l select=1:ncpus=1:gpu_id=4
#PBS -l place=shared
#PBS -o log/out.out				
#PBS -e log/err.out		
#PBS -N SGAN_128_DI

cd ~/Opensees_based_damage_pattern_generation		

source ~/.bashrc			
conda activate hysteresis	

module load cuda-12.4


python3 train_mixed.py --lr 1e-4 --epoch 3000 --batch_size 4 --architecture HystereticCGAN --loss_type SGAN \
        --latent_conditioning False --extractor_path ./results/HystereticPrediction/HystereticGRU/2024-06-12_18-13-47 \
        --regressor_path ./results/DamageQuantification/vit_b_16/2024-06-12_14-01-03 --add_DI_loss False \
        --image_size 128 --z_dim 100 --G_h_size 128 --D_h_size 128 --SELU False --spectral_D False \
        --no_batch_norm False 

