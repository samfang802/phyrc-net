#!/bin/bash
#PBS -l select=1:ncpus=1:gpu_id=3
#PBS -l place=shared
#PBS -o log/out.out				
#PBS -e log/err.out		
#PBS -N RSGAN_512_latent

cd ~/Opensees_based_damage_pattern_generation		

source ~/.bashrc			
conda activate hysteresis	

module load cuda-12.4


python3 train_mixed.py --lr 1e-4 --epoch 3000 --batch_size 4 --architecture HystereticCGAN --loss_type RSGAN --data_path id_list_old.json\
        --latent_condition True --direction_conditioning True --extractor_path ./results/HystereticPrediction/HystereticGRU/2025-08-01_13-28-08 \
        --add_DI_loss False --image_size 512 --z_dim 100 --G_h_size 128 --D_h_size 128 --SELU False --spectral_D False \
        --no_batch_norm False 

