#!/bin/bash
#PBS -l select=1:ncpus=1:gpu_id=1
#PBS -l place=shared
#PBS -o log/out.out				
#PBS -e log/err.out		
#PBS -N VQGAN

cd ~/Opensees_based_damage_pattern_generation		

source ~/.bashrc			
conda activate hysteresis	

module load cuda-12.4


python3 train_reconstruction.py --lr 4.5e-06 --epoch 1000 --disc_start 200 --batch_size 8 --embed_dim 4 --n_embed 256 --resolution 256 --architecture VQGAN

