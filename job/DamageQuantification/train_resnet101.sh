#!/bin/bash
#PBS -l select=1:ncpus=1:gpu_id=2
#PBS -l place=shared
#PBS -o log/out.out				
#PBS -e log/err.out		
#PBS -N resnet101

cd ~/Opensees_based_damage_pattern_generation		

source ~/.bashrc			
conda activate hysteresis	

module load cuda-12.4


python3 train_quantification.py --lr 1e-6 --epoch 500 --batch_size 8 --backbone resnet101 --pretrained

