#!/bin/bash
#PBS -l select=1:ncpus=1:gpu_id=2
#PBS -l place=shared
#PBS -o log/out.out				
#PBS -e log/err.out		
#PBS -N vit_l_16

cd ~/Opensees_based_damage_pattern_generation		

source ~/.bashrc			
conda activate hysteresis	

module load cuda-12.4


python3 train_quantification.py --lr 1e-6 --epoch 500 --batch_size 8 --backbone vit_l_16 --pretrained 2>&1 | tee log/train_vit_l_16.log

