#!/bin/bash
#PBS -l select=1:ncpus=2:gpu_id=0
#PBS -l place=shared
#PBS -o log/out.out				
#PBS -e log/err.out		
#PBS -N eval_val

cd ~/Opensees_based_damage_pattern_generation		

source ~/.bashrc			
conda activate hysteresis	

module load cuda-12.4


python3 eval_mixed.py 2>&1|tee log/eval_val.log
