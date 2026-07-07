# Physics-Guided GAN-Based Prognostic Framework for Hysteresis Behavior and Damage Pattern Predictions in RC Bridge Columns

This is a repository for Physics Based Damage Patterns Generation.


## Training

All the training processes were controlled by the bash files, which can be found in `./job`.
The details of the arguments can be found in `./argument`.


## Experiment and Result
### Experiment of HysGRU
| Simulation Involved | Force Unit | Force Normalization | Result directory |
| :---: | :---: | :---: | :---: |
| False | kN | False | ./results/HystereticPrediction/HystereticGRU/2024-06-12_18-12-05|
| True | kN | False | ./results/HystereticPrediction/HystereticGRU/2024-06-12_18-12-54|
| False | MN | False | ./results/HystereticPrediction/HystereticGRU/2024-06-13_16-28-18|
| True | MN | False | ./results/HystereticPrediction/HystereticGRU/2024-06-13_16-28-08|
| True | None | True | ./results/HystereticPrediction/HystereticGRU/2024-06-12_18-13-47|


### Experiment of HysGAN
| Condition | Loss Function | Image Resolution | Result directory |
| :---: | :---: | :---: | :---: |
| Info. + DI| End-to-End| 128 | ./results/DamagePatternGeneration/Baseline/2024-06-15_20-03-23 |
| Info. + DI| End-to-End| 512 | ./results/DamagePatternGeneration/Baseline/2024-06-17_12-17-37 |
| Latent | End-to-End| 128 | ./results/DamagePatternGeneration/Baseline/2024-06-15_20-07-08 |
| Latent | End-to-End| 512 | ./results/DamagePatternGeneration/Baseline/2024-06-17_12-14-26 |
| Info. + DI| SGAN | 128 | ./results/DamagePatternGeneration/SGAN/2024-06-15_19-36-55 |
| Latent| SGAN | 128 | ./results/DamagePatternGeneration/SGAN/2024-06-15_01-25-09 |
| Latent | SGAN| 256 | ./results/DamagePatternGeneration/SGAN/2024-06-15_01-23-07 |
| Latent | SGAN| 512 | ./results/DamagePatternGeneration/SGAN/2024-06-15_01-21-08 |
| Info. + DI| RSGAN | 128 | ./results/DamagePatternGeneration/RSGAN/2024-06-15_19-36-55 |
| Latent| RSGAN | 128 | ./results/DamagePatternGeneration/RSGAN/2024-06-15_01-25-10 |
| Latent | RSGAN| 256 | ./results/DamagePatternGeneration/RSGAN/2024-06-15_01-23-09 |
| Latent | RSGAN| 512 | ./results/DamagePatternGeneration/RSGAN/2024-06-15_01-21-11 |
| Info. + DI| RasGAN | 128 | ./results/DamagePatternGeneration/RaSGAN/2024-06-15_19-36-55 |
| Latent| RasGAN | 128 | ./results/DamagePatternGeneration/RaSGAN/2024-06-15_01-25-13 |
| Latent | RasGAN| 256 | ./results/DamagePatternGeneration/RaSGAN/2024-06-15_01-23-11 |
| Latent | RasGAN| 512 | ./results/DamagePatternGeneration/RaSGAN/2024-06-15_01-21-13 |


### Experiment of DI prediction network
| Architecture |  Result directory |
| :---: | :---: |
| vit_b_16 | ./results/DamageQuantification/vit_b_16/2024-06-12_14-01-03 |
| vit_b_32 | ./results/DamageQuantification/vit_b_32/2024-06-12_15-59-56 |
| vit_l_16 | ./results/DamageQuantification/vit_l_16/2024-06-12_14-01-08 |
| vit_l_32 | ./results/DamageQuantification/vit_l_32/2024-06-12_15-59-50 |

