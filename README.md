# Physics-Guided Damage Prediction and Generation for RC Bridge Columns

This repository provides deep learning pipelines for:

- hysteresis force prediction from structural response signals,
- damage index (DI) regression from images and metadata,
- conditional damage pattern generation (GAN-based),
- damage image reconstruction (VQGAN/VAE).

The codebase is built with PyTorch and PyTorch Lightning.

## Project Layout

```text
intership-project/
|- argument/                        # CLI argument definitions
|- job/                             # Example shell scripts for training
|- Data/
|  |- Hysteresis/                   # Hysteresis csv/xml data and metadata
|  |- DamagePattern/                # Damage images, labels, split json
|- src/
|  |- dataset.py                    # Dataset classes for all tasks
|  |- model/
|  |  |- HystereticPrediction.py    # LSTM/GRU models
|  |  |- DamageQuantification.py    # DI regression model
|  |  |- ImageGeneration.py         # Conditional GAN models
|  |  |- ImageReconstruction.py     # VQGAN/VAE models
|  |  |- Component/                 # Shared building blocks
|  |- losses/                       # Perceptual/adversarial losses
|- train_hysteresis.py
|- train_quantification.py
|- train_mixed.py
|- train_reconstruction.py
|- test_*.py / eval_mixed.py        # Inference and evaluation scripts
```

## Environment Setup

### 1) Create and activate a virtual environment

```bash
python -m venv .venv
```

Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

macOS/Linux:

```bash
source .venv/bin/activate
```

### 2) Install dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 3) Install PyTorch (recommended from official selector)

If your GPU/CUDA setup requires a specific wheel, install from:
[https://pytorch.org/get-started/locally/](https://pytorch.org/get-started/locally/)

Example (CUDA 12.1):

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

### 4) Optional: Weights & Biases login

```bash
wandb login
```

## Data Requirements

By default, scripts expect:

- `./Data/Hysteresis/...`
- `./Data/DamagePattern/...`

Key files used by datasets include:

- `Data/Hysteresis/id_list_*.json`
- `Data/Hysteresis/info_w_mander_*.csv`
- `Data/Hysteresis/statistics_*.csv`
- `Data/DamagePattern/label.csv`
- split json file referenced by `--data_path` for generation

## How to Run

All commands below are executed from the project root.

### 1) Train Hysteresis Predictor

```bash
python train_hysteresis.py --architecture HystereticLSTM --epoch 100 --batch_size 8
```

Alternative (GRU):

```bash
python train_hysteresis.py --architecture HystereticGRU --epoch 100 --batch_size 8
```

### 2) Train Damage Quantification (DI Regression)

```bash
python train_quantification.py --backbone vit_b_16 --pretrained --epoch 100 --batch_size 8
```

### 3) Train Damage Pattern Generator (Mixed Pipeline)

Before running this stage, make sure you have:

- a trained hysteresis model checkpoint directory (`--extractor_path`),
- a DI model checkpoint directory (`--regressor_path`).

Example:

```bash
python train_mixed.py \
  --architecture HystereticCGAN \
  --extractor_path ./results/HystereticPrediction/HystereticLSTM/<run_id> \
  --regressor_path ./results/DamageQuantification/vit_b_16/<run_id> \
  --data_path id_list.json \
  --latent_conditioning true \
  --direction_conditioning true \
  --loss_type SGAN \
  --add_DI_loss true \
  --image_size 128 \
  --epoch 3000
```

### 4) Train Image Reconstruction (VQGAN / VAE)

VQGAN:

```bash
python train_reconstruction.py --architecture VQGAN --resolution 256 --epoch 100
```

VAE:

```bash
python train_reconstruction.py --architecture VAE --resolution 256 --epoch 100
```

## Colab-Friendly Notes

- By default, all training scripts now use `CSVLogger` and **do not require W&B login**.
- Add `--use_wandb` only when you want remote experiment tracking.
- For `train_mixed.py`, both `--extractor_path` and `--regressor_path` are required.
- You can lower loader workers on Colab if needed:

```bash
python train_hysteresis.py --num_workers 1
```

- A quick Colab starter script is available at `scripts/colab_quickstart.sh`.

## Evaluation and Testing

Examples:

```bash
python test_hysteresis.py
python test_quantification.py
python test_mixed.py
python eval_mixed.py
```

`plot_peak.py` can be used to export videos/images for selected experiments.

## Outputs

Training outputs are written under `./results/` with task-based subfolders, for example:

- `results/HystereticPrediction/...`
- `results/DamageQuantification/...`
- `results/DamagePatternGeneration/...`
- `results/ImageReconstruction/...`

Each run stores `args.json`, checkpoints, and logged artifacts.

## Notes

- Legacy latent-diffusion helper blocks under `src/model/Component/diffusion/` were removed during cleanup because they were not connected to the active training pipeline.
- `argument/argument.py` and cross-validation training entrypoints were removed to reduce maintenance and keep a single primary training path.

## Citation

If you use this repository in academic work, please cite your associated paper/project and include a link to this repository.

