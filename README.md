# Physics-Guided Damage Prediction and Generation for RC Bridge Columns

This repository trains a **frozen hysteresis GRU** and a **conditional flow-matching (CFM)** generator that synthesizes RC-column damage images from structural signals.

The default generator is **V9 / selfcondnoDI**: pixel-space ADM U-Net, split gated AdaGN, zeroed DI, classic self-conditioning, VSS at L3/U3, and Patch Diffusion. See [docs/ARCHITECTURE_UPDATE_V9.md](docs/ARCHITECTURE_UPDATE_V9.md).

Built with PyTorch and PyTorch Lightning.

## Project Layout

```text
intership-project/
|- argument/
|  |- mixed.py                         # Stage-3 CFM flags
|  |- argument_mixed.py                # re-export used by train_mixed.py
|- job/DamagePatternGeneration/
|  |- common.sh                        # conda, CUDA, nice
|  |- train_ADMCFM.sh                  # V9 default job
|- src/
|  |- data/                            # dataset implementations
|  |- dataset.py                       # import path kept for trainers
|  |- model/
|  |  |- adm.py                        # ADMUNetCFM
|  |  |- vss.py
|  |  |- cfm/                          # default CFM
|  |  |- exp/                          # wavelet, Laplacian, FreCaS, MAFP, ...
|  |  |- ImageGeneration.py
|- scripts/
|  |- eval/  viz/  export/
|- train_mixed.py
|- docs/                               # current notes; older ones in docs/archive/
```

Project notes are in [docs/](docs/README.md). Historical architecture notes are in `docs/archive/`.

## Environment Setup



### 1) Create and activate a virtual environment

```bash
python -m venv .venv
source .venv/bin/activate
```



### 2) Install dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```



### 3) Install PyTorch

Use the wheel that matches your CUDA: [https://pytorch.org/get-started/locally/](https://pytorch.org/get-started/locally/)

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```



### 4) Optional: Weights & Biases

The cluster job `train_ADMCFM.sh` expects wandb login (`wandb login` or `WANDB_API_KEY`). Local runs can omit `--use_wandb` and use CSV logs.

## Data Requirements

Scripts expect:

- `./Data/Hysteresis/...`
- `./Data/DamagePattern/...`

Typical files:

- `Data/Hysteresis/id_list_*.json`
- `Data/Hysteresis/info_w_mander_*.csv`
- `Data/Hysteresis/statistics_*.csv`
- `Data/DamagePattern/label.csv`
- `Data/DamagePattern/mask/`
- split json via `--data_path` (V9: `id_list_val_R307.json`)

Generation also needs a frozen extractor at `./phase1_extractor/` (`best.ckpt` + `args.json`). Prepare with:

```bash
bash job/DamagePatternGeneration/prepare_extractor_from_hf.sh
```



## How to Run

Run all commands from the project root.

### 1) Train hysteresis predictor (Stage 1)

Only needed if you are not using `phase1_extractor`:

```bash
python train_hysteresis.py --architecture HystereticGRU --epoch 100 --batch_size 8
```



### 2) Train damage-pattern generator (Stage 3, V9)

This version uses nearly 20 GB of VRAM.

Cluster (matches `results/DamagePatternGeneration/CFMADM/selfcondnoDI`):

```bash
qsub job/DamagePatternGeneration/train_ADMCFM.sh
```

Equivalent local flags (subset):

```bash
python train_mixed.py \
  --architecture CFMADM \
  --loss_type CFMADM \
  --extractor_path ./phase1_extractor \
  --data_path id_list_val_R307.json \
  --condition_mode split \
  --zero_di_condition True \
  --self_condition True \
  --use_wavelet_unet False \
  --use_physics_cross_attn False \
  --use_vss_l3_u3 True \
  --use_patch_diffusion True \
  --use_damage_mask_condition True \
  --use_object_loss_mask False \
  --latent_conditioning True \
  --direction_conditioning True \
  --include_di_conditioning True \
  --image_size 128 \
  --max_steps 15000
```

`--self_condition` is an architecture bool (alias `--use_self_conditioning`). Default is `True`. Set `False` only for a no-self-cond ablation; that checkpoint **cannot** load selfcondnoDI weights.

### 3) Test a CFM checkpoint

```bash
bash job/DamagePatternGeneration/test_ADMCFM.sh
```

Default checkpoint: `results/DamagePatternGeneration/CFMADM/selfcondnoDI/checkpoint_step_014000.pt`. Override with `CKPT_PATH` / `RESULT_DIR`.

### 4) Optional: DI regressor (evaluation, not used inside V9 CFM loss)

```bash
python train_resnet18_siamese.py
python test_resnet18_siamese.py
```

Older ViT `train_quantification.py` remains available; V9 generation does not load it (`add_DI_loss=False`).

## Evaluation scripts

Under `scripts/eval/`, for example:

- `eval_fid_ssim_lpips_cfm_cgan.py`
- `eval_cfm_generated_r2_siamese.py`
- `compute_test_lpips.py`

Point them at a CFM `result_dir` that contains `args.json` and a `.pt` checkpoint.

## Outputs

Training writes under `./results/`:

- `results/HystereticPrediction/...`
- `results/DamagePatternGeneration/CFMADM/...`
- `results/DamageQuantification/...`

Each run stores `args.json`, checkpoints, and (if enabled) wandb artifacts.

Reference V9 run: `results/DamagePatternGeneration/CFMADM/selfcondnoDI/`.

## Notes

- V9 is **not** the V8 CrossAttention U-Net and **not** the WaveletUNet job. Do not mix those checkpoints with `train_ADMCFM.sh` as it stands now.
- Laplacian / cascade / VAE-latent jobs under `job/DamagePatternGeneration/` are optional experiments.
- `--regressor_path` is only required when `--add_DI_loss True`.



## Citation

If you use this repository in academic work, please cite your associated paper/project and include a link to this repository.