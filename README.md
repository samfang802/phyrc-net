# Physics-Guided Damage Image Generation for RC Bridge Columns

Generates image sequences of damage on reinforced-concrete (RC) bridge columns under cyclic loading. The input is the loading history (hysteresis loop and drift); the output is one damage image per loading peak.

Built with PyTorch and PyTorch Lightning. Most design notes in [`docs/`](docs/README.md) are written in Traditional Chinese.

## Pipeline

| Stage | What it does | Entry point | Notes |
|---|---|---|---|
| 1. Damage state | Encodes the hysteresis history into a per-frame damage state (`z` + damage scalars + loading sign) | `train_damage_encoder.py` (older GRU: `train_hysteresis.py`) | [DAMAGE_STATE_ENCODER.md](docs/DAMAGE_STATE_ENCODER.md) |
| 2. Single-frame generator | Pixel-space ADM U-Net trained with conditional flow matching; V12 adds a spalling-layout input channel | `train_mixed.py` via `job/DamagePatternGeneration/train_ADMCFM_v12.sh` | [V11](docs/ARCHITECTURE_UPDATE_V11.md), [V12](docs/ARCHITECTURE_UPDATE_V12.md) |
| 3. Temporal refiner | Redraws frame *t* from the previous frame *S<sub>t−1</sub>* so damage accumulates over the sequence; rfmv2 adds autoregressive resampling and rule-based memory | `train_mixed.py` via `job/DamagePatternGeneration/train_ADMCFM_v12refiner.sh` | [RFMV2_ARCHITECTURE.md](docs/RFMV2_ARCHITECTURE.md) |
| 4. Evaluation | Generates samples, removes test-rig interference, then scores realism, condition consistency, damage texture, copying and temporal behaviour | `job/evaluation/eval_suite.sh` | [docs/eval/](docs/eval/README.md) |

Earlier versions (V9–V10.6) are described in `docs/ARCHITECTURE_UPDATE_V*.md`. The latest results for all versions are in [docs/eval/results.md](docs/eval/results.md).

A separate DI (damage index) regressor, used only for evaluation, lives in [`quantification/`](quantification/).

## Repository layout

```text
argument/                 command-line flags (mixed.py = generator / refiner)
src/
  data/                   datasets, damage-state conditioning, layout I/O
  model/                  ADM U-Net, CFM mixins (model/cfm/), damage-state encoder, regressors
  metrics/                KID / DINO / damage-texture metrics
scripts/
  eval/                   evaluation suite and diagnostic probes
  viz/  export/  data/    figures, dataset export, caches
quantification/           DI regressors (run as python -m quantification.<script>)
job/                      PBS job scripts, see job/README.md
docs/                     architecture and experiment notes
train_mixed.py            generator / refiner training
test_cfm_checkpoint.py    sampling from a CFM checkpoint
train_damage_encoder.py   damage-state encoder
train_hysteresis.py       hysteresis GRU (stage 1, older conditioning)
```

## Setup

```bash
conda env create -f environment.yml -n internship
conda activate internship
```

Or with pip: `pip install -r requirements.txt`, then install the PyTorch wheel that matches your CUDA ([pytorch.org](https://pytorch.org/get-started/locally/)).

Training jobs log to Weights & Biases when run with `--use_wandb` (needs `wandb login`); otherwise they write CSV logs.

## Data

Expected layout (run everything from the repository root):

- `Data/Hysteresis/`: specimen info (`info.csv`, `info_w_mander_big.csv`), statistics, splits (`id_list_big.json`, `fold.json`), simulated hysteresis loops (`Simulate/`)
- `Data/DamagePattern/`: `label.csv`, splits (`id_list_*.json`), damage masks (`mask/`), hand-labelled spalling polygons (`anylabel/`, `anylabel_r_spall.csv`)

**Not included in this repository:** the original column photographs (`Data/DamagePattern/images/`), the measured hysteresis loops and their source XML (`Data/Hysteresis/Experiment/`, `Data/Hysteresis/xml/`), and the Mander / PDSHM source spreadsheets. Training the generator needs the photographs.

## Running

The scripts under `job/` are PBS jobs for our cluster (`qsub ...`). They `cd` to the submission directory and activate the `internship` conda environment through `job/DamagePatternGeneration/common.sh`; each script's header lists its options. See [job/README.md](job/README.md).

```bash
# Stage 1: damage-state encoder
qsub -v MODEL=damage_state job/HysteresisPrediction/train_hysteresis.sh

# Stage 2: V12 single-frame generator
qsub -v LAYOUT_CACHE=<layout cache> job/DamagePatternGeneration/train_ADMCFM_v12.sh

# Stage 3: temporal refiner (rfmv2 recipe)
qsub -v RECIPE=rfmv2 job/DamagePatternGeneration/train_ADMCFM_v12refiner.sh

# Stage 4: evaluation of the final checkpoints
qsub -v CONFIG=job/evaluation/configs/eval_suite_final.json job/evaluation/eval_suite.sh
```

To run without PBS, call the `python` command inside the job script directly from the repository root. The generator needs about 20 GB of GPU memory at 128 × 128.

Outputs go to `results/` (one folder per run with `args.json`, checkpoints and logs). `results/` is not tracked.
