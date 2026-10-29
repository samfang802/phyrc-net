# ARCHITECTURE_UPDATE_V10

## Delta Scope (V10 Only)

V10 adds three things on top of the V9 single-frame generator (`train_ADMCFM.sh`). None of them changes the V9 network or its defaults; every feature is opt-in.

1. **Autoguidance sampling**, with a time schedule, a parallel-part weight and frequency-decoupled weights.
2. **Data pipeline**: an optional in-memory cache, and batches that carry the frame two steps back.
3. **Temporal refiner**, in two variants:
  - **A. Paired bridge with a model-generated start** (first trial, finished).
  - **B. Residual refiner with a locked state channel, a stop-gradient state track and a** `Δg` **condition** (current).



### Status at writing


| Item                             | State                                                                                                                                                                        |
| -------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Autoguidance                     | Implemented. 6x6 High-DI grids produced. **No quantitative evaluation.**                                                                                                     |
| Data cache / three-frame batches | Implemented and used by variant B. Per-frame cost measured; **resulting step speed not measured on the final configuration.**                                                |
| Refiner A (bridge)               | Trained 3000 steps. **Not evaluated.**                                                                                                                                       |
| Refiner B (residual, with `Δg`)  | A 6-step smoke test passed (warm start, training, validation, 64-frame sampling). The full 3000-step run (job `56093`) was just submitted. **No loss curve, no evaluation.** |


Nothing in this file is a measured result unless it says so.

---



## 1. Autoguidance



### Idea

Guide the main CFM velocity `v` with a weaker copy of itself (Karras et al. 2024) instead of an unconditional branch. V9 has no unconditional branch (`CFG_G_DROP_PROB=0.0`), so classifier-free guidance is not available.

The weak ("bad") model has the same architecture, data and conditions as the main model, with 16 instead of 32 channels and a short schedule. At sampling both models see the same `z_t`, `t`, `c`, `g` and patch coordinates.

### Update rule

```
d      = v - v_bad
d_low, d_high = split(d)                  # 2x down/up resize, same split as FA-CFG
d      = wl * d_low + wh * d_high         # frequency-decoupled weights
d      = d_perp + eta * d_par             # d_par is the part of d parallel to v
w(t)   = max(w_min, w * (1 - t)^gamma)    # t=0 noise, t=1 data
v      = v + w(t) * d
```

Defaults are no-ops: `wl = wh = eta = 1`, `gamma = 0`, `w_min = 0`. With only `autoguide_weight` set, the update is the plain `v + w (v - v_bad)`.

### Parameters


| Parameter                                      | Meaning                                                                                              |
| ---------------------------------------------- | ---------------------------------------------------------------------------------------------------- |
| `autoguide_checkpoint`, `autoguide_result_dir` | Frozen bad model and the run dir holding its `args.json`                                             |
| `autoguide_weight`                             | `w` (0 disables)                                                                                     |
| `autoguide_eta`                                | Weight on the part of `d` parallel to `v`. The tertile script defaults to 0, other entry points to 1 |
| `autoguide_gamma`, `autoguide_w_min`           | Time schedule                                                                                        |
| `autoguide_wl`, `autoguide_wh`                 | Low/high frequency weights                                                                           |




### Code

- `src/model/cfm/ode_mixin.py`, `_guided_velocity`: applied right after the main prediction and before the other guidance terms. Heun stage 1 and stage 2 both use it. **ODE only**; `sample_with_sde` is unchanged.
- `src/model/cfm/lifecycle_mixin.py`: `_load_frozen_cfm` (shared loader, also used by the refiner) and `_attach_autoguide`. The bad model is held in a list so it is not an `nn.Module` child and is not trained or saved with the main model. If the checkpoint's `scalar_mlp` has 3 inputs while `zero_di_condition` is on, the loader sets `keep_zero_di_slot` so the model matches the checkpoint.
- `src/model/cfm/options.py`: the fields above.
- Entry points with CLI flags: `scripts/viz/export_di_tertile_overview_grids.py`, `scripts/eval/eval_cfm_generated_r2_siamese.py`.



### Jobs and outputs

- Bad model: `train_ADMCFM.sh` with overrides — `qsub -v CFM_MODEL_CHANNELS=16,MAX_STEPS=6000,EARLY_STOP_PATIENCE=3000,CHECKPOINT_EVERY_N_STEPS=1000,CHECKPOINT_START_STEP=1000,RESUME_CKPT=none,RUN_TAG=autoguide_bad job/DamagePatternGeneration/train_ADMCFM.sh` (the separate `train_ADMCFM_autoguide_bad.sh` was merged into it; last version in git `80f71fa`) (16 channels, `num_res_blocks=1`, dropout 0.1, checkpoints every 1000 steps). The first recipe was about 3000 steps; the script now sets `MAX_STEPS=6000`. Checkpoints of run `2026-10-12_03-59-39_autoguide_bad` at step 3000 and 6000 have both been used.
- 6x6 High-DI grids: `qsub -v MODE=autoguide_fdg job/DamagePatternGeneration/viz_high_di_grid.sh`. It runs three variants (no guidance, autoguidance, autoguidance + FDG with `wh=0.5`). Choose the bad checkpoint with `-v BAD_STEP=003000` or `006000`.
- Output names carry `_ag..`, `_g..`, `_fdgL..H..` and `_bad<step>` suffixes. Files made before the `_bad` suffix existed do not record which bad checkpoint they used.
- `results/.../1024-dadaGNsplit/inference_samples/` is grouped into `autoguide/`, `other_guidance/`, `flowedit_sdedit/` and `baseline_6x6/`. The JSON files and `fixed_di_*.png` stay at the top level because scripts read and write them there (the High-DI JSON locks the cell order).



### Caveats

- The main model is `1024-dadaGNsplit/.../chjqbp9m` (no spall mask). The bad model must match the main model's input layout (spall mask, self-conditioning).
- Guidance strength was chosen by eye. Earlier observation: conditional velocity differences are small (mean about 0.055), and a high-frequency weight of 2 amplified grain in 30-step Heun. Keep `wh <= 1` for first tries.
- `gamma = 4` comes from the image-CFG literature; with only 30 steps it makes guidance almost vanish in the last steps.

---



## 2. Data pipeline

Files: `src/data/pattern_loop.py` (dataset), `argument/mixed.py` (flags).

### 2.1 In-memory cache (`--cache_data`, default False)

**Where the time went.** Measured on one CPU core for a roughly 750x800 RGB image (login node, 20-30 frames):


| Step                                                           | Cost per frame |
| -------------------------------------------------------------- | -------------- |
| Decode from disk                                               | 2.5 ms         |
| Resize to 128, grayscale, CLAHE (all deterministic)            | 12.8 ms        |
| Random brightness/contrast jitter, blur, dtype, channel repeat | 0.25 ms        |


A refiner sample has 3 frames, so one batch of 128 is 384 frames, about 5.8 s of CPU work; with 2 workers about 2.9 s per step. The first residual-refiner run showed about 0.36 it/s at step 5, which matches this. The first version of the cache stored only decoded arrays and saved just the 2.5 ms decode, so it did not help.

**Current cache.** At dataset build, per split:


| Cached                                                                                             | Where                                      | Notes                                                                                                                                                                                                            |
| -------------------------------------------------------------------------------------------------- | ------------------------------------------ | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Output of the deterministic head (Resize, ToImage, Grayscale, CLAHE) as a uint8 `1x128x128` tensor | `_prefix_cache`                            | 16 KB per image, so about 10 MB per split. `MaskedDamageImageTransform.enable_prefix_cache` splits the pipeline, `precompute` runs the head once, and `__call__(..., image_is_prefixed=True)` runs only the tail |
| Decoded arrays                                                                                     | `_img_cache`                               | Fallback only, when the transform cannot be split                                                                                                                                                                |
| Object mask tensors                                                                                | `_mask_cache`, keyed by (file, image size) | Returned as a clone                                                                                                                                                                                              |
| `info` row per experiment                                                                          | `_info_cache`                              | Always on, regardless of the flag                                                                                                                                                                                |


Flip, jitter, blur and random erasing still run on every call, so the training distribution is unchanged. Equivalence was checked with fixed seeds: the cached path and the full pipeline give identical image, loss mask and flip flag (8 images x 3 seeds, with flip, erasing and an object mask). The cache is filled before the DataLoader workers start.

Expected cost per frame after caching is about 0.3 ms plus tensor handling. A run with this cache showed the two loader workers at about 14% CPU each, so loading stopped being the bottleneck. In that run the main process was at about 93% CPU, and the step rate averaged about 0.64 it/s (555 steps in about 14.5 minutes including start-up and a validation every 5 steps). The old 0.36 it/s was an instantaneous reading at step 5, so the two figures are not directly comparable.

### 2.3 CPU budget and loader settings

The refiner job requests **4 CPUs**: the main process, 2 loader workers, and a little headroom. Changes in `train_mixed.py` and the job script:

- Each worker sets `torch.set_num_threads(1)` and `cv2.setNumThreads(0)` (`_worker_init`), so workers do not each spawn a full thread pool.
- The job exports `OMP_NUM_THREADS=2` and `MKL_NUM_THREADS=2` for the main process.
- Train loader: persistent workers, `prefetch_factor=4`, `pin_memory` on CUDA. Val loader: same worker init and `pin_memory`.
- `NUM_WORKERS` defaults to 2 in the job script and can be overridden.

The main process, not the workers, is what is busy. The cause is not confirmed; kernel-launch overhead in the U-Net is a guess. The GPU is an RTX A6000 and `torch.set_float32_matmul_precision` is not set (TF32 / mixed precision untested).

### 2.2 Frame two steps back (`use_refiner_residual` turns it on)

Existing temporal pairing gives `image_prev` (same experiment and direction, previous timestep). When the refiner is on, the dataset also returns, for the frame before that one:

`image_prev2`, `hidden_state_prev2`, `direction_prev2`, `DI_prev2`, `has_prev2`.

The same horizontal flip is applied to all frames of one sample. Where a frame does not exist (the first two frames of a column), the previous frame's tensors are copied in as a placeholder and `has_prev` / `has_prev2` are 0. `_unpack_generation_batch` in `condition_mixin.py` passes these through in `ar_pack`.

---



## 3. Temporal refiner



### 3.1 Problem

A pure incremental target (`x_t - x_{t-1}`) can at best copy the true increment. The single-frame model's own bias (global under-estimation, suppressed high-damage regions) is never corrected and accumulates along the chain. A refiner must predict the current frame from a state that carries the base model's real inference error, and must be trained on states that look like the ones it will see at inference.

Both variants use a frozen single-frame model as the "base" (`chjqbp9m`, loaded by `_attach_refiner_base`). The base is held in a list, has no gradient, and is conditioned with its own `_unpack_conditions`.

### 3.2 Variant A: paired bridge with a model-generated start

Reuse the paired bridge (`docs/archive/PAIRED_BRIDGE.md`) and change only where the bridge start comes from.

```
S_prev = frozen single-frame model sample for the previous frame
x0     = (1 - s) * S_prev + s * eps,    s ~ U(0, paired_bridge_noise_max)
z_t    = (1 - t) * x0 + t * x_target
v      = x_target - x0
```

- `S_prev`: with probability `1 - refiner_roll_prob` (0.85) per batch, one Euler step from noise; with probability `refiner_roll_prob` (0.15), a `refiner_roll_nfe` (10) evaluation Heun rollout, which is 5 Heun steps. In evaluation it is always the full rollout and `s = 0`.
- Warm start from the same checkpoint through `VELOCITY_INIT`.
- Code: `_refiner_base_prev_state` in `loss_mixin.py`; the paired-bridge branch of `_flow_matching_loss_for_batch` in `training_mixin.py` swaps in the base sample when `refiner_base_checkpoint` is set.
- Job: `job/DamagePatternGeneration/train_ADMCFM_refiner.sh` (removed with the job cleanup; `git show 80f71fa:job/DamagePatternGeneration/train_ADMCFM_refiner.sh`). Sampling uses the existing `sample_paired_bridge_sequence` unchanged.
- Run history: job `56082` (GPU 5) was stopped at step 1195 and resumed from a saved Lightning checkpoint as job `56083` on GPU 0, which reached 3000 steps. Weights: `results/DamagePatternGeneration/CFMADM/2026-10-12_23-13-34_refiner/checkpoint_step_003000.pt`. This is the control for variant B.
- Weakness: the bridge integrates from `S_prev`, so the previous frame's brightness is inherited along the path (see 3.6).



### 3.3 Variant B: residual refiner with a locked state channel

Enabled by `--use_refiner_residual True` (requires `--refiner_base_checkpoint`). It cannot be combined with paired bridge, spall mask, AR conditioning, self-conditioning or cascade SR; `options.py` rejects those combinations.

**Network input.** The state `S_{t-1}` is one more input block of the same channel count as the image, placed after the image channels:

```
[ z (C) | S_{t-1} (C) | coord_x, coord_y | damage mask (added inside forward) ]
```

It reuses the cascade `lr_cond` path (`_build_velocity_net_input`, `_ensure_cascade_lr_cond`, `_apply_patch_crop(z_aux=...)`), so patch crops keep the state aligned with the noisy image. `net_in_channels` in `options.py` gains `C` for this block.

**Output composition.** The network samples a residual from noise; the frame is `clamp(S_{t-1} + residual, -1, 1)`, added outside the network by `_compose_cascade_sample`.

**Loss track** (`refiner_residual_flow_matching_loss`, gradient flows):

```
r   = x_t - S_in                     # target residual, S_in is the clean state
x0  ~ N(0, I)
z_tau = (1 - tau) x0 + tau r,   tau ~ U(0, 1)
v   = r - x0
input state channel = S_in + noise,  noise std ~ U(0, refiner_state_noise_max)   (training only)
loss = MSE(v_pred, v), Min-SNR weighted, averaged over samples with has_prev = 1
```

**State track** (same function, `no_grad`): builds `S_in` for the loss track.

- *Single hop* (probability `1 - refiner_chain_prob`): `S_in` = frozen base sample for frame `t-1`, conditioned on that frame's `c`, `g`, direction and DI.
- *Chain* (probability `refiner_chain_prob`, 0.5 per batch; always in evaluation): `S0` = base sample for frame `t-2`; then this network, with `S0` locked and frame `t-1`'s conditions, produces `S1 = clamp(S0 + residual)`; `S_in = S1`. The state therefore carries the refiner's own error. For samples without a frame `t-2`, `S0` is the base sample of frame `t-1` and the hop is skipped.
- Each rollout is one Euler step from noise, or with probability `refiner_roll_prob` (0.15, drawn independently per call) a `refiner_roll_nfe` (10) evaluation Heun rollout.
- The network runs in training mode during the state track, so dropout also perturbs the state.

**Warm start** (`_load_velocity_init_with_state_channels` in `train_mixed.py`), from the single-frame checkpoint:

1. `in_conv` grows from 6 to 9 input channels: the `C` new state channels are inserted right after the image channels with zero weights.
2. The 3-input `scalar_mlp` of the old checkpoint is converted to 2 inputs by the existing load hook in `adm.py` (drops the leading always-zero DI slot).
3. `out_conv` **weights and bias are set to zero** (`--refiner_zero_init_out`, default True), so the initial velocity is 0.

Why step 3, and a correction: with only step 1, a first run showed every generated frame saturating to white at step 5. The single-frame head outputs a whole image; used as a residual it adds a full image to `S_{t-1}` at every frame. Zeroing the output layer was meant to start from "residual = 0". **That reasoning was wrong for a noise-start flow.** Zero velocity integrates to the start point, so an untrained sample is its own Gaussian noise: the step-5 panel of the run with this init shows frame 0 (base) intact and every later frame as colour noise. Neither initialisation is a good starting point; the zero one should recover quickly because the target `r - x0` is close to `-x0` and easy to learn, but this has not been checked. The check is the step-1200 / 2000 / 3000 panels, not step 5. If the zero init trains slowly, rerun with `--refiner_zero_init_out False` and compare.

**The** `Δg` **condition** (`--use_delta_g`, off by default, on in the refiner job). Without it the network sees only the current frame's `g` and has to work out what changed by comparing `g_t` with the state. With it:

```
h = e_c + alpha * e_g + alpha_dg * Linear_delta(g_t - g_{t-1})
```

- `Linear_delta` is a `1024 -> cond_dim` linear layer **without bias and without a normalisation layer**, so the size of the change is kept (a LayerNorm would erase it). `alpha_dg` is zero-initialised (scalar, or a vector if `gated_fusion_alpha_dim=vector`), so a warm-started model starts exactly as before.
- Loss track: `Δg = g_t - g_{t-1}` (target frame minus previous frame). State track, chain first hop: `Δg = g_1 - g_0`. Sampling: `g_i - g_{i-1}` for frame `i >= 1`; frame 0 and the base model take none.
- Code: `adm.py` (`use_delta_g`, `physics_linear_delta`, `alpha_dg`, `gru_delta` in `forward` and `_build_cond`), `patch_mixin._predict_velocity(gru_delta=)`, `ode_mixin.sample_with_ode(gru_delta=)` and `sample_refiner_sequence`, `loss_mixin` (loss and state track). It is only supported by the ADM U-Net with split conditioning, and the argument is passed to the network only when on.
- Warm start loads with `strict=False`, but only the two new keys may be missing; any other mismatch raises.
- Not checked: whether `Δg` helps. The planned ablation is to zero or shuffle it at sampling time.

**Sampling** (`sample_refiner_sequence` in `ode_mixin.py`): frame 0 comes from the base; each later frame is drawn from noise by this network with the previous output locked as `lr_cond`, and `S_t = clamp(S_{t-1} + residual)`. The base needs a 3-dimensional scalar condition, so the refiner's 2-dimensional `c` gets a zero column prepended for frame 0. Training-time validation walks 64 val frames of one column (`exp R307`) through `_run_bridge_sequence_inference` and saves `fixed_di_step_XXXXX.png`. SDE sampling is not supported in this mode.

**Validation schedule.** `VAL_CHECK_INTERVAL` is 1000 in the refiner job (it was 5). A validation pass runs the full 10-evaluation chain for its states and rewrites the two `last.ckpt` files, which was costly every 5 steps. The 64-frame panel was already tied to `inference_every_n_epochs=1000`, matched against the global step. A consequence: with validation every 1000 steps there is no early check of the sampling path, so a code change should be smoke-tested first (`MAX_STEPS=6 VAL_CHECK_INTERVAL=5`, as was done for `Δg`).

**Logging**: `train/refiner_chain` (whether the batch used the chain), `train/refiner_resid_std` (std of the target residual).

**Run history of variant B** (all use `chjqbp9m` as the base):


| Job     | What happened                                                                                                                                                             |
| ------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `56087` | Deleted while still queued                                                                                                                                                |
| `56088` | Failed at the step-5 validation: the base model needs a 3-dimensional scalar condition and was given the refiner's 2-dimensional one (fixed in `sample_refiner_sequence`) |
| `56089` | Stopped after its step-5 panel saturated to white (no output-layer zero init)                                                                                             |
| `56090` | Stopped to switch to the cache of the deterministic head (the first cache saved little)                                                                                   |
| `56091` | Stopped to add `Δg`, validation every 1000 steps and the CPU settings. It was at roughly step 1000; its run directory is kept                                             |
| `56092` | 6-step smoke test of the `Δg` build: passed                                                                                                                               |
| `56093` | Current: 3000 steps with `Δg`                                                                                                                                             |




### 3.4 Flags


| Flag                                                     | Default  | Meaning                                                       |
| -------------------------------------------------------- | -------- | ------------------------------------------------------------- |
| `--use_refiner_residual`                                 | False    | Variant B                                                     |
| `--refiner_base_checkpoint`, `--refiner_base_result_dir` | empty    | Frozen single-frame model and the dir with its `args.json`    |
| `--refiner_chain_prob`                                   | 0.5      | Per-batch probability of the two-hop chain                    |
| `--refiner_roll_prob`, `--refiner_roll_nfe`              | 0.15, 10 | Full rollout instead of one Euler step; Heun steps = nfe // 2 |
| `--refiner_state_noise_max`                              | 0.05     | Max noise std on the state input channel                      |
| `--refiner_zero_init_out`                                | True     | Zero the output layer at warm start                           |
| `--use_delta_g`                                          | False    | Zero-gated `g_t - g_{t-1}` branch                             |
| `--cache_data`                                           | False    | Section 2.1                                                   |


Variant B job: `job/DamagePatternGeneration/train_ADMCFM_refiner_residual.sh` (3000 steps, batch 128, spall mask off, `CACHE_DATA=True`, `USE_DELTA_G=True`, `VAL_CHECK_INTERVAL=1000`, 4 CPUs and 2 workers, checkpoints every 1000 steps; the GPU is set at `qsub` time, e.g. `qsub -l select=1:ncpus=4:mem=15gb:gpu_id=0 ...`). `MAX_STEPS` and `VAL_CHECK_INTERVAL` can be overridden with `-v`.

### 3.5 Code map


| File                                                                 | Change                                                                                                                                 |
| -------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------- |
| `src/data/pattern_loop.py`                                           | Cache, `image_prev2` fields                                                                                                            |
| `src/model/cfm/options.py`, `argument/mixed.py`                      | Flags, validation, extra input channels                                                                                                |
| `src/model/cfm/lifecycle_mixin.py`                                   | `_load_frozen_cfm`, `_attach_refiner_base`                                                                                             |
| `src/model/cfm/loss_mixin.py`                                        | `_refiner_draw_full`, `_refiner_base_sample`, `_refiner_state_step`, `_refiner_base_prev_state`, `refiner_residual_flow_matching_loss` |
| `src/model/cfm/training_mixin.py`                                    | Dispatch to the new loss, `info` passed down, validation and logging                                                                   |
| `src/model/cfm/patch_mixin.py`, `ode_mixin.py`, `inference_mixin.py` | State channel gating, residual composition, `sample_refiner_sequence`, validation sequence                                             |
| `src/model/adm.py`                                                   | `Δg` branch and gate                                                                                                                   |
| `src/utils.py`                                                       | Prefix split of the image transform for the cache                                                                                      |
| `train_mixed.py`                                                     | Warm start with the extra channels, zero output layer and missing `Δg` keys; loader workers and threads                                |




### 3.6 Photometric propagation

The residual form adds to `S_{t-1}` and so inherits its brightness. The bridge form inherits it along the integration path. Neither has a built-in fix. What exists:

- A sampling-time corrector for the bridge (`_bridge_correct_frame`, see `docs/archive/TEMPORAL_FINAL.md`): re-noise to `s = 0.35`, re-draw with the frozen single-frame model, take high frequency from the first frame, and pull a dark global mean back. Its recorded result keeps mean gray stable (about 142-151) but the dark bottom region still widens over time.

Options not implemented: photometric augmentation on `S_{t-1}` only, low-frequency anchoring to a single-frame sample at the same `g`, longer chains in training. First step should be a measurement: mean gray and dark-area curves over time for GT, single-frame direct and the refiner chain.

### 3.7 Differences from the original idea


| Original idea                                                                   | Variant B                                                                                    |
| ------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------- |
| `S_{t-1}` locked input channel, residual target, accumulate outside the network | Done                                                                                         |
| Stop-gradient side track producing `S_t` from pure noise with one Euler step    | Done, as a two-hop chain with the base for the first hop; 15% of rollouts use 10 evaluations |
| Chain across many time steps                                                    | Two hops only                                                                                |
| Condition includes the change `Δg`                                              | Done, as a zero-gated AdaGN branch on `g_t - g_{t-1}`                                        |




### 3.8 Open items

- Check that variant B's training loss falls and that the step-1200 / 2000 / 3000 validation panels no longer saturate.
- Compare variant B, variant A and the single-frame direct output on absolute damage area and intensity, especially in high-damage regions, and with `Δg` ablations (zero or shuffle the condition change at sampling) to check that the refiner is not just copying `S_{t-1}`.
- Risks: a one-step sample is blurrier than a 30-step sample, so the training state is distributed differently from the inference state (the 15% full rollouts and the chain are the mitigation); the extra no-grad forward passes make each step slower than variant A.
- Next if it helps: longer chains, photometric handling from 3.6, TF32 / mixed precision, and moving base-model state generation offline.

