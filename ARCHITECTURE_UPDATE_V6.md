# ARCHITECTURE_UPDATE_V6

## Delta Scope (V6 Only)

V6 builds on the V5 scaled CFM U-Net (`cfm_model_channels=32`, `cfm_cond_dim=128`) and introduces three major architectural / training changes:

1. **Dual-path condition injection**
   - Scalar: `DI` only (`condition_mode=di_only`) → AdaGN via `cond_mlp`
   - Spatial: DI-driven 2D Gaussian `damage_mask` → channel concat at `in_conv` (Zero-Conv init)
2. **Masked flow-matching loss** using rebar object masks (`Data/DamagePattern/mask`)
3. **Training / logging / augmentation pipeline overhaul**
   - Step-based training clock, W&B step metrics, gradient clipping
   - True spatial affine augmentation with joint image + mask transform

V5 U-Net width scaling is unchanged. V6 changes **what conditions the model**, **how loss is computed**, and **how training is monitored**.

---

## 1) Condition Architecture (V6 Active Path)

### 1.1 Scalar condition: `di_only`

Active script flag:

```bash
--condition_mode di_only
```

`build_generation_condition()` short-circuits to DI only:

```text
if condition_mode == "di_only":
    return DI                          # [B, 1]
```

`set_dataset_args()` therefore sets:

- `args.condition_dim = 1`

`ADMUNetCFM.cond_mlp` becomes `Linear(1 → 128) → SiLU → Linear(128 → 128)`.

**Not injected into U-Net under `di_only`:**

| Field          | Shape    | Still loaded in batch? | Used by CFM? |
| -------------- | -------- | ---------------------- | ------------ |
| `hidden_state` | `[1024]` | Yes (extractor cache)  | No           |
| `direction`    | `[2]`    | Yes                    | No           |
| `info`         | `[17]`   | Yes                    | No           |
| `DI`           | `[1]`    | Yes                    | **Yes**      |

Flags `--latent_conditioning True --direction_conditioning True --include_di_conditioning True` remain in the script for compatibility, but are **ignored** when `condition_mode=di_only`.

To restore V5-style full conditioning:

```bash
--condition_mode full
# → c = hidden_state[1024] + direction[2] + DI[1] = [1027]
```

---

### 1.2 Spatial condition: DI-driven Gaussian `damage_mask`

New flag:

```bash
--use_damage_mask_condition True
```

#### Mask generation (`PatternLoopDataset._build_damage_mask`)

For each sample, mask is built from the (possibly perturbed) DI value:

```text
di_scaled = clamp(DI / damage_mask_max_di, 0, 1)

# Active profile (linear):
intensity_scale = di_scaled
spread_scale    = di_scaled

sigma   = sigma_min + spread_scale * (sigma_max - sigma_min)
sigma_y = sigma
sigma_x = sigma * width_scale          # anisotropic: wider horizontally

gaussian = exp(-((x-cx)²/(2σx²) + (y-cy)²/(2σy²)))
damage_mask = intensity_scale * gaussian     # [1, H, W]
```

Active geometry (128×128 images):

| Parameter | Active value | Effect |
| --------- | ------------ | ------ |
| Center `(cx, cy)` | `(63.5, 127)` | Bottom-center of image |
| `damage_mask_sigma_min` | `80.0` | Minimum spread at DI≈0 |
| `damage_mask_sigma_max` | `320.0` | Maximum spread at DI≈1 |
| `damage_mask_width_scale` | `2.0` | Horizontal spread ×2 vs vertical |
| Profile | **Linear** | Intensity and spread both ∝ DI |

Note: CLI args `damage_mask_split_di`, `damage_mask_elastic_*`, `damage_mask_quadratic_scale` are retained for future two-stage profiles but **not used** by the current linear `_two_stage_damage_profile()`.

#### Zero-Conv injection at U-Net input

`ADMUNetCFM.in_conv` expands from `C → C+1` input channels:

```text
h = in_conv( concat([z_t, damage_mask], dim=1) )
     ^              ^         ^
     |              |         └─ [B, 1, 128, 128] spatial condition
     |              └─ [B, 3, 128, 128] flow state
     └─ only the extra mask channel is zero-initialized
```

Implementation: `_expand_conv_in_with_zero_mask_channel()` copies existing weights and zero-inits the new channel.

Backward-compatible checkpoint load: `_load_from_state_dict()` pads old `in_conv.weight` `[out, C, k, k]` with a zero channel when loading into the expanded layer.

---

### 1.3 End-to-end condition flow (V6)

```text
label.csv → DI ∈ R¹
        │
        ├─► cond_mlp(DI) ──► c_emb ──┐
        │                            ├─► joint_mlp(t_emb + c_emb) ──► AdaGN (all ResBlocks)
        │   t ──► time_mlp ──► t_emb ─┘
        │
        └─► _build_damage_mask(DI) ──► damage_mask [1,H,W] ──► in_conv concat (Zero-Conv)
```

AdaGN path (scalar) and spatial concat path (mask) are **independent** — condition does not enter spatial features except via the dedicated mask channel.

---

## 2) Masked Flow-Matching Loss

### 2.1 Object mask source

Directory: `Data/DamagePattern/mask/`

Resolution order per image filename:

```text
{stem}_mask.png  →  {stem}.png  →  {filename}
```

Mask convention:

- **White (255)** = rebar (object)
- **Black (0)** = concrete (valid generation region)

Converted to loss mask:

```text
valid_mask = 1.0 - mask / 255.0      # concrete=1, rebar=0
```

Flag: `--use_object_loss_mask True` (dataset); `use_loss_mask=True` forced in `train_mixed.py` for CFMADM.

### 2.2 Augmentation-aware loss mask

`MaskedDamageImageTransform` (`src/utils.py`) returns:

```text
{ "target_image": image, "loss_mask": mask, "flipped": bool }
```

`loss_mask` is transformed **jointly** with the image:

- Same `TF.affine` translate/scale on image (bilinear) and masks (nearest)
- Same horizontal flip
- Final mask = augmented all-ones mask × object mask

Batch return (train):

```text
image, hidden_state, direction, exp, timestep, DI, info, loss_mask, damage_mask
```

### 2.3 Loss computation

```python
unreduced_loss = F.mse_loss(pred_v, target_u, reduction="none")   # [B, C, H, W]
masked_loss    = unreduced_loss * valid_mask                         # broadcast on channel dim
loss           = masked_loss.sum() / (valid_mask.sum() + 1e-8)
```

`damage_mask` is **not** used in the loss — it is a generation condition only.

---

## 3) V6 U-Net Input/Output (pixel space)

Active config: `use_sd3_vae_latent=False`, `image_size=128`.

```text
z_t           [B, 3, 128, 128]   flow state
damage_mask   [B, 1, 128, 128]   spatial condition (concat at input)
c             [B, 1]             scalar DI condition (AdaGN)
t             [B]                flow time

in_conv input: [B, 4, 128, 128]  = concat(z_t, damage_mask)
in_conv:       4 → 32

... (same down/mid/up path as V5, cfm_model_channels=32) ...

pred_v        [B, 3, 128, 128]
```

Internal channel layout is identical to V5 §3; only `in_conv.in_channels` increases from 3 to 4.

---

## 4) Training Pipeline Changes

### 4.1 Step-based training clock

| Parameter | Active value | Notes |
| --------- | ------------ | ----- |
| `max_steps` | `15000` | Primary stop criterion (`max_epochs=-1`) |
| `warmup_steps` | `800` | LR warmup for step scheduler |
| `val_check_interval` | `5` | Validation every 5 steps |
| `train_virtual_repeat` | `100` | Dataset length multiplier |

LR schedule: `warmup_cosine` with `min_lr_ratio=0.1`, driven by **optimizer step** (not epoch).

### 4.2 Gradient clipping

```bash
--grad_clip_max_norm 2.0
```

Applied in `CFMADM.on_before_optimizer_step()` via `clip_grad_norm_`.

Pre-clip L2 norm logged as `train/grad_norm_L2` (uncapped, for W&B diagnostics).

### 4.3 W&B step-aligned logging

On `CFMADM` setup:

```python
wandb.define_metric("trainer/global_step")
wandb.define_metric("*", step_metric="trainer/global_step")
```

Logged step metrics include:

- `train/loss`, `train/fm_loss`, `train/lr`, `train/grad_norm_L2`
- `val/loss` (epoch-aggregated but step-indexed)
- `CFM/FixedDI/step_{NNNNN}` (unique key per inference step)

### 4.4 Inference panels (fixed DI)

| Parameter | Active value |
| --------- | ------------ |
| `inference_every_n_epochs` | `500` (interpreted as **steps**) |
| `inference_num_images` | `4` |
| `fixed_di_values` | `0.0, 0.4, 0.75, 0.95` |
| `inference_anchor_exp` | `R615` |
| `inference_anchor_timestep` | `-1` (auto) |

Behavior:

1. During validation, collect `anchor_candidates` (GT image + metadata) from val batches
2. `_select_anchors_for_di_values()` picks **4 distinct GT samples**, each nearest one target DI
3. For each anchor, run ODE sampling with **target DI** (not GT DI) and matching `damage_mask`
4. Save panel: `{pair_count} rows × 2 cols` → GT | Inference
5. Output: `results/.../inference_samples/fixed_di_step_{NNNNN}.png`

This yields **4 different GT + 4 inference = 8 panels** per logging step.

### 4.5 Checkpoints and early stopping

- `ModelCheckpoint(monitor="val/loss")` — primary best checkpoint for CFMADM
- `EarlyStopping(patience=300, monitor="val/loss")` — active in script
- FID/SSIM checkpoints only added when `metrics_every_n_epochs <= val_check_interval`

---

## 5) Augmentation Overhaul

Centralized in `job/DamagePatternGeneration/train_ADMCFM.sh` and implemented via `MaskedDamageImageTransform`.

### 5.1 Active augmentation config

| Augmentation | Active value | Notes |
| ------------ | ------------ | ----- |
| ColorJitter | brightness/contrast/saturation/hue | Via `build_damage_image_transform` |
| Gaussian blur | prob=0.15, kernel=3 | |
| Horizontal flip | prob=0.5 | Also flips direction + signed info features |
| RandomAffine translate | `0.10` (10%) | **True spatial** `TF.affine`, not cutout |
| RandomAffine scale | `1.0` (disabled) | No scaling |
| RandomErasing | **disabled** (`prob=0.0`) | Removed from transform path |

### 5.2 Asymmetric Y translation

```text
dx ∈ [-max_dx, +max_dx]     # symmetric horizontal
dy ∈ [-max_dy, 0]           # upward only (no downward shift)
```

Rationale: damage patterns concentrate at the column base; downward translation would move content into padded/empty regions.

### 5.3 Joint transform guarantee

Image and all mask tensors (`loss_mask`, `object_mask`) receive identical affine parameters, preserving pixel alignment between loss masking and augmented geometry.

Preview saved at train start:

```text
results/DamagePatternGeneration/CFMADM/{timestamp}/data_check/train_aug_preview_10x10.png
```

---

## 6) Condition Perturbation (train only)

Active:

```bash
--condition_perturb
--condition_perturb_dist uniform
--condition_perturb_min 0.01
--condition_perturb_max 0.03
```

Multiplicative noise `tensor * (1 + ε)`, `ε ∈ ±[1%, 3%]`.

Under `di_only`, only DI perturbation affects UNet conditioning (hidden/info are perturbed in dataset but unused by `process_condition`).

Because `damage_mask` is rebuilt from perturbed DI in `__getitem__`, scalar and spatial conditions stay consistent.

---

## 7) Bug Fixes and Compatibility

| Issue | Fix |
| ----- | --- |
| `TypeError: multiple values for use_damage_mask_condition` | `train_mixed.py` builds `dataset_kwargs` once; avoids duplicate kwarg via `**vars(args)` |
| W&B panels overwriting each other | Unique log keys `CFM/FixedDI/step_{NNNNN}` |
| W&B x-axis using epoch instead of step | `wandb.define_metric(..., step_metric="trainer/global_step")` |
| GroupNorm crash with `cfm_model_channels=32` | Dynamic `_num_groups(channels)` in `cfm_adm.py` |
| Gray-box "translation" artifact | Replaced cutout-style erasing with true `TF.affine`; erasing disabled |
| Old checkpoint missing mask channel | `_load_from_state_dict` zero-pads `in_conv` weights |

---

## 8) Updated Files (V6 touch list)

| File | Change |
| ---- | ------ |
| `src/model/Component/cfm_adm.py` | Zero-Conv `in_conv` expand, `damage_mask` forward arg, dynamic GroupNorm, checkpoint compat |
| `src/model/ImageGeneration.py` | `build_generation_condition` di_only, masked FM loss, damage mask builder, anchor selection, grad clip, W&B step logging |
| `src/dataset.py` | `_build_damage_mask`, object mask loading, condition perturbation, batch tuple extension |
| `src/utils.py` | `MaskedDamageImageTransform`, joint affine, asymmetric Y translate |
| `argument/argument_mixed.py` | New CLI args: condition_mode, damage mask, object loss mask, grad clip, aug params |
| `train_mixed.py` | `dataset_kwargs`, loss-mask transform wiring, aug preview grid |
| `job/DamagePatternGeneration/train_ADMCFM.sh` | Full V6 active config |

---

## 9) Active Runtime Config Summary (`train_ADMCFM.sh`)

```text
Architecture:     CFMADM
Condition:        di_only (DI scalar) + damage_mask (spatial)
Loss:             masked MSE on concrete pixels
Batch size:       128
Max steps:        15000
U-Net width:      cfm_model_channels=32, cfm_cond_dim=128
Grad clip:        2.0
Inference:        every 500 steps, DI=[0, 0.4, 0.75, 0.95], anchor exp=R615
Data split:       id_list_no_leak.json
Extractor:        phase1_extractor/best.ckpt (hidden_state still cached, not used in di_only)
W&B:              enabled (requires login)
```

---

## 10) Resume / Migration Notes

### From V5 full-condition checkpoint → V6 di_only

- `cond_mlp` input shape changes: `1027 → 1` — **not shape-compatible**; train from scratch or keep `condition_mode=full`
- `in_conv` input shape changes: `3 → 4` — partially compatible via zero-pad loader if only spatial path is new

### From V6 di_only checkpoint

- Resuming requires same `condition_mode`, `use_damage_mask_condition`, and mask geometry params
- Changing `damage_mask_sigma_*` or `width_scale` at resume is safe (mask is recomputed, not stored in checkpoint)

### From pre-V5 wide U-Net (`cfm_model_channels=128`)

- Still incompatible with V5/V6 scaled architecture (see V5 §6)

---

## 11) V5 → V6 Quick Comparison

| Aspect | V5 | V6 |
| ------ | -- | -- |
| Scalar condition | `hidden+dir+DI` (1027-D) | `DI` only (1-D) |
| Spatial condition | None | Gaussian `damage_mask` at `in_conv` |
| Loss | Full-image MSE | Masked MSE (concrete only) |
| Translation aug | N/A / cutout-like | True affine, Y upward-only |
| Training clock | Epoch-oriented defaults | Step-based (`max_steps`) |
| Inference logging | Epoch-keyed | Step-keyed, distinct GT per DI |
| Grad clip | None | L2 norm 2.0 |
