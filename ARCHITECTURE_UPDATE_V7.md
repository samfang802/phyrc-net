# ARCHITECTURE_UPDATE_V7

## Delta Scope (V7 Only)

V7 builds on the V5 scaled CFM U-Net (`cfm_model_channels=32`, `cfm_cond_dim=128`) and adds **Patch Diffusion** training for pixel-space CFM:

1. **Coordinate channels** — absolute XY grids (full-canvas `[-1, 1]`) concatenated to the flow state
2. **Stochastic patch scheduling** — per mini-batch random crop size (`128 / 64 / 32`)
3. **Image-only CFM loss** — MSE on image channels only; predicted coordinate channels are discarded

V5 width scaling and prior conditioning / masking plumbing are unchanged. V7 changes **what spatial resolution the U-Net sees during training** and **how absolute position is encoded**.

Active script flags:

```bash
--use_patch_diffusion True
--patch_size_probs 0.5,0.3,0.2
```

---

## 1) Coordinate Channels

### 1.1 Construction

For a full canvas of size `H × W` (active: `128 × 128`):

```text
xs = linspace(-1, 1, W)
ys = linspace(-1, 1, H)
grid_x, grid_y = meshgrid(xs, ys)     # each [H, W]
```

When a patch is cropped at `(top, left)` with size `P`:

```text
coords = stack([
    grid_x[top:top+P, left:left+P],
    grid_y[top:top+P, left:left+P],
], dim=0)                              # [2, P, P]
```

Values remain **absolute** w.r.t. the full image (not re-normalized to the patch). A `64×64` crop from the bottom-right still carries coordinates near `+1`.

### 1.2 Concatenate as model input

```text
z_t_image : [B, C, P, P]               # C=3 (pixel-space RGB)
coords    : [B, 2, P, P]
model_in  = cat([z_t_image, coords])   # [B, C+2, P, P]
```

U-Net I/O when Patch Diffusion is enabled:

| Tensor | Shape (train, example `P=64`) |
| ------ | ----------------------------- |
| Flow state `z_t` (image) | `[B, 3, 64, 64]` |
| Coords | `[B, 2, 64, 64]` |
| `in_conv` input | `[B, 5, 64, 64]` |
| U-Net raw output | `[B, 5, 64, 64]` |
| Velocity used for CFM | `[B, 3, 64, 64]` (= first `C` channels) |

Implementation: `CFMADM._build_full_coordinate_maps`, `_concat_coord_channels`, `_predict_velocity`.

---

## 2) Stochastic Patch Scheduling

### 2.1 Per mini-batch size draw

During **training only** (`self.training=True`), each mini-batch draws one shared crop size:

| Probability | Patch size | Relative to `image_size=128` |
| ----------- | ---------- | ---------------------------- |
| **50%** | `128 × 128` | full |
| **30%** | `64 × 64` | half |
| **20%** | `32 × 32` | quarter |

CLI: `--patch_size_probs 0.5,0.3,0.2` (order = full / half / quarter; auto-normalized to sum 1).

```text
u ~ Uniform(0, 1)
if u < p_full:          P = 128
elif u < p_full+p_half: P = 64
else:                   P = 32
```

### 2.2 Per-sample crop location

Within a batch, **size `P` is shared** (tensor stackable), but **`(top, left)` is independent per sample**:

```text
top  ~ Uniform{0, …, H-P}
left ~ Uniform{0, …, W-P}
```

`loss_mask` / `valid_mask` is cropped with the same window so masked MSE stays aligned.

### 2.3 Eval / inference

Validation FM loss and ODE sampling **always use full `128 × 128`** with the full-canvas coordinate grid (no stochastic crop).

### 2.4 U-Net resolution compatibility

With `channel_mult=(1, 2, 4, 4)` the spatial downsample factor is `8`:

| Patch | Bottleneck spatial |
| ----- | ------------------ |
| `128` | `16 × 16` |
| `64` | `8 × 8` |
| `32` | `4 × 4` |

All valid. Guard: `model_spatial_size % 8 == 0`.

---

## 3) Loss Computation (Image Channels Only)

Flow matching still operates on **image channels only**:

```text
z0, z1, z_t, target_u = z1 - z0     # all [B, C, P, P]
pred_full = velocity_net(cat([z_t, coords]), t, c)   # [B, C+2, P, P]
pred_v    = pred_full[:, :C]                         # drop coords
loss      = masked_MSE(pred_v, target_u)             # optional Min-SNR weight
```

Coordinate channels are **not** noised and are **not** part of the CFM target. They are positional conditioning only.

Entry points:

- `CFMADM._image_channels_only`
- `CFMADM.flow_matching_loss` → `_apply_patch_crop` then `_predict_velocity`
- `CFMADM.sample_with_ode` → fixed full-size coords; ODE state stays `[B, C, H, W]`

---

## 4) End-to-End Training Step (V7)

```text
batch image [B, 3, 128, 128], valid_mask, condition c
        │
        ▼
encode_to_target_space  (pixel-space identity when VAE off)
        │
        ▼
stochastic schedule → P ∈ {128, 64, 32}          (train only)
random crop (image + mask) + crop-aligned XY coords
        │
        ▼
z0 ~ N(0,I),  z_t = (1-t) z0 + t z1
model_in = cat([z_t, coords])                    # C+2
        │
        ▼
ADMUNetCFM → pred [B, C+2, P, P]
pred_v = pred[:, :C]
        │
        ▼
masked MSE(pred_v, z1 - z0)  (+ optional Min-SNR)
```

Inference ODE:

```text
z ~ N(0,I) at full 128×128
coords = full-canvas XY in [-1, 1]
for each solver step:
    v = velocity_net(cat([z, coords]), t, c)[:, :C]
    z ← z + dt · v   (Heun / Euler)
decode / visualize
```

---

## 5) U-Net Channel Diagram (V7 + Patch Diffusion)

Active config: `cfm_model_channels=32`, `channel_mult=(1,2,4,4)`, `num_res_blocks=1`, Patch Diffusion ON, mask-as-condition ON, `cfm_mid_attn_heads=4`, `use_vss_l3_u3=True`.

Example below uses **full patch `P=128`** (val / ODE / 50% train batches). Replace `128` with `P` for stochastic crops (`64` or `32`); channel widths are unchanged, only `H×W` scales (bottleneck `P/8`).

### 5.1 CFM-side inputs (before U-Net)

```text
z_t_image : [B, 3, P, P]                         # noised flow state (RGB)
coords    : [B, 2, P, P]                         # absolute XY ∈ [-1, 1]
model_in  = cat([z_t_image, coords])             # [B, 5, P, P]
condition_mask (object mask) : [B, 1, P, P]      # train: Data/DamagePattern/mask (aug-aligned)
                                                 # val/test/ODE: all-ones
```

Inside `ADMUNetCFM.forward`:

```text
in_conv_input = cat([model_in, condition_mask])  # [B, 6, P, P]
```

CFM loss / ODE state remain **image-only** `[B, 3, P, P]`; coord + mask channels are conditioning only. Predicted coord channels are dropped: `pred_v = pred_full[:, :3]`.

### 5.2 U-Net tensor flow (`P=128`)

```text
Input model_in [B, 5, 128, 128]                  # 3 image + 2 coords
condition_mask [B, 1, 128, 128]
  └─ in_conv: cat → [B, 6, 128, 128] → 32
      h0 [B, 32, 128, 128]

Down path (channel_mult: 1, 2, 4, 4)
  L0: ResBlock 32 -> 32, save skip0 [B, 32, 128, 128]
      Downsample -> [B, 32, 64, 64]
  L1: ResBlock 32 -> 64, save skip1 [B, 64, 64, 64]
      Downsample -> [B, 64, 32, 32]
  L2: ResBlock 64 -> 128, save skip2 [B, 128, 32, 32]
      Downsample -> [B, 128, 16, 16]
  L3: ResBlock 128 -> 128
      l3_vss (VSSBlock2d, dim=128): BCHW -> BHWC -> VSSBlock -> BCHW
      save skip3 [B, 128, 16, 16]          # no Downsample after L3

Mid path
  mid_block1: [B, 128, 16, 16]
  mid_attn: SelfAttention2d (4 heads × 32 dim, 256 tokens) on [B, 128, 16, 16]
  mid_block2: [B, 128, 16, 16]

Up path (with skip concat)
  U3: concat(h, skip3) => [B, 256, 16, 16] -> ResBlock -> [B, 128, 16, 16]
      u3_vss (VSSBlock2d, dim=128): BCHW -> BHWC -> VSSBlock -> BCHW
      Upsample -> [B, 128, 32, 32]
  U2: concat(h, skip2) => [B, 256, 32, 32] -> ResBlock -> [B, 128, 32, 32]
      Upsample -> [B, 128, 64, 64]
  U1: concat(h, skip1) => [B, 192, 64, 64] -> ResBlock -> [B, 64, 64, 64]
      Upsample -> [B, 64, 128, 128]
  U0: concat(h, skip0) => [B, 96, 128, 128] -> ResBlock -> [B, 32, 128, 128]

Output
  out_norm + SiLU + out_conv: 32 -> 5
  pred_full [B, 5, 128, 128]
  pred_v    = pred_full[:, :3]                   # [B, 3, 128, 128] used by CFM / ODE
```

Scalar condition `c` (AdaGN, `cond_dim=128`) modulates every ResBlock; it does **not** alter the spatial/channel diagram above.

### 5.3 Spatial sizes by patch schedule

| Patch `P` | `in_conv` spatial | Bottleneck (`mid_*`) | `pred_full` spatial |
| --------- | ----------------- | -------------------- | ------------------- |
| `128` | `128 × 128` | `16 × 16` (256 tokens) | `128 × 128` |
| `64` | `64 × 64` | `8 × 8` (64 tokens) | `64 × 64` |
| `32` | `32 × 32` | `4 × 4` (16 tokens) | `32 × 32` |

Channel layout in §5.2 is identical for all `P`; only spatial dims change.

L3 / U3 VSS spatial sizes follow bottleneck resolution `P/8`:

| Patch `P` | L3 / U3 spatial (after 3× Downsample) | VSS token count `H×W` |
| --------- | ------------------------------------- | --------------------- |
| `128` | `16 × 16` | 256 |
| `64` | `8 × 8` | 64 |
| `32` | `4 × 4` | 16 |

### 5.4 Vision Mamba VSS at L3 / U3

**Source / implementation:** `src/model/Component/vss_block.py` — **not** the official [VMamba](https://github.com/MzeroMiko/VMamba) repo. It is a pure-PyTorch, dependency-free reimplementation inspired by the VSS block design (SS2D cross-scan + selective mixing + MLP), with a simplified selective scan so it runs at any `H×W ≥ 1` without `mamba-ssm` CUDA kernels.

| Module | Location | Input / output | Role |
| ------ | -------- | -------------- | ---- |
| `self.l3_vss` | Down L3, after ResBlock | `[B, 128, H, W]` → same | Global mixing before saving `skip3` |
| `self.u3_vss` | Up U3, after ResBlock | `[B, 128, H, W]` → same | Global mixing before Upsample |

**Tensor layout adapter (`VSSBlock2d`):**

```text
(B, C, H, W)  --permute(0,2,3,1)-->  (B, H, W, C)
    └─ VSSBlock: LayerNorm → SS2D → residual; LayerNorm → MLP → residual
(B, H, W, C)  --permute(0,3,1,2)-->  (B, C, H, W)
```

CLI: `--use_vss_l3_u3 True` (default). Resume from pre-VSS checkpoints: `l3_vss` / `u3_vss` weights are **missing** and initialize randomly.

---

## 6) Constraints

| Constraint | Behavior |
| ---------- | -------- |
| Pixel-space only | `use_patch_diffusion` + `use_sd3_vae_latent` → `ValueError` |
| Square canvas | Crop helper expects `H == W` |
| Divisible by 8 | Required so half/quarter patches remain U-Net-valid |

---

## 7) Updated Files (V7 touch list)

| File | Change |
| ---- | ------ |
| `src/model/Component/cfm_adm.py` | Generalized `_load_from_state_dict` pad/trim for `in_conv` / `out_conv` when channel count differs (mask / coords); L3/U3 `VSSBlock2d` hooks |
| `src/model/Component/vss_block.py` | Pure-PyTorch VSS / SS2D (VMamba-inspired, not official repo) + BCHW adapter |
| `src/model/ImageGeneration.py` | Coord grids, stochastic crop, image-only velocity/loss/ODE path; passes `use_vss_l3_u3` |
| `argument/argument_mixed.py` | `--use_patch_diffusion`, `--patch_size_probs`, `--use_vss_l3_u3` |
| `job/DamagePatternGeneration/train_ADMCFM.sh` | Enables Patch Diffusion + `0.5,0.3,0.2` schedule |
| `resume_job.sh` | Same Patch Diffusion flags |

---

## 8) Active Runtime Config Summary (`train_ADMCFM.sh`)

```text
Architecture:     CFMADM (pixel space)
U-Net width:      cfm_model_channels=32, cfm_cond_dim=128 (V5)
U-Net VSS:        use_vss_l3_u3=True (l3_vss + u3_vss, dim=128 at L3/U3)
Patch Diffusion:  ON
  schedule:       128²=0.50, 64²=0.30, 32²=0.20
  coords:         XY ∈ [-1, 1] over full canvas, concat → in/out C+2=5
  loss:           image channels only (drop predicted coords)
Condition:        condition_mode=di_only; damage_mask=False
Loss mask:        object/valid-pixel mask (cropped with patch)
Batch size:       128
Max steps:        8000
ODE:              Heun, 30 steps, full 128² + full coords
Data split:       id_list_no_leak.json
Extractor:        phase1_extractor/best.ckpt
```

---

## 9) Resume / Migration Notes

### From V5/V6 checkpoint **without** Patch Diffusion (`in/out = 3`)

- `in_conv`: `3 → 5` input channels — loader zero-pads the extra 2 coord channels
- `out_conv`: `3 → 5` output channels — loader zero-pads the extra 2 output channels / bias
- Prefer **train from scratch** for a clean Patch Diffusion run; zero-pad resume is a soft compatibility path only

### From V7 Patch Diffusion checkpoint

- Keep `--use_patch_diffusion True` and the same `patch_size_probs` (probs only affect train-time sampling, not weight shapes)
- Turning Patch Diffusion **off** after a V7 run requires channel trim on load (`5 → 3`); not recommended mid-experiment

### From pre-VSS checkpoint (no `l3_vss` / `u3_vss` weights)

- Enable with `--use_vss_l3_u3 True` (default): Lightning loads existing weights; VSS modules are **new parameters** (random init)
- Disable with `--use_vss_l3_u3 False` for strict weight compatibility with older runs

### With VAE latent CFM

- Not supported under V7 Patch Diffusion (see §6)

---

## 10) V6 → V7 Quick Comparison

| Aspect | V6 | V7 |
| ------ | -- | -- |
| Train spatial size | Always `128 × 128` | Stochastic `128 / 64 / 32` |
| Position encoding | None (or optional damage mask) | Absolute XY coord channels |
| U-Net `in/out` channels | `3` (+1 if damage mask) | `5` (= `3+2` coords; +1 more if damage mask) |
| CFM target / MSE | Image channels | Image channels only (explicitly drop coord preds) |
| Val / ODE resolution | `128 × 128` | `128 × 128` (unchanged) |
| U-Net width | V5 scaled (`32` / mid `128`) | Unchanged |
