# ARCHITECTURE_UPDATE_V4

## Delta Scope (V4 Only)

V4 introduces one architectural shift on top of V3: **CFM can now run in SD3 VAE latent space** instead of only pixel space.

This update is implemented as a runtime-selectable path and does not remove V3 pixel-space behavior.

---

## 1) New Latent-Space CFM Path

### What changed

- Added `--use_sd3_vae_latent` switch to Stage-3 generation.
- When enabled:
  1. input image `x` is encoded by SD3 VAE into latent `z1`
  2. flow matching is trained on latent tensors
  3. ODE sampling produces latent predictions
  4. predicted latents are decoded back to image space for visualization/metrics



### Why this matters

- Moves generation objective into a more compact semantic space.
- Improves compatibility with future diffusion-first training pipelines.
- Keeps downstream reporting (PSNR/SSIM/FID and visualization) in image space via VAE decode.

---



## 2) Tensor Pipeline Delta



### Active run config (`train_ADMCFM.sh`)

- `batch_size=64`, `image_size=128`, `use_sd3_vae_latent=False` (pixel-space CFM)
- Dataset split: `id_list_no_leak.json` → train **521** / val **140** / test **124**
- Per epoch: train **~9** steps (`ceil(521/64)`), val **~3** steps (`ceil(140/64)`)
- Condition flags: `latent_conditioning=True`, `direction_conditioning=True`, `include_di_conditioning=True`



### Condition tensor (per sample)

Built by `process_condition()` from Phase-1 extractor + labels:


| Tensor           | Shape   | Source                                                                 |
| ---------------- | ------- | ---------------------------------------------------------------------- |
| `hidden_state`   | `[512]` | `HystereticGRU` (`phase1_extractor/best.ckpt`, `hidden_state_dim=512`) |
| `direction`      | `[2]`   | one-hot Pull/Push                                                      |
| `DI`             | `[1]`   | `label.csv`                                                            |
| `c` **(concat)** | `[515]` | `512 + 2 + 1` → fed to `ADMUNetCFM.cond_mlp`                           |


Batch condition: `c` → `[B, 515]` (e.g. `[64, 515]`).

### V3 path — pixel space (current default)

```
real_data [64, 3, 128, 128]          # normalized to [-1, 1]
    │
    ├─ hidden_state [64, 512], direction [64, 2], DI [64, 1]
    └─ c = concat → [64, 515]
         │
z1 = real_data                        # target latent = pixels (no VAE)
z0 ~ N(0, I)                          # [64, 3, 128, 128]
t ~ U(0, 1)                           # [64]
zt = (1-t)*z0 + t*z1                  # [64, 3, 128, 128]
         │
velocity_net(zt, t, c)              # ADMUNetCFM in/out [64, 3, 128, 128]
         │
loss = MSE(pred_v, z1 - z0)
```

**Velocity U-Net full visualization (active config)**  
`cfm_model_channels=128`, `channel_mult=(1,2,4,4)`, `num_res_blocks=1`, `use_mid_attention=False`

```text
Input z_t [B, 3, 128, 128]
  └─ in_conv: 3 -> 128
      h0 [B, 128, 128, 128]

Down path
  L0 ResBlock: 128 -> 128, save skip0 [B, 128, 128, 128]
  Downsample -> [B, 128, 64, 64]
  L1 ResBlock: 128 -> 256, save skip1 [B, 256, 64, 64]
  Downsample -> [B, 256, 32, 32]
  L2 ResBlock: 256 -> 512, save skip2 [B, 512, 32, 32]
  Downsample -> [B, 512, 16, 16]
  L3 ResBlock: 512 -> 512, save skip3 [B, 512, 16, 16]

Mid path
  mid_block1: [B, 512, 16, 16]
  mid_attn: Identity (attention disabled)
  mid_block2: [B, 512, 16, 16]

Up path (with skip concat)
  U3: concat(h, skip3) => [B, 1024, 16, 16] -> ResBlock -> [B, 512, 16, 16]
      Upsample -> [B, 512, 32, 32]
  U2: concat(h, skip2) => [B, 1024, 32, 32] -> ResBlock -> [B, 512, 32, 32]
      Upsample -> [B, 512, 64, 64]
  U1: concat(h, skip1) => [B, 768, 64, 64] -> ResBlock -> [B, 256, 64, 64]
      Upsample -> [B, 256, 128, 128]
  U0: concat(h, skip0) => [B, 384, 128, 128] -> ResBlock -> [B, 128, 128, 128]

Output
  out_norm + SiLU + out_conv: 128 -> 3
  pred_v [B, 3, 128, 128]
```

| Block | Spatial | Channels |
| ----- | ------- | -------- |
| Input `z_t` | 128×128 | 3 |
| Stem `in_conv` | 128×128 | 128 |
| Down L0 | 128×128 | 128 |
| Down L1 | 64×64 | 256 |
| Down L2 | 32×32 | 512 |
| Down L3 | 16×16 | 512 |
| Mid block1/2 | 16×16 | 512 |
| Up U3 | 16×16 | 512 |
| Up U2 | 32×32 | 512 |
| Up U1 | 64×64 | 256 |
| Up U0 | 128×128 | 128 |
| Output `pred_v` | 128×128 | 3 |

If `num_res_blocks=2`, each level has 2 ResBlocks instead of 1, but the stagewise channel plan above stays the same.


**ODE inference** (`cfm_ode_steps=30`, `cfm_solver=heun`):

```
z ~ N(0,I) [1, 3, 128, 128]  +  c [1, 515]  +  fixed_di ∈ {0.0, 0.4, 0.75, 0.95}
    → 30-step Heun integration
    → sampled [1, 3, 128, 128]
    → clamp(*0.5+0.5) → vis [1, 3, 128, 128] ∈ [0, 1]
```

Fixed-DI panel (every **50** epochs): **4** GT + **4** inference = **8** images (GT chosen by closest `GT_DI` to each target DI).

Metrics (PSNR / SSIM / FID) computed every **100** epochs on full val set (`sampled_vis` vs `real_vis`, both `[B, 3, 128, 128]`).

### V4 path — SD3 VAE latent space (opt-in, not active in current job)

```
real_data [64, 3, 128, 128]
    → VAE.encode → z1 [64, 16, 16, 16]     # Cz=16, f=8 for SD3-medium VAE
    → flow matching on z1
    → ODE sample → sampled_z [64, 16, 16, 16]
    → VAE.decode → sampled_img [64, 3, 128, 128]
```

Guardrail: `128 % 8 == 0` ✓

---



## 3) New Runtime Arguments

Added in `argument/argument_mixed.py`:

- `--use_sd3_vae_latent` (bool): enable latent-space CFM
- `--sd3_vae_model_id` (str): default `stabilityai/stable-diffusion-3-medium-diffusers`
- `--sd3_vae_subfolder` (str): default `vae`
- `--sd3_vae_use_fp16` (bool): use fp16 VAE on CUDA for memory efficiency

Dependency update:

- `requirements.txt`: add `diffusers>=0.30`

---



## 4) Implementation Trace (V4 Additions)

- `src/model/ImageGeneration.py` (`CFMADM`)
  - SD3 VAE loading and freeze
  - latent encode/decode helpers
  - dynamic CFM in/out channel sizing from VAE latent channels
  - latent-sized initial noise for ODE sampling
  - train/val flow-matching target switched to latent when enabled
  - metric and DI-regressor paths decode sampled latent back to image space

---



## 5) Backward Compatibility

- Default behavior is unchanged (`--use_sd3_vae_latent=False`).
- Existing V3 training commands remain valid.
- Latent path is opt-in and can be introduced incrementally in experiments.

---



## 6) Suggested Command (Latent CFM)

```bash
python train_mixed.py \
  --architecture CFMADM \
  --use_sd3_vae_latent True \
  --sd3_vae_model_id stabilityai/stable-diffusion-3-medium-diffusers \
  --sd3_vae_subfolder vae \
  --sd3_vae_use_fp16 True \
  --extractor_path "<phase1_extractor_dir>" \
  --latent_conditioning True \
  --direction_conditioning True \
  --include_di_conditioning True
```

