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

### V3 path (unchanged fallback)

`real_data [B,3,H,W] -> flow matching in pixel space -> sampled [B,3,H,W]`

### V4 latent path

`real_data [B,3,H,W] -> VAE.encode -> z1 [B,Cz,H/f,W/f] -> flow matching in latent space -> sampled_z [B,Cz,H/f,W/f] -> VAE.decode -> sampled_img [B,3,H,W]`

Where:

- `Cz = vae.config.latent_channels`
- `f = vae downsample factor`

Guardrails added:

- `image_size % f == 0` check at model init.
- shape mismatch protection remains in existing validation/image logging path.

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
