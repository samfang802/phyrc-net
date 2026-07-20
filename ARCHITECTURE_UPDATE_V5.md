# ARCHITECTURE_UPDATE_V5

## Delta Scope (V5 Only)

V5 applies **uniform channel scaling by 1/4** to the CFM U-Net backbone so that the bottleneck (mid) width becomes 128 channels.

The active scaling is:

- `cfm_model_channels`: `128 -> 32`
- `cfm_cond_dim`: `512 -> 128`
- `channel_mult`: unchanged `(1, 2, 4, 4)`
- `num_res_blocks`: unchanged from your run config (train script uses 2)

---

## 1) Uniform Scaling Rule

With `channel_mult=(1,2,4,4)`, each stage width is:

- stage width = `cfm_model_channels * mult`

So after scaling:

- level-0: `32 * 1 = 32`
- level-1: `32 * 2 = 64`
- level-2: `32 * 4 = 128`
- level-3: `32 * 4 = 128`
- mid dim: `128` (target achieved)

---

## 2) Active V5 Tensor Pipeline (pixel space)

Current training path is still pixel-space CFM (`use_sd3_vae_latent=False`), so:

- input `z_t`: `[B, 3, 128, 128]`
- condition `c`: `[B, 515]` from `hidden_state[512] + direction[2] + DI[1]`
- output `pred_v`: `[B, 3, 128, 128]`

The 1/4 scaling changes the U-Net internal channels only.

---

## 3) V5 U-Net Visualization (`cfm_model_channels=32`)

```text
Input z_t [B, 3, 128, 128]
  └─ in_conv: 3 -> 32
      h0 [B, 32, 128, 128]

Down path (channel_mult: 1,2,4,4)
  L0: ResBlock(s) 32 -> 32, save skip0 [B, 32, 128, 128]
      Downsample -> [B, 32, 64, 64]
  L1: ResBlock(s) 32 -> 64, save skip1 [B, 64, 64, 64]
      Downsample -> [B, 64, 32, 32]
  L2: ResBlock(s) 64 -> 128, save skip2 [B, 128, 32, 32]
      Downsample -> [B, 128, 16, 16]
  L3: ResBlock(s) 128 -> 128, save skip3 [B, 128, 16, 16]

Mid path
  mid_block1: [B, 128, 16, 16]
  mid_attn: Identity (attention disabled)
  mid_block2: [B, 128, 16, 16]

Up path (with skip concat)
  U3: concat(h, skip3) => [B, 256, 16, 16] -> ResBlock(s) -> [B, 128, 16, 16]
      Upsample -> [B, 128, 32, 32]
  U2: concat(h, skip2) => [B, 256, 32, 32] -> ResBlock(s) -> [B, 128, 32, 32]
      Upsample -> [B, 128, 64, 64]
  U1: concat(h, skip1) => [B, 192, 64, 64] -> ResBlock(s) -> [B, 64, 64, 64]
      Upsample -> [B, 64, 128, 128]
  U0: concat(h, skip0) => [B, 96, 128, 128] -> ResBlock(s) -> [B, 32, 128, 128]

Output
  out_norm + SiLU + out_conv: 32 -> 3
  pred_v [B, 3, 128, 128]
```

---

## 4) Parameter/Compute Effect (relative)

Uniform width scaling by 1/4 roughly gives:

- convolutional parameter scale: about `1/16`
- activation memory scale: about `1/4` to `1/8` in major blocks (depends on concat + optimizer states)
- expected throughput improvement: typically higher, hardware-dependent

---

## 5) Updated Runtime Defaults and Scripts

Updated defaults in `argument/argument_mixed.py`:

- `--cfm_model_channels` default `32`
- `--cfm_cond_dim` default `128`

Updated scripts:

- `job/DamagePatternGeneration/train_ADMCFM.sh`
  - `--cfm_model_channels 32`
  - `--cfm_cond_dim 128`
- `resume_job.sh`
  - `--cfm_model_channels 32`
  - `--cfm_cond_dim 128`

---

## 6) Note on Resume Compatibility

If an old checkpoint was trained with `cfm_model_channels=128` / `cfm_cond_dim=512`, it is not shape-compatible with V5 scaled architecture and cannot be resumed strictly with full state restore.
