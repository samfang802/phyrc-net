# ARCHITECTURE_UPDATE_V5

## Delta Scope (V5 Only)

V5 applies **uniform channel scaling by 1/4** to the CFM U-Net backbone so that the bottleneck (mid) width becomes 128 channels.

The active scaling is:

- `cfm_model_channels`: `128 -> 32`
- `cfm_cond_dim`: `512 -> 128`
- `channel_mult`: unchanged `(1, 2, 4, 4)`
- `num_res_blocks`: unchanged from your run config (train script uses `1`)

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
- condition `c`: `[B, 1027]` from `hidden_state[1024] + direction[2] + DI[1]`
  - Phase-1 `HystereticGRU` uses `hidden_state_dim=512` with `bidirectional=True`, so GRU output width is `512 × 2 = 1024`
- output `pred_v`: `[B, 3, 128, 128]`

The 1/4 scaling changes the U-Net internal channels only.

---



## 2.1) Condition Injection From `best.ckpt` Into U-Net (Detailed)

This section documents the full path used by the active CFMADM training script:

- extractor checkpoint: `phase1_extractor/best.ckpt`
- flags: `latent_conditioning=True`, `direction_conditioning=True`, `include_di_conditioning=True`
- U-Net: `ADMUNetCFM` with `latent_dim=condition_dim=1027`, `cond_dim=cfm_cond_dim=128`



### A. Load Phase-1 extractor (`best.ckpt`)

Entry point: `train_mixed.py`.

1. Read extractor config from `phase1_extractor/args.json`
  - `architecture=HystereticGRU`
  - `hidden_state_dim=512`
  - `bidirectional=True`
  - `layers=4`
2. Construct `HystereticGRU(**extractor_args)`
3. Resolve checkpoint: `phase1_extractor/**/best.ckpt`
4. Load Lightning `state_dict` into extractor
5. Freeze extractor:
  - `extractor.eval()`
  - `requires_grad=False` for all parameters

Important: Phase-1 weights are **not** updated during CFM training. They only produce conditioning features.

```text
phase1_extractor/args.json
        │
        ▼
  HystereticGRU architecture
        │
phase1_extractor/best.ckpt  ──load_state_dict──►  frozen extractor (L + P)
```



### B. Offline / dataset-time hidden-state extraction

In `PatternLoopDataset.get_hidden_state()`:

1. For each experiment id (`exp`) in the split, build one hysteresis sequence via `HystereticDataset`
  - input to GRU `L` is the simulated loop tensor (column info + drift/force features)
2. Forward only the encoder RNN:
  - `output, _ = extractor.L(simulated_loop)`
  - `output` shape: `[1, T, 1024]` (batch=1, sequence length `T`, bidirectional hidden=1024)
3. Convert to CPU matrix with `_to_hidden_state_matrix(output)` → `[T, 1024]`
4. Cache as `self.hidden_state[exp] = [T, 1024]`

Optional: extractor predictor `P(output)` is only used for hysteresis visualization (`data_check/*.svg`), not for CFM condition.

```text
Simulate CSV (Drift, Force) + column info
        │
        ▼
HystereticDataset  → simulated_loop [1, T, F]
        │
        ▼
extractor.L (GRU, frozen from best.ckpt)
        │
        ▼
hidden_state[exp] = [T, 1024]
```



### C. Per-sample condition assembly in `__getitem__`

For one damage-pattern sample from `label.csv`:


| Field          | Shape    | Source                                                                       |
| -------------- | -------- | ---------------------------------------------------------------------------- |
| `hidden_state` | `[1024]` | `self.hidden_state[exp][timestep, :]` indexed by image timestep              |
| `direction`    | `[2]`    | one-hot: Pull=`[0,1]`, Push=`[1,0]` (may flip if horizontal flip is enabled) |
| `DI`           | `[1]`    | damage index from `label.csv`                                                |
| `info`         | `[17]`   | standardized column features (used only when `latent_conditioning=False`)    |


With current flags, `info` is **not** concatenated into CFM condition.

Optional train-only augmentation: `--condition_perturb` applies multiplicative ±1%~±3% noise on selected tensors (`hidden_state` / `info` / `DI`).

Returned batch tensors (train with loss mask):

```text
image, hidden_state, direction, exp, timestep, DI, info, loss_mask
```



### D. Build generation condition vector `c`

In `CFMADM.process_condition()` → `build_generation_condition()`:

```text
if latent_conditioning:
    parts = [hidden_state]                 # [B, 1024]
    if direction_conditioning:
        parts += [direction]               # [B, 2]
    if include_di_conditioning:
        parts += [DI]                      # [B, 1]
    c = cat(parts, dim=1)                  # [B, 1027]
```

`set_dataset_args()` measures this concat width once and stores:

- `args.condition_dim = 1027`

When constructing `CFMADM`, this becomes U-Net `latent_dim`:

```python
ADMUNetCFM(
    ...,
    cond_dim=cfm_cond_dim,      # 128 (shared AdaGN / joint embedding width)
    latent_dim=condition_dim,   # 1027 (raw condition vector width)
)
```



### E. Inject `c` into U-Net with timestep (AdaGN path)

Training / validation call:

```text
c = process_condition(...)                 # [B, 1027]
pred_v = velocity_net(z_t, t, c)           # ADMUNetCFM.forward
```

Inside `ADMUNetCFM._build_cond(t, c)`:

1. Timestep embedding
  - `t_emb0 = sinusoidal_timestep_embedding(t, model_channels=32)` → `[B, 32]`
  - `t_emb = time_mlp(t_emb0)` → `[B, 128]`
2. Condition embedding
  - `c_emb = cond_mlp(c)` with `Linear(1027 → 128) → SiLU → Linear(128 → 128)` → `[B, 128]`
3. Joint fusion
  - `cond = joint_mlp(t_emb + c_emb)` → `[B, 128]`

This fused `cond` is broadcast into **every** `AdaGNResBlock` in down / mid / up paths.

### F. How AdaGN applies condition inside each ResBlock

For each `AdaGNResBlock(x, cond)`:

1. `h = GroupNorm(x)`
2. `ada1(cond)` → `[B, 2 * C_in]` split into `gamma, beta`
3. AdaGN modulation:
  - `h = SiLU(h) * (gamma + 1) + beta`
4. Conv3×3
5. Repeat with `ada2` on output channels
6. Residual add

So condition does **not** concatenate into spatial feature maps. It modulates channel-wise scale/shift after GroupNorm (FiLM / AdaGN style), jointly with flow time `t`.

```text
c [B, 1027] ──cond_mlp──► c_emb [B, 128]
t [B] ──sin+time_mlp──► t_emb [B, 128]
                │
                ▼
         joint_mlp(t_emb + c_emb)
                │
                ▼
           cond [B, 128]
                │
    ┌───────────┼───────────┐
    ▼           ▼           ▼
 Down AdaGN   Mid AdaGN   Up AdaGN
 (all ResBlocks receive the same cond vector)
```



### G. End-to-end summary (one training step)

```text
best.ckpt (HystereticGRU)
   └─ L(simulated_loop) → hidden_state[exp][t] ∈ R^{1024}
label.csv → direction ∈ R^{2}, DI ∈ R^{1}
        │
        ▼
c = concat(hidden, direction, DI) ∈ R^{1027}
        │
        ▼
ADMUNetCFM:
  c → cond_mlp → 128
  t → time_mlp → 128
  cond = joint_mlp(sum) ∈ R^{128}
        │
        ▼
AdaGN ResBlocks modulate z_t features → pred_v
loss = masked MSE(pred_v, z1 - z0)
```



### H. What is / is not coming from `best.ckpt`


| Comes from `best.ckpt`            | Does not come from `best.ckpt`     |
| --------------------------------- | ---------------------------------- |
| GRU encoder `L` weights           | CFM U-Net weights (`velocity_net`) |
| Resulting `hidden_state` features | `direction`, `DI` labels           |
| (optional viz) predictor `P`      | AdaGN / time embeddings            |


In short: `best.ckpt` contributes the **physics/hysteresis latent condition**; U-Net learns to turn `(z_t, t, c)` into velocity under flow matching.

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