# ARCHITECTURE_UPDATE_V8

## Delta Scope (V8 Only)

V8 builds on the V7 CFM U-Net (`cfm_model_channels=32`, `cfm_cond_dim=128`, Patch Diffusion ON) and introduces **split conditioning (特徵分流)**:

1. **Scalar path** — `time` + `DI` + `direction` → MLP → AdaGN (global modulation on every ResBlock)
2. **Spatial path** — object `condition_mask` → channel concat at `in_conv` (Zero-Conv init, unchanged from V6/V7)
3. **Physics path** — GRU `hidden_state [1024]` → token sequence → CrossAttention at mid / U3 / U2

V7 Patch Diffusion (coord channels, stochastic crops, image-only CFM loss) and VSS at L3/U3 are unchanged. V8 changes **how conditions are routed into the U-Net** and **where GRU physics is injected**.

Active script flags:

```bash
--condition_mode split
--use_damage_mask_condition True
--use_patch_diffusion True
--cfm_physics_num_tokens 16
--cfm_physics_token_dim 128
```

---

## 1) Three Injection Mechanisms

V8 uses three distinct conditioning routes. Each carries a different semantic role and enters the U-Net at a different point.

| Route | Source signal | Injection site | Mechanism | Role |
| ----- | ------------- | -------------- | --------- | ---- |
| **AdaGN** | `time` + `DI` + `direction` | Every `AdaGNResBlock` | Global `(γ, β)` on GroupNorm features | Timestep, damage intensity, load direction |
| **in_conv** | Object `condition_mask` | U-Net input | Channel concat + Zero-Conv | Spatial layout / valid region |
| **CrossAttention** | GRU `hidden_state` | mid, U3, U2 | Q=image, K/V=physics tokens | Macro physics alignment & coarse generation |

```text
Batch fields
  hidden_state [B, 1024]  ──────────────────────────► Physics path ──► CrossAttention
  DI           [B, 1]     ─┐
  direction    [B, 2]     ─┼──► Scalar path ──► AdaGN (all ResBlocks)
  t            [B]        ─┘       (+ sinusoidal time_emb)
  condition_mask [B,1,H,W] ──────────────────────────► Spatial path ──► in_conv concat
  z_t + coords [B,5,H,W]  ──────────────────────────► Flow state (not conditioning)
```

Implementation: `build_scalar_condition`, `build_physics_condition` in `ImageGeneration.py`; `ADMUNetCFM` in `cfm_adm.py`.

---

## 2) Condition Routing (`condition_mode=split`)

### 2.1 Scalar path → AdaGN

`build_generation_condition()` short-circuits when `condition_mode == "split"`:

```text
scalar_c = cat([DI, direction])          # [B, 3]  (DI=1 + direction=2)
```

Inside `ADMUNetCFM._build_cond`:

```text
time_emb = sinusoidal_timestep_embedding(t, model_channels=32)
time_emb = time_mlp(time_emb)            # [B, 128]
scalar_emb = scalar_mlp(scalar_c)          # Linear(3→128) → SiLU → Linear(128→128)
AdaGN_cond = joint_mlp(time_emb + scalar_emb)   # [B, 128]
```

`AdaGN_cond` modulates **every** `AdaGNResBlock` via learned `(γ, β)`:

```text
h = SiLU(GroupNorm(x)) * (γ + 1) + β
```

**Not fed into AdaGN under `split`:**

| Field | Shape | Loaded in batch? | Injection |
| ----- | ----- | ---------------- | --------- |
| `hidden_state` | `[1024]` | Yes | CrossAttention only |
| `info` | `[17]` | Yes | No (CFM) |

`set_dataset_args()` sets `condition_dim = 3`, `physics_dim = 1024`.

### 2.2 Spatial path → in_conv

Unchanged from V6/V7. Object mask from `Data/DamagePattern/mask` (train: aug-aligned; val/test/ODE: all-ones):

```text
in_conv_input = cat([model_in, condition_mask], dim=1)
              = cat([z_t_image, coords, condition_mask])
              = [B, 6, P, P]     # 3 + 2 + 1
h0 = in_conv(in_conv_input)       # [B, 32, P, P]
```

The extra mask channel is **zero-initialized** (`_expand_conv_in_with_zero_mask_channel`). Existing coord/image weights are copied; only the new channel starts at zero.

### 2.3 Physics path → CrossAttention tokens

GRU extractor output is serialized into a token sequence for cross-attention:

```text
gru_feat [B, 1024]
  → LayerNorm(1024)
  → Linear(1024 → 16×128 = 2048)
  → reshape
gru_tokens [B, seq_len=16, dim=128]
```

Cross-attention at each injection site:

```text
Q  ← image features [B, 128, H, W]   reshaped to spatial tokens
K,V ← gru_tokens     [B, 16, 128]
out = x + proj( softmax(QK^T / √d) · V )
```

Heads: `cfm_mid_attn_heads=4` → `head_dim = 128 / 4 = 32`.

**Physics injection is stopped at U1.** U1/U0 rely on skip connections + AdaGN + conv to restore high-frequency detail without GRU interference.

---

## 3) U-Net Architecture

Active config: `cfm_model_channels=32`, `channel_mult=(1,2,4,4)`, `num_res_blocks=1`, Patch Diffusion ON, mask-as-condition ON, `cfm_mid_attn_heads=4`, `use_vss_l3_u3=True`, `condition_mode=split`.

Example below uses **full patch `P=128`**. Replace spatial dims with `P` for stochastic crops (`64` / `32`); channel widths are unchanged (bottleneck = `P/8`).

### 3.1 Inputs (before U-Net)

```text
z_t_image      : [B, 3, P, P]       # noised flow state (RGB)
coords         : [B, 2, P, P]       # absolute XY ∈ [-1, 1]  (V7 Patch Diffusion)
model_in       = cat([z_t_image, coords])          # [B, 5, P, P]
condition_mask : [B, 1, P, P]       # object mask → in_conv injection
scalar_c       : [B, 3]             # DI + direction → AdaGN injection
gru_feat       : [B, 1024]           # hidden_state → CrossAttention injection
gru_tokens     : [B, 16, 128]       # derived once per forward, shared by all cross-attn modules
```

### 3.2 U-Net tensor flow (`P=128`)

```text
Input model_in [B, 5, 128, 128]
condition_mask [B, 1, 128, 128]
  └─ in_conv: cat → [B, 6, 128, 128] → 32          ★ Spatial injection
      h0 [B, 32, 128, 128]

Down path (pure visual extraction; AdaGN only)
  L0: ResBlock(AdaGN) 32 → 32, save skip0 [B, 32, 128, 128]
      Downsample → [B, 32, 64, 64]
  L1: ResBlock(AdaGN) 32 → 64, save skip1 [B, 64, 64, 64]
      Downsample → [B, 64, 32, 32]
  L2: ResBlock(AdaGN) 64 → 128, save skip2 [B, 128, 32, 32]
      Downsample → [B, 128, 16, 16]
  L3: ResBlock(AdaGN) 128 → 128
      l3_vss (VSSBlock2d, dim=128)
      save skip3 [B, 128, 16, 16]

Mid path (macro physics alignment)
  mid_block1:  ResBlock(AdaGN)  [B, 128, 16, 16]
  mid_attn:    SelfAttention2d  (4 heads × 32 dim, 256 spatial tokens)
  mid_cross_attn: CrossAttention2d (Q=image, K/V=gru_tokens)  ★ Physics injection 1
  mid_block2:  ResBlock(AdaGN)  [B, 128, 16, 16]

Up path (physics-guided generation → high-freq handoff to conv)
  U3: concat(h, skip3) → [B, 256, 16, 16] → ResBlock(AdaGN) → [B, 128, 16, 16]
      u3_cross_attn: CrossAttention2d (Q=image, K/V=gru_tokens)  ★ Physics injection 2
      u3_vss (VSSBlock2d, dim=128)
      Upsample → [B, 128, 32, 32]

  U2: concat(h, skip2) → [B, 256, 32, 32] → ResBlock(AdaGN) → [B, 128, 32, 32]
      u2_cross_attn: CrossAttention2d (Q=image, K/V=gru_tokens)  ★ Physics injection 3
      Upsample → [B, 128, 64, 64]

  U1: concat(h, skip1) → [B, 192, 64, 64] → ResBlock(AdaGN) → [B, 64, 64, 64]
      (no cross-attn — conv restores mid-freq detail)
      Upsample → [B, 64, 128, 128]

  U0: concat(h, skip0) → [B, 96, 128, 128] → ResBlock(AdaGN) → [B, 32, 128, 128]
      (no cross-attn — conv restores high-freq detail)

Output
  out_norm + SiLU + out_conv: 32 → 5
  pred_full [B, 5, 128, 128]
  pred_v    = pred_full[:, :3]                       # [B, 3, 128, 128] used by CFM / ODE
```

### 3.3 Injection map by U-Net stage

| Stage | Spatial size | AdaGN | in_conv | CrossAttention |
| ----- | ------------ | ----- | ------- | -------------- |
| Input | `P × P` | — | ★ mask concat | — |
| L0–L3 (Down) | `P` → `P/8` | ★ every ResBlock | — | — |
| Mid | `P/8 × P/8` | ★ | — | ★ `mid_cross_attn` |
| U3 | `P/8 × P/8` | ★ | — | ★ `u3_cross_attn` |
| U2 | `P/4 × P/4` | ★ | — | ★ `u2_cross_attn` |
| U1 | `P/2 × P/2` | ★ | — | — |
| U0 | `P × P` | ★ | — | — |

### 3.4 Spatial sizes by patch schedule

With `channel_mult=(1,2,4,4)`, downsample factor = 8:

| Patch `P` | `in_conv` spatial | Bottleneck (`mid_*`, U3 cross-attn) | U2 cross-attn spatial | CrossAttn image tokens |
| --------- | ----------------- | ----------------------------------- | --------------------- | ---------------------- |
| `128` | `128 × 128` | `16 × 16` (256 tokens) | `32 × 32` (1024 tokens) | mid/U3: 256; U2: 1024 |
| `64` | `64 × 64` | `8 × 8` (64 tokens) | `16 × 16` (256 tokens) | mid/U3: 64; U2: 256 |
| `32` | `32 × 32` | `4 × 4` (16 tokens) | `8 × 8` (64 tokens) | mid/U3: 16; U2: 64 |

`gru_tokens` count is fixed at **16** regardless of patch size; only the image-side spatial token count changes.

Channel layout is identical for all `P`; only spatial dims scale.

---

## 4) End-to-End Condition + U-Net Forward

```text
batch: image, hidden_state, direction, DI, condition_mask, ...
        │
        ├─ scalar_c  = cat([DI, direction])           [B, 3]     → AdaGN (all ResBlocks)
        ├─ gru_feat  = hidden_state                   [B, 1024]  → gru_tokens [B, 16, 128]
        └─ condition_mask                             [B, 1, P, P] → in_conv concat
        │
        ▼
z_t = (1-t)·z0 + t·z1
model_in = cat([z_t, coords])                         [B, 5, P, P]
        │
        ▼
ADMUNetCFM.forward(model_in, t, scalar_c, gru_feat, damage_mask=condition_mask)
  in_conv:  cat([model_in, mask]) → h0                 ★ spatial
  down/up:  ResBlock(h, AdaGN_cond)                    ★ scalar (every block)
  mid/U3/U2: CrossAttention2d(h, gru_tokens)          ★ physics (3 sites)
        │
        ▼
pred_full [B, 5, P, P]  →  pred_v = pred_full[:, :3]  →  masked MSE vs (z1 - z0)
```

Inference ODE uses the same three injection routes at full `128 × 128` with all-ones mask.

---

## 5) Active Runtime Config Summary (`train_ADMCFM.sh`)

```text
Architecture:       CFMADM (pixel space, V7 + V8 split conditioning)
Condition mode:     split
  Scalar (AdaGN):   DI [1] + direction [2] → cond_dim=128
  Spatial (in_conv): object mask channel (Zero-Conv)
  Physics (CrossAttn): hidden_state [1024] → 16 tokens × 128 dim
U-Net width:        cfm_model_channels=32, cfm_cond_dim=128
CrossAttn heads:    cfm_mid_attn_heads=4 (mid + u3 + u2)
U-Net VSS:          use_vss_l3_u3=True (l3_vss + u3_vss, dim=128)
Patch Diffusion:    ON (128²=0.50, 64²=0.30, 32²=0.20)
Loss:               image channels only (drop predicted coords)
Mask condition:     use_damage_mask_condition=True
Batch size:         128
Max steps:          15000
ODE:                Heun, 30 steps, full 128² + full coords
```

---

## 6) Updated Files (V8 touch list)

| File | Change |
| ---- | ------ |
| `src/model/Component/cfm_adm.py` | `CrossAttention2d`; split scalar/physics paths; `mid_cross_attn`, `u3_cross_attn`, `u2_cross_attn` |
| `src/model/ImageGeneration.py` | `build_scalar_condition`, `build_physics_condition`; `_unpack_conditions`; pass `gru_feat` through loss/ODE |
| `argument/argument_mixed.py` | `condition_mode=split`; `--cfm_physics_num_tokens`, `--cfm_physics_token_dim`, `--physics_dim` |
| `job/DamagePatternGeneration/train_ADMCFM.sh` | `CONDITION_MODE=split` |

---

## 7) Resume / Migration Notes

### From V7 (`condition_mode=full`, 1027-dim AdaGN)

- `cond_mlp` (1027 → 128) replaced by `scalar_mlp` (3 → 128) — **incompatible weights**
- New modules: `physics_norm`, `physics_proj`, `mid_cross_attn`, `u3_cross_attn`, `u2_cross_attn` — random init
- **Train from scratch** for a clean V8 run

### From V7 without split (same U-Net body)

- Patch Diffusion coord channels, VSS, mask-as-condition weights are reusable
- Only conditioning head + cross-attn modules differ

### From pre-VSS checkpoint under V8

- Enable `--use_vss_l3_u3 True`: VSS weights missing → random init (same as V7 note)

---

## 8) V7 → V8 Quick Comparison

| Aspect | V7 | V8 |
| ------ | -- | -- |
| Condition routing | Single vector → AdaGN | Split: scalar / spatial / physics |
| GRU `hidden_state` | Concat into AdaGN (`full` mode) or unused (`di_only`) | Tokenized → CrossAttention |
| DI + direction | Part of AdaGN vector or unused | Dedicated scalar → AdaGN |
| Object mask | in_conv concat | in_conv concat (unchanged) |
| Mid attention | SelfAttention only | SelfAttention + CrossAttention |
| Up attention | None | CrossAttention at U3, U2 only |
| Down path | AdaGN ResBlocks | AdaGN only (no physics) |
| Patch Diffusion | ON | ON (unchanged) |
| CFM loss target | Image channels only | Image channels only (unchanged) |
