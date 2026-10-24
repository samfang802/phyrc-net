# ARCHITECTURE_UPDATE_V11

## Delta Scope (v11 Only)

v11 changes the **condition** of the single-frame generator (gn, `1024-dadaGNsplit`). The network, loss, data, augmentation and schedule are gn's. Three things change:

1. **Physics condition.** The GRU hidden state g (1024) is replaced by the damage-state encoder: its state z (256) through the existing gated branch, and four damage scalars through a new gated branch.
2. **Loading sign.** The sign of the drift at the frame's peak (+1 / −1, 0 before loading) is appended to the face one-hot in the scalar path.
3. **Horizontal flip keeps the face label.** gn swapped the face one-hot on every flipped sample.

Only the full v11 run (called G2 in the planning discussion) is trained. The intermediate runs (gn + flip fix only; z + sign without damage scalars) are not, so a change against gn cannot be attributed to one of the three items above.

Nothing in this file is a measured result unless it says so.

---

## 1. Why

From `docs/eval/condition_g.md` and `docs/DAMAGE_STATE_ENCODER.md` (all measured):

- **g is organised by loading direction.** In the raw GRU state, the sign of the drift at the frame explains 37–46% of the variance; consecutive frames of one face (which alternate sign) have cosine −0.30. The stacked bidirectional GRU also mixes the future of the loop into every g_t (cosine 0.53–0.80 between the full-loop and truncated-loop states).
- **g barely tells specimens apart in damage terms.** A ridge probe from g to spalling over-predicts R307 (val) by +0.23 / +0.31 at 6 / 8% drift and under-predicts CTR1 (test) by −0.20 / −0.21, and the generated damage areas err in the same directions.
- **Δg was unused.** α_dg in v10.6-F ended at 0.000055; consecutive-frame Δg was dominated by the sign flip.
- **The damage-state encoder fixes the structure.** The selected encoder (A_info, below) has sign variance 0.020, consecutive-frame cosine 0.99, a spalling leave-one-specimen-out RMSE of 0.055 on the four train specimens (GRU 0.073) and roughly halves the R307 bias at 6% drift (3-seed means). It does **not** improve CTR1, and it predicts peak force less accurately than the GRU (median RMSE 0.105 vs 0.077 on 30 held-out specimens).
- **The loading sign is visible.** At drift ≥ 3%, opposite-sign neighbouring frames differ more than same-sign frames two steps apart (mean |Δ| 26.2 vs 23.5 grey levels), although they are closer in time. Since the encoder removes the sign, it is given back explicitly.
- **The face flip was wrong.** The photographed faces are the two faces normal to the loading direction, not side faces (§3). gn's flip therefore mislabelled the face on half of the training samples.

## 2. Condition

```
e_c = LN( MLP( t, face one-hot (2), loading sign (1) ) )                 no gate (as gn, one more input)
e_g = LN( Linear_256→128( z ) )                                          × α     (as gn, z instead of g)
e_d = LN( MLP_4→64→128( strength, stiffness, energy, log1p max drift ) ) × α_d   (new, α_d = 0 at init)
h_cond = e_c + α·e_g + α_d·e_d          → AdaGN in every ResBlock (unchanged)
```

| Input | Source | Notes |
|---|---|---|
| face one-hot | label `direction` (Push / Pull face) | unchanged, but not swapped on flip |
| loading sign | sign of the simulated drift at the frame timestep | 0 at timestep 0 |
| z | damage-state encoder, causal GRU state at the latest load peak ≤ t | zero state before the first peak |
| strength ratio, stiffness ratio, cumulative energy | encoder heads, standardized target units; stiffness clipped ≥ 0 | predictions from the simulated loop (the experimental loop is not available at inference) |
| log1p(max drift so far, %) | simulated loop | |
| specimen code s | **not given** to the generator | only ~3 distinct values among the four train specimens, so it would act as a specimen ID; CTR1's s is far outside (Vn ≈ +3.9 σ). z already reads s inside the encoder. |
| DI | still zeroed (`zero_di_condition`) | overlaps the damage scalars; R307 shows high DI does not mean much spalling |

Design choices:

- **Sign with the face, in one MLP.** Their effect is an interaction (at a positive peak one face is in tension, the other in compression), which a shared MLP can form and two additive gated branches cannot easily.
- **z by Linear.** On the 30 held-out specimens a linear read-out of z reaches the encoder's own MLP heads (R² within ±0.04 for all five targets), so the information is linear in z; a larger projection would mainly add capacity to memorise four specimens. Same form as gn's g projection.
- **Damage scalars by a small MLP, own gate.** Four inputs through a Linear would span only a 4-d subspace and could not form thresholds (e.g. after strength loss). The separate α_d makes their use readable after training, as α / α_dg were in v10.6.
- **Zero-init gates.** α and α_d start at 0; with gn's zero-initialised AdaGN projections, the condition enters gradually. Gates did not block learning before (α reached −0.2 in v10.6-F).

The physics vector passed through the code is `[z (256) | 4 damage scalars]` = 260 numbers in the `hidden_state` slot, so sampling, evaluation and the dataset interface are unchanged. `ADMUNetCFM(damage_scalar_dim=4)` splits off the last four.

### 2.1 Encoder

`results/DamageStateEncoder/A_info/2026-10-23_22-34-47/` (`docs/DAMAGE_STATE_ENCODER.md` §8): half-cycle token input, static code from 7 dimensionless descriptors + 17 column-info features + failure type, causal 2-layer GRU (hidden 256), trained on 124 hysteresis specimens with the six imaged specimens excluded from training and selection; seed 1 of 3, chosen by val loss. It is frozen; its per-frame outputs are computed once when the dataset is built (`src/data/damage_condition.py`).

## 3. Face labels under horizontal flip

Each specimen is photographed from two opposite faces (`Push`: NE or NW, `Pull`: SW or SE). Two possibilities:

- side faces (normal ⊥ loading): the tension side alternates left / right with the loading sign, and a flip is equivalent to the other face;
- flexural faces (normal ∥ loading): the whole face is in tension or in compression, left-right is a symmetry, and a flip changes nothing.

Measured on GT images at drift ≥ 3% (dark-pixel fraction as a crude damage proxy):

- The left-right asymmetry of each face is the same at positive and negative peaks (e.g. R315 Pull +0.088 / +0.091; correlation with the sign between −0.34 and +0.35, inconsistent in direction). It does not follow the loading sign.
- The two faces respond to the sign in opposite ways in 11 of 12 face-specimen pairs: the Push face is darker at negative peaks (e.g. R315 0.308 vs 0.286), the Pull face at positive peaks (0.248 vs 0.238); CTR1's Pull face is flat. The Push face is darker overall (0.28–0.33 vs 0.22–0.27).

So the faces are the flexural faces, and a flip must keep the face label (and the sign). With gn's 0.5 flip probability, half of its training samples carried the wrong face label. `--flip_keeps_direction True` fixes this.

## 4. Training

The gn recipe of `job/DamagePatternGeneration/train_ADMCFM.sh` (ADM U-Net 32 channels, split gated AdaGN, zero DI, patch diffusion, no self-conditioning, batch 128, lr 1e-4 warmup + cosine, 15000 steps, Min-SNR, condition perturbation 1–3%, hflip 0.5), **from scratch**, with `V11=True`:

```
--damage_condition_encoder results/DamageStateEncoder/A_info/2026-10-23_22-34-47
--damage_scalar_dim 4 --load_sign_condition True --flip_keeps_direction True
```

`physics_dim` (260) and the scalar `condition_dim` (3) are set from the data. Logging adds `train/damage_alpha` next to `train/gated_fusion_alpha`.

The condition perturbation multiplies the whole 260-vector by (1 + U(0.01, 0.03)), damage scalars included.

## 5. Flags and switches

| Flag | Default | v11 | Meaning |
|---|---|---|---|
| `--damage_condition_encoder` | "" | encoder run | replace the GRU state by `[z | damage scalars]` |
| `--damage_scalar_dim` | 0 | 4 | trailing damage scalars of the physics vector, own gate α_d |
| `--load_sign_condition` | False | True | append the loading sign to the face one-hot |
| `--flip_keeps_direction` | False | True | flip keeps the face label |

Job switch: `V11` in `train_ADMCFM.sh`, **default True**, so a plain `qsub job/DamagePatternGeneration/train_ADMCFM.sh` trains v11 (PBS header: 2 CPUs, 15 GB, `gpu_id` 6, log `log/train_v11.{out,err}`; `NUM_WORKERS` 1, `RUN_TAG` v11, `RESUME_CKPT` none). `V11=False` gives the previous GRU recipe; `V11_ENCODER=<run>` swaps the encoder; `V11_ENCODER=none` keeps the GRU g and adds only the loading sign and the flip fix (§9.3).

## 6. Code map

| File | Change |
|---|---|
| `src/data/damage_condition.py` (new) | per-timestep `[z | damage scalars]` from the encoder; loading signs |
| `src/data/pattern_loop.py` | the three options; `_direction_to_tensor(..., exp, timestep)` appends the sign and honours `flip_keeps_direction` |
| `src/model/adm.py` | `damage_scalar_dim`: split the physics vector, `damage_mlp` + `e_d_ln` + `alpha_d`; `get_damage_alpha` |
| `src/model/cfm/options.py`, `training_mixin.py` | pass `damage_scalar_dim`; log `train/damage_alpha` |
| `src/model/cfm/refiner_cache.py` | passes exp / timestep to `_direction_to_tensor` (needed if a v11 base is used for a refiner) |
| `argument/mixed.py` | the four flags |
| `job/DamagePatternGeneration/train_ADMCFM.sh` | `V11` switch |

CPU unit test: with α = α_d = 0 the output ignores z and the damage scalars; the sign changes it through the scalar path; with non-zero gates both z and the scalars change it (AdaGN projections randomised for the test, since they are zero-initialised).

## 7. Runs

| | Job | PBS `gpu_id` |
|---|---|---|
| Smoke (6 steps) | `56849` failed: out of memory at 6.1 GB, because PBS `gpu_id` 4 is the A5000 that also holds a non-PBS vLLM process (16.65 GB). Not a v11 problem. | 4 |
| Smoke (6 steps) | `56850` passed: dataset with the 260-d condition, `condition_dim` 3, 6 training steps, checkpoints, validation samples. Peak GPU memory 19.1 GB. | 5 |
| v11, 15000 steps | `results/DamagePatternGeneration/CFMADM/2026-10-24_05-55-15_v11`, stopped and resumed as `2026-10-24_12-36-59_v11` (checkpoints 12000, 14000, `last.ckpt`). Needs ~19 GB free on the card. | 6 (24 GB A5000, set in the script header) |
| v10.6 refiner on the v11 base | `2026-10-24_15-09-56_v11_refiner`, base `v11/checkpoint_step_012000.pt`, 15000 steps (`train_ADMCFM_refiner_residual.sh`) | 5 |
| Ablation `v11_gru` (§9.3) | `2026-10-24_22-07-50_v11_gru`, job `56894`; eval `56895` queued after it (`eval_suite_v11_gru.json`) | 6 |

## 8. Evaluation plan

Quick suite on test and val, paired against gn (its samples already exist: `results/eval_suite/v105/test/samples/gn.npz`, `results/eval_suite/v106/val/samples/gn.npz`).

| Question | Read | Expectation (not measured) |
|---|---|---|
| Does damage follow drift better? | progression ratio / corr, neighbour contrast | better than gn |
| Is R307 over-drawn less? | val high-DI damage area (GT 0.079, gn 0.085, v106F 0.103) | lower or equal |
| Does CTR1 change? | test damage area (GT 0.164, gn 0.106) | about unchanged: the encoder does not carry CTR1's spalling propensity |
| Realism | KID | about unchanged (the oracle condition improved KID by only 0.009) |
| Are the new inputs used? | `train/gated_fusion_alpha`, `train/damage_alpha` | both away from 0 |
| Face × sign | GT pattern of §3 (Push darker at negative peaks) in the samples | reproduced, unlike gn |

If v11 helps, the next steps are the v10.6 refiner on top of a v11 base (state cache rebuilt; Δz instead of Δg) and the physics-based spalling indicator for CTR1 (`docs/DAMAGE_STATE_ENCODER.md`, `docs/eval/condition_g.md` §11).

## 9. Results (measured, eval-v1, DINOv3, K = 8)

Sources: `results/eval_suite/v11/` (v11 vs gn), `results/eval_suite/v11_refiner/` (refiner checkpoints vs v11_12k), `results/eval_suite/v106C/` (v106C_6k). Rows from different suite runs share the protocol and reference set (gn reproduces to ±0.001 KID across runs) but have no paired z between them.

### 9.1 v11 base vs gn

| | KID ↓ | progression ratio →1 | progression corr ↑ | GT neighbour ↑ | high-DI damage area →GT | high-DI fog ↓ | texture score ↓ |
|---|---|---|---|---|---|---|---|
| **test (CTR1)** GT / reference | other column 0.072 | 0.53 | 0.83 | 100% | 0.164 | 0.086 | train column 1.20 |
| v11 (15k) | 0.210 | 0.49 | 0.79 | 67% | 0.085 | 0.466 | 3.57 |
| v11_12k | 0.202 | 0.50 | 0.79 | 74% | 0.096 | 0.324 | 2.41 |
| gn | 0.195 | 0.38 | 0.81 | 58% | 0.106 | 0.135 | 0.85 |
| **val (R307)** GT / reference | | 0.96 | 0.90 | 100% | 0.079 | 0.032 | train column 1.60 |
| v11 (15k) | 0.122 | 0.78 | 0.87 | 75% | 0.088 | 0.413 | 3.94 |
| v11_12k | 0.116 | 0.81 | 0.88 | 79% | 0.099 | 0.311 | 2.95 |
| gn | 0.119 | 0.66 | 0.86 | 53% | 0.085 | 0.216 | 2.02 |

Paired against gn (block jackknife): progression ratio +0.11 / +0.12 (z 3.6 / 3.6, test / val); GT neighbour +0.06 / +0.18 (z 1.3 / 5.5); KID +0.015 (z 2.3) on test, tied on val. v11_12k: ratio +0.13 / +0.16 (z 4.4 / 4.3), neighbour +0.11 / +0.21 (z 2.8 / 5.6), KID tied.

Against the §8 expectations:

- **Damage follows drift better**: yes, on both columns (ratio, neighbour contrast). Corr is unchanged.
- **R307 over-drawn less**: yes, the val high-DI area is 0.088 (GT 0.079), against 0.103 for v106F and 0.142 for the V10 refiner. gn was already at 0.085.
- **CTR1 unchanged**: it is slightly lower (0.085 vs gn 0.106; GT 0.164), so CTR1 spalling is still under-drawn, as the encoder screen predicted.
- **Realism**: KID about unchanged (slightly worse on test at 15k).
- **New inputs used**: at 15k, α = −0.130 and α_d = −0.061; both are away from 0.
- **Face × sign**: not measured.
- **Not expected: the high-DI damage turns into grey fog.** The fog share is 3.5× gn's on test and 1.9× on val, the fine-texture energy E drops (7.5 vs 11.0) and the texture score is the worst of all our models (same level as cGAN, 3.55). It also gets worse from 12k to 15k. Which of the three changes causes it is not known (see Delta Scope), hence §9.3.

### 9.2 v10.6 refiner on the v11 base

Checkpoints 6k / 9k / 13k of `2026-10-24_15-09-56_v11_refiner`, paired against its base v11_12k:

| | KID ↓ | ratio →1 | corr ↑ | GT neighbour ↑ | high-DI fog ↓ | texture score ↓ | tone jump →GT | damage regress ↓ | copy excess ↓ |
|---|---|---|---|---|---|---|---|---|---|
| **test** GT | | | | | 0.086 | | 4.07 | 0.12 | |
| v11r_6k | 0.134 | 0.65 | 0.86 | 101% | 0.136 | 1.25 | 3.12 | 0.15 | 0.121 |
| v11r_9k | 0.135 | 0.66 | 0.87 | 98% | 0.200 | 1.54 | 2.71 | 0.15 | 0.091 |
| v11r_13k | 0.142 | 0.67 | 0.85 | 94% | 0.242 | 1.89 | 4.15 | 0.13 | 0.112 |
| v106C_6k (gn base) | 0.137 | 0.48 | 0.90 | 81% | 0.060 | 1.09 | 3.90 | 0.26 | 0.111 |
| **val** GT | | | | | 0.032 | | 7.42 | 0.29 | |
| v11r_6k | 0.066 | 1.01 | 0.96 | 116% | 0.111 | 2.34 | 3.09 | 0.14 | 0.098 |
| v11r_9k | 0.068 | 1.00 | 0.95 | 115% | 0.237 | 2.71 | 3.15 | 0.16 | 0.055 |
| v11r_13k | 0.065 | 1.00 | 0.95 | 119% | 0.212 | 2.75 | 4.63 | 0.17 | 0.076 |
| v106C_6k (gn base) | 0.079 | 0.82 | 0.90 | 68% | 0.091 | 2.17 | 4.39 | 0.21 | 0.087 |

- All three checkpoints beat v11_12k on KID, neighbour contrast, progression ratio and corr on both columns (|z| ≥ 2.4), and remove most of the base's flicker (tone jump 15.7 → ~3).
- **6k is the best checkpoint**: same condition metrics as 9k / 13k, least fog. Fog and the texture score worsen with training, in line with the training-time KID (`kid_log.jsonl`: 0.113 at 6k / 9k, 0.117 at 12k).
- **Against v106C_6k** (same refiner recipe, gn base): v11r_6k follows the condition much better (neighbour 101% / 116% vs 81% / 68%, ratio 0.65 / 1.01 vs 0.48 / 0.82) and regresses less, but its high-DI texture is worse (fog 0.136 vs 0.060 on test, R_spall probe 0.33 vs 0.46). The texture gap is inherited from the base (§9.1).
- Drift × DI panels (`inference_samples/drift_di_{v11r_6k,v106C_6k}_{SW,NE}.png` in each run): v11r_6k is clean at 0 % and spalls gradually from DI ≈ 0.9 at 6 %. v106C_6k already has cracks at 0 % and spalls abruptly, with blown-out white blocks at 8 % NE. Neither reproduces the NE spall that CTR1 already has at 4 % (DI 0.59).
- Still open: CTR1 spalling is under-drawn (high-DI area 0.099 vs 0.164), and the refiner copies more training-column crack layout than its base (copy excess ~0.1 vs ~0).

### 9.3 Ablation: GRU g + loading sign + flip fix (`v11_gru`)

Question: is the fog of §9.1 caused by replacing g with the encoder state, or by the sign / flip changes? `v11_gru` is the v11 recipe with `V11_ENCODER=none`: GRU g (1024) through the α gate as in gn, the loading sign in the scalar path, `flip_keeps_direction`, from scratch, 15000 steps. Evaluated with `eval_suite_v11_gru.json` (baseline v11; gn and v11_12k included).

Reading: texture close to gn and progression close to v11 → the fog comes from z, so keep g (with the sign) or feed g and z through separate gates. Fog as in v11 → the sign / flip changes cause it, and the condition can stay as it is.

Results: pending (training job `56894`, eval `56895`).

## Status (2026-10-24)

Trained and evaluated (§9). The condition change does what it was meant to (progression, R307), and the v10.6 refiner on top of it is the best model so far on condition and temporal metrics (v11r_6k). Open: high-DI fog inherited from the v11 base (§9.3 ablation running) and CTR1 spalling, which no condition-side change has improved (`docs/DAMAGE_STATE_ENCODER.md` §8).
