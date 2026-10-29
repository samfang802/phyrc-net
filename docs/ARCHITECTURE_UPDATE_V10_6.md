# ARCHITECTURE_UPDATE_V10_6

## Delta Scope (v10.6 Only)

v10.6 goes back to the V10 refiner layout (`S_{t-1}` concatenated to the input, `docs/ARCHITECTURE_UPDATE_V10.md` §3.3) and changes two things:

1. **Target x1.** The network generates `x_t` itself (`v = x_t - x0`), as in v10.5. There is no `S_t = clamp(S_{t-1} + residual)`.
2. **The v10.5 exposure package.** Offline full-ODE base states, state perturbation, one photometric draw per sample, tone loss, more full rollouts in the state track, shared low-frequency start noise (`docs/ARCHITECTURE_UPDATE_V10_5.md` §3-§4).

Unlike v10.5, **the refiner does not learn the single-frame task**. The base (frozen gn) only provides states: frame 0 at inference and the base hop of the state track. This is the cascade-diffusion split the refiner was designed from: the second stage is trained only to go from the first stage's (augmented) output to the target, and never to generate on its own.

There are three versions. F and S differ only in initialisation; C differs from F only in how `S_{t-1}` enters (§2.1):

| | 10.6-F (gn features) | 10.6-S (scratch) |
|---|---|---|
| Init | gn weights; the new `S_{t-1}` input channels zero; `out_conv` zeroed (as V10) | random |
| Question | do gn's internal features help the refiner? | does a cascade-style refiner trained from scratch avoid being tied to gn? |
| Switches | `REFINER_TARGET_X1=True` | `REFINER_TARGET_X1=True REFINER_FROM_SCRATCH=True` |

| | 10.6-C (control) |
|---|---|
| Init | as F (gn weights, `out_conv` zeroed); control branch random |
| `S_{t-1}` enters | a fresh ControlNet branch in the nycu-project layout, not the input channels |
| Question | is a separate encoder for `S_{t-1}` better than concatenation, with everything else as F? |
| Switches | `REFINER_TARGET_X1=True REFINER_IMPLICIT=True CONTROL_HINT_ONLY=True CONTROL_INJECT_ENCODER=True TRUNK_LR_MULT=1.0` |

Nothing in this file is a measured result unless it says so.

---

## 1. Why (from the v10.5 evaluation)

> 2026-10-29: DINOv3 metrics in this section (KID, progression ratio, neighbour contrast, saturated frames) are JPEG-q75 aligned (`<suite>/<split>/kid_jpeg75/`, see `docs/eval/protocol.md`). Texture and temporal metrics are unaffected.

Measured on test (`results/eval_suite/v105/test/summary_jpeg75.md`, `results/eval_suite/quick/test/summary_jpeg75.md`; gn / V10 refiner from `results/eval_suite/jpeg75/test/summary.md`, unpaired):

- The v10.5 exposure package fixed the tone. `tone_jump` fell from 24.1 (V10 refiner) to 4.8-6.1 (GT 4.1), saturated frames from 10% to 1-2%, and frame-to-frame change came close to GT (0.89-1.07 x GT; gn_shared 0.41 x).
- Realism and condition following fell back to gn's level. KID was 0.136 (A) and 0.142 (B), against gn 0.138 and the V10 refiner 0.113. The V10 refiner was also better on progression ratio (0.43 vs 0.39-0.40), but its neighbour-contrast lead is small (87% vs 79-82%; paired z = +0.7 against A, not significant).
- v10.5-A kept gn's whole single-frame model (frozen trunk, same target, same head) and its KID stayed at gn's level. The V10 refiner also loaded gn's weights, but it zeroed the output layer and learned a different target (the residual). The two differ in more than this, so this is not a controlled comparison. It does point the same way as the hypothesis that keeping gn's single-frame behaviour ties the refiner to gn.
- 15% of v10.5 samples (control dropout), plus every first frame, trained the single-frame task, which is never used at inference: frame 0 comes from gn.
- The step-3000 checkpoints of v10.5 scored as well as or better than step 15000 on test (KID 0.131 vs 0.140 for A, z = -3.3). Longer training roughly doubled `copy_excess` (A: 0.064 → 0.122), and B's grey fog (0.195 → 0.434) only appeared late. These checkpoints are from the 15000-step schedule, so they are not equivalent to a run that is only 3000 steps long.

**Why not ControlNet here.** ControlNet's main benefit is protecting a pretrained trunk with zero convs. Without a trunk worth protecting, it becomes a second encoder fused by addition at every scale (+2.2 M parameters, one more encoder pass per evaluation). Concatenation is the cascade (CDM / SR3) layout and the V10 layout, so v10.6 changes only the target and the training around it. A from-scratch ControlNet against a from-scratch concat net remains a possible follow-up; there is no evidence either way.

**A caveat on F.** Only `out_conv` (the last 3x3 conv) is zeroed. Every other layer keeps gn's weights, so the features just before the head are gn's features for generating x1. With an x1 target, the head can likely relearn gn's projection quickly, and F may settle near "gn + a little `S_{t-1}`". This is an inference, not a measurement. In V10 the same initialisation was less tied to gn because the residual target is a different quantity.

---

## 2. Network

```
z_t ──────────┐
S_{t-1} ──────┼─ concat ─► ADM U-Net (gn layout: 32 ch, VSS L3/U3, mid attention) ─► v      v = x_t - x0
coords, mask ─┘            AdaGN cond: t + c + α·g_t + α_dg·Linear(Δg)
sampling: S_t is the ODE sample itself (no S + r, no clamp)
```

- Input layout `[z (3) | S_{t-1} (3) | coords (2) | mask (1)]`, the V10 layout (`_build_velocity_net_input`).
- `Δg` in the trunk AdaGN, zero-gated, as V10.
- `_compose_cascade_sample` returns the sample unchanged when `refiner_target_x1` is on.

### 2.1 10.6-C: control branch in the nycu-project layout

```
S_{t-1} ─► in_conv (own, 3 → 32) ─► copy of the trunk encoder layout (random init) ─┐
                                                   zero 1x1 conv per encoder block   │
z_t, coords, mask ─► trunk in_conv ─► enc ─(⊕ after each block, before the skip)─► mid ─► dec ─► v
```

- Follows `nycu-project/src/model/Component/cfm_adm.py` (`ControlNet`): the branch reads the hint alone (not `z_t`), its weights are not copied from the trunk, and its outputs are added to the trunk encoder features before they are stored as skips. There is no mid output.
- `ControlBranch(hint_only=True)` in `src/model/adm.py`; `copy_from_trunk` refuses a hint-only branch. 1.42 M parameters (the v10.5 copied branch had 2.24 M).
- The trunk has no `S_{t-1}` input channels (`in_conv` keeps gn's shape). `Δg` goes to the control branch (`α_dg · Linear(Δg)`, zero-gated), as in v10.5.
- Everything else as F: gn warm start, `out_conv` zeroed, x1 target, `has_prev` weighting (no control dropout, no single-frame task), exposure package, all parameters at lr 1e-4 (`TRUNK_LR_MULT=1.0`).
- With zero convs at zero, step 0 is the gn trunk with a zeroed head, i.e. v = 0, the same start as F.

---

## 3. Training

`refiner_implicit_flow_matching_loss` (shared with v10.5; the branch is chosen by `refiner_implicit`):

```
z_τ = (1 - τ) x0 + τ x_t,   v = x_t - x0
input S_{t-1} = S_in perturbed (v10.5 §3.3) + Gaussian noise std ~ U(0, 0.05)
loss = Σ_b w_b · MinSNR(τ_b) · MSE_b / Σ_b w_b   +   1.0 · Σ_b w_b τ_b |LP(x̂1) - LP(x_t)|²_b / Σ_b w_b
w_b = has_prev_b          (no single-frame task; V10 weighting)
```

| Item | Setting | Same as |
|---|---|---|
| State track | two hops, probability 0.5 each; base hop from the offline gn cache (Heun 30, 4 samples per frame); self hop 1 Euler step, or a 5-step Heun with probability 0.3 | v10.5 |
| State perturbation, shared photometric draw, tone loss (1.0) | on | v10.5 |
| Optimizer | all parameters, AdamW lr 1e-4, warmup 1000, cosine to 0.1 x | V10 |
| Steps, batch | 15000, 128 | V10, v10.5 |
| Monitor | KID + exposure stats every 3000 steps (v10.5 §5) | v10.5 |
| Resources | 2 CPUs, 1 loader worker, 15 GB, PBS `gpu_id` 6 / 7 (24 GB A5000) | — |

From scratch the AdaGN gates (`alpha`, `alpha_dg`) start at zero, as they did when gn itself was trained.

---

## 4. Sampling

Frame 0 from gn, later frames from this network with the previous output concatenated, Heun 30, shared low-frequency start noise (v10.5 §4.1).

**Open question.** The low-frequency noise sharing was never ablated on v10.5. Whether it costs KID is unknown. The evaluation should run each v10.6 model with and without it (`--set refiner_shared_lowfreq_noise=0`).

---

## 5. Flags and switches

| Flag | Default | F / S | Meaning |
|---|---|---|---|
| `--refiner_target_x1` | False | True | v10.6; requires `--use_refiner_residual`. With `--refiner_implicit` (C), `S_{t-1}` goes through the control branch |
| `--control_hint_only` | False | C only | nycu-project control branch: hint alone, fresh weights, no mid; requires `--refiner_implicit` |
| `--velocity_init` | — | gn / empty | F and C warm start (zeroed `out_conv`) / S random init |
| v10.5 exposure flags | off | on | follow `REFINER_X1` in the job script |

Job script (`job/DamagePatternGeneration/train_ADMCFM_refiner_residual.sh`, no new job file): `REFINER_TARGET_X1`, `REFINER_FROM_SCRATCH`, `CONTROL_HINT_ONLY`. `REFINER_X1` (implicit or target x1) switches on the exposure package, roll probability 0.3, tone weight 1.0 and 15000 steps.

---

## 6. Code map

| File | Change |
|---|---|
| `src/model/adm.py` | `ControlBranch(hint_only=...)`, `ADMUNetCFM(control_hint_only=...)`; no mid residual when hint-only (C) |
| `src/model/cfm/options.py` | `refiner_target_x1`, `control_hint_only` and their validation |
| `src/model/cfm/loss_mixin.py` | x1 loss shared with v10.5; `has_prev` weighting and no control dropout whenever `refiner_target_x1` |
| `train_mixed.py` | `_load_velocity_init_with_control`: no copy for a hint-only branch; `out_conv` zeroed with `refiner_target_x1` (C) |
| `src/model/cfm/patch_mixin.py` | no `S + r` composition when `refiner_target_x1` |
| `src/model/cfm/training_mixin.py` | dispatch and logging (`train/refiner_fm_loss`, `train/refiner_tone_loss`) |
| `argument/mixed.py` | `--refiner_target_x1` |
| job script | `REFINER_TARGET_X1`, `REFINER_FROM_SCRATCH`, `REFINER_X1` |

---

## 7. Runs

| | Job | PBS `gpu_id` |
|---|---|---|
| Smoke F / S (6 steps, val and KID at step 5) | `56684` / `56685`, both passed | 6 / 7 |
| 10.6-F, 15000 steps | `56686` | 6 |
| 10.6-S, 15000 steps | `56687` | 7 |
| Smoke C | `56744`, passed; peak 24.7 GB (A6000) | 0 |
| 10.6-C, 15000 steps | `56745` | 0 (48 GB A6000) |

Peak GPU memory in the smoke runs, read from `nvidia-smi` (PyTorch allocator reserve, validation and KID included): about 21.5 GB of 24.5 GB on both. It fits, but with about 3 GB of headroom.

---

## 8. Evaluation plan

Quick suite (`job/evaluation/configs/eval_suite_quick.json`: test only, no second backbone, no interference removal or GT-mask placebo; about 4 minutes of generation per model). Models: 10.6-F, 10.6-S, each with and without shared low-frequency noise, plus the V10 refiner as a paired baseline. Its test samples already exist in `results/eval_suite/v105/test/samples/refiner.npz`; copy it into the quick output directory to skip regeneration.

| Question | Read |
|---|---|
| Does dropping gn's single-frame behaviour recover the V10 refiner's realism? | KID, progression ratio, neighbour contrast vs the V10 refiner (paired z) |
| Is the tone fix kept? | `tone_jump`, saturated frames, frame-to-frame change / GT |
| F vs S | all of the above, plus `copy_excess` |
| Is S just copying `S_{t-1}`? | progression ratio; `Δg` zeroed / shuffled at sampling |
