# ARCHITECTURE_UPDATE_V10_5

## Delta Scope (v10.5 Only)

v10.5 changes the V10 residual refiner (variant B, `docs/ARCHITECTURE_UPDATE_V10.md` §3.3) in two ways:

1. **Implicit residual.** `S_{t-1}` no longer enters as extra input channels with `S_t = clamp(S_{t-1} + residual)` composed outside the network. It enters a ControlNet branch, and the network generates the frame `x_t` itself, with the plain CFM target `v = x_t - x0`.
2. **Exposure error.** Five changes aimed at the tone problem in `docs/eval/results.md` (refiner section "偏亮與飽和偽影"): the refiner is brighter than GT (mean 179 vs 159 on test), 22% of its frames have a mean above 200, its `tone_jump` is 24 against 4 for GT, and the spikes are random rather than accumulated.

There are two versions. They use the same script and settings and differ only in two switches:

| | 10.5-A (basic) | 10.5-B (retrained) |
|---|---|---|
| Question | Can a control branch alone, with gn frozen, hold tone and structure across frames? | Does training the trunk as well also fix gn's own bias (high-DI spall at 60-70% of GT)? |
| Control residuals go to | the decoder skips and the mid output | the trunk encoder features before they are stored as skips (so they also reach deeper encoder levels), and the mid output; the nycu-project layout |
| Trunk | frozen gn | warm-started from gn, trained at 0.2 x lr |
| Switches | `CONTROL_INJECT_ENCODER=False FREEZE_TRUNK=True` | `CONTROL_INJECT_ENCODER=True FREEZE_TRUNK=False` |

A and B differ in two factors at once (injection point and trunk training), so their comparison answers "is B worth its cost", not "which factor helps". If B wins, a third run (encoder injection, frozen trunk) separates the two.

Nothing in this file is a measured result unless it says so.

### Status at writing

| Item | State |
|---|---|
| Code | Implemented. CPU unit test of the control branch passed (a fresh branch leaves the trunk output bit-identical in both injection modes; `control_scale=0` gives the trunk alone). |
| Smoke runs (6 steps, validation and KID at step 5, K = 1) | Passed for both (A `56659`, B `56658`): cache build, training, sequence panel and periodic KID all ran. A's first attempt (`56657`) failed because a loaded cache left the base on the CPU; fixed in `load_or_build_state_cache`. |
| Full runs | 15000 steps each. First submission (`56660` / `56661`, 4 CPUs, 2 workers) was cancelled after 11 minutes: PBS showed 137-143% CPU use (main process about 100%, each training worker about 22%) and 27-28 GB resident memory against a 15 GB request. Resubmitted as `56664` (A) and `56665` (B) with 2 CPUs, 1 worker, 15 GB (the queue maximum; a 32 GB request is refused). Resident memory was above the request before and nothing was killed, so the limit does not appear to be enforced, but that is not confirmed. Both finished 15000 steps. |
| Evaluation | Eval suite on A and B only (`job/DamagePatternGeneration/eval_suite_v105.json`, test + val, K = 8): `results/eval_suite/v105/{test,val}/summary.md`; drift x DI panels in `results/eval_suite/v105/test/panels/`. No other model was in the run, so comparisons with gn / V10 refiner use the numbers in `docs/eval/results.md` and carry no paired z. |

---

## 1. Why the tone jumps (diagnosis behind the changes)

Three causes, the first measured, the others inferred:

1. **Global tone is set by the low-frequency part of the start noise.** gn draws fresh noise per frame and has `tone_jump` 31; gn_shared, the same model with one start noise per trajectory, has 2.5. The V10 refiner draws each frame's residual from fresh noise, so the DC part of the residual is random.
2. **Training data teaches random tone changes (hypothesis).** `ColorJitter` (brightness / contrast 0.2 in the refiner job) was drawn independently for the target and the previous frames; only the flip was shared (`pattern_loop.py`, `force_flip`). The target therefore carried a random brightness offset relative to the state, which the residual had to learn. The size, about ±0.2 of the range, is of the order of the observed jumps. Not checked directly.
3. **The state track never shows the network an over-exposed state (inference).** 85% of V10 state-track samples were one Euler step, which is close to the conditional mean and has a smooth tone. The spikes come from full 30-step ODE samples with a random low-frequency start. The network never learned to pull a too-bright `S_{t-1}` back, and `clamp(S + r)` loses the information once a frame saturates.

---

## 2. Network: ControlNet branch

```
                 ┌── trunk (ADM U-Net, gn weights) ─────────────────────┐
z_t, coords ───► │ in_conv → enc ─(⊕ B)→ skips → mid ⊕ → dec(⊕ A) ──► v │   v = x_t - x0
mask             └────────────▲──────────────────▲──────▲──────────────┘
                              │ zero 1x1 conv per skip, one after mid
S_{t-1} ─► hint_block ─(+)─► control copy of in_conv + enc + mid
           (zero last conv)   cond = cond_trunk(t, c, g) + α_dg · Linear(Δg)
```

- **Class** `ControlBranch` in `src/model/adm.py`, built by `ADMUNetCFM(use_control_branch=True)`. It deep-copies the trunk's `in_conv`, `down_blocks`, `downsamples`, `l2_vss` / `l3_vss`, `mid_block1`, `mid_attn`, `mid_block2`.
- **Input** is the trunk's own input after the mask is concatenated (`z_t | coords | mask`) plus the hint `S_{t-1}` (3 channels, gray replicated) through `hint_block` (conv 3x3, SiLU, conv 3x3 zero-init). This is the original ControlNet layout. The nycu-project branch reads the hint alone; here the copied encoder weights only make sense on the trunk's input.
- **Outputs** are one zero-init 1x1 conv per trunk skip (in skip order) and one after mid. In decoder mode (A) they are added to the skips as the decoder pops them. In encoder mode (B) each is added to the trunk feature right after its block (and after the L3 VSS), before the feature is stored as a skip. The mid residual is added after `mid_block2` in both modes.
- **Conditioning.** The control branch uses the trunk's cond embedding. `Δg = g_t - g_{t-1}` enters only the control branch, as `α_dg · Linear(Δg)` (no bias, no norm, `α_dg = 0` at init), the same form as the V10 trunk branch. With `--refiner_implicit` the trunk is built without its own `Δg` layers.
- **Control scale.** `forward(..., control_hint, control_scale)` scales all control residuals per sample; `0` is the trunk alone. No hint means the branch is not run.
- **Initialisation** (`_load_velocity_init_with_control` in `train_mixed.py`). The gn checkpoint loads into the trunk unchanged: no input channels are added, so `in_conv` keeps its shape and the output layer is not zeroed. Then `copy_from_trunk` copies the encoder + mid into the control branch. Zero convs, the hint conv and `α_dg` stay zero, so step 0 is exactly gn. This also removes the V10 initialisation problem (§3.3 of the V10 doc: an un-zeroed head saturated to white, a zeroed head sampled noise).
- **Size.** With `model_channels = 32`: trunk 4.31 M parameters, control branch 2.24 M (measured in the unit test).
- **Freezing** (`--freeze_trunk`, set in `options.py` at construction, so it also holds on resume) turns off `requires_grad` for everything outside `control.`, the cond embedding included. The frozen trunk keeps its 0.1 dropout active in training, like the trainable trunk of B.
- **Optimizer** (`configure_optimizers`): control parameters at `--lr`, an unfrozen trunk at `trunk_lr_mult · lr` (0.2, so 2e-5). Both groups follow the same warmup + cosine schedule. gn ended its own schedule at 1e-5; 2e-5 lets the trunk move without restarting it from a high rate.

### Output

`_compose_cascade_sample` returns the sample unchanged in implicit mode: no `S + r`, no clamp. gn does not clamp its samples either, so the state passed to the next frame has the same range as gn's output and as the cached base states.

---

## 3. Training

### 3.1 Loss track

`refiner_implicit_flow_matching_loss` in `loss_mixin.py`:

```
z_τ = (1 - τ) x0 + τ x_t,   v = x_t - x0,   x0 ~ N(0, I),   τ ~ U(0, 1)
control hint = S_in, perturbed (3.3) + Gaussian noise std ~ U(0, 0.05)
control_scale = has_prev AND (u ≥ refiner_control_drop_prob)        u ~ U(0, 1) per sample
loss = Min-SNR weighted MSE(v_pred, v), mean over the batch
     + refiner_tone_loss_weight · mean_b( τ_b · mean |LP(x̂1) - LP(x_t)|² ),   x̂1 = z_τ + (1 - τ) v_pred
```

- Every sample counts. Samples without a previous frame, and 15% of the rest (`refiner_control_drop_prob`), run with the control branch off, which is the single-frame task. V10 weighted samples without a previous frame by 0. In A these samples give no gradient (the trunk is frozen); the setting is kept for parity with B, where they keep the trunk a single-frame model.
- `LP` is the mean over 16 x 16 blocks of the gray channel (`refiner_lowfreq_block`). The tone loss is weighted by `τ` because `x̂1` is only a blurred mean near the noise end. Weight 1.0 in both runs (set in the job script).
- Patch diffusion crops `x_t`, `x0` and the hint with the same offsets (`z_aux`), as in V10.

### 3.2 State track (two-hop, as V10)

`_refiner_state_track` (shared with the V10 loss, whose behaviour is unchanged) builds `S_in` without gradient:

- **Single hop** (probability 0.5): `S_in` = a base sample of frame t-1.
- **Chain** (probability 0.5, always in validation): a base sample of frame t-2, then one hop of this network to frame t-1 (`Δg = g_{t-1} - g_{t-2}`).
- **Base hop from the offline cache** (`--refiner_state_cache`). `src/model/cfm/refiner_cache.py` builds, once, 4 samples of the frozen gn for every train frame, with the 30-step Heun ODE used at evaluation. Each sample uses the frame's clean conditions, no flip and the frame's own object mask. They are stored as float16 in `results/DamagePatternGeneration/state_cache/train_n4_heun30_s128_<hash>.pt`, about 70 MB, keyed by base checkpoint and split frames. A file lock stops two jobs from building it twice. The dataset returns one random cached sample per frame (`state_prev`, `state_prev2`), flipped with the sample. The base hop therefore carries gn's real full-ODE error, tone spikes included, at no training cost. Validation batches have no cache and fall back to on-the-fly base sampling (5-step Heun in eval).
- **Self hop.** One Euler step from noise, or with probability `REFINER_ROLL_PROB_IMPLICIT = 0.3` (V10: 0.15) a 10-evaluation (5-step) Heun rollout.

### 3.3 State perturbation (`--refiner_state_perturb`)

`_perturb_refiner_state`, training only, applied to the hint, never to the target:

| Step | Draw (per sample) | Models |
|---|---|---|
| Contrast gain about the image mean | `U(1 - 0.15, 1 + 0.15)` | contrast error |
| Exposure offset, half of the batch | `U(-0.2, 0.4)` in model units (range [-1, 1]; 0.4 is about +51 gray levels) | the over-exposed frames; skewed bright because the observed error is bright |
| Smooth brightness field | Gaussian on 32 px cells, bicubic, std 0.08 | uneven exposure |
| Dark (damage) pixels pulled toward the median | factor `U(0.7, 1)` on pixels darker than median - 0.3 | spall drawn too weakly (gn draws 60-70% of GT) |
| Gaussian blur | prob 0.3, σ `U(0.5, 1.5)` | the smooth 1-step states |

The result is clipped to [-1, 1], so a pushed-up frame also saturates as a real one does. Because the target is untouched, the network is trained to ignore the tone of `S_{t-1}` and to take tone from the condition (and from the start noise, 4.1). The ranges are first guesses, not fitted to `measure_base_vs_gt_shift.py`.

### 3.4 One photometric draw per sample (`--share_photometric_aug`)

`MaskedDamageImageTransform.__call__(rng_state=...)` runs the photometric tail (ColorJitter, blur) from a given torch RNG state and restores the caller's stream afterwards. `PatternLoopDataset.__getitem__` takes the RNG state once and passes it to the target, previous and second-previous frames, so all three get the same brightness, contrast and blur. The flip was already shared. Train split only.

---

## 4. Sampling

### 4.1 Shared low-frequency start noise (`--refiner_shared_lowfreq_noise`)

`sample_refiner_sequence` draws one noise image `ε_s` per trajectory and, for every frame (frame 0 included):

```
z0_t = ε_t - P ε_t + P ε_s,    P = 16 x 16 block mean (avg-pool, nearest upsample)
```

`P` is an orthogonal projection, and for white Gaussian noise `Pε` and `(I - P)ε` are independent, so each `z0_t` is exactly N(0, I). The model sees in-distribution noise, so no retraining is needed for this. Neighbouring frames share the 8 x 8 grid of block means, which sets the global tone, and draw everything finer fresh. This sits between gn (all fresh: `tone_jump` 31) and gn_shared (all shared: `tone_jump` 2.5, but only 0.4x GT's frame-to-frame change). It also applies to a V10 checkpoint if the attribute is set at sampling.

### 4.2 Frames

Frame 0 is drawn by the frozen gn in both versions, so frame 0 is identical between A and B and differences come only from later frames. Later frames are drawn by this network with the previous output as the control hint. Heun, 30 steps, as all models in `results.md`.

### 4.3 Control guidance (`--control_guidance`, sampling only)

`v = v_off + w (v_on - v_off)`, where `v_off` is the trunk alone (the same cond, control branch skipped). `w = 1` (default) is plain control and costs nothing extra. It is meant for a sweep after training: `w < 1` gives more freedom per frame, `w > 1` follows the previous frame more closely. It is set through the eval script's `--set control_guidance=...`.

---

## 5. In-training KID (`--kid_every_n_steps`)

`PeriodicKID` (callback in `train_mixed.py`, logic in `src/metrics/train_kid.py`) runs every 3000 steps (`KID_EVERY_N_STEPS`, default in the refiner job, also for V10 runs from it):

- Generates the whole val split (R307, 128 frames, both faces) with the same sequence sampler and per-trajectory seeding as `generate_eval_samples.py`, K = 8 trajectories. `generate_eval_samples._generate_refiner` now calls the same function (`refiner_sequence_samples`). Non-refiner models get K independent single-frame samples.
- Scores `kid_global` exactly as `eval_dinov3_kid.py`: DINOv3 ViT-S/16, [CLS ; mean patch], polynomial kernel, block U-statistic with one group per condition, against the inpainted reference `Data/DamagePattern_inpaint_selfcondnoDI/images`. It is the val column of `results.md`: gn 0.119, V10 refiner 0.090, V9 0.096. One difference: val fakes there were not inpainted, and neither are these.
- Also logs `mean_gray`, `bright_frac` (frames with mean > 200), `saturated_frac` (> 1% of pixels at ≤ 5 or ≥ 250, the `results.md` definition) and `tone_jump` (gray levels, along trajectories, as `src/metrics/temporal.py`).
- Writes one JSON line per run to `<result_dir>/kid_log.jsonl` and the same values as `val/*` to the logger. Training RNG state and train/eval mode are restored afterwards.
- No jackknife SE: this is a monitor. Report numbers come from the eval suite.

---

## 6. Flags

| Flag | Default | Job value (A / B) | Meaning |
|---|---|---|---|
| `--refiner_implicit` | False | True / True | v10.5; requires `--use_refiner_residual` |
| `--control_inject_encoder` | False | False / True | Injection point (§2) |
| `--freeze_trunk` | False | True / False | Freeze everything outside `control.` |
| `--trunk_lr_mult` | 0.2 | 0.2 | LR of an unfrozen trunk relative to `--lr` |
| `--refiner_control_drop_prob` | 0.15 | 0.15 | Control off per sample (training) |
| `--control_guidance` | 1.0 | 1.0 | Sampling only (§4.3) |
| `--refiner_state_cache`, `--refiner_state_cache_n` | False, 4 | True, 4 | Offline base hop (§3.2) |
| `--refiner_state_perturb` (+ `bias_min/max`, `gain_jitter`, `lowfreq_std`, `damage_shrink`, `blur_prob`) | False | True | §3.3 |
| `--refiner_tone_loss_weight` | 0.0 | 1.0 | §3.1 |
| `--share_photometric_aug` | False | True | §3.4 |
| `--refiner_shared_lowfreq_noise`, `--refiner_lowfreq_block` | False, 16 | True, 16 | §4.1 |
| `--kid_every_n_steps`, `--kid_k`, `--kid_image_dir` | 0, 8, inpainted dir | 3000, 8 | §5 |

Job script switches (`job/DamagePatternGeneration/train_ADMCFM_refiner_residual.sh`, no new job file): `REFINER_IMPLICIT`, `CONTROL_INJECT_ENCODER`, `FREEZE_TRUNK`. `REFINER_STATE_CACHE`, `REFINER_STATE_PERTURB`, `SHARE_PHOTOMETRIC_AUG` and `REFINER_SHARED_LOWFREQ_NOISE` default to the value of `REFINER_IMPLICIT`. With `REFINER_IMPLICIT=True` the script also sets `MAX_STEPS=15000`, roll probability 0.3 and tone weight 1.0. Each can be overridden with `-v`. `KID_EVERY_N_STEPS` defaults to 3000.

Matched between A and B: data, split, augmentation, state track, cache, perturbation, losses, batch 128, 15000 steps, warmup 1000 + cosine to 0.1, control LR 1e-4, Heun 30 at evaluation, frame 0 from gn, KID seed and K.

---

## 7. Code map

| File | Change |
|---|---|
| `src/model/adm.py` | `ControlBranch`; `ADMUNetCFM(use_control_branch, control_hint_channels, control_delta_g, control_inject_encoder)`; `forward(control_hint, control_scale)` |
| `src/model/cfm/options.py` | Flags, validation, no state input channels in implicit mode, control branch construction, trunk freezing |
| `src/model/cfm/patch_mixin.py` | `lr_cond` is the control hint in implicit mode; control guidance; no `S + r` composition |
| `src/model/cfm/loss_mixin.py` | `_refiner_state_track` (shared, reads the cache), `_refiner_state_condition`, `_perturb_refiner_state`, `_lowpass_tone`, `refiner_implicit_flow_matching_loss` |
| `src/model/cfm/training_mixin.py` | Dispatch, logging (`train/refiner_control_frac`, `train/refiner_tone_loss`), two-LR optimizer |
| `src/model/cfm/ode_mixin.py` | `_refiner_start_noise`, shared low-frequency noise in `sample_refiner_sequence` |
| `src/model/cfm/condition_mixin.py` | `state_prev`, `state_prev2` in `ar_pack` |
| `src/model/cfm/refiner_cache.py` | New: offline base states |
| `src/data/pattern_loop.py`, `src/utils.py` | Shared photometric draw; `attach_state_cache`, cached states in items |
| `src/metrics/train_kid.py` | New: sequence generation (shared with the eval script), periodic KID and exposure stats |
| `train_mixed.py` | Control warm start, cache build before the loaders fork, `PeriodicKID` |
| `argument/mixed.py` | Flags |
| `scripts/eval/generate_eval_samples.py` | Uses the shared generator; records `refiner_implicit` and the noise setting in the npz meta |

---

## 8. Runs

Submitted from the same script (2 CPUs: main process + 1 loader worker, 1 BLAS thread; 15 GB), PBS `gpu_id` 0 and 1 (physical GPUs 4 and 5, see `docs/GPU.md`):

```
qsub -l select=1:ncpus=2:mem=15gb:gpu_id=0 -o log/train_v105A.out -e log/train_v105A.err -N CFM_v105A \
  -v REFINER_IMPLICIT=True,CONTROL_INJECT_ENCODER=False,FREEZE_TRUNK=True,RUN_TAG=v105A,NUM_WORKERS=1,OMP_NUM_THREADS=1,MKL_NUM_THREADS=1 \
  job/DamagePatternGeneration/train_ADMCFM_refiner_residual.sh
qsub -l select=1:ncpus=2:mem=15gb:gpu_id=1 -o log/train_v105B.out -e log/train_v105B.err -N CFM_v105B \
  -v REFINER_IMPLICIT=True,CONTROL_INJECT_ENCODER=True,FREEZE_TRUNK=False,RUN_TAG=v105B,NUM_WORKERS=1,OMP_NUM_THREADS=1,MKL_NUM_THREADS=1 \
  job/DamagePatternGeneration/train_ADMCFM_refiner_residual.sh
```

---

## 9. Evaluation plan and pass marks

Against the V10 refiner, gn and gn_shared, with the suite in `docs/eval/` (test and val, K = 8):

| Metric | Pass mark |
|---|---|
| `tone_jump` | near GT (4-7) |
| Saturated frames | < 2% |
| `persistence`, `crack_keep` | above the V10 refiner (0.59 / 0.50 on test) |
| `flicker_ratio` | closer to 1 than gn_shared (0.41) |
| KID, progression ratio, neighbour contrast | not worse than the V10 refiner |

Ablations for both versions: `Δg` zeroed or shuffled at sampling (if nothing changes, the model copies `S_{t-1}`); with and without the shared low-frequency noise; a `control_guidance` sweep.

**B over A** only if high-DI damage area rises clearly above gn's 0.106 (test, high DI tertile) without KID falling back to gn's level. If B wins only on temporal metrics, A is enough.

## 10. Risks and open items

- **A may not pass enough detail through the control branch** (crack positions). `crack_keep` is the number to watch; if it is low, B or an unfrozen decoder is the answer.
- **B may learn to copy `S_{t-1}`**, since copying lowers the loss. Watch the progression ratio and the `Δg` shuffle ablation. Remedies: higher control dropout, stronger state perturbation.
- **State perturbation ranges are guesses.** Measuring gn vs GT tone and damage shifts (`scripts/eval/measure_base_vs_gt_shift.py`) and fitting the ranges is the next step if the tone does not settle.
- **The cache fixes 4 base samples per frame.** The self hop and the perturbation add variety, but the base hop sees only those 4.
- **Hypothesis 2 of §1** (independent jitter teaching tone jumps) is not tested on its own; `SHARE_PHOTOMETRIC_AUG=False` on an otherwise identical run would test it.
