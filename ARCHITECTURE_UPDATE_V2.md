# ARCHITECTURE_UPDATE_V2

## 1. Executive Summary

This update introduces a major architectural transition for the RC bridge column damage-generation pipeline, moving from legacy recurrent + GAN-centered modeling to a sequence-aware conditional flow framework. The primary motivation is twofold:

- **Mitigate sequential forgetting** in long-horizon hysteresis representation learning.
- **Eliminate adversarial instability** (mode collapse, discriminator-generator imbalance, fragile convergence) observed in SGAN/RSGAN-style training.

From a research perspective, this milestone reframes the project around a **physics-aware sequence latent representation** and a **continuous-time generative process**. From an engineering perspective, it consolidates training/inference flow, improves reproducibility, and provides a cleaner handover path for subsequent model extensions.

---

## 2. Core Architecture Evolution

### Sequence Modeling: Transition to Mamba

The sequence module is upgraded toward a **Mamba-based state-space sequence encoder** to better preserve long-range temporal dependencies from hysteresis trajectories under non-stationary loading patterns.

- **Why Mamba**: linear-time sequence processing, stronger long-context memory retention, and improved stability over conventional gated RNN updates in very long loading histories.
- **Expected outcome**: richer latent condition embedding for downstream damage generation under variable loading paths.

> Engineering note: where full Mamba rollout is phased, interfaces are kept compatible with existing latent-conditioning APIs so the generative backend can be validated independently.

### Generative Framework: CFM + ADM U-Net with adaGN Conditioning

The generative backend is migrated from adversarial training to **Conditional Flow Matching (CFM)** with an **ADM-style U-Net**.

- Target vector field learning objective:
  - $Z_0 \sim \mathcal{N}(0, I)$
  - $Z_1$: target damage image
  - $Z_t = (1-t)Z_0 + tZ_1$
  - $u_t = Z_1 - Z_0$
  - Minimize $\mathbb{E}v_\theta(Z_t, t, c) - u_t^2$
- Conditioning path:
  - timestep embedding (sinusoidal) + condition embedding projected into shared latent space
  - injected into residual blocks using **adaGN** (scale-shift modulation)

This design removes discriminator dependencies while preserving strong conditional controllability.

### Inference: ODE Solver with Latent Space Time-Conditioned Universal Guidance

Sampling is performed by integrating the learned velocity field from noise to image:

- **Primary solver**: Heun (improved stability over Euler at moderate step counts)
- **Guidance concept**: latent-space, time-conditioned control through condition vector $c$ across integration trajectory
- **Practical effect**: smoother generation trajectory and better alignment between condition semantics and output damage morphology

---

## 3. Tensor Dimension Transformations (CRITICAL)

Below is the canonical shape flow (batch-first convention):

```text
Input sequence:                [B, L, C_seq]
-> Sequence Extractor (Mamba): [B, L, D_seq]
-> Temporal pooling/select:    [B, D_latent]

Metadata / condition:
Direction one-hot:             [B, 2]
Damage index (DI):             [B, 1]
Condition concat:              [B, D_cond]

Flow matching branch:
Target image Z1:               [B, C_img, H, W]
Noise Z0:                      [B, C_img, H, W]
Interpolated Zt:               [B, C_img, H, W]
timestep t:                    [B]

ADM U-Net input:
(Zt, t, c) -> velocity vθ:     [B, C_img, H, W]
Target velocity ut = Z1 - Z0:  [B, C_img, H, W]
```

Reference pipeline form:

```text
Input [B, L, C] -> Mamba Extractor -> Latent Condition [B, D]
-> ADM U-Net (CFM) -> Output Velocity [B, C, H, W]
-> ODE Integration -> Generated Image [B, C, H, W]
```

Implementation-level condition examples:

- **Latent-only**: $c = [h]$
- **Latent + direction**: $c = [h, d]$
- **Latent + direction + DI**: $c = [h, d, \mathrm{DI}]$

---

## 4. Loss Functions & Optimization Strategy

### Adaptive Loss Weighting

Training prioritizes flow-matching velocity supervision while optionally combining auxiliary terms (e.g., DI-related consistency) via adaptive or scheduled weighting:

- Main objective: $\mathcal{L}*{FM} = v*\theta(Z_t, t, c) - (Z_1 - Z_0)_2^2$
- Optional composite form: $\mathcal{L} = \lambda_{fm}\mathcal{L}*{FM} + \lambda*{aux}\mathcal{L}_{aux}$

Weighting can be ramped or normalized to prevent domination by high-variance auxiliary losses.

### Optimizer: AdamW

- **AdamW** for decoupled weight decay and improved generalization stability
- Recommended defaults:
  - `lr`: $1\times10^{-4}$ (task-dependent)
  - `betas`: $(0.9, 0.999)$
  - `weight_decay`: tuned per model scale

### Learning Rate Schedule: Cosine Annealing with Warmup

- Warmup for initial optimization stabilization
- Cosine decay toward `min_lr_ratio`
- Expected behavior:
  - reduced early divergence risk
  - smoother late-stage convergence
  - more reliable overfit diagnostics on single-sample sanity runs

---

## 5. Paradigm Shift in Evaluation Metrics

The project shifts from low-level scalar agreement (area-based errors/MSE-only checks) toward **representation-aware verification** to reduce shortcut learning.

- **Legacy risk**: models can optimize pixel-level proxies while failing to preserve structural/topological semantics of damage evolution.
- **Updated direction**: incorporate **DINO v3 semantic and topological verification**:
  - semantic consistency between generated and target damage morphology
  - topology-sensitive feature alignment over crack/region structure
- **Practical recommendation**:
  - keep PSNR/SSIM/FID as secondary diagnostics
  - use DINO-based similarity and topology checks as primary quality gates for research conclusions

---

## 6. Pseudocode (Training and Inference)

### End-to-End Training (CFM + Conditional Sequence Latent)

```text
Given:
  Sequence input X_seq [B, L, C_seq]
  Target damage image Z1 [B, C_img, H, W]
  Direction d [B, 2], damage index DI [B, 1]

Initialize:
  Sequence extractor f_seq (Mamba-compatible interface)
  Velocity network v_theta (ADM U-Net with adaGN)
  Optimizer AdamW, scheduler warmup + cosine

for each training step:
  h_seq = f_seq(X_seq)                              # [B, L, D_seq]
  h = temporal_select_or_pool(h_seq)               # [B, D_latent]
  c = concat(h, d, DI)                             # [B, D_cond]

  t ~ Uniform(0, 1)                                # [B]
  Z0 ~ N(0, I)                                     # [B, C_img, H, W]
  Zt = (1 - t) * Z0 + t * Z1
  u_t = Z1 - Z0

  v_pred = v_theta(Zt, t, c)
  L_fm = MSE(v_pred, u_t)

  L_total = lambda_fm * L_fm + lambda_aux * L_aux  # optional auxiliary branch
  Backprop(L_total)
  Optimizer.step()
  Scheduler.step()
```

### Inference with ODE Solver (Heun)

```text
Given condition c [B, D_cond], num_steps = N:
  z = sample_gaussian([B, C_img, H, W])  # z at t=0
  dt = 1 / N

  for i in 0..N-1:
    t_i = i / N
    v_i = v_theta(z, t_i, c)

    # Heun predictor-corrector
    z_pred = z + dt * v_i
    t_next = (i + 1) / N
    v_next = v_theta(z_pred, t_next, c)
    z = z + 0.5 * dt * (v_i + v_next)

  return z  # generated image approximation to Z1
```

### Overfit Sanity Check (Single Sample)

```text
Select one training sample index k
Repeat same sample for train and validation
Train CFM for 100 epochs (or more) with fixed condition pathway
Monitor:
  train/fm_loss -> should decrease toward near-zero
  val/psnr, val/ssim -> should increase markedly
Use GT vs inference plotting as final qualitative verification
```

---

## 7. Deprecations

The following legacy components are considered obsolete for the new baseline:

- **Adversarial discriminators** as primary training drivers:
  - SGAN / RSGAN / RaSGAN discriminator-centric optimization
- Legacy diffusion trial scripts and disconnected blocks not used by current CFM path
- Any training/evaluation scripts tightly coupled to deprecated discriminator checkpoints

Deprecation policy:

- Keep read-only archival compatibility where needed for reproducibility.
- Do not use deprecated modules for new benchmark claims.

---

## 8. Known Issues & Future Work

### Known Issues

- **High-noise endpoint instability** near $t \approx 1$:
  - occasional gradient/trajectory sensitivity
  - can degrade generated quality under coarse ODE step settings
- Metric mismatch risk:
  - low flow-matching loss does not always imply low FID
- Runtime dependencies:
  - FID metric requires `torch-fidelity`; missing package causes runtime failure

### Future Work

- Full integration of **Mamba sequence extractor** as default production path
- Time-adaptive ODE step scheduling for faster yet stable inference
- DINO v3 metric suite integration into automated validation reports
- Calibration of condition composition ($[h]$, $[h,d]$, $[h,d,DI]$) under controlled ablations
- Robustness experiments under cross-experiment and cross-loading generalization settings

---

## Closing Note

This V2 update establishes a cleaner research and engineering baseline: sequence-aware conditioning, flow-based generation, and stronger evaluation rigor. It is intended to support both immediate reproducibility and forward-compatible extension toward higher-fidelity structural damage intelligence workflows.